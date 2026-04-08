from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import threading
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentence_transformers import losses
from sentence_transformers import SentenceTransformer
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.fit_mixin import SaveModelCallback
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback, TrainerControl, TrainerState
from legal_st.config import dump_config, load_config
from legal_st.data import (
    load_triplet_records,
    records_to_input_examples,
    split_records_by_query,
)
from legal_st.evaluation import LossEvaluator
from legal_st.modeling import build_sentence_transformer
from legal_st.retrieval import (
    evaluate_dense_retrieval_datasets,
    results_to_markdown,
    results_to_readme,
    write_multi_results_artifacts,
)
from legal_st.utils import ensure_dir, set_seed


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when eval metric stops improving for *patience* evaluations."""

    def __init__(self, patience: int, primary_metric: str, greater_is_better: bool = False) -> None:
        self.patience = patience
        self.primary_metric = primary_metric
        self.greater_is_better = greater_is_better
        self._best: float | None = None
        self._wait: int = 0

    def on_evaluate(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ) -> None:
        key = f"eval_{self.primary_metric}"
        score = metrics.get(key)
        if score is None:
            return

        improved = (
            self._best is None
            or (self.greater_is_better and score > self._best)
            or (not self.greater_is_better and score < self._best)
        )

        if improved:
            self._best = score
            self._wait = 0
        else:
            self._wait += 1
            log(
                f"[EarlyStopping] No improvement for {self._wait}/{self.patience} evals "
                f"(best {key}={self._best:.6f}, current={score:.6f})"
            )
            if self._wait >= self.patience:
                log(f"[EarlyStopping] Triggered — stopping training.")
                control.should_training_stop = True


def dataloader_to_hf_dataset(dataloader: NoDuplicatesDataLoader) -> Dataset:
    """Drain a NoDuplicatesDataLoader into a HuggingFace Dataset."""
    original_collate = dataloader.collate_fn
    dataloader.collate_fn = lambda batch: batch  # identity – keep InputExamples raw
    texts_list: list[tuple[str, ...]] = []
    labels_list: list[float] = []
    for batch in dataloader:
        for example in batch:
            texts_list.append(tuple(example.texts))
            labels_list.append(example.label)
    dataloader.collate_fn = original_collate

    n_texts = len(texts_list[0])
    data: dict[str, list] = {
        f"sentence_{i}": [t[i] for t in texts_list] for i in range(n_texts)
    }
    if set(labels_list) != {0}:
        data["label"] = labels_list
    return Dataset.from_dict(data)


def get_rank() -> int:
    value = os.getenv("RANK")
    return int(value) if value is not None else 0


def get_local_rank() -> int:
    value = os.getenv("LOCAL_RANK")
    return int(value) if value is not None else 0


def is_main_process() -> bool:
    return get_rank() == 0


def log(message: str) -> None:
    if is_main_process():
        print(message)


def run_post_train_retrieval_eval(output_dir: Path, config) -> None:
    eval_model = SentenceTransformer(str(output_dir))
    eval_model.max_seq_length = min(config.max_seq_length, eval_model.max_seq_length)
    dataset_results = evaluate_dense_retrieval_datasets(
        model=eval_model,
        config=config,
        truncate_dims=config.truncate_dims,
        limit_queries=config.retrieval_eval_limit_queries,
        extra_corpus_docs=config.retrieval_eval_extra_corpus_docs,
    )
    eval_output_dir = output_dir / "retrieval_eval"
    write_multi_results_artifacts(
        output_dir=eval_output_dir,
        dataset_results=dataset_results,
        config=config,
        model_path=str(output_dir),
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        results_to_readme(dataset_results, config, str(output_dir)),
        encoding="utf-8",
    )
    for name, rows in dataset_results:
        log(f"\n--- {name} ---")
        log(results_to_markdown(rows, config))
    log(f"Saved retrieval evaluation to: {eval_output_dir}")
    log(f"Updated model card at: {readme_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Vietnamese legal embedding model"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--hf-token", help="Optional Hugging Face token override")
    parser.add_argument("--hf-repo-id", help="Optional Hugging Face repo override")
    return parser.parse_args()


class HubSync:
    def __init__(
        self,
        repo_id: str,
        output_dir: Path,
        checkpoint_dir: Path,
        token: str | None = None,
        private: bool = False,
        poll_interval: float = 10.0,
    ) -> None:
        self.repo_id = repo_id
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.token = token
        self.private = private
        self.poll_interval = poll_interval
        self.api = HfApi(token=token)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.uploaded_checkpoints: set[str] = set()
        self.output_signature: tuple[tuple[str, int, int], ...] | None = None

    def start(self) -> None:
        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            private=self.private,
            exist_ok=True,
        )
        self.thread = threading.Thread(
            target=self._run, name="hf-hub-sync", daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def _run(self) -> None:
        while not self.stop_event.wait(self.poll_interval):
            self.sync_once()
        self.sync_once()

    def sync_once(self) -> None:
        self._sync_checkpoints()
        self._sync_output_dir()

    def _sync_checkpoints(self) -> None:
        if not self.checkpoint_dir.exists():
            return

        checkpoints = sorted(
            [
                path
                for path in self.checkpoint_dir.glob("checkpoint-*")
                if path.is_dir()
            ],
            key=lambda path: int(path.name.split("-")[-1]),
        )
        for checkpoint in checkpoints:
            if checkpoint.name in self.uploaded_checkpoints:
                continue
            if not any(checkpoint.iterdir()):
                continue

            self.api.upload_folder(
                repo_id=self.repo_id,
                repo_type="model",
                folder_path=str(checkpoint),
                path_in_repo=f"checkpoints/{checkpoint.name}",
                commit_message=f"Upload {checkpoint.name}",
            )
            self.uploaded_checkpoints.add(checkpoint.name)
            print(f"Uploaded {checkpoint.name} to Hugging Face Hub: {self.repo_id}")

    def _sync_output_dir(self) -> None:
        marker = self.output_dir / "modules.json"
        if not marker.exists():
            return

        files = sorted(
            path
            for path in self.output_dir.rglob("*")
            if path.is_file()
            and "checkpoints" not in path.relative_to(self.output_dir).parts
        )
        signature = tuple(
            (
                str(path.relative_to(self.output_dir)).replace("\\", "/"),
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in files
        )
        if not signature or signature == self.output_signature:
            return

        self.api.upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=str(self.output_dir),
            path_in_repo=".",
            ignore_patterns=["checkpoints/*"],
            commit_message="Update exported model",
        )
        self.output_signature = signature
        print(f"Uploaded model snapshot to Hugging Face Hub: {self.repo_id}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)

    local_rank = get_local_rank()
    if torch.cuda.is_available() and os.getenv("LOCAL_RANK") is not None:
        torch.cuda.set_device(local_rank)

    hf_token = (
        args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    hf_repo_id = args.hf_repo_id or config.hf_repo_id
    if not hf_token:
        hf_repo_id = None
        log("No HF token found — HuggingFace Hub sync disabled.")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if config.precision == "bf16":
            torch.set_autocast_dtype("cuda", torch.bfloat16)
        elif config.precision == "fp16":
            torch.set_autocast_dtype("cuda", torch.float16)

    use_amp = config.use_amp and config.precision != "fp32"

    output_dir = ensure_dir(config.output_dir)
    if is_main_process():
        dump_config(config, output_dir / "resolved_config.yaml")
        shutil.copy2(args.config, output_dir / Path(args.config).name)

    records = load_triplet_records(config)
    train_records, validation_records = split_records_by_query(
        records,
        validation_size=config.validation_size,
        seed=config.seed,
    )

    log(f"Loaded {len(records):,} triplets from {config.train_dataset}")
    log(f"Train triplets: {len(train_records):,}")
    log(f"Validation triplets: {len(validation_records):,}")

    model = build_sentence_transformer(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        pooling=config.pooling,
        normalize_embeddings=config.normalize_embeddings,
    )

    train_examples = records_to_input_examples(train_records)
    train_dataloader = NoDuplicatesDataLoader(
        train_examples, batch_size=config.train_batch_size
    )

    inner_loss = (
        losses.CachedMultipleNegativesRankingLoss(model)
        if config.use_cached_mnrl
        else losses.MultipleNegativesRankingLoss(model)
    )
    train_loss = (
        losses.MatryoshkaLoss(model, inner_loss, matryoshka_dims=config.matryoshka_dims)
        if config.matryoshka_dims
        else inner_loss
    )

    evaluator = None
    if validation_records:
        selected_validation_records = validation_records
        if (
            config.validation_subset is not None
            and len(selected_validation_records) > config.validation_subset
        ):
            selected_validation_records = selected_validation_records[
                : config.validation_subset
            ]

        validation_examples = records_to_input_examples(selected_validation_records)
        evaluator = LossEvaluator(
            examples=validation_examples,
            loss_model=train_loss,
            name="validation",
            batch_size=config.eval_batch_size,
        )

    warmup_steps = math.ceil(
        len(train_dataloader) * config.num_train_epochs * config.warmup_ratio
    )
    checkpoint_dir = output_dir / "checkpoints"
    hub_sync = None
    if is_main_process() and hf_repo_id is not None:
        hub_sync = HubSync(
            repo_id=hf_repo_id,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            token=hf_token,
            private=config.hf_private,
        )
        hub_sync.start()

    log(f"Warmup steps: {warmup_steps}")
    log(f"Output dir: {output_dir}")
    log(f"Precision: {config.precision}")
    if hf_repo_id is not None:
        log(f"HF repo: {hf_repo_id}")
    if os.getenv("LOCAL_RANK") is not None:
        log("Distributed mode: torchrun/DDP")

    # Convert the NoDuplicatesDataLoader to a HuggingFace Dataset so that
    # SentenceTransformerTrainer can handle batching, DDP sharding, and AMP.
    train_dataset = dataloader_to_hf_dataset(train_dataloader)
    log(f"Train dataset converted: {len(train_dataset):,} examples")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        # Precision: honour the config field instead of conflating use_amp → fp16
        bf16=(config.precision == "bf16"),
        fp16=(config.precision == "fp16"),
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        max_grad_norm=1.0,
        eval_strategy="steps" if evaluator is not None else "no",
        eval_steps=config.evaluation_steps if evaluator is not None else 0,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=config.checkpoint_save_steps,
        save_total_limit=config.checkpoint_save_total_limit,
        metric_for_best_model=(
            f"eval_{evaluator.primary_metric}" if evaluator is not None else None
        ),
        greater_is_better=False,
        load_best_model_at_end=False,
        seed=config.seed,
        disable_tqdm=False,
        report_to="none",
    )

    callbacks = []
    if evaluator is not None:
        callbacks.append(SaveModelCallback(str(output_dir), evaluator, save_best_model=True))
        if config.early_stopping_patience is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    patience=config.early_stopping_patience,
                    primary_metric=evaluator.primary_metric,
                    greater_is_better=evaluator.greater_is_better,
                )
            )
            log(f"Early stopping: patience={config.early_stopping_patience} evals")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
        callbacks=callbacks,
    )

    try:
        trainer.train()
    finally:
        if hub_sync is not None:
            hub_sync.stop()

    # Reload the best checkpoint saved by SaveModelCallback so that
    # the in-memory model matches what is on disk (matters when early
    # stopping fires or training overshoots the best eval step).
    best_marker = output_dir / "modules.json"
    if is_main_process() and best_marker.exists():
        log("Reloading best model from output_dir into memory...")
        best_state = SentenceTransformer(str(output_dir)).state_dict()
        model.load_state_dict(best_state)
        log("Best model reloaded.")

    if is_main_process() and config.run_retrieval_eval_after_train:
        log("Running retrieval evaluation on saved best/final model...")
        run_post_train_retrieval_eval(output_dir, config)
        if hub_sync is not None:
            hub_sync.sync_once()

    log("Training finished.")
    log(f"Best or final model saved to: {output_dir}")


if __name__ == "__main__":
    main()
