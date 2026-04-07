from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import threading
from pathlib import Path

import torch
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentence_transformers import losses
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.evaluation import TripletEvaluator

from legal_st.config import dump_config, load_config
from legal_st.data import (
    build_triplet_evaluator_payload,
    load_triplet_records,
    records_to_input_examples,
    split_records_by_query,
)
from legal_st.modeling import build_sentence_transformer
from legal_st.utils import ensure_dir, set_seed


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
        anchors, positives, negatives = build_triplet_evaluator_payload(
            validation_records,
            subset_size=config.validation_subset,
        )
        evaluator = TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            name="validation",
            batch_size=config.eval_batch_size,
        )

    warmup_steps = math.ceil(
        len(train_dataloader) * config.num_train_epochs * config.warmup_ratio
    )
    checkpoint_dir = output_dir / "checkpoints"
    hub_sync = None
    if is_main_process() and (config.hf_push_on_save or hf_repo_id is not None):
        if hf_repo_id is None:
            raise ValueError("hf_repo_id is required when Hugging Face sync is enabled")
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

    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=config.num_train_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": config.learning_rate},
            weight_decay=config.weight_decay,
            output_path=str(output_dir),
            save_best_model=evaluator is not None,
            use_amp=use_amp,
            checkpoint_path=str(checkpoint_dir),
            checkpoint_save_steps=config.checkpoint_save_steps,
            checkpoint_save_total_limit=config.checkpoint_save_total_limit,
            evaluation_steps=config.evaluation_steps if evaluator is not None else 0,
            show_progress_bar=True,
        )
    finally:
        if hub_sync is not None:
            hub_sync.stop()

    log("Training finished.")
    log(f"Best or final model saved to: {output_dir}")


if __name__ == "__main__":
    main()
