from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import torch

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Vietnamese legal embedding model"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)

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
    dump_config(config, output_dir / "resolved_config.yaml")
    shutil.copy2(args.config, output_dir / Path(args.config).name)

    records = load_triplet_records(config)
    train_records, validation_records = split_records_by_query(
        records,
        validation_size=config.validation_size,
        seed=config.seed,
    )

    print(f"Loaded {len(records):,} triplets from {config.train_dataset}")
    print(f"Train triplets: {len(train_records):,}")
    print(f"Validation triplets: {len(validation_records):,}")

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
    print(f"Warmup steps: {warmup_steps}")
    print(f"Output dir: {output_dir}")
    print(f"Precision: {config.precision}")

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
        checkpoint_path=str(output_dir / "checkpoints"),
        checkpoint_save_steps=config.checkpoint_save_steps,
        checkpoint_save_total_limit=config.checkpoint_save_total_limit,
        evaluation_steps=config.evaluation_steps if evaluator is not None else 0,
        show_progress_bar=True,
    )

    print("Training finished.")
    print(f"Best or final model saved to: {output_dir}")


if __name__ == "__main__":
    main()
