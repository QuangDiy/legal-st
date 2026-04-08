"""Standalone retrieval evaluation script.

With a config file:
    uv run scripts/evaluate_retrieval.py \\
        --model-path outputs/bert-tiny-stage2-sbert \\
        --config configs/bert-tiny-stage2-hf.yaml

Without a config file (uses built-in defaults + two standard eval datasets):
    uv run scripts/evaluate_retrieval.py \\
        --model-path dangvantuan/vietnamese-embedding

CLI overrides (apply on top of config or defaults):
    --max-seq-length, --eval-batch-size, --truncate-dims, --recall-at-k
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentence_transformers import SentenceTransformer

from legal_st.config import ExperimentConfig, load_config
from legal_st.retrieval import (
    evaluate_dense_retrieval_datasets,
    results_to_markdown,
    results_to_readme,
    write_multi_results_artifacts,
)

# ---------------------------------------------------------------------------
# Built-in eval suite used when no --config is given
# ---------------------------------------------------------------------------
_DEFAULT_EVAL_DATASETS = [
    {
        "dataset": "taidng/UIT-ViQuAD2.0",
        "name": "UIT-ViQuAD2.0",
        "format": "squad",
        "split": "validation",
    },
    {
        "dataset": "GreenNode/zalo-ai-legal-text-retrieval-vn",
        "name": "Zalo-Legal",
        "corpus_config": "corpus",
        "queries_config": "queries",
        "labels_config": "qrels",
        "split": "test",
    },
]


def _default_config(model_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="eval",
        model_name=model_name,
        output_dir=".",
        eval_datasets=_DEFAULT_EVAL_DATASETS,
        top_k=[1, 3, 5, 10],
        recall_at_k=[5, 10, 100],
        map_at_k=100,
        eval_batch_size=32,
        max_seq_length=512,
        truncate_dims=[],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense retrieval performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-path", required=True, help="Local path or HF model id")
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML experiment config (optional — omit to use built-in defaults)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for results (default: <model-path>/retrieval_eval or ./retrieval_eval)",
    )
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--extra-corpus-docs", type=int, default=None)
    # CLI overrides
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Override max_seq_length from config")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Override eval_batch_size from config")
    parser.add_argument("--truncate-dims", type=int, nargs="+", default=None,
                        help="Override truncate_dims from config (e.g. --truncate-dims 768 512 256)")
    parser.add_argument("--recall-at-k", type=int, nargs="+", default=None,
                        help="Override recall_at_k from config (e.g. --recall-at-k 5 10 100)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        print(f"No --config provided. Using built-in eval defaults.")
        config = _default_config(args.model_path)

    # Apply CLI overrides
    if args.max_seq_length is not None:
        config.max_seq_length = args.max_seq_length
    if args.eval_batch_size is not None:
        config.eval_batch_size = args.eval_batch_size
    if args.truncate_dims is not None:
        config.truncate_dims = args.truncate_dims
    if args.recall_at_k is not None:
        config.recall_at_k = args.recall_at_k

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Loading model: {args.model_path}")
    model = SentenceTransformer(args.model_path)

    # Only override max_seq_length when the user explicitly asked for it
    # (via --max-seq-length or --config). When using the built-in defaults,
    # keep the model's native value so models like PhoBERT (max 256) don't
    # receive out-of-range position indices.
    if args.max_seq_length is not None:
        model.max_seq_length = args.max_seq_length
    elif args.config is not None:
        model.max_seq_length = min(config.max_seq_length, model.max_seq_length)

    print(f"max_seq_length: {model.max_seq_length}")

    dataset_results = evaluate_dense_retrieval_datasets(
        model=model,
        config=config,
        truncate_dims=config.truncate_dims or None,
        limit_queries=args.limit_queries,
        extra_corpus_docs=args.extra_corpus_docs,
    )

    model_path = Path(args.model_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (model_path / "retrieval_eval" if model_path.exists() else Path("retrieval_eval"))
    )
    write_multi_results_artifacts(
        output_dir=output_dir,
        dataset_results=dataset_results,
        config=config,
        model_path=str(args.model_path),
    )

    if model_path.exists():
        readme_path = model_path / "README.md"
        readme_path.write_text(
            results_to_readme(dataset_results, config, str(args.model_path)),
            encoding="utf-8",
        )
        print(f"README updated : {readme_path}")

    for name, rows in dataset_results:
        print(f"\n--- {name} ---")
        print(results_to_markdown(rows, config))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
