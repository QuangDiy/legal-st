from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentence_transformers import SentenceTransformer

from legal_st.config import load_config
from legal_st.retrieval import (
    evaluate_dense_retrieval_datasets,
    results_to_markdown,
    results_to_readme,
    write_multi_results_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval performance")
    parser.add_argument("--model-path", required=True, help="Local path or HF model id")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--output-dir", default=None, help="Directory for results (default: <model-path>/retrieval_eval)"
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Optional query limit for quick tests",
    )
    parser.add_argument(
        "--extra-corpus-docs",
        type=int,
        default=None,
        help="Optional extra corpus docs beyond gold matches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = SentenceTransformer(args.model_path)
    model.max_seq_length = config.max_seq_length

    dataset_results = evaluate_dense_retrieval_datasets(
        model=model,
        config=config,
        truncate_dims=config.truncate_dims,
        limit_queries=args.limit_queries,
        extra_corpus_docs=args.extra_corpus_docs,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path) / "retrieval_eval"
    write_multi_results_artifacts(
        output_dir=output_dir,
        dataset_results=dataset_results,
        config=config,
        model_path=args.model_path,
    )

    readme_path = Path(args.model_path) / "README.md"
    readme_path.write_text(
        results_to_readme(dataset_results, config, args.model_path),
        encoding="utf-8",
    )

    for name, rows in dataset_results:
        print(f"\n--- {name} ---")
        print(results_to_markdown(rows, config))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
