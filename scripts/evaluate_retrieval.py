from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentence_transformers import SentenceTransformer

from legal_st.config import load_config
from legal_st.retrieval import (
    evaluate_dense_retrieval,
    results_to_markdown,
    write_results_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval performance")
    parser.add_argument("--model-path", required=True, help="Local path or HF model id")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for JSON and markdown results"
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
        help="Optional extra corpus docs beyond gold matches for quick tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model = SentenceTransformer(args.model_path)
    model.max_seq_length = config.max_seq_length

    rows = evaluate_dense_retrieval(
        model=model,
        config=config,
        truncate_dims=config.truncate_dims,
        limit_queries=args.limit_queries,
        extra_corpus_docs=args.extra_corpus_docs,
    )
    write_results_artifacts(
        output_dir=args.output_dir,
        rows=rows,
        config=config,
        model_path=args.model_path,
    )

    print(results_to_markdown(rows, config))
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
