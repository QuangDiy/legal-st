"""Merge several evaluate_retrieval.py runs into GreenNode-style comparison tables.

Each input is a results directory (or its results.json) written by
evaluate_retrieval.py / train_embedding.py. Prefix with ``label=`` to name a row:

    python scripts/compare_results.py \\
        tiny=results/gn-trvn/bert-tiny \\
        base=results/gn-trvn/bert-base \\
        bamibert=results/gn-trvn/bamibert \\
        greennode=results/gn-trvn/greennode-large \\
        --dims all --output results/gn-trvn/comparison.md

``--dims max`` (default) keeps only each model's full embedding size,
``--dims all`` adds one row per Matryoshka truncation, and ``--dims 256``
keeps a single size. BM25 rows are included when present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from legal_st.retrieval import benchmark_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval results across models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("runs", nargs="+", help="[label=]path/to/results(.json)")
    parser.add_argument("--k", type=int, default=5,
                        help="Cutoff for MAP/MRR/NDCG/Recall (needs map_at_k == k)")
    parser.add_argument("--hit-ks", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--dims", default="max", help="max | all | <int>")
    parser.add_argument("--output", default=None, help="Optional markdown output path")
    return parser.parse_args()


def _load_run(spec: str) -> tuple[str, dict]:
    label, _, path_text = spec.rpartition("=")
    path = Path(path_text)
    if path.is_dir():
        path = path / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "datasets" not in payload:
        raise ValueError(f"{path} is not a combined results.json (missing 'datasets')")
    return label or Path(payload["model_path"]).name, payload


def _select_rows(rows: list[dict], dims: str) -> list[dict]:
    bm25 = [row for row in rows if row.get("truncate_dim") == -1]
    dense = [row for row in rows if row.get("truncate_dim") != -1]
    if dims == "all" or not dense:
        return bm25 + dense
    if dims == "max":
        return bm25 + [max(dense, key=lambda row: row["truncate_dim"])]
    return bm25 + [row for row in dense if row["truncate_dim"] == int(dims)]


def main() -> None:
    args = parse_args()
    by_dataset: dict[str, list[tuple[str, dict]]] = {}
    seen_bm25: set[str] = set()

    for spec in args.runs:
        label, payload = _load_run(spec)
        for entry in payload["datasets"]:
            labelled = by_dataset.setdefault(entry["dataset"], [])
            for row in _select_rows(entry["results"], args.dims):
                if row.get("truncate_dim") == -1:
                    # BM25 does not depend on the model; show it once per dataset.
                    if entry["dataset"] in seen_bm25:
                        continue
                    seen_bm25.add(entry["dataset"])
                    labelled.append(("BM25", row))
                    continue
                suffix = "" if args.dims == "max" else f" @{row['truncate_dim']}"
                labelled.append((f"{label}{suffix}", row))

    sections = []
    for dataset, labelled in by_dataset.items():
        sections.append(
            f"### {dataset}\n\n{benchmark_to_markdown(labelled, args.k, args.hit_ks)}\n"
        )
    markdown = "\n".join(sections)
    print(markdown)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
