"""Controlled GN-TRVN retrieval evaluation over document token windows.

Example:
    python scripts/evaluate_context_lengths.py \
      --model-path QuangDuy/gn-trvn-bert-base-stage2 \
      --config configs/gn-trvn-bert-base-stage2.yaml \
      --output-dir results/gn-trvn/context-length-base

The checkpoint, query embeddings, corpus, qrels, embedding dimension, and
ranking procedure stay fixed. Only the document encoding window changes.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from sentence_transformers import util

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from legal_st.config import load_config
from legal_st.data import load_retrieval_dataset_from_spec
from legal_st.modeling import load_sentence_transformer
from legal_st.retrieval import average_precision_at_k, ndcg_at_k
from legal_st.utils import ensure_dir, safe_max_seq_length, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="One fine-tuned SentenceTransformer checkpoint")
    parser.add_argument("--config", required=True, help="GN-TRVN evaluation YAML")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="GN-TRVN")
    parser.add_argument("--document-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--query-length", type=int, default=512,
                        help="Fixed query window across all document-window runs")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def token_lengths(tokenizer, texts: list[str], batch_size: int = 128) -> list[int]:
    """Count special tokens and full, untruncated input tokens."""
    lengths = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[start:start + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in inputs["input_ids"])
    return lengths


def query_groups(
    query_ids: list[str], qrels: dict[str, dict[str, int]],
    corpus_lengths: dict[str, int],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Freeze length strata before evaluation using the relevant document length."""
    groups = {name: [] for name in ("all", "le_512", "gt_512", "513_1024", "1025_2048", "gt_2048")}
    for index, query_id in enumerate(query_ids):
        relevant_lengths = [
            corpus_lengths[doc_id]
            for doc_id, relevance in qrels[query_id].items()
            if relevance > 0 and doc_id in corpus_lengths
        ]
        if not relevant_lengths:
            raise ValueError(f"No relevant corpus document for query {query_id}")
        # GN-TRVN has one relevant table per query. For multi-positive datasets,
        # use the longest relevant document and record that rule in the output.
        relevant_length = max(relevant_lengths)
        groups["all"].append(index)
        if relevant_length <= 512:
            groups["le_512"].append(index)
        else:
            groups["gt_512"].append(index)
            if relevant_length <= 1024:
                groups["513_1024"].append(index)
            elif relevant_length <= 2048:
                groups["1025_2048"].append(index)
            else:
                groups["gt_2048"].append(index)
    return groups, {name: len(indices) for name, indices in groups.items()}


def summarize_hits(
    hits: list[list[dict]], corpus_ids: list[str], query_ids: list[str],
    qrels: dict[str, dict[str, int]], groups: dict[str, list[int]],
) -> dict[str, dict[str, float | int | None]]:
    per_query = []
    for index, query_id in enumerate(query_ids):
        ranked_ids = [corpus_ids[hit["corpus_id"]] for hit in hits[index]]
        relevant = {doc_id for doc_id, score in qrels[query_id].items() if score > 0}
        per_query.append({
            "MAP@5": average_precision_at_k(ranked_ids, relevant, 5),
            "NDCG@5": ndcg_at_k(ranked_ids, relevant, 5),
            "Recall@5": sum(doc_id in relevant for doc_id in ranked_ids[:5]) / len(relevant),
        })

    summary = {}
    for name, indices in groups.items():
        summary[name] = {"n_queries": len(indices)}
        for metric in ("MAP@5", "NDCG@5", "Recall@5"):
            summary[name][metric] = (
                sum(per_query[index][metric] for index in indices) / len(indices)
                if indices else None
            )
    return summary


def markdown_report(payload: dict) -> str:
    lines = [
        "# Controlled document-window retrieval evaluation",
        "",
        f"Checkpoint: `{payload['model_path']}`  ",
        f"Dataset: `{payload['dataset']}` ({payload['split']})  ",
        f"Queries: {payload['n_queries']}; corpus documents: {payload['n_corpus']}  ",
        f"Fixed query window: {payload['query_length']} tokens; full embedding dimension: {payload['embedding_dim']}  ",
        "Only the document token window varies. The ranking corpus and query set are identical in every row.",
        "",
        "| Document window | Query group | Queries | MAP@5 | NDCG@5 | Recall@5 |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for window, result in payload["results"].items():
        for group, values in result.items():
            formatted = ["--" if values[key] is None else f"{100 * values[key]:.2f}"
                         for key in ("MAP@5", "NDCG@5", "Recall@5")]
            lines.append(f"| {window} | {group} | {values['n_queries']} | " + " | ".join(formatted) + " |")
    lines.extend([
        "",
        "Query groups are fixed using the untruncated token length of the longest relevant document.",
        "`gt_512` is the primary long-document subgroup; bins are disjoint except for `all` and `gt_512`.",
        "Scores are multiplied by 100. Empty groups are shown as `--`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    lengths = args.document_lengths
    if len(set(lengths)) != len(lengths) or any(length <= 0 for length in lengths):
        raise ValueError("Document lengths must be unique positive integers")
    if args.query_length <= 0 or args.query_length > min(lengths):
        raise ValueError("Query length must be positive and no greater than the shortest document window")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    config = load_config(args.config)
    spec = next((item for item in config.eval_datasets if item.get("name") == args.dataset_name), None)
    if spec is None:
        raise ValueError(f"Dataset {args.dataset_name!r} is absent from {args.config}")
    corpus, queries, qrels = load_retrieval_dataset_from_spec(spec)
    if not corpus or not queries:
        raise ValueError("Evaluation dataset has no documents or labeled queries")
    corpus_ids, query_ids = list(corpus), list(queries)
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
    query_texts = [queries[query_id] for query_id in query_ids]

    model = load_sentence_transformer(args.model_path, config.attn_implementation)
    model.eval()
    model.truncate_dim = None
    embedding_dim = model.get_sentence_embedding_dimension()
    if embedding_dim is None:
        raise ValueError("Could not determine the model's full embedding dimension")
    model.truncate_dim = embedding_dim
    tokenizer = model[0].tokenizer
    corpus_lengths = dict(zip(corpus_ids, token_lengths(tokenizer, corpus_texts)))
    query_token_lengths = token_lengths(tokenizer, query_texts)
    groups, group_sizes = query_groups(query_ids, qrels, corpus_lengths)

    model.max_seq_length = args.query_length
    if safe_max_seq_length(model) != args.query_length:
        raise ValueError("The requested query window exceeds this model's positional limit")
    with torch.inference_mode():
        query_embeddings = model.encode(
            query_texts, batch_size=args.batch_size, convert_to_tensor=True,
            normalize_embeddings=True, show_progress_bar=True,
        )

    output_dir = ensure_dir(args.output_dir)
    payload = {
        "model_path": args.model_path,
        "config": str(Path(args.config).resolve()),
        "dataset": args.dataset_name,
        "dataset_id": spec["dataset"],
        "split": spec.get("split", "test"),
        "n_queries": len(query_ids),
        "n_corpus": len(corpus_ids),
        "query_length": args.query_length,
        "queries_over_query_window": sum(length > args.query_length for length in query_token_lengths),
        "embedding_dim": embedding_dim,
        "batch_size": args.batch_size,
        "document_lengths": lengths,
        "corpus_documents_over_512": sum(length > 512 for length in corpus_lengths.values()),
        "group_sizes": group_sizes,
        "relevant_document_group_rule": "longest relevant document, full token length including special tokens",
        "results": {},
    }
    max_k = max(5, config.map_at_k, *config.top_k, *config.recall_at_k)
    for length in lengths:
        model.max_seq_length = length
        if safe_max_seq_length(model) != length:
            raise ValueError(f"Document window {length} exceeds this model's positional limit")
        print(f"Evaluating {args.dataset_name} at document window {length} tokens")
        with torch.inference_mode():
            corpus_embeddings = model.encode(
                corpus_texts, batch_size=args.batch_size, convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=True,
            )
            hits = util.semantic_search(
                query_embeddings, corpus_embeddings, top_k=max_k,
                score_function=util.dot_score,
            )
        payload["results"][str(length)] = summarize_hits(hits, corpus_ids, query_ids, qrels, groups)
        save_json(payload, output_dir / "results.json")
        (output_dir / "results.md").write_text(markdown_report(payload), encoding="utf-8")
        del corpus_embeddings, hits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Saved controlled length comparison to {output_dir}")


if __name__ == "__main__":
    main()
