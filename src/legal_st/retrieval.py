from __future__ import annotations

import gc
import math
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util
from tabulate import tabulate

from .config import ExperimentConfig
from .data import load_retrieval_dataset
from .utils import ensure_dir, save_json


def average_precision_at_k(
    ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int
) -> float:
    if not relevant_doc_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            precision_sum += hits / rank

    denominator = min(len(relevant_doc_ids), k)
    if denominator == 0:
        return 0.0
    return precision_sum / denominator


def ndcg_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    hits = [1 if doc_id in relevant_doc_ids else 0 for doc_id in ranked_doc_ids[:k]]
    dcg = 0.0
    for idx, hit in enumerate(hits):
        if hit:
            dcg += 1.0 / math.log2(idx + 2)

    ideal_hits = min(len(relevant_doc_ids), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def first_relevant_reciprocal_rank(
    ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int
) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def evaluate_dense_retrieval(
    model: SentenceTransformer,
    config: ExperimentConfig,
    truncate_dims: list[int] | None = None,
    limit_queries: int | None = None,
    extra_corpus_docs: int | None = None,
) -> list[dict[str, float | int]]:
    corpus, queries, qrels = load_retrieval_dataset(
        config,
        limit_queries=limit_queries,
        extra_corpus_docs=extra_corpus_docs,
    )

    corpus_ids = list(corpus)
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
    query_ids = list(queries)
    query_texts = [queries[query_id] for query_id in query_ids]

    if truncate_dims is None or not truncate_dims:
        truncate_dims = [model.get_sentence_embedding_dimension()]

    max_k = max(max(config.top_k), config.map_at_k)
    rows: list[dict[str, float | int]] = []
    original_truncate_dim = getattr(model, "truncate_dim", None)

    for truncate_dim in truncate_dims:
        model.truncate_dim = truncate_dim

        query_embeddings = model.encode(
            query_texts,
            batch_size=config.eval_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=config.eval_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        hits = util.semantic_search(
            query_embeddings,
            corpus_embeddings,
            top_k=max_k,
            score_function=util.dot_score,
        )

        totals: dict[str, float] = {f"cosine_accuracy@{k}": 0.0 for k in config.top_k}
        totals.update({f"cosine_precision@{k}": 0.0 for k in config.top_k})
        totals.update({f"cosine_recall@{k}": 0.0 for k in config.top_k})
        totals.update({f"cosine_ndcg@{k}": 0.0 for k in config.top_k if k != 1})
        totals.update({f"cosine_mrr@{k}": 0.0 for k in config.top_k if k != 1})
        totals[f"cosine_map@{config.map_at_k}"] = 0.0

        for query_index, query_id in enumerate(query_ids):
            relevant_doc_ids = {
                doc_id for doc_id, score in qrels[query_id].items() if score > 0
            }
            ranked_doc_ids = [
                corpus_ids[item["corpus_id"]] for item in hits[query_index]
            ]

            for k in config.top_k:
                top_k_docs = ranked_doc_ids[:k]
                hit_count = sum(
                    1 for doc_id in top_k_docs if doc_id in relevant_doc_ids
                )
                totals[f"cosine_accuracy@{k}"] += 1.0 if hit_count > 0 else 0.0
                totals[f"cosine_precision@{k}"] += hit_count / k
                totals[f"cosine_recall@{k}"] += hit_count / len(relevant_doc_ids)
                if k != 1:
                    totals[f"cosine_ndcg@{k}"] += ndcg_at_k(
                        top_k_docs, relevant_doc_ids, k
                    )
                    totals[f"cosine_mrr@{k}"] += first_relevant_reciprocal_rank(
                        top_k_docs, relevant_doc_ids, k
                    )

            totals[f"cosine_map@{config.map_at_k}"] += average_precision_at_k(
                ranked_doc_ids,
                relevant_doc_ids,
                config.map_at_k,
            )

        query_count = float(len(query_ids))
        row: dict[str, float | int] = {"truncate_dim": truncate_dim}
        for key, value in totals.items():
            row[key] = value / query_count
        rows.append(row)

        del query_embeddings
        del corpus_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.truncate_dim = original_truncate_dim
    return rows


def results_to_markdown(
    rows: list[dict[str, float | int]], config: ExperimentConfig
) -> str:
    headers = [
        "dim",
        "Accuracy@1",
        "Accuracy@3",
        "Accuracy@5",
        "Accuracy@10",
        "NDCG@3",
        "NDCG@5",
        "NDCG@10",
        "MRR@3",
        "MRR@5",
        "MRR@10",
        f"MAP@{config.map_at_k}",
    ]

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                int(row["truncate_dim"]),
                _format_score(row.get("cosine_accuracy@1", 0.0)),
                _format_score(row.get("cosine_accuracy@3", 0.0)),
                _format_score(row.get("cosine_accuracy@5", 0.0)),
                _format_score(row.get("cosine_accuracy@10", 0.0)),
                _format_score(row.get("cosine_ndcg@3", 0.0)),
                _format_score(row.get("cosine_ndcg@5", 0.0)),
                _format_score(row.get("cosine_ndcg@10", 0.0)),
                _format_score(row.get("cosine_mrr@3", 0.0)),
                _format_score(row.get("cosine_mrr@5", 0.0)),
                _format_score(row.get("cosine_mrr@10", 0.0)),
                _format_score(row.get(f"cosine_map@{config.map_at_k}", 0.0)),
            ]
        )

    return tabulate(table_rows, headers=headers, tablefmt="github")


def write_results_artifacts(
    output_dir: str | Path,
    rows: list[dict[str, float | int]],
    config: ExperimentConfig,
    model_path: str,
) -> None:
    output_path = ensure_dir(output_dir)
    payload = {
        "model_path": model_path,
        "dataset": config.eval_dataset,
        "eval_split": config.eval_split,
        "results": rows,
    }
    save_json(payload, output_path / "results.json")
    markdown = results_to_markdown(rows, config)
    (output_path / "results.md").write_text(markdown + "\n", encoding="utf-8")


def results_to_readme(
    rows: list[dict[str, float | int]], config: ExperimentConfig, model_path: str
) -> str:
    table = results_to_markdown(rows, config)
    return "\n".join(
        [
            "# " + Path(model_path).name,
            "",
            "SentenceTransformer checkpoint fine-tuned for Vietnamese legal retrieval.",
            "",
            "## Evaluation",
            "",
            f"- Dataset: `{config.eval_dataset}`",
            f"- Split: `{config.eval_split}`",
            f"- Truncate dims: `{config.truncate_dims or []}`",
            "",
            table,
            "",
        ]
    )


def _format_score(value: float | int) -> str:
    return f"{float(value):.6f}"
