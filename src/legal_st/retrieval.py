from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from tabulate import tabulate

from .config import ExperimentConfig
from .data import load_retrieval_dataset, load_retrieval_dataset_from_spec
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


def _tokenize_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def _run_bm25_metrics(
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    config: ExperimentConfig,
) -> dict[str, float | int]:
    """Compute retrieval metrics using BM25Okapi as a sparse baseline."""
    corpus_ids = list(corpus)
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
    query_ids = list(queries)
    query_texts = [queries[qid] for qid in query_ids]

    recall_ks: list[int] = list(config.recall_at_k) if config.recall_at_k else [5, 10, 100]
    max_k = max(max(config.top_k), config.map_at_k, *recall_ks)

    print("  Building BM25 index …")
    tokenized_corpus = [_tokenize_for_bm25(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    totals: dict[str, float] = {f"bm25_accuracy@{k}": 0.0 for k in config.top_k}
    totals.update({f"bm25_ndcg@{k}": 0.0 for k in config.top_k if k != 1})
    totals.update({f"bm25_mrr@{k}": 0.0 for k in config.top_k if k != 1})
    totals.update({f"bm25_recall@{k}": 0.0 for k in recall_ks})
    totals[f"bm25_map@{config.map_at_k}"] = 0.0

    for query_id, query_text in zip(query_ids, query_texts):
        tokenized_query = _tokenize_for_bm25(query_text)
        scores = np.array(bm25.get_scores(tokenized_query))
        # Efficient top-k using argpartition (avoids full sort of large corpora)
        n = min(max_k, len(scores))
        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        ranked_doc_ids = [corpus_ids[int(i)] for i in top_indices]

        relevant_doc_ids = {
            doc_id for doc_id, rel in qrels[query_id].items() if rel > 0
        }

        for k in config.top_k:
            top_k_docs = ranked_doc_ids[:k]
            hit_count = sum(1 for d in top_k_docs if d in relevant_doc_ids)
            totals[f"bm25_accuracy@{k}"] += 1.0 if hit_count > 0 else 0.0
            if k != 1:
                totals[f"bm25_ndcg@{k}"] += ndcg_at_k(ranked_doc_ids, relevant_doc_ids, k)
                totals[f"bm25_mrr@{k}"] += first_relevant_reciprocal_rank(
                    ranked_doc_ids, relevant_doc_ids, k
                )

        n_relevant = max(len(relevant_doc_ids), 1)
        for k in recall_ks:
            hit_count = sum(1 for d in ranked_doc_ids[:k] if d in relevant_doc_ids)
            totals[f"bm25_recall@{k}"] += hit_count / n_relevant

        totals[f"bm25_map@{config.map_at_k}"] += average_precision_at_k(
            ranked_doc_ids, relevant_doc_ids, config.map_at_k
        )

    query_count = float(len(query_ids))
    row: dict[str, float | int] = {"truncate_dim": -1}  # sentinel: BM25 baseline
    for key, value in totals.items():
        row[key] = value / query_count
    return row


def _run_metrics(
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    model: SentenceTransformer,
    config: ExperimentConfig,
    truncate_dims: list[int] | None,
    batch_size: int | None = None,
) -> list[dict[str, float | int]]:
    corpus_ids = list(corpus)
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
    query_ids = list(queries)
    query_texts = [queries[query_id] for query_id in query_ids]

    if truncate_dims is None or not truncate_dims:
        truncate_dims = [model.get_sentence_embedding_dimension()]

    effective_batch_size = batch_size if batch_size is not None else config.eval_batch_size
    recall_ks: list[int] = list(config.recall_at_k) if config.recall_at_k else [5, 10, 100]
    max_k = max(max(config.top_k), config.map_at_k, *recall_ks)
    rows: list[dict[str, float | int]] = []
    original_truncate_dim = getattr(model, "truncate_dim", None)

    for truncate_dim in truncate_dims:
        model.truncate_dim = truncate_dim

        query_embeddings = model.encode(
            query_texts,
            batch_size=effective_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=effective_batch_size,
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
        totals.update({f"cosine_ndcg@{k}": 0.0 for k in config.top_k if k != 1})
        totals.update({f"cosine_mrr@{k}": 0.0 for k in config.top_k if k != 1})
        totals.update({f"cosine_recall@{k}": 0.0 for k in recall_ks})
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
                if k != 1:
                    totals[f"cosine_ndcg@{k}"] += ndcg_at_k(
                        ranked_doc_ids, relevant_doc_ids, k
                    )
                    totals[f"cosine_mrr@{k}"] += first_relevant_reciprocal_rank(
                        ranked_doc_ids, relevant_doc_ids, k
                    )

            n_relevant = max(len(relevant_doc_ids), 1)
            for k in recall_ks:
                hit_count = sum(
                    1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_doc_ids
                )
                totals[f"cosine_recall@{k}"] += hit_count / n_relevant

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


def evaluate_dense_retrieval(
    model: SentenceTransformer,
    config: ExperimentConfig,
    truncate_dims: list[int] | None = None,
    limit_queries: int | None = None,
    extra_corpus_docs: int | None = None,
    dataset_spec: dict | None = None,
    run_bm25: bool = True,
) -> list[dict[str, float | int]]:
    """Evaluate retrieval on a single dataset.

    Pass *dataset_spec* to load from an explicit spec dict instead of config's
    ``eval_dataset`` / ``eval_*_config`` / ``eval_split`` fields.

    When *run_bm25* is True, a BM25Okapi baseline row is prepended to the
    returned list (``truncate_dim == -1``).
    """
    if dataset_spec is not None:
        corpus, queries, qrels = load_retrieval_dataset_from_spec(
            dataset_spec,
            limit_queries=limit_queries,
            extra_corpus_docs=extra_corpus_docs,
        )
    else:
        corpus, queries, qrels = load_retrieval_dataset(
            config,
            limit_queries=limit_queries,
            extra_corpus_docs=extra_corpus_docs,
        )

    effective_dims = truncate_dims if truncate_dims is not None else config.truncate_dims
    spec_batch_size = int(dataset_spec["eval_batch_size"]) if dataset_spec and "eval_batch_size" in dataset_spec else None
    rows = _run_metrics(corpus, queries, qrels, model, config, effective_dims, batch_size=spec_batch_size)

    if run_bm25:
        print("  Running BM25 baseline …")
        bm25_row = _run_bm25_metrics(corpus, queries, qrels, config)
        rows = [bm25_row] + rows

    return rows


def evaluate_bm25_retrieval_datasets(
    config: ExperimentConfig,
    limit_queries: int | None = None,
    extra_corpus_docs: int | None = None,
) -> list[tuple[str, list[dict[str, float | int]]]]:
    """Run BM25-only evaluation on all datasets defined in config.

    Returns the same ``(name, rows)`` format as
    :func:`evaluate_dense_retrieval_datasets` so downstream helpers
    (``write_multi_results_artifacts``, ``results_to_markdown``, …) work unchanged.
    Each result list contains a single BM25 row (``truncate_dim == -1``).
    """
    specs = config.eval_datasets
    if not specs:
        specs = [
            {
                "dataset": config.eval_dataset,
                "name": config.eval_dataset,
                "corpus_config": config.eval_corpus_config,
                "queries_config": config.eval_queries_config,
                "labels_config": config.eval_labels_config,
                "split": config.eval_split,
            }
        ]

    results: list[tuple[str, list[dict[str, float | int]]]] = []
    for spec in specs:
        name: str = spec.get("name") or spec["dataset"]
        print(f"\n=== BM25 on: {name} ===")
        corpus, queries, qrels = load_retrieval_dataset_from_spec(
            spec,
            limit_queries=limit_queries,
            extra_corpus_docs=extra_corpus_docs,
        )
        bm25_row = _run_bm25_metrics(corpus, queries, qrels, config)
        results.append((name, [bm25_row]))
    return results


def evaluate_dense_retrieval_datasets(
    model: SentenceTransformer,
    config: ExperimentConfig,
    truncate_dims: list[int] | None = None,
    limit_queries: int | None = None,
    extra_corpus_docs: int | None = None,
    run_bm25: bool = True,
) -> list[tuple[str, list[dict[str, float | int]]]]:
    """Evaluate on all datasets defined in config.

    Returns a list of (dataset_name, rows) tuples.  When ``config.eval_datasets``
    is non-empty those datasets are used; otherwise falls back to the single
    ``config.eval_dataset`` entry.

    When *run_bm25* is True, each dataset also gets a BM25 baseline row
    prepended (``truncate_dim == -1``).
    """
    specs = config.eval_datasets
    if not specs:
        specs = [
            {
                "dataset": config.eval_dataset,
                "name": config.eval_dataset,
                "corpus_config": config.eval_corpus_config,
                "queries_config": config.eval_queries_config,
                "labels_config": config.eval_labels_config,
                "split": config.eval_split,
            }
        ]

    results: list[tuple[str, list[dict[str, float | int]]]] = []
    for spec in specs:
        name: str = spec.get("name") or spec["dataset"]
        print(f"\n=== Evaluating on: {name} ===")
        rows = evaluate_dense_retrieval(
            model=model,
            config=config,
            truncate_dims=truncate_dims,
            limit_queries=limit_queries,
            extra_corpus_docs=extra_corpus_docs,
            dataset_spec=spec,
            run_bm25=run_bm25,
        )
        results.append((name, rows))
    return results


def results_to_markdown(
    rows: list[dict[str, float | int]], config: ExperimentConfig
) -> str:
    recall_ks: list[int] = sorted(config.recall_at_k) if config.recall_at_k else [5, 10, 100]

    headers = (
        ["method"]
        + [f"Accuracy@{k}" for k in config.top_k]
        + [f"NDCG@{k}" for k in config.top_k if k != 1]
        + [f"MRR@{k}" for k in config.top_k if k != 1]
        + [f"Recall@{k}" for k in recall_ks]
        + [f"MAP@{config.map_at_k}"]
    )

    table_rows = []
    for row in rows:
        is_bm25 = row.get("truncate_dim") == -1
        prefix = "bm25" if is_bm25 else "cosine"
        label = "BM25" if is_bm25 else str(int(row["truncate_dim"]))
        cells = [label]
        cells += [_fmt(row.get(f"{prefix}_accuracy@{k}", 0.0)) for k in config.top_k]
        cells += [_fmt(row.get(f"{prefix}_ndcg@{k}", 0.0)) for k in config.top_k if k != 1]
        cells += [_fmt(row.get(f"{prefix}_mrr@{k}", 0.0)) for k in config.top_k if k != 1]
        cells += [_fmt(row.get(f"{prefix}_recall@{k}", 0.0)) for k in recall_ks]
        cells += [_fmt(row.get(f"{prefix}_map@{config.map_at_k}", 0.0))]
        table_rows.append(cells)

    return tabulate(table_rows, headers=headers, tablefmt="github")


def write_results_artifacts(
    output_dir: str | Path,
    rows: list[dict[str, float | int]],
    config: ExperimentConfig,
    model_path: str,
    dataset_name: str | None = None,
    dataset_id: str | None = None,
) -> None:
    output_path = ensure_dir(output_dir)
    payload = {
        "model_path": model_path,
        "dataset": dataset_id or config.eval_dataset,
        "eval_split": config.eval_split,
        "results": rows,
    }
    save_json(payload, output_path / "results.json")
    markdown = results_to_markdown(rows, config)
    (output_path / "results.md").write_text(markdown + "\n", encoding="utf-8")


def write_multi_results_artifacts(
    output_dir: str | Path,
    dataset_results: list[tuple[str, list[dict[str, float | int]]]],
    config: ExperimentConfig,
    model_path: str,
) -> None:
    """Write per-dataset subdirs and a combined results.json."""
    output_path = ensure_dir(output_dir)
    all_results = []
    for name, rows in dataset_results:
        safe_name = name.replace("/", "_")
        write_results_artifacts(
            output_dir=output_path / safe_name,
            rows=rows,
            config=config,
            model_path=model_path,
            dataset_name=name,
            dataset_id=name,
        )
        all_results.append({"dataset": name, "results": rows})

    save_json({"model_path": model_path, "datasets": all_results}, output_path / "results.json")


def results_to_readme(
    dataset_results: list[tuple[str, list[dict[str, float | int]]]],
    config: ExperimentConfig,
    model_path: str,
) -> str:
    lines = [
        "# " + Path(model_path).name,
        "",
        "SentenceTransformer checkpoint fine-tuned for Vietnamese legal retrieval.",
        "",
        "## Evaluation",
        "",
        f"- Truncate dims: `{config.truncate_dims or []}`",
        "",
    ]
    for name, rows in dataset_results:
        lines += [
            f"### {name}",
            "",
            results_to_markdown(rows, config),
            "",
        ]
    return "\n".join(lines)


def _fmt(value: float | int) -> str:
    return f"{float(value):.6f}"


def _format_score(value: float | int) -> str:
    return _fmt(value)
