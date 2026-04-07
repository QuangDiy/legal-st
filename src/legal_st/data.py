from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from datasets import load_dataset
from sentence_transformers import InputExample

from .config import ExperimentConfig
from .utils import normalize_text


def _trim_triplet_row(row: dict, include_hard_negatives: bool) -> dict[str, str]:
    payload = {
        "query": normalize_text(row["query"]),
        "positive": normalize_text(row["positive"]),
        "negative": normalize_text(row["negative"]),
    }
    if not include_hard_negatives:
        payload.pop("negative")
    return payload


def load_triplet_records(config: ExperimentConfig) -> list[dict[str, str]]:
    dataset = load_dataset(config.train_dataset, split=config.train_split)
    return [
        _trim_triplet_row(row, include_hard_negatives=config.include_hard_negatives)
        for row in dataset
    ]


def split_records_by_query(
    records: list[dict[str, str]],
    validation_size: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if validation_size <= 0:
        return records, []

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        grouped[row["query"]].append(row)

    queries = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(queries)

    validation_query_count = max(1, int(len(queries) * validation_size))
    validation_queries = set(queries[:validation_query_count])

    train_records: list[dict[str, str]] = []
    validation_records: list[dict[str, str]] = []
    for query, rows in grouped.items():
        if query in validation_queries:
            validation_records.extend(rows)
        else:
            train_records.extend(rows)

    return train_records, validation_records


def records_to_input_examples(records: Iterable[dict[str, str]]) -> list[InputExample]:
    examples: list[InputExample] = []
    for row in records:
        texts = [row["query"], row["positive"]]
        if "negative" in row:
            texts.append(row["negative"])
        examples.append(InputExample(texts=texts))
    return examples


def build_triplet_evaluator_payload(
    records: list[dict[str, str]],
    subset_size: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    if subset_size is not None and len(records) > subset_size:
        records = records[:subset_size]

    anchors = [row["query"] for row in records]
    positives = [row["positive"] for row in records]
    negatives = [row.get("negative", row["positive"]) for row in records]
    return anchors, positives, negatives


def load_retrieval_dataset(
    config: ExperimentConfig,
    limit_queries: int | None = None,
    extra_corpus_docs: int | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    corpus_rows = load_dataset(
        config.eval_dataset, config.eval_corpus_config, split=config.eval_split
    )
    query_rows = load_dataset(
        config.eval_dataset, config.eval_queries_config, split=config.eval_split
    )
    qrel_rows = load_dataset(
        config.eval_dataset, config.eval_labels_config, split=config.eval_split
    )

    queries: dict[str, str] = {}
    selected_query_ids: list[str] = []
    for row in query_rows:
        query_id = row["query_id"]
        if limit_queries is not None and len(selected_query_ids) >= limit_queries:
            break
        selected_query_ids.append(query_id)
        queries[query_id] = normalize_text(row["question"])

    selected_query_ids_set = set(selected_query_ids)
    relevant_docs: dict[str, dict[str, int]] = defaultdict(dict)
    required_corpus_ids: set[str] = set()
    for row in qrel_rows:
        query_id = row["query_id"]
        if query_id not in selected_query_ids_set:
            continue
        corpus_id = row["corpus_id"]
        score = int(row["score"])
        relevant_docs[query_id][corpus_id] = score
        if score > 0:
            required_corpus_ids.add(corpus_id)

    corpus: dict[str, str] = {}
    extras_remaining = None if extra_corpus_docs is None else extra_corpus_docs
    for row in corpus_rows:
        corpus_id = row["id"]
        include_row = corpus_id in required_corpus_ids
        if not include_row and extras_remaining is not None and extras_remaining > 0:
            include_row = True
            extras_remaining -= 1
        elif not include_row and extras_remaining is not None:
            continue

        title = normalize_text(row.get("title") or "")
        text = normalize_text(row.get("text") or "")
        corpus[corpus_id] = build_corpus_text(title=title, text=text)

    filtered_qrels: dict[str, dict[str, int]] = {}
    corpus_ids = set(corpus)
    for query_id, docs in relevant_docs.items():
        kept = {doc_id: score for doc_id, score in docs.items() if doc_id in corpus_ids}
        if kept:
            filtered_qrels[query_id] = kept

    filtered_queries = {
        query_id: text
        for query_id, text in queries.items()
        if query_id in filtered_qrels
    }
    return corpus, filtered_queries, filtered_qrels


def build_corpus_text(title: str, text: str) -> str:
    if title and text:
        return f"{title}\n{text}"
    return title or text
