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


def _load_one_triplet_dataset(
    dataset_id: str, split: str, include_hard_negatives: bool
) -> list[dict[str, str]]:
    dataset = load_dataset(dataset_id, split=split)
    return [
        _trim_triplet_row(row, include_hard_negatives=include_hard_negatives)
        for row in dataset
    ]


def load_triplet_records(config: ExperimentConfig) -> list[dict[str, str]]:
    dataset_ids = (
        config.train_dataset
        if isinstance(config.train_dataset, list)
        else [config.train_dataset]
    )
    all_records: list[dict[str, str]] = []
    for dataset_id in dataset_ids:
        records = _load_one_triplet_dataset(
            dataset_id, config.train_split, config.include_hard_negatives
        )
        all_records.extend(records)
    return all_records


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


_ID_ALIASES: list[str] = ["_id", "id", "query_id", "qid", "corpus_id"]
_TEXT_ALIASES: list[str] = ["text", "question", "query", "sentence"]
_QREL_QID_ALIASES: list[str] = ["query-id", "query_id", "qid", "q_id"]
_QREL_DID_ALIASES: list[str] = ["corpus-id", "corpus_id", "doc_id", "did", "document_id"]
_SCORE_ALIASES: list[str] = ["score", "relevance", "label"]


def _resolve_col(row_keys: list[str], preferred: str, aliases: list[str], label: str) -> str:
    """Return *preferred* if present in *row_keys*, else first matching alias."""
    if preferred in row_keys:
        return preferred
    for alias in aliases:
        if alias in row_keys:
            return alias
    raise KeyError(
        f"Cannot find column for '{label}'. "
        f"Tried: {[preferred] + aliases}. "
        f"Available columns: {row_keys}"
    )


def _build_retrieval_splits(
    dataset_id: str,
    corpus_config: str,
    queries_config: str,
    labels_config: str,
    split: str,
    corpus_id_col: str,
    corpus_title_col: str,
    corpus_text_col: str,
    query_id_col: str,
    query_text_col: str,
    qrel_query_id_col: str,
    qrel_corpus_id_col: str,
    qrel_score_col: str,
    limit_queries: int | None,
    extra_corpus_docs: int | None,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    corpus_rows = load_dataset(dataset_id, corpus_config, split=split)
    query_rows = load_dataset(dataset_id, queries_config, split=split)
    qrel_rows = load_dataset(dataset_id, labels_config, split=split)

    # Auto-detect column names using aliases when the configured name is absent
    q_keys = query_rows.column_names
    query_id_col = _resolve_col(q_keys, query_id_col, _ID_ALIASES, "query id")
    query_text_col = _resolve_col(q_keys, query_text_col, _TEXT_ALIASES, "query text")

    qr_keys = qrel_rows.column_names
    qrel_query_id_col = _resolve_col(qr_keys, qrel_query_id_col, _QREL_QID_ALIASES, "qrel query id")
    qrel_corpus_id_col = _resolve_col(qr_keys, qrel_corpus_id_col, _QREL_DID_ALIASES, "qrel corpus id")
    qrel_score_col = _resolve_col(qr_keys, qrel_score_col, _SCORE_ALIASES, "qrel score")

    c_keys = corpus_rows.column_names
    corpus_id_col = _resolve_col(c_keys, corpus_id_col, _ID_ALIASES, "corpus id")

    queries: dict[str, str] = {}
    selected_query_ids: list[str] = []
    for row in query_rows:
        query_id = str(row[query_id_col])
        if limit_queries is not None and len(selected_query_ids) >= limit_queries:
            break
        selected_query_ids.append(query_id)
        queries[query_id] = normalize_text(row[query_text_col])

    selected_query_ids_set = set(selected_query_ids)
    relevant_docs: dict[str, dict[str, int]] = defaultdict(dict)
    required_corpus_ids: set[str] = set()
    for row in qrel_rows:
        query_id = str(row[qrel_query_id_col])
        if query_id not in selected_query_ids_set:
            continue
        corpus_id = str(row[qrel_corpus_id_col])
        score = int(row[qrel_score_col])
        relevant_docs[query_id][corpus_id] = score
        if score > 0:
            required_corpus_ids.add(corpus_id)

    corpus: dict[str, str] = {}
    extras_remaining = None if extra_corpus_docs is None else extra_corpus_docs
    for row in corpus_rows:
        corpus_id = str(row[corpus_id_col])
        include_row = corpus_id in required_corpus_ids
        if not include_row and extras_remaining is not None and extras_remaining > 0:
            include_row = True
            extras_remaining -= 1
        elif not include_row and extras_remaining is not None:
            continue

        title = normalize_text(row.get(corpus_title_col) or "")
        text = normalize_text(row.get(corpus_text_col) or "")
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


def load_retrieval_dataset(
    config: ExperimentConfig,
    limit_queries: int | None = None,
    extra_corpus_docs: int | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    return _build_retrieval_splits(
        dataset_id=config.eval_dataset,
        corpus_config=config.eval_corpus_config,
        queries_config=config.eval_queries_config,
        labels_config=config.eval_labels_config,
        split=config.eval_split,
        corpus_id_col="id",
        corpus_title_col="title",
        corpus_text_col="text",
        query_id_col="query_id",
        query_text_col="question",
        qrel_query_id_col="query_id",
        qrel_corpus_id_col="corpus_id",
        qrel_score_col="score",
        limit_queries=limit_queries,
        extra_corpus_docs=extra_corpus_docs,
    )


def _load_squad_format(
    dataset_id: str,
    split: str,
    context_col: str,
    question_col: str,
    id_col: str,
    title_col: str,
    limit_queries: int | None,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Load a SQuAD-style dataset (single split, context + question per row).

    Corpus documents are built from unique context passages; queries from
    questions; qrels map each question to its containing passage.
    """
    rows = load_dataset(dataset_id, split=split)

    corpus: dict[str, str] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    context_to_id: dict[str, str] = {}

    for row in rows:
        context = normalize_text(row[context_col])
        title = normalize_text(row.get(title_col) or "")

        if context not in context_to_id:
            corpus_id = f"corpus_{len(context_to_id)}"
            context_to_id[context] = corpus_id
            corpus[corpus_id] = build_corpus_text(title=title, text=context)

        corpus_id = context_to_id[context]
        query_id = str(row[id_col])
        question = normalize_text(row[question_col])

        if limit_queries is not None and len(queries) >= limit_queries:
            break

        if query_id not in queries:
            queries[query_id] = question
        qrels[query_id][corpus_id] = 1

    return corpus, queries, dict(qrels)


def load_retrieval_dataset_from_spec(
    spec: dict,
    limit_queries: int | None = None,
    extra_corpus_docs: int | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Load a retrieval dataset from a spec dict (as used in config.eval_datasets).

    Required key: ``dataset``.
    Optional ``format`` key selects the loader:
      - ``"beir"`` (default): separate corpus / queries / qrels configs.
      - ``"squad"``: single split with context + question columns (SQuAD-style).

    BEIR optional keys (with defaults):
      ``corpus_config`` ("corpus"), ``queries_config`` ("queries"),
      ``labels_config`` ("qrels"), ``split`` ("test"), plus column-name
      overrides ``corpus_id_col``, ``corpus_title_col``, ``corpus_text_col``,
      ``query_id_col``, ``query_text_col``, ``qrel_query_id_col``,
      ``qrel_corpus_id_col``, ``qrel_score_col``.

    SQuAD optional keys (with defaults):
      ``split`` ("validation"), ``context_col`` ("context"),
      ``question_col`` ("question"), ``id_col`` ("id"), ``title_col`` ("title").
    """
    fmt = spec.get("format", "beir")

    if fmt == "squad":
        return _load_squad_format(
            dataset_id=spec["dataset"],
            split=spec.get("split", "validation"),
            context_col=spec.get("context_col", "context"),
            question_col=spec.get("question_col", "question"),
            id_col=spec.get("id_col", "id"),
            title_col=spec.get("title_col", "title"),
            limit_queries=limit_queries,
        )

    return _build_retrieval_splits(
        dataset_id=spec["dataset"],
        corpus_config=spec.get("corpus_config", "corpus"),
        queries_config=spec.get("queries_config", "queries"),
        labels_config=spec.get("labels_config", "qrels"),
        split=spec.get("split", "test"),
        corpus_id_col=spec.get("corpus_id_col", "id"),
        corpus_title_col=spec.get("corpus_title_col", "title"),
        corpus_text_col=spec.get("corpus_text_col", "text"),
        query_id_col=spec.get("query_id_col", "query_id"),
        query_text_col=spec.get("query_text_col", "question"),
        qrel_query_id_col=spec.get("qrel_query_id_col", "query_id"),
        qrel_corpus_id_col=spec.get("qrel_corpus_id_col", "corpus_id"),
        qrel_score_col=spec.get("qrel_score_col", "score"),
        limit_queries=limit_queries,
        extra_corpus_docs=extra_corpus_docs,
    )


def build_corpus_text(title: str, text: str) -> str:
    if title and text:
        return f"{title}\n{text}"
    return title or text
