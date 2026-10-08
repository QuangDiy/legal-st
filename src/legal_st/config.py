from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    run_name: str
    model_name: str
    output_dir: str
    train_dataset: str | list[str] = "batmangiaicuuthegioi/zalo-legal-triplets"
    train_split: str = "train"
    eval_dataset: str = "another-symato/VMTEB-Zalo-legel-retrieval"
    eval_corpus_config: str = "corpus"
    eval_queries_config: str = "queries"
    eval_labels_config: str = "data_ir"
    eval_split: str = "train"
    seed: int = 42
    max_seq_length: int = 512
    pooling: str = "mean"
    normalize_embeddings: bool = True
    include_hard_negatives: bool = True
    num_train_epochs: int = 3
    train_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    # "auto" resolves at runtime: bf16 on Ampere+ GPUs, fp16 on older GPUs (T4), fp32 on CPU.
    precision: str = "bf16"
    # "auto" uses flash_attention_2 when the GPU (Ampere+), flash_attn and the
    # architecture all support it, otherwise sdpa.
    attn_implementation: str = "auto"
    use_amp: bool = True
    use_cached_mnrl: bool = False
    cached_mnrl_mini_batch_size: int = 32
    # Share in-batch negatives across GPUs under DDP (torchrun).
    gather_across_devices: bool = False
    validation_size: float = 0.05
    # Column used to group rows before the train/validation split. Use "positive"
    # when several queries share one positive document, so it never leaks.
    validation_group_key: str = "query"
    validation_subset: int | None = 1024
    evaluation_steps: int = 250
    checkpoint_save_steps: int = 250
    checkpoint_save_total_limit: int = 2
    early_stopping_patience: int | None = None
    hf_repo_id: str | None = None
    hf_private: bool = False
    hf_push_on_save: bool = False
    run_retrieval_eval_after_train: bool = True
    run_bm25_baseline: bool = True
    retrieval_eval_limit_queries: int | None = None
    retrieval_eval_extra_corpus_docs: int | None = None
    matryoshka_dims: list[int] = field(default_factory=list)
    truncate_dims: list[int] = field(default_factory=list)
    top_k: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    recall_at_k: list[int] = field(default_factory=lambda: [5, 10, 100])
    map_at_k: int = 100
    eval_batch_size: int = 128
    # List of extra retrieval eval dataset specs. Each entry is a dict with keys:
    #   dataset, name (optional), corpus_config, queries_config, labels_config,
    #   split, and optional column-name overrides (see data.py for full list).
    # When non-empty, these datasets are used for post-training retrieval eval
    # instead of (or in addition to) the single eval_dataset field.
    eval_datasets: list[dict] = field(default_factory=list)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Config file is empty: {config_path}")

    valid_keys = {item.name for item in fields(ExperimentConfig)}
    unknown_keys = sorted(set(payload) - valid_keys)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unknown config key(s) in {config_path}: {joined}")

    config = ExperimentConfig(**payload)
    config.precision = config.precision.lower()
    if config.precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported precision in {config_path}: {config.precision}. "
            "Expected one of: auto, fp32, fp16, bf16"
        )
    if config.attn_implementation not in {"auto", "flash_attention_2", "sdpa", "eager"}:
        raise ValueError(
            f"Unsupported attn_implementation in {config_path}: {config.attn_implementation}. "
            "Expected one of: auto, flash_attention_2, sdpa, eager"
        )
    if config.validation_group_key not in {"query", "positive"}:
        raise ValueError(
            f"Unsupported validation_group_key in {config_path}: {config.validation_group_key}. "
            "Expected one of: query, positive"
        )

    if not config.truncate_dims:
        config.truncate_dims = list(config.matryoshka_dims)
    return config


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        item.name: getattr(config, item.name) for item in fields(config)
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
