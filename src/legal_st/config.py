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
    train_dataset: str = "batmangiaicuuthegioi/zalo-legal-triplets"
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
    use_amp: bool = True
    use_cached_mnrl: bool = False
    validation_size: float = 0.05
    validation_subset: int | None = 1024
    evaluation_steps: int = 250
    checkpoint_save_steps: int = 250
    checkpoint_save_total_limit: int = 2
    matryoshka_dims: list[int] = field(default_factory=list)
    truncate_dims: list[int] = field(default_factory=list)
    top_k: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    map_at_k: int = 100
    eval_batch_size: int = 128

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
