from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(payload: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def safe_max_seq_length(model) -> int:
    """Return the largest seq length that won't overflow the model's position table.

    SentenceTransformer configs sometimes carry a ``max_seq_length`` that exceeds
    the underlying transformer's ``max_position_embeddings`` (e.g. PhoBERT has 258
    positions but some HF wrappers advertise 512).  Passing sequences longer than
    the position table causes a CUDA scatter/gather index-out-of-bounds crash.
    """
    current = model.max_seq_length
    try:
        transformer_module = model[0]
        hf_config = transformer_module.auto_model.config
        max_pos = hf_config.max_position_embeddings
        # RoBERTa-family reserves positions 0 and 1 (padding + start offset)
        if getattr(hf_config, "model_type", "").lower() in ("roberta", "phobert"):
            max_pos = max_pos - 2
        if max_pos < current:
            print(
                f"[safe_max_seq_length] Clamping max_seq_length {current} → {max_pos} "
                f"(model max_position_embeddings={hf_config.max_position_embeddings})"
            )
            return max_pos
    except Exception:
        pass
    return current
