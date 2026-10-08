from __future__ import annotations

import importlib.util
from typing import Callable, TypeVar

import torch
from sentence_transformers import SentenceTransformer, models
from transformers import AutoConfig

T = TypeVar("T")


def flash_attention_available() -> bool:
    """FlashAttention-2 needs an Ampere-or-newer GPU and the flash_attn package."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        return False
    return importlib.util.find_spec("flash_attn") is not None


def _attn_candidates(requested: str) -> list[str]:
    if requested != "auto":
        return [requested]
    if flash_attention_available():
        return ["flash_attention_2", "sdpa", "eager"]
    return ["sdpa", "eager"]


def _load_with_attn_fallback(load: Callable[[str], T], requested: str) -> T:
    """Call *load* with each candidate attention implementation in turn.

    With ``requested="auto"`` an architecture that does not implement
    FlashAttention-2 (e.g. RoBERTa in transformers 4.57) falls back to sdpa,
    and one without sdpa falls back to eager.
    An explicitly requested implementation is never silently replaced.
    """
    candidates = _attn_candidates(requested)
    for index, attn in enumerate(candidates):
        try:
            result = load(attn)
        except (ValueError, ImportError) as exc:
            if index == len(candidates) - 1:
                raise
            print(f"[attention] {attn} unavailable ({exc}); trying {candidates[index + 1]}")
            continue
        print(f"[attention] using {attn}")
        return result
    raise RuntimeError("No attention implementation candidates")


def build_sentence_transformer(
    model_name: str,
    max_seq_length: int = 512,
    pooling: str = "mean",
    normalize_embeddings: bool = True,
    attn_implementation: str = "auto",
) -> SentenceTransformer:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    config_args = {"trust_remote_code": False}
    if getattr(config, "model_type", None) == "modernbert":
        config_args["reference_compile"] = False

    transformer = _load_with_attn_fallback(
        lambda attn: models.Transformer(
            model_name,
            max_seq_length=max_seq_length,
            model_args={"trust_remote_code": False, "attn_implementation": attn},
            config_args=config_args,
            tokenizer_args={"use_fast": True},
        ),
        attn_implementation,
    )

    pooling = pooling.lower()
    if pooling not in {"mean", "cls", "max"}:
        raise ValueError(f"Unsupported pooling strategy: {pooling}")

    pooling_model = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=pooling == "mean",
        pooling_mode_cls_token=pooling == "cls",
        pooling_mode_max_tokens=pooling == "max",
    )

    modules = [transformer, pooling_model]
    if normalize_embeddings:
        modules.append(models.Normalize())

    model = SentenceTransformer(modules=modules)
    model.max_seq_length = max_seq_length
    return model


def load_sentence_transformer(
    model_name_or_path: str, attn_implementation: str = "auto"
) -> SentenceTransformer:
    """Load a saved/hub SentenceTransformer with the requested attention backend."""
    return _load_with_attn_fallback(
        lambda attn: SentenceTransformer(
            model_name_or_path, model_kwargs={"attn_implementation": attn}
        ),
        attn_implementation,
    )
