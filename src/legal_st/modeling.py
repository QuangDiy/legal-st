from __future__ import annotations

from sentence_transformers import SentenceTransformer, models


def build_sentence_transformer(
    model_name: str,
    max_seq_length: int = 512,
    pooling: str = "mean",
    normalize_embeddings: bool = True,
) -> SentenceTransformer:
    transformer = models.Transformer(
        model_name,
        max_seq_length=max_seq_length,
        model_args={"trust_remote_code": False},
        tokenizer_args={"use_fast": True},
    )

    pooling = pooling.lower()
    pooling_model = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=pooling == "mean",
        pooling_mode_cls_token=pooling == "cls",
        pooling_mode_max_tokens=pooling == "max",
    )

    if pooling not in {"mean", "cls", "max"}:
        raise ValueError(f"Unsupported pooling strategy: {pooling}")

    modules = [transformer, pooling_model]
    if normalize_embeddings:
        modules.append(models.Normalize())

    model = SentenceTransformer(modules=modules)
    model.max_seq_length = max_seq_length
    return model
