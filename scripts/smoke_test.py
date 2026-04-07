from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from legal_st.config import load_config
from legal_st.modeling import build_sentence_transformer
from legal_st.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick dependency and model wiring check"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)

    model = build_sentence_transformer(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        pooling=config.pooling,
        normalize_embeddings=config.normalize_embeddings,
    )

    sample_texts = [
        "Muc phat khi dieu khien xe o to khong co giay phep lai xe",
        "Quy dinh ve thoi gian nghi phep hang nam cua nguoi lao dong",
    ]
    embeddings = model.encode(sample_texts, normalize_embeddings=True)

    print("Smoke test passed.")
    print(f"Model: {config.model_name}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Max sequence length: {model.max_seq_length}")


if __name__ == "__main__":
    main()
