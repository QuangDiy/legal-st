#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

conda_env="${CONDA_ENV:-legal-st-fa2}"
result_root="${RESULT_ROOT:-results/gn-trvn}"

# Columns: Hugging Face model ID, embedding dimension, sequence length,
# evaluation batch size, output directory name.
while read -r model_id dim seq_length batch_size name; do
  output_dir="$result_root/$name"
  mkdir -p "$output_dir"

  echo "=== Evaluating $model_id (length=$seq_length, dim=$dim, batch=$batch_size) ==="
  conda run --no-capture-output -n "$conda_env" python -u scripts/evaluate_retrieval.py \
    --config configs/gn-trvn-bert-base-stage2.yaml \
    --model-path "$model_id" \
    --truncate-dims "$dim" \
    --max-seq-length "$seq_length" \
    --eval-batch-size "$batch_size" \
    --no-bm25 \
    --output-dir "$output_dir" 2>&1 | tee "$output_dir/eval.log"
done <<'MODELS'
GreenNode/GreenNode-Embedding-Large-VN-V1 1024 8192 16 greennode-large
VoVanPhuc/sup-SimCSE-VietNamese-phobert-base 768 256 32 simcse-phobert
keepitreal/vietnamese-sbert 768 256 32 vietnamese-sbert
bkai-foundation-models/vietnamese-bi-encoder 768 256 32 bkai-bi-encoder
aisingapore/SEA-LION-ModernBERT-Embedding-600M 1024 8192 16 sealion-600m
aisingapore/SEA-LION-ModernBERT-Embedding-300M 768 8192 32 sealion-300m
MODELS
