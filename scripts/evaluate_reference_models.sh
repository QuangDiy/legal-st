#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

conda_env="${CONDA_ENV:-legal-st-fa2}"
result_root="${RESULT_ROOT:-results/gn-trvn}"
failed_models=()

# Columns: Hugging Face model ID, embedding dimension, sequence length,
# evaluation batch size, output directory name.
while read -r model_id dim seq_length batch_size name; do
  output_dir="$result_root/$name"
  if [[ -f "$output_dir/results.json" ]]; then
    echo "=== Skipping $model_id: $output_dir/results.json already exists ==="
    continue
  fi
  mkdir -p "$output_dir"

  echo "=== Evaluating $model_id (length=$seq_length, dim=$dim, batch=$batch_size) ==="
  if conda run --no-capture-output -n "$conda_env" python -u scripts/evaluate_retrieval.py \
      --config configs/gn-trvn-bert-base-stage2.yaml \
      --model-path "$model_id" \
      --truncate-dims "$dim" \
      --max-seq-length "$seq_length" \
      --eval-batch-size "$batch_size" \
      --no-bm25 \
      --output-dir "$output_dir" 2>&1 | tee "$output_dir/eval.log"; then
    echo "=== Finished $model_id ==="
  else
    failed_models+=("$model_id")
    echo "=== Failed $model_id; see $output_dir/eval.log ===" >&2
  fi
done <<'MODELS'
GreenNode/GreenNode-Embedding-Large-VN-V1 1024 8192 16 greennode-large
VoVanPhuc/sup-SimCSE-VietNamese-phobert-base 768 256 32 simcse-phobert
keepitreal/vietnamese-sbert 768 256 32 vietnamese-sbert
bkai-foundation-models/vietnamese-bi-encoder 768 256 32 bkai-bi-encoder
aisingapore/SEA-LION-ModernBERT-Embedding-600M 1024 8192 16 sealion-600m
aisingapore/SEA-LION-ModernBERT-Embedding-300M 768 8192 32 sealion-300m
QuangDuy/gn-trvn-bert-base-stage2 768 4096 2 gn-trvn-bert-base-stage2
QuangDuy/gn-trvn-bamibert 768 2048 4 gn-trvn-bamibert
QuangDuy/gn-trvn-bert-tiny-stage2 384 4096 8 gn-trvn-bert-tiny-stage2
MODELS

if (( ${#failed_models[@]} )); then
  printf 'Evaluation failed for: %s\n' "${failed_models[*]}" >&2
  exit 1
fi
