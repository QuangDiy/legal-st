# Legal ST

Training and evaluation scaffold for Vietnamese legal embedding models built with Sentence Transformers.

This workspace is set up to fine-tune and benchmark:

- `QuangDuy/bert-tiny-stage2-hf`
- `QuangDuy/bert-base-stage2-hf`

The pipeline mirrors the reference repo at a high level:

- training data: `batmangiaicuuthegioi/zalo-legal-triplets`
- loss: `MatryoshkaLoss(MultipleNegativesRankingLoss)`
- evaluation benchmark: `another-symato/VMTEB-Zalo-legel-retrieval-wseg`
- retrieval metrics: Accuracy@k, Precision@k, Recall@k, NDCG@k, MRR@k, MAP@100

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
```

Validate the install:

```bash
conda run -n legal-st python scripts/smoke_test.py --config configs/bert-tiny-stage2-hf.yaml
```

Notes:

- The default environment is CPU-safe.
- If you have NVIDIA CUDA available, reinstall `torch` inside the env with the CUDA wheel that matches your machine.

## Train

Train the tiny model:

```bash
conda run -n legal-st python scripts/train_embedding.py --config configs/bert-tiny-stage2-hf.yaml
```

Train the base model:

```bash
conda run -n legal-st python scripts/train_embedding.py --config configs/bert-base-stage2-hf.yaml
```

Artifacts are written to `outputs/`.

## Evaluate

Evaluate the tiny checkpoint:

```bash
conda run -n legal-st python scripts/evaluate_retrieval.py \
  --model-path outputs/bert-tiny-stage2-sbert \
  --config configs/bert-tiny-stage2-hf.yaml \
  --output-dir results/bert-tiny-stage2-sbert
```

Evaluate the base checkpoint:

```bash
conda run -n legal-st python scripts/evaluate_retrieval.py \
  --model-path outputs/bert-base-stage2-sbert \
  --config configs/bert-base-stage2-hf.yaml \
  --output-dir results/bert-base-stage2-sbert
```

The evaluation script writes:

- `results.json`
- `results.md`

## Configs

- `configs/bert-tiny-stage2-hf.yaml`
- `configs/bert-base-stage2-hf.yaml`

The main differences are batch size and Matryoshka dimensions:

- tiny: `[384, 256, 128, 64]`
- base: `[768, 512, 256, 128]`

## GN-TRVN Table Retrieval (Matryoshka)

Fine-tune `bert-tiny-stage2`, `bert-base-stage2` and `Qualcomm-AI-Research/BamiBERT` on
[GreenNode-Table-Markdown-Retrieval-VN](https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN)
and report the same metrics as the
[GreenNode-Embedding-Large-VN-V1](https://huggingface.co/GreenNode/GreenNode-Embedding-Large-VN-V1)
model card: MAP/MRR/NDCG/Recall@5, their Mean, and Hit Rate@1/5/10/20 on
GN-TRVN, Zalo Legal and MTEB VieQuADRetrieval.

GPU environment (Python 3.11, PyTorch 2.4.0 + CUDA 12.4, transformers 4.57.1):

```bash
conda env create -f environment-fa2.yml
conda activate legal-st-fa2
pip install "flash_attn==2.6.3" --no-build-isolation   # RTX 4090 only; skip on T4
```

`precision: auto` and `attn_implementation: auto` pick bf16 + FlashAttention-2 on
Ampere-or-newer GPUs and fp16 + sdpa on T4. FlashAttention-2 only applies to the
ModernBERT models; BamiBERT (RoBERTa) always runs sdpa.

1. Mine hard negatives from the train split (train documents only, no test leakage)
   by running `notebooks/mine_gn_trvn_negatives.ipynb` from the `notebooks/` folder.
   Parameters sit in its first code cell (default: `BAAI/bge-m3`, 3 negatives). It writes
   `data/gn-trvn-hard-negatives/train.parquet` (at most 143,106 rows) and
   `mining_args.json` with the final row count and file size. On Kaggle (GPU T4 x2,
   Internet on, `HF_TOKEN` secret) the notebook installs the pinned libraries, clones
   the repo, encodes on both GPUs and writes to `/kaggle/working/data/gn-trvn-hard-negatives`.

2. Train (each run ends with the retrieval evaluation in `outputs/<run>/retrieval_eval`):

   ```bash
   # 1x RTX 4090
   python scripts/train_embedding.py --config configs/gn-trvn-bert-base-stage2.yaml
   # 2x T4
   torchrun --nproc_per_node 2 scripts/train_embedding.py --config configs/gn-trvn-bert-base-stage2.yaml
   ```

   Configs: `configs/gn-trvn-bert-tiny-stage2.yaml`, `configs/gn-trvn-bert-base-stage2.yaml`,
   `configs/gn-trvn-bamibert.yaml`. Lower `cached_mnrl_mini_batch_size` on out-of-memory;
   it changes memory use, not the loss.

3. Evaluate reference models with the same config, then compare:

   ```bash
   python scripts/evaluate_retrieval.py --config configs/gn-trvn-bert-base-stage2.yaml \
     --model-path GreenNode/GreenNode-Embedding-Large-VN-V1 --max-seq-length 1024 \
     --truncate-dims 1024 --no-bm25 --output-dir results/gn-trvn/greennode-large

   python scripts/compare_results.py \
     tiny=outputs/gn-trvn-bert-tiny-stage2/retrieval_eval \
     base=outputs/gn-trvn-bert-base-stage2/retrieval_eval \
     bamibert=outputs/gn-trvn-bamibert/retrieval_eval \
     greennode=results/gn-trvn/greennode-large \
     --dims all --output results/gn-trvn/comparison.md
   ```

## Project Layout

- `scripts/train_embedding.py`: fine-tune a Sentence Transformers model
- `scripts/evaluate_retrieval.py`: run dense retrieval evaluation on the configured benchmarks
- `notebooks/mine_gn_trvn_negatives.ipynb`: build the GN-TRVN hard-negative training set
- `scripts/compare_results.py`: merge evaluation runs into GreenNode-style comparison tables
- `scripts/smoke_test.py`: quick dependency and model wiring check
- `src/legal_st/`: reusable loaders, metrics, config, and model builder
