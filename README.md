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

## Project Layout

- `scripts/train_embedding.py`: fine-tune a Sentence Transformers model
- `scripts/evaluate_retrieval.py`: run dense retrieval evaluation on the legal benchmark
- `scripts/smoke_test.py`: quick dependency and model wiring check
- `src/legal_st/`: reusable loaders, metrics, config, and model builder
