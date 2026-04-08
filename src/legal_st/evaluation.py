from __future__ import annotations

import contextlib
import csv
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader


def _is_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"


@contextlib.contextmanager
def _patch_model_in_loss(loss_module: nn.Module, model: SentenceTransformer):
    """
    Temporarily replace every '.model' attribute inside *loss_module* that is
    an nn.Module with the freshly-updated *model* passed by the Trainer.

    Background: SentenceTransformerTrainer.compute_loss() replaces loss.model
    with the DDP-wrapped model on the first training step.  The evaluator
    receives the *unwrapped* SentenceTransformer, but loss_model.model still
    points to the (potentially stale) DDP wrapper.  This context manager
    patches the reference for the duration of the eval forward pass so that
    the evaluator always uses the model weights that the Trainer considers
    up-to-date.
    """
    saved: list[tuple[nn.Module, nn.Module]] = []

    def _replace(m: nn.Module) -> None:
        for name, child in list(m.named_children()):
            if name == "model" and isinstance(child, nn.Module):
                saved.append((m, child))
                setattr(m, name, model)
            else:
                _replace(child)

    _replace(loss_module)
    try:
        yield
    finally:
        for parent, original in saved:
            parent.model = original


class LossEvaluator(SentenceEvaluator):
    def __init__(
        self,
        examples: Iterable[InputExample],
        loss_model,
        *,
        batch_size: int = 32,
        name: str = "validation",
    ) -> None:
        self.examples = list(examples)
        self.loss_model = loss_model
        self.batch_size = batch_size
        self.name = name
        self.primary_metric = f"{name}_loss"
        self.greater_is_better = False

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        if not self.examples:
            return {self.primary_metric: 0.0}

        dataloader = DataLoader(
            self.examples,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.smart_batching_collate,
        )

        was_training = model.training
        model.eval()
        self.loss_model.eval()

        total_loss = 0.0
        total_batches = 0

        # Patch loss_model so its internal .model references point to the
        # current (unwrapped) SentenceTransformer instead of a DDP wrapper.
        with torch.no_grad(), _patch_model_in_loss(self.loss_model, model):
            for features, labels in dataloader:
                features = [
                    {
                        key: value.to(model.device) if hasattr(value, "to") else value
                        for key, value in sentence_features.items()
                    }
                    for sentence_features in features
                ]
                if hasattr(labels, "to"):
                    labels = labels.to(model.device)

                loss = self.loss_model(features, labels)
                total_loss += loss.detach().float().item()
                total_batches += 1

        if was_training:
            model.train()
        self.loss_model.train()

        mean_loss = total_loss / max(total_batches, 1)

        if _is_main_process():
            print(
                f"[{self.name}] loss={mean_loss:.6f}  epoch={epoch}  steps={steps}"
            )

            if output_path is not None:
                csv_path = Path(output_path) / f"{self.name}_results.csv"
                write_header = not csv_path.exists()
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["epoch", "steps", self.primary_metric]
                    )
                    if write_header:
                        writer.writeheader()
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "steps": steps,
                            self.primary_metric: f"{mean_loss:.6f}",
                        }
                    )

        return {self.primary_metric: mean_loss}
