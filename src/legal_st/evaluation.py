from __future__ import annotations

from typing import Iterable

import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader


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

        loss_model = self.loss_model.to(model.device)
        was_training = model.training
        loss_was_training = loss_model.training
        model.eval()
        loss_model.eval()

        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
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

                loss = loss_model(features, labels)
                total_loss += loss.detach().float().item()
                total_batches += 1

        if was_training:
            model.train()
        if loss_was_training:
            loss_model.train()

        mean_loss = total_loss / max(total_batches, 1)
        if output_path:
            print(
                f"Validation Loss on {self.name}: {mean_loss:.6f} "
                f"(epoch={epoch}, steps={steps})"
            )

        return {self.primary_metric: mean_loss}
