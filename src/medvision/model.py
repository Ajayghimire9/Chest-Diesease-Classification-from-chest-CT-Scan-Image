from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class CompactCNN(nn.Module):
    """Small reference classifier; replaceable without changing the serving layer."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class ModelRunner:
    def __init__(self, weights: str | Path | None = None, num_classes: int = 2) -> None:
        self.model = CompactCNN(num_classes=num_classes).eval()
        if weights:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)

    @torch.inference_mode()
    def predict(self, image_chw: torch.Tensor) -> tuple[int, float]:
        logits = self.model(image_chw.unsqueeze(0) if image_chw.ndim == 3 else image_chw)
        probs = torch.softmax(logits, dim=-1)[0]
        index = int(torch.argmax(probs))
        return index, float(probs[index])
