"""モデル定義（ResNet18 をベースに最終層のみ置換）。"""

from __future__ import annotations

import torch


def build_resnet18_head(num_classes: int, device: torch.device) -> torch.nn.Module:
    """ResNet18（weights=None）に最終全結合を付け替えて返す。"""
    import torchvision.models as models
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features  # type: ignore[attr-defined]
    model.fc = torch.nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    return model.to(device)
