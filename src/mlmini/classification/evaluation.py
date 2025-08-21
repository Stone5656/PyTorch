"""評価用ユーティリティ。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def evaluate_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """精度（accuracy）と予測/正解の配列を返す。"""
    model.eval()
    correct = 0
    total = 0
    all_targets: List[int] = []
    all_predictions: List[int] = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predicted.detach().cpu().tolist())
    accuracy = correct / max(1, total)
    return accuracy, np.array(all_targets), np.array(all_predictions)


def evaluate_with_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """精度と平均損失を返す。"""
    model.eval()
    correct = 0
    total = 0
    summed_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            summed_loss += float(loss.item()) * targets.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predicted.detach().cpu().tolist())
    accuracy = correct / max(1, total)
    average_loss = summed_loss / max(1, total)
    return accuracy, average_loss, np.array(all_targets), np.array(all_predictions)
