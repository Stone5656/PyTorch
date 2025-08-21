"""データ読み込みと前処理の定義。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import torch


# 学習・検証の前処理で使用する正規化（ResNet 系の標準値）
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]


@dataclass
class DataLoaders:
    """学習と検証用の DataLoader とクラス名。"""
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    class_names: List[str]


def ensure_torchvision() -> None:
    """torchvision の存在を確認し、無い場合は分かりやすいエラーにする。"""
    try:
        import torchvision  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "torchvision が見つかりません。画像分類には torchvision が必要です。\n"
            "インストール例: pip install torchvision"
        ) from exc


def create_data_loaders(dataset_directory: str, batch_size: int, num_workers: int) -> DataLoaders:
    """ImageFolder 形式のデータセットから DataLoader を構築する。

    仕様:
        - dataset/train と dataset/val があればそれらを使用。
        - なければ dataset/ 直下を 8:2 に分割して学習/検証を作成。
    """
    ensure_torchvision()
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])
    transform_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])

    train_directory = os.path.join(dataset_directory, "train")
    valid_directory = os.path.join(dataset_directory, "val")

    if os.path.isdir(train_directory) and os.path.isdir(valid_directory):
        dataset_train = torchvision.datasets.ImageFolder(train_directory, transform=transform_train)
        dataset_valid = torchvision.datasets.ImageFolder(valid_directory, transform=transform_valid)
        class_names = list(dataset_train.classes)
    else:
        dataset_full = torchvision.datasets.ImageFolder(dataset_directory, transform=transform_train)
        class_names = list(dataset_full.classes)
        dataset_size = len(dataset_full)
        valid_size = max(1, int(dataset_size * 0.2))
        train_size = dataset_size - valid_size
        dataset_train, dataset_valid_raw = random_split(dataset_full, [train_size, valid_size])
        # 検証セットには検証用 transform を適用
        dataset_valid_raw.dataset.transform = transform_valid  # type: ignore[attr-defined]
        dataset_valid = dataset_valid_raw

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return DataLoaders(train_loader=train_loader, valid_loader=valid_loader, class_names=class_names)
