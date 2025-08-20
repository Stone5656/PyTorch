"""分類モデルの学習処理（スケルトン）。"""

from __future__ import annotations


def train_classification_model(
    dataset_directory: str,
    output_base_directory: str = "./out",
    epochs: int = 5,
    device: str = "cpu",
) -> str:
    """分類モデルを学習して成果物を保存する（プレースホルダ）。"""
    _ = (dataset_directory, epochs, device)
    return output_base_directory


def cli_classification_train(args) -> None:  # type: ignore[no-untyped-def]
    """CLI から分類モデルを学習するラッパ（プレースホルダ）。"""
    _ = args
    print("clf-train (スケルトン): まだ処理は未実装です。")


def train_main() -> None:
    """既存 train_classifier.py から呼ばれるエントリポイント（プレースホルダ）。"""
    print("train_main (分類, スケルトン): まだ処理は未実装です。")
