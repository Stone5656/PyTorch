"""回帰モデルの学習処理（スケルトン）。

既存の linner.py のロジックを今後移植します。
"""

from __future__ import annotations


def train_regression_model(output_base_directory: str = "./out", device: str = "cpu") -> str:
    """回帰モデルを学習して成果物を保存する（プレースホルダ）。"""
    return output_base_directory


def cli_regression_train(args) -> None:  # type: ignore[no-untyped-def]
    """CLI から回帰モデルを学習するラッパ（プレースホルダ）。"""
    _ = args
    print("regression-train (スケルトン): まだ処理は未実装です。")


def train_main() -> None:
    """既存 linner.py から呼ばれるエントリポイント（プレースホルダ）。"""
    print("train_main (回帰, スケルトン): まだ処理は未実装です。")
