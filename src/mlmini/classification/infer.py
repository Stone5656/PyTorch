"""分類モデルの推論処理（スケルトン）。"""

from __future__ import annotations
from typing import Callable, Tuple


def load_classification_predictor(model_directory: str) -> Callable[[str], Tuple[str, float]]:
    """保存済み分類モデルを読み込み推論器を返す（プレースホルダ）。"""
    def _predict(image_path: str) -> Tuple[str, float]:
        _ = image_path
        return "unknown", 0.0
    return _predict


def cli_classification_predict(args) -> None:  # type: ignore[no-untyped-def]
    """CLI から分類推論を行うラッパ（プレースホルダ）。"""
    _ = args
    print('clf-predict (スケルトン): {"label": "unknown", "score": 0.0}')
