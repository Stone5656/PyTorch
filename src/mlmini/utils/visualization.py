"""可視化用ユーティリティ（スケルトン）。

Matplotlib等を用いた可視化関数をここにまとめます。
初回コミットでは関数の形だけを定義しています。
"""

from __future__ import annotations
from typing import Sequence


def plot_ideal_line(true_values, predicted_values, output_path: str) -> None:  # type: ignore[no-untyped-def]
    """理想線（y=x）と予測値の散布図を描画するプレースホルダ。"""
    pass


def plot_feature_importance(weights, feature_names: Sequence[str], output_path: str) -> None:  # type: ignore[no-untyped-def]
    """特徴量の重要度や回帰係数を棒グラフで描画するプレースホルダ。"""
    pass


def plot_confusion_matrix(confusion_matrix, labels: Sequence[str], output_path: str) -> None:  # type: ignore[no-untyped-def]
    """混同行列を可視化するプレースホルダ。"""
    pass
