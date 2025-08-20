"""可視化用ユーティリティ。

Matplotlib を用いた可視化関数を提供します。
このモジュールは学習フローから呼び出され、画像ファイルを出力します。
"""

from __future__ import annotations

from typing import Sequence, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_ideal_line(true_values: Iterable[float], predicted_values: Iterable[float], output_path: str) -> None:
    """理想線（y=x）と予測結果の散布図を保存する。

    Args:
        true_values: 実測値配列。
        predicted_values: 予測値配列。
        output_path: 画像の出力先パス。
    """
    true_values = np.asarray(list(true_values), dtype=float)
    predicted_values = np.asarray(list(predicted_values), dtype=float)

    min_val = float(np.min([true_values.min(), predicted_values.min()]))
    max_val = float(np.max([true_values.max(), predicted_values.max()]))

    plt.figure()
    plt.scatter(true_values, predicted_values, s=8, alpha=0.6)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (Ideal Line)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(weights: Sequence[float], feature_names: Sequence[str], output_path: str) -> None:
    """特徴量の重要度（回帰係数）を棒グラフで保存する。

    Args:
        weights: 各特徴量の係数。
        feature_names: 特徴量名。
        output_path: 画像の出力先パス。
    """
    weights = np.asarray(list(weights), dtype=float)
    indices = np.arange(len(weights))

    plt.figure(figsize=(max(6, len(weights) * 0.6), 4))
    plt.bar(indices, weights)
    plt.xticks(indices, feature_names, rotation=45, ha="right")
    plt.ylabel("Coefficient")
    plt.title("Feature Importance (Linear Regression Coefficients)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_prediction_grid(
    feature_matrix: np.ndarray,
    target_vector: np.ndarray,
    predicted_vector: np.ndarray,
    feature_names: Sequence[str],
    output_path: str,
) -> None:
    """各特徴量ごとの「実測 vs 予測」の散布図グリッドを保存する。

    Note:
        x軸に各特徴量、y軸に価格（実測/予測）を取り、比較のために同じ図内に重ねる。
        ここでは単純な散布図比較（全特徴量一括学習の予測値を使用）を行う。

    Args:
        feature_matrix: 形状 (N, D) の特徴量（**スケーリング前の元スケール**推奨）。
        target_vector: 実測値 (N,)。
        predicted_vector: 予測値 (N,)。
        feature_names: 特徴量名のシーケンス。
        output_path: 画像の出力先パス。
    """
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    target_vector = np.asarray(target_vector, dtype=float)
    predicted_vector = np.asarray(predicted_vector, dtype=float)

    num_features = feature_matrix.shape[1]
    cols = 3
    rows = int(np.ceil(num_features / cols))

    plt.figure(figsize=(cols * 4, rows * 3))
    for i in range(num_features):
        ax = plt.subplot(rows, cols, i + 1)
        x = feature_matrix[:, i]
        ax.scatter(x, target_vector, s=6, alpha=0.4, label="True")
        ax.scatter(x, predicted_vector, s=6, alpha=0.4, label="Predicted")
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel("Target / Predicted")
        ax.set_title(f"{feature_names[i]} vs Target/Pred")
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_confusion_matrix(confusion_matrix, labels, output_path: str):  # type: ignore[no-untyped-def]
    """混同行列を画像として保存する。

    Args:
        confusion_matrix: 2次元配列（shape: [num_classes, num_classes]）
        labels: クラス名のシーケンス（軸ラベルに使用）
        output_path: 出力ファイルパス（.png など）
    """
    import numpy as _np
    cm = _np.asarray(confusion_matrix)

    plt.figure(figsize=(max(4, len(labels)*0.8), max(3, len(labels)*0.8)))
    ax = plt.gca()
    im = ax.imshow(cm, aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # セル内に数値を表示
    thresh = cm.max()/2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, f"{int(val)}",
                ha="center", va="center",
                fontsize=8,
                color="white" if val > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
