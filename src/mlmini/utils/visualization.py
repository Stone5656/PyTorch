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

# === 追加: 分類用の拡張可視化 ===
def plot_learning_curves(
    train_losses, valid_losses,
    train_accuracies=None, valid_accuracies=None,
    output_path: str = "learning_curve.png",
):
    """学習曲線（loss と accuracy）を一枚にプロットして保存する。

    Args:
        train_losses: 各エポックの学習損失のシーケンス。
        valid_losses: 各エポックの検証損失のシーケンス。
        train_accuracies: 各エポックの学習精度。None の場合は描画しない。
        valid_accuracies: 各エポックの検証精度。None の場合は描画しない。
        output_path: 出力ファイルパス。
    """
    import numpy as _np
    epochs = _np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    # Loss
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, valid_losses, label="Valid Loss")
    # Accuracy (任意)
    if train_accuracies is not None and valid_accuracies is not None:
        ax2 = plt.gca().twinx()
        ax2.plot(epochs, train_accuracies, linestyle=":", label="Train Acc")
        ax2.plot(epochs, valid_accuracies, linestyle=":", label="Valid Acc")
        ax2.set_ylabel("Accuracy")
        # 凡例を両軸から集約
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc="best")
    else:
        plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix_normalized(confusion_matrix, labels, output_path: str):
    """行方向に正規化した混同行列を画像として保存する。

    Args:
        confusion_matrix: 2次元配列（shape: [num_classes, num_classes]）
        labels: クラス名のシーケンス
        output_path: 出力ファイルパス
    """
    import numpy as _np
    cm = _np.asarray(confusion_matrix, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    # 0除算を回避（ゼロ行はゼロのまま）
    cm_norm = _np.divide(cm, row_sums, out=_np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(max(4, len(labels)*0.8), max(3, len(labels)*0.8)))
    ax = plt.gca()
    im = ax.imshow(cm_norm, aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized)")

    # セル数値（小数2桁）
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_misclassified_grid(
    images, targets, predictions, class_names, output_path: str, max_samples: int = 16
):
    """誤分類サンプルをグリッドで可視化して保存する。

    Args:
        images: 画像テンソル (N, C, H, W) または PIL 画像の配列。
        targets: 真のラベル（int列）。
        predictions: 予測ラベル（int列）。
        class_names: クラス名リスト。
        output_path: 出力ファイルパス。
        max_samples: 最大表示枚数。
    """
    import numpy as _np
    import torch as _torch

    # 誤分類インデックス抽出
    targets = _np.asarray(targets)
    predictions = _np.asarray(predictions)
    wrong_idx = _np.where(targets != predictions)[0]

    if wrong_idx.size == 0:
        plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "No misclassifications", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    take = min(max_samples, wrong_idx.size)
    sel = wrong_idx[:take]

    # 画像を numpy(H, W, C) に変換
    def _to_numpy(img):
        if isinstance(img, _torch.Tensor):
            # (C,H,W) -> (H,W,C), 0-1範囲に仮定
            arr = img.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = _np.transpose(arr, (1, 2, 0))
            return _np.clip(arr, 0, 1)
        try:
            # PIL 画像
            return _np.asarray(img) / 255.0
        except Exception:
            return _np.zeros((64, 64, 3), dtype=float)

    imgs_np = [_to_numpy(images[i]) for i in sel]
    rows = int(_np.ceil(take / 4))
    cols = min(4, take)

    plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for k, i in enumerate(sel):
        ax = plt.subplot(rows, cols, k + 1)
        ax.imshow(imgs_np[k])
        true_name = class_names[int(targets[i])]
        pred_name = class_names[int(predictions[i])]
        ax.set_title(f"T:{true_name} / P:{pred_name}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_input_and_detected_grids(
    images,
    targets,
    predictions,
    class_names,
    output_path_input: str,
    output_path_detected: str,
    normalization_mean=None,
    normalization_std=None,
    max_samples: int = 32,
):
    """入力グリッドと検出結果グリッドの2枚を保存する。"""
    import numpy as _np
    import torch as _torch
    import matplotlib.pyplot as plt

    def _to_numpy(img):
        if isinstance(img, _torch.Tensor):
            arr = img.detach().cpu().float().clone()
            if normalization_mean is not None and normalization_std is not None and arr.ndim == 3 and arr.shape[0] in (1, 3):
                for c in range(arr.shape[0]):
                    arr[c] = arr[c] * float(normalization_std[c]) + float(normalization_mean[c])
                arr = arr.clamp(0.0, 1.0)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.numpy()
                arr = _np.transpose(arr, (1, 2, 0))
                return _np.clip(arr, 0.0, 1.0)
            return _np.zeros((224, 224, 3), dtype=float)
        try:
            arr = _np.asarray(img).astype("float32") / 255.0
            if arr.ndim == 2:
                arr = _np.stack([arr] * 3, axis=-1)
            return _np.clip(arr, 0.0, 1.0)
        except Exception:
            return _np.zeros((224, 224, 3), dtype=float)

    targets_arr = _np.asarray(targets)
    preds_arr = _np.asarray(predictions)
    n = len(images)
    if n == 0:
        for path in (output_path_input, output_path_detected):
            plt.figure(figsize=(5, 3))
            plt.text(0.5, 0.5, "No samples", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
        return

    take = int(min(max_samples, n))
    import math as _math
    cols = min(8, max(3, int(_math.ceil(take ** 0.5))))
    rows = int(_math.ceil(take / cols))
    imgs = [_to_numpy(images[i]) for i in range(take)]

    # 入力
    plt.figure(figsize=(cols * 2.4, rows * 2.4))
    for k in range(take):
        ax = plt.subplot(rows, cols, k + 1)
        ax.imshow(imgs[k]); ax.axis("off")
    plt.tight_layout(); plt.savefig(output_path_input, dpi=150); plt.close()

    # 検出結果
    plt.figure(figsize=(cols * 2.4, rows * 2.6))
    for k in range(take):
        ax = plt.subplot(rows, cols, k + 1)
        ax.imshow(imgs[k]); ax.axis("off")
        ti = int(targets_arr[k]) if k < len(targets_arr) else 0
        pi = int(preds_arr[k]) if k < len(preds_arr) else 0
        tname = class_names[ti] if ti < len(class_names) else str(ti)
        pname = class_names[pi] if pi < len(class_names) else str(pi)
        ok = (ti == pi)
        ax.set_title(f"T:{tname} / P:{pname}", fontsize=8)
        color = "tab:green" if ok else "tab:red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.0)
    plt.tight_layout(); plt.savefig(output_path_detected, dpi=150); plt.close()
