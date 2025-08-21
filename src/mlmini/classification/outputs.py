"""成果物（図・レポート・モデル・クラス名・メトリクス）の保存。

本モジュールに、片側専用の可視化関数も集約する：
- plot_inputs_grid: 入力（訓練）グリッドのみ
- plot_detected_grid_only: 検出結果（T/P）グリッドのみ
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from mlmini.classification.data import NORMALIZATION_MEAN, NORMALIZATION_STD
from mlmini.classification.data import DataLoaders
from mlmini.classification.evaluation import evaluate_accuracy
from mlmini.utils.visualization import (
    plot_confusion_matrix,
    plot_confusion_matrix_normalized,
    plot_learning_curves,
)

# ========= 片側専用の可視化関数（ここに集約） =========

def plot_inputs_grid(
    images,
    output_path: str,
    normalization_mean=None,
    normalization_std=None,
    max_samples: int = 32,
):
    """入力画像だけを並べたグリッドを保存する（注釈なし）。"""
    import numpy as _np
    import torch as _torch
    import matplotlib.pyplot as plt
    import math as _math

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

    n = len(images)
    if n == 0:
        plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "No training samples", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    take = int(min(max_samples, n))
    cols = min(8, max(3, int(_math.ceil(take ** 0.5))))
    rows = int(_math.ceil(take / cols))
    imgs = [_to_numpy(images[i]) for i in range(take)]

    plt.figure(figsize=(cols * 2.4, rows * 2.4))
    for k in range(take):
        ax = plt.subplot(rows, cols, k + 1)
        ax.imshow(imgs[k])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_detected_grid_only(
    images,
    targets,
    predictions,
    class_names,
    output_path_detected: str,
    normalization_mean=None,
    normalization_std=None,
    max_samples: int = 32,
):
    """検出結果（T/P）だけを1枚のグリッド画像として保存する。

    表示仕様:
        - 左上タイトルは「T: 真値」（常に黒）
        - 右肩に「P: 予測」を表示し、誤検出時のみ赤色・正解時は黒
        - 画像の枠色は 正=緑 / 誤=赤
        - 入力画像はデノーマライズして表示（mean/std が与えられた場合）
    """
    import numpy as _np
    import torch as _torch
    import matplotlib.pyplot as plt
    import math as _math

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
        plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "No samples", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path_detected, dpi=150)
        plt.close()
        return

    take = int(min(max_samples, n))
    cols = min(8, max(3, int(_math.ceil(take ** 0.5))))
    rows = int(_math.ceil(take / cols))
    imgs = [_to_numpy(images[i]) for i in range(take)]

    plt.figure(figsize=(cols * 2.4, rows * 2.6))
    for k in range(take):
        ax = plt.subplot(rows, cols, k + 1)
        ax.imshow(imgs[k]); ax.axis("off")

        true_idx = int(targets_arr[k]) if k < len(targets_arr) else 0
        pred_idx = int(preds_arr[k]) if k < len(preds_arr) else 0
        true_name = class_names[true_idx] if true_idx < len(class_names) else str(true_idx)
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        correct = (true_idx == pred_idx)

        # 枠色
        frame_color = "tab:green" if correct else "tab:red"
        for spine in ax.spines.values():
            spine.set_edgecolor(frame_color)
            spine.set_linewidth(2.0)

        # 左: 真値（黒）
        ax.set_title(f"T:{true_name}", fontsize=8, loc="left", color="black")
        # 右: 予測（誤検出のみ赤）
        ax.text(
            0.98, 1.02, f"P:{pred_name}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8, color=("tab:red" if not correct else "black"),
        )

    plt.tight_layout()
    plt.savefig(output_path_detected, dpi=150)
    plt.close()

# ========= ここまで片側関数 =========


def save_all_evaluation_artifacts(
    model: torch.nn.Module,
    loaders: DataLoaders,
    device: torch.device,
    output_weight_directory: str,
    history_train_losses: Sequence[float],
    history_valid_losses: Sequence[float],
    history_train_accuracies: Sequence[float],
    history_valid_accuracies: Sequence[float],
    best_valid_accuracy: float,
    epochs: int,
) -> Tuple[float, str]:
    """評価・可視化・保存を一括で行い、最終精度と保存先を返す。"""
    # --- 最終評価 ---
    valid_accuracy, targets_array, predictions_array = evaluate_accuracy(model, loaders.valid_loader, device)

    # --- 混同行列 ---
    if targets_array.size == 0:
        num_classes = len(loaders.class_names)
        cm = np.zeros((num_classes, num_classes), dtype=int)
    else:
        cm = confusion_matrix(
            targets_array, predictions_array,
            labels=list(range(len(loaders.class_names))),
        )

    # --- 可視化（混同行列・学習曲線） ---
    plot_confusion_matrix(
        confusion_matrix=cm,
        labels=loaders.class_names,
        output_path=os.path.join(output_weight_directory, "confusion_matrix.png"),
    )
    plot_confusion_matrix_normalized(
        confusion_matrix=cm,
        labels=loaders.class_names,
        output_path=os.path.join(output_weight_directory, "confusion_matrix_normalized.png"),
    )
    plot_learning_curves(
        train_losses=history_train_losses,
        valid_losses=history_valid_losses,
        train_accuracies=history_train_accuracies,
        valid_accuracies=history_valid_accuracies,
        output_path=os.path.join(output_weight_directory, "learning_curve.png"),
    )

    # --- 片側グリッド: 訓練=入力, 検証=検出 ---
    # 入力グリッド（訓練バッチの先頭）
    try:
        train_first_batch = next(iter(loaders.train_loader))
        train_images_batch = train_first_batch[0]
        plot_inputs_grid(
            images=train_images_batch,
            output_path=os.path.join(output_weight_directory, "validation_inputs_grid.png"),
            normalization_mean=NORMALIZATION_MEAN,
            normalization_std=NORMALIZATION_STD,
            max_samples=32,
        )
    except StopIteration:
        pass

    # 検出グリッド（検証バッチの先頭）
    try:
        valid_first_batch = next(iter(loaders.valid_loader))
        images_batch, targets_batch = valid_first_batch
        with torch.no_grad():
            outputs_batch = model(images_batch.to(device))
            _, predictions_batch = torch.max(outputs_batch, dim=1)
        plot_detected_grid_only(
            images=images_batch,
            targets=targets_batch,
            predictions=predictions_batch,
            class_names=loaders.class_names,
            output_path_detected=os.path.join(output_weight_directory, "validation_detected_grid.png"),
            normalization_mean=NORMALIZATION_MEAN,
            normalization_std=NORMALIZATION_STD,
            max_samples=32,
        )
    except StopIteration:
        pass

    # --- モデル＆クラス名 ---
    model_path = os.path.join(output_weight_directory, "model.pt")
    classes_path = os.path.join(output_weight_directory, "classes.json")
    torch.save(model.state_dict(), model_path)
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(loaders.class_names, f, ensure_ascii=False, indent=2)

    # --- メトリクス ---
    metrics_path = os.path.join(output_weight_directory, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_valid_accuracy": float(best_valid_accuracy),
                "final_valid_accuracy": float(valid_accuracy),
                "epochs": int(epochs),
                "num_classes": int(len(loaders.class_names)),
                "class_names": loaders.class_names,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --- クラス別レポート（pandas が無くても落とさない） ---
    try:
        from sklearn.metrics import classification_report
        import pandas as _pd  # noqa: N816
        report_dict = classification_report(
            targets_array,
            predictions_array,
            labels=list(range(len(loaders.class_names))),
            target_names=loaders.class_names,
            output_dict=True,
            zero_division=0,
        )
        report_dataframe = _pd.DataFrame(report_dict).transpose()
        report_dataframe.to_csv(
            os.path.join(output_weight_directory, "classification_report.csv"),
            index=True,
            encoding="utf-8",
        )
    except Exception:
        pass

    return valid_accuracy, output_weight_directory
