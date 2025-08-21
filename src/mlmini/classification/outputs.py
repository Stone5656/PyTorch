"""成果物（図・レポート・モデル・クラス名・メトリクス）の保存。"""

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
    plot_input_and_detected_grids,
)


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

    # --- 可視化 ---
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

    # 入力/検出 2 グリッド（検証バッチの先頭のみ・最大32枚）
    try:
        first_batch = next(iter(loaders.valid_loader))
        images_batch, targets_batch = first_batch
        with torch.no_grad():
            outputs_batch = model(images_batch.to(device))
            _, predictions_batch = torch.max(outputs_batch, dim=1)
        plot_input_and_detected_grids(
            images=images_batch,
            targets=targets_batch,
            predictions=predictions_batch,
            class_names=loaders.class_names,
            output_path_input=os.path.join(output_weight_directory, "validation_inputs_grid.png"),
            output_path_detected=os.path.join(output_weight_directory, "validation_detected_grid.png"),
            normalization_mean=NORMALIZATION_MEAN,
            normalization_std=NORMALIZATION_STD,
            max_samples=32,
        )
    except StopIteration:
        # 検証データなし
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
