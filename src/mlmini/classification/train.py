"""分類モデルの学習処理（実装版）。

目的:
    画像分類タスクを最小構成で学習できるようにします。
    依存は PyTorch + torchvision（可能なら）を利用します。
    既存の挙動を壊さないため、成果物の保存はシンプルに行います。

入出力:
    - 入力: ImageFolder 形式のデータセット（`--dataset-directory`）
      例1) dataset/{classA,classB,...}               -> 内部で 8:2 の学習/検証分割
      例2) dataset/train/{class...}, dataset/val/{class...} -> そのまま使用
    - 出力（out/weightN/ 直下に作成）:
        * model.pt                         (state_dict)
        * classes.json                     (クラス名のリスト)
        * confusion_matrix.png             (検証データに対する混同行列)
        * confusion_matrix_normalized.png  (行正規化した混同行列)
        * learning_curve.png               (学習曲線)
        * classification_report.csv        (各クラスのレポート)
        * metrics.json                     (学習/検証の最終精度など)

注意:
    - デバイスは "cpu" / "cuda" を受け付けます（利用可否は環境依存）。
    - torchvision が無い場合はエラーメッセージを出して終了します（依存を暗黙追加しない方針）。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from mlmini.utils.file_io import allocate_next_weight_directory, ensure_output_directory
from mlmini.utils.utilities_common import set_global_random_seed, elapsed_timer
from mlmini.utils.visualization import (
    plot_confusion_matrix,
    plot_confusion_matrix_normalized,
    plot_learning_curves,
    plot_misclassified_grid,
)


@dataclass
class DataLoaders:
    """学習と検証用の DataLoader セット。"""
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    class_names: List[str]


def _ensure_torchvision() -> None:
    """torchvision の存在を確認し、無ければ分かりやすい例外を投げる。"""
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "torchvision が見つかりません。分類の学習には torchvision が必要です。\n"
            "インストール例: pip install torchvision"
        ) from exc


def _create_data_loaders(dataset_directory: str, batch_size: int, num_workers: int) -> DataLoaders:
    """ImageFolder 形式のデータセットから DataLoader を構築する。

    ルール:
        - dataset/train と dataset/val が存在する場合はそれらを使用。
        - そうでなければ dataset/ を丸ごと読み、内部で 8:2 に分割する。
    """
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split

    # 画像前処理（ResNet18 を想定）
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])
    transform_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])

    train_directory = os.path.join(dataset_directory, "train")
    valid_directory = os.path.join(dataset_directory, "val")
    if os.path.isdir(train_directory) and os.path.isdir(valid_directory):
        dataset_train = torchvision.datasets.ImageFolder(train_directory, transform=transform_train)
        dataset_valid = torchvision.datasets.ImageFolder(valid_directory, transform=transform_valid)
        class_names = list(dataset_train.classes)
    else:
        # ルート直下がクラスディレクトリのケース
        dataset_full = torchvision.datasets.ImageFolder(dataset_directory, transform=transform_train)
        class_names = list(dataset_full.classes)
        dataset_size = len(dataset_full)
        valid_size = max(1, int(dataset_size * 0.2))
        train_size = dataset_size - valid_size
        dataset_train, dataset_valid_raw = random_split(dataset_full, [train_size, valid_size])
        # 検証セットには検証用の transform を適用する
        dataset_valid_raw.dataset.transform = transform_valid  # type: ignore[attr-defined]
        dataset_valid = dataset_valid_raw

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return DataLoaders(train_loader=train_loader, valid_loader=valid_loader, class_names=class_names)


def _build_model(num_classes: int, device: torch.device) -> torch.nn.Module:
    """ResNet18（weights=None）をベースに出力数をクラス数へ合わせる。"""
    import torchvision.models as models
    model = models.resnet18(weights=None)  # 追加依存を避けるため事前学習なし
    # 最終全結合を差し替え
    in_features = model.fc.in_features  # type: ignore[attr-defined]
    model.fc = torch.nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    return model.to(device)


def _evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
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


def _evaluate_with_loss(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, criterion: torch.nn.Module) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """精度と損失（average）を併せて返す評価関数。"""
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


def train_classification_model(
    dataset_directory: str,
    output_base_directory: str = "./out",
    epochs: int = 5,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 2,
    learning_rate: float = 1e-3,
) -> str:
    """分類モデルを学習して成果物（モデル、クラス名、混同行列、メトリクス）を保存する。

    Args:
        dataset_directory: ImageFolder 形式のデータセットディレクトリ。
        output_base_directory: 成果物を保存する基準ディレクトリ（内部で weightN を採番）。
        epochs: 学習エポック数。
        device: "cpu" または "cuda" を指定。
        batch_size: ミニバッチサイズ。
        num_workers: DataLoader のワーカー数。
        learning_rate: 最適化の学習率。

    Returns:
        作成された out/weightN ディレクトリの絶対パス。
    """
    _ensure_torchvision()
    set_global_random_seed(42)

    output_base_directory = ensure_output_directory(output_base_directory)
    output_weight_directory = allocate_next_weight_directory(output_base_directory)

    torch_device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    with elapsed_timer("prepare_data"):
        loaders = _create_data_loaders(dataset_directory=dataset_directory, batch_size=batch_size, num_workers=num_workers)
        num_classes = len(loaders.class_names)

    with elapsed_timer("build_model"):
        model = _build_model(num_classes=num_classes, device=torch_device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_valid_accuracy = 0.0
    history_train_losses: List[float] = []
    history_train_accuracies: List[float] = []
    history_valid_losses: List[float] = []
    history_valid_accuracies: List[float] = []

    for epoch in range(1, epochs + 1):
        with elapsed_timer(f"epoch_{epoch}_train"):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            for inputs, targets in loaders.train_loader:
                inputs = inputs.to(torch_device, non_blocking=True)
                targets = targets.to(torch_device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * targets.size(0)
                _, predicted = torch.max(outputs, dim=1)
                running_correct += (predicted == targets).sum().item()
                running_total += targets.size(0)

            train_loss = running_loss / max(1, running_total)
            train_accuracy = running_correct / max(1, running_total)

        with elapsed_timer(f"epoch_{epoch}_eval"):
            valid_accuracy, _, _ = _evaluate(model, loaders.valid_loader, torch_device)

        print(f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} valid_acc={valid_accuracy:.4f}")

        history_train_losses.append(float(train_loss))
        history_train_accuracies.append(float(train_accuracy))
        # 検証損失は loss 付き評価で計算
        valid_accuracy_with_loss, valid_loss, _, _ = _evaluate_with_loss(model, loaders.valid_loader, torch_device, criterion)
        history_valid_losses.append(float(valid_loss))
        history_valid_accuracies.append(float(valid_accuracy_with_loss))

        # 簡易にベスト精度を追跡（必要であればここで都度保存も可能）
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy

    # 最終評価と混同行列の作成
    with elapsed_timer("final_eval_and_save"):
        valid_accuracy, targets_array, predictions_array = _evaluate(model, loaders.valid_loader, torch_device)

        # 検証データが空の場合の安全策
        if targets_array.size == 0:
            print("warning: validation set is empty; skipping evaluation plots.")
            targets_array = np.array([], dtype=int)
            predictions_array = np.array([], dtype=int)
            cm = np.zeros((num_classes, num_classes), dtype=int)
        else:
            cm = confusion_matrix(targets_array, predictions_array, labels=list(range(len(loaders.class_names))))

        # 可視化の保存
        confusion_matrix_path = os.path.join(output_weight_directory, "confusion_matrix.png")
        plot_confusion_matrix(confusion_matrix=cm, labels=loaders.class_names, output_path=confusion_matrix_path)
        confusion_matrix_norm_path = os.path.join(output_weight_directory, "confusion_matrix_normalized.png")
        plot_confusion_matrix_normalized(confusion_matrix=cm, labels=loaders.class_names, output_path=confusion_matrix_norm_path)

        # 学習曲線の保存
        learning_curve_path = os.path.join(output_weight_directory, "learning_curve.png")
        plot_learning_curves(
            train_losses=history_train_losses,
            valid_losses=history_valid_losses,
            train_accuracies=history_train_accuracies,
            valid_accuracies=history_valid_accuracies,
            output_path=learning_curve_path,
        )

        # 誤分類サンプルのグリッドを保存（最大16枚）
        mis_grid_path = os.path.join(output_weight_directory, "misclassified_samples.png")
        # 検証ローダから最初のバッチ画像を取得してグリッド化（全検証分を並べると巨大になるため）
        try:
            first_batch = next(iter(loaders.valid_loader))
            images_batch, targets_batch = first_batch
            with torch.no_grad():
                outputs_batch = model(images_batch.to(torch_device))
                _, predictions_batch = torch.max(outputs_batch, dim=1)
            plot_misclassified_grid(
                images=images_batch,
                targets=targets_batch,
                predictions=predictions_batch,
                class_names=loaders.class_names,
                output_path=mis_grid_path,
            )
        except StopIteration:
            # 検証データが無いケース
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 3))
            plt.text(0.5, 0.5, "No validation samples", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(mis_grid_path, dpi=150)
            plt.close()

        # モデルとクラス名を保存
        model_path = os.path.join(output_weight_directory, "model.pt")
        classes_path = os.path.join(output_weight_directory, "classes.json")
        torch.save(model.state_dict(), model_path)
        with open(classes_path, "w", encoding="utf-8") as file:
            json.dump(loaders.class_names, file, ensure_ascii=False, indent=2)

        # メトリクス保存
        metrics_path = os.path.join(output_weight_directory, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "best_valid_accuracy": float(best_valid_accuracy),
                    "final_valid_accuracy": float(valid_accuracy),
                    "epochs": int(epochs),
                    "num_classes": int(len(loaders.class_names)),
                    "class_names": loaders.class_names,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        # クラス別レポート（CSV）を保存（pandas が無ければスキップ）
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
            report_path = os.path.join(output_weight_directory, "classification_report.csv")
            report_dataframe.to_csv(report_path, index=True, encoding="utf-8")
        except Exception:
            pass

    print(f"saved to: {output_weight_directory}")
    return output_weight_directory


def cli_classification_train(args) -> None:  # type: ignore[no-untyped-def]
    """CLI から分類モデルを学習するラッパ。"""
    dataset_directory = getattr(args, "dataset_directory", None)
    if not dataset_directory:
        raise SystemExit("dataset_directory が指定されていません。--dataset-directory を指定してください。")

    output_base_directory = getattr(args, "output_directory", "./out")
    device = getattr(args, "device", "cpu")
    epochs = getattr(args, "epochs", 5)

    train_classification_model(
        dataset_directory=dataset_directory,
        output_base_directory=output_base_directory,
        epochs=epochs,
        device=device,
    )


def train_main() -> None:
    """既存 train_classifier.py から呼ばれるエントリポイント（互換ラッパ用）。

    Note:
        コミット3のラッパにより、環境変数 MLMINI_REDIRECT=1 のときにのみこの関数が実行されます。
        既存互換として `--data`, `--epochs`, `--out`, `--device` を受け付けます。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="dataset_directory", required=True)
    parser.add_argument("--epochs", dest="epochs", type=int, default=5)
    parser.add_argument("--out", dest="output_directory", default="./out")
    parser.add_argument("--device", dest="device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    train_classification_model(
        dataset_directory=args.dataset_directory,
        output_base_directory=args.output_directory,
        epochs=args.epochs,
        device=args.device,
    )
