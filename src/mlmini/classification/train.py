"""分類モデルの学習オーケストレーション（tqdm + 色付きログ対応）。"""

from __future__ import annotations

import argparse
from typing import List

import torch
from tqdm import tqdm

from mlmini.classification.data import DataLoaders, create_data_loaders
from mlmini.classification.modeling import build_resnet18_head
from mlmini.classification.evaluation import evaluate_accuracy, evaluate_with_loss
from mlmini.classification.outputs import save_all_evaluation_artifacts
from mlmini.utils.file_io import allocate_next_weight_directory, ensure_output_directory
from mlmini.utils.utilities_common import set_global_random_seed, elapsed_timer
from mlmini.utils.logging_utils import get_logger

logger = get_logger("image_classification")


def train_classification_model(
    dataset_directory: str,
    output_base_directory: str = "./out",
    epochs: int = 5,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 2,
    learning_rate: float = 1e-3,
) -> str:
    """分類モデルを学習し、成果物を out/weightN/ に保存する。"""
    set_global_random_seed(42)

    output_base_directory = ensure_output_directory(output_base_directory)
    output_weight_directory = allocate_next_weight_directory(output_base_directory)

    torch_device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    with elapsed_timer("prepare_data"):
        loaders: DataLoaders = create_data_loaders(
            dataset_directory=dataset_directory,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        num_classes = len(loaders.class_names)
        # 情報ログ（色付き）
        logger.info(f"[bold]クラス数[/]: {num_classes} → {loaders.class_names}")
        try:
            n_train = len(loaders.train_loader.dataset)
            n_valid = len(loaders.valid_loader.dataset)
            logger.info(f"[bold]訓練画像数[/]: {n_train}, [bold]検証画像数[/]: {n_valid}")
        except Exception:
            pass
        if num_classes == 1:
            logger.warning("[yellow]注意[/]: クラスが1種類のみです。混同行列や一部メトリクスが退化します。")

    with elapsed_timer("build_model"):
        model = build_resnet18_head(num_classes=num_classes, device=torch_device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_valid_accuracy = 0.0
    history_train_losses: List[float] = []
    history_train_accuracies: List[float] = []
    history_valid_losses: List[float] = []
    history_valid_accuracies: List[float] = []

    for epoch in range(1, epochs + 1):
        # ===== Train =====
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        # tqdm プログレスバー（バッチ単位）
        bar = tqdm(
            loaders.train_loader,
            total=len(loaders.train_loader),
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            colour="cyan",
            leave=False,
        )
        for inputs, targets in bar:
            inputs = inputs.to(torch_device, non_blocking=True)
            targets = targets.to(torch_device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * targets.size(0)
            _, predicted = torch.max(outputs, dim=1)
            running_correct += (predicted == targets).sum().item()
            running_total += targets.size(0)

            # バー上に現在の平均損失を表示
            if running_total:
                bar.set_postfix(loss=f"{running_loss/running_total:.4f}")

        train_loss = running_loss / max(1, running_total)
        train_accuracy = running_correct / max(1, running_total)

        # ===== Eval =====
        valid_accuracy, _, _ = evaluate_accuracy(model, loaders.valid_loader, torch_device)
        valid_accuracy_with_loss, valid_loss, _, _ = evaluate_with_loss(
            model, loaders.valid_loader, torch_device, criterion
        )

        history_train_losses.append(float(train_loss))
        history_train_accuracies.append(float(train_accuracy))
        history_valid_losses.append(float(valid_loss))
        history_valid_accuracies.append(float(valid_accuracy_with_loss))

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy

        logger.info(
            f"[bold]Epoch {epoch}/{epochs}[/] "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={valid_loss:.4f} val_acc={valid_accuracy:.4f}"
        )

    # まとめて出力（図・モデル・メトリクス・レポート）
    with elapsed_timer("final_eval_and_save"):
        final_valid_accuracy, saved_dir = save_all_evaluation_artifacts(
            model=model,
            loaders=loaders,
            device=torch_device,
            output_weight_directory=output_weight_directory,
            history_train_losses=history_train_losses,
            history_valid_losses=history_valid_losses,
            history_train_accuracies=history_train_accuracies,
            history_valid_accuracies=history_valid_accuracies,
            best_valid_accuracy=best_valid_accuracy,
            epochs=epochs,
        )

    logger.info(f"[green]saved to[/]: {saved_dir}")
    return saved_dir


# === CLI ラッパ（既存互換） ===
def cli_classification_train(args) -> None:  # type: ignore[no-untyped-def]
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
    """既存 train_classifier.py から呼ばれるエントリポイント（互換）。"""
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
