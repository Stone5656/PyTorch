from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from utils.train.early_stopping import EarlyStopping
from utils.visualize.image_classification import plot_learning_curve


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Callable,
    device: torch.device,
    logger: Optional[Callable] = None,
    epoch: int = 0,
) -> float:
    model.train()
    running_loss = 0.0

    loop = tqdm(dataloader, desc=f"[Train] Epoch {epoch}", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
         # --- Mixup適用 ---
        images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4)
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader.dataset)
    if logger:
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
    logger: Optional[Callable] = None,
    epoch: int = 0,
) -> tuple[float, float]:
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"[Val] Epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = val_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    if logger:
        logger.info(f"Epoch {epoch} - Val Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    return avg_loss, accuracy


from tqdm.auto import tqdm

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    logger: Optional[Callable] = None,
    checkpoint_path: Path = None,
):
    train_losses = []
    val_losses = []
    val_accuracies = []
    early_stopping = EarlyStopping(patience=5, delta=1e-4, path=checkpoint_path)

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, logger, epoch
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, logger, epoch
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            if logger:
                logger.info(f"Early stopping at epoch {epoch}")
            break

    if logger:
        logger.info("学習曲線をプロットします")
    plot_learning_curve(
        train_losses,
        val_losses,
        val_accuracies,
        save_path=Path("out/learning_curve.png")
    )

def mixup_data(x, y, alpha=1.0):
    """
    Mixupによる入力・ラベルの合成
    """
    if alpha <= 0:
        return x, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
