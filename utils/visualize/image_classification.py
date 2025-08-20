import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    混同行列をヒートマップで表示・保存する。

    Parameters
    ----------
    y_true : list[int]
        正解ラベルのリスト
    y_pred : list[int]
        予測ラベルのリスト
    class_names : list[str]
        クラス名のリスト
    save_path : Path or None
        保存先パス（指定されなければ表示のみ）
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_sample_predictions(images, labels, preds=None, class_names=None, max_display=16, save_path: Path = None):
    """
    ラベル・予測付きの画像サンプルをグリッドで保存。

    Parameters
    ----------
    images : Tensor[B, C, H, W]
        バッチ画像
    labels : Tensor[B]
        正解ラベル
    preds : Tensor[B] or None
        予測ラベル（省略可能）
    class_names : list[str]
        クラス名
    max_display : int
        表示最大枚数
    save_path : Path or None
        保存パス。Noneなら保存しない
    """
    num = min(len(images), max_display)
    images = images[:num].cpu()
    labels = labels[:num].cpu()
    if preds is not None:
        preds = preds[:num].cpu()

    cols = 10
    rows = math.ceil(num / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i in range(rows * cols):
        ax = axes[i]
        if i < num:
            img = TF.to_pil_image(images[i])
            ax.imshow(img)
            label_str = class_names[labels[i]]

            if preds is not None:
                pred_str = class_names[preds[i]]
                correct = labels[i] == preds[i]
                color = "green" if correct else "red"
                ax.set_title(f"T:{label_str}\nP:{pred_str}", fontsize=8, color=color)
            else:
                ax.set_title(label_str, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_misclassified_images(
    images,
    labels,
    preds,
    class_names,
    output_dir: Path,
    max_save: int = 20
):
    """
    誤分類された画像を指定ディレクトリに保存する。

    Parameters
    ----------
    images : Tensor[B, C, H, W]
        入力画像バッチ
    labels : Tensor[B]
        正解ラベル
    preds : Tensor[B]
        予測ラベル
    class_names : list[str]
        クラス名
    output_dir : Path
        出力先ディレクトリ
    max_save : int
        最大保存数
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i in range(len(images)):
        if labels[i] != preds[i]:
            img = TF.to_pil_image(images[i].cpu())
            true_label = class_names[labels[i]]
            pred_label = class_names[preds[i]]
            filename = output_dir / f"{i:03d}_T_{true_label}_P_{pred_label}.png"
            img.save(filename)
            count += 1
            if count >= max_save:
                break

def plot_learning_curve(
    train_losses: list[float],
    val_losses: list[float],
    val_accuracies: list[float],
    save_path: Path = None
):
    """
    学習曲線（損失と精度）を描画・保存する。

    Parameters
    ----------
    train_losses : list of float
        各エポックの学習損失
    val_losses : list of float
        各エポックの検証損失
    val_accuracies : list of float
        各エポックの検証精度（0.0〜1.0）
    save_path : Path or None
        保存先パス（Noneならplt.show()）
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 損失（左軸）
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o')
    ax1.plot(epochs, val_losses, label='Val Loss', color='orange', marker='o')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # 精度（右軸）
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accuracies, label='Val Accuracy', color='green', marker='x')
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')

    plt.title("Learning Curve")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_loader_samples(loader, class_names, title="", max_display=16):
    images = []
    labels = []

    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
        if sum(len(b) for b in images) >= max_display:
            break

    images = torch.cat(images)[:max_display]
    labels = torch.cat(labels)[:max_display]

    output_dir = Path("out")

    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"{title.lower().replace(' ', '_')}.png"
    plot_sample_predictions(images, labels, preds=None, class_names=class_names, max_display=max_display, save_path=save_path)
