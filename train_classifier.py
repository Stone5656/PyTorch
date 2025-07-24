from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from utils.lib_conf.logging_conf import get_logger
from utils.lib_conf.matplotlib_conf import setup_matplotlib_style
from utils.lib_conf.numpy_conf import configure_numpy_print
from utils.train.image_classification import train_model
from utils.predata.image_dataloader import get_image_loaders
from utils.predata.model_factory import get_model
from utils.visualize.image_classification import (
    plot_confusion_matrix,
    plot_sample_predictions,
    save_misclassified_images,
    visualize_loader_samples,
)

# ===== 設定 =====
IMAGE_DIR = Path("dataset")
IMAGE_SIZE = 224
BATCH_SIZE = 10
NUM_EPOCHS = 50
MODEL_NAME = "resnet18"  # "cnn" or "resnet18" or "mobilenet_v2"
PRETRAINED = True
LR = 1e-3
LOG_NAME = "image_classification"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = Path("out/best_model.pt")

# ===== 環境準備 =====
configure_numpy_print()
setup_matplotlib_style()
logger = get_logger(name=LOG_NAME)

# ===== データ準備 =====
train_loader, val_loader, class_names = get_image_loaders(
    IMAGE_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    augment=True,
)

logger.info(f"クラス数: {len(class_names)} → {class_names}")
logger.info(f"訓練画像数: {len(train_loader.dataset)}, 検証画像数: {len(val_loader.dataset)}")

# ✅ 最大20枚まで表示して保存
visualize_loader_samples(train_loader, class_names, title="Train Dataset Sample", max_display=len(train_loader.dataset))
visualize_loader_samples(val_loader, class_names, title="Validation Dataset Sample", max_display=len(val_loader.dataset))

# ===== モデル定義 =====
model = get_model(MODEL_NAME, num_classes=len(class_names), pretrained=PRETRAINED)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===== 学習開始 =====
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    logger=logger,
    checkpoint_path=checkpoint_path,
)

# ===== 推論 & 可視化 =====
all_preds = []
all_labels = []
all_images = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_images.append(images.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
all_images = torch.cat(all_images)

# ✅ 混同行列を保存
plot_confusion_matrix(
    all_labels.numpy(),
    all_preds.numpy(),
    class_names,
    save_path=Path("out/confusion_matrix.png")
)

# ✅ 正解/予測画像を保存（色分け表示）
plot_sample_predictions(
    images=all_images,
    labels=all_labels,
    preds=all_preds,
    class_names=class_names,
    save_path=Path("out/sample_predictions.png"),
)

# ✅ 間違えた画像を保存
save_misclassified_images(
    all_images,
    all_labels,
    all_preds,
    class_names,
    output_dir=Path("out/misclassified")
)
