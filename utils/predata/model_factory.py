# utils/model_factory.py

import torch.nn as nn
import torchvision.models as models


def get_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    指定されたモデル名に応じた画像分類モデルを返す。

    Parameters
    ----------
    name : str
        モデル名（例: "cnn", "resnet18", "mobilenet_v2"）
    num_classes : int
        出力クラス数
    pretrained : bool
        torchvisionモデルをpretrained weightsで初期化するか

    Returns
    -------
    nn.Module
        構築されたPyTorchモデル
    """
    if name == "cnn":
        return BetterCNN(num_classes=num_classes)
    elif name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model name: {name}")


class BetterCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 最終特徴量を [batch, 256, 1, 1] に固定化
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # → [batch, 256]
            nn.Linear(256, 128), nn.ReLU(),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

