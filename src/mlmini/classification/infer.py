"""分類モデルの推論処理（実装版）。

学習で保存した以下の成果物を読み込み、画像1枚の推論（ラベルとスコア）を返します。
- model.pt           : state_dict（ResNet18の最終層をクラス数に合わせて差し替え）
- classes.json       : クラス名のリスト

CLI:
    mlmini classification-predict --model-directory ./out/weightN --image-path ./samples/cat.jpg
"""

from __future__ import annotations

import json
import os
from typing import Callable, List, Tuple

import torch


def _ensure_torchvision() -> None:
    """torchvision が利用可能かを確認し、無ければ分かりやすい例外を投げる。"""
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "torchvision が見つかりません。分類の推論には torchvision が必要です。\n"
            "インストール例: pip install torchvision"
        ) from exc


def _build_model_for_inference(num_classes: int, device: torch.device) -> torch.nn.Module:
    """ResNet18（weights=None）に最終全結合を付け替えて返す。"""
    import torchvision.models as models
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features  # type: ignore[attr-defined]
    model.fc = torch.nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    model.eval()
    return model.to(device)


def _build_preprocess_transform():
    """学習時と整合する前処理 Transform を構築して返す。"""
    from torchvision import transforms
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])


def load_classification_predictor(model_directory: str) -> Callable[[str], Tuple[str, float]]:
    """保存済み分類モデルを読み込み、画像パス→(ラベル, スコア) を返す推論関数を返す。

    Args:
        model_directory: out/weightN ディレクトリのパス。

    Returns:
        predict_image(image_path) -> (label: str, score: float)
    """
    _ensure_torchvision()
    import json
    import torch
    from PIL import Image

    classes_path = os.path.join(model_directory, "classes.json")
    model_path = os.path.join(model_directory, "model.pt")
    if not os.path.exists(classes_path):
      raise FileNotFoundError(f"classes.json が見つかりません: {classes_path}")
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"model.pt が見つかりません: {model_path}")

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names: List[str] = json.load(f)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model_for_inference(num_classes=num_classes, device=device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    preprocess = _build_preprocess_transform()

    def predict_image(image_path: str) -> Tuple[str, float]:
        """画像1枚の予測ラベルとスコア（softmaxの最大値）を返す。"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
        image = Image.open(image_path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            score, index = torch.max(probabilities, dim=0)
            label = class_names[int(index)]
            return label, float(score)

    return predict_image


def cli_classification_predict(args) -> None:  # type: ignore[no-untyped-def]
    """CLI から分類推論を行う。"""
    model_directory = getattr(args, "model_directory", None)
    image_path = getattr(args, "image_path", None)
    if not model_directory or not image_path:
        raise SystemExit("使用法: mlmini classification-predict --model-directory ./out/weightN --image-path ./path/to/image.jpg")

    predictor = load_classification_predictor(model_directory)
    label, score = predictor(image_path)
    print(json.dumps({"label": label, "score": score}, ensure_ascii=False))
