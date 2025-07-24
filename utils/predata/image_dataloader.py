import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def get_image_loaders(
    data_dir: Path,
    image_size: int = 224,
    batch_size: int = 32,
    augment: bool = True,
    val_split: float = 0.2,
):
    """
    任意の画像フォルダから学習用・検証用DataLoaderを返す。
    """
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 画像を指定サイズにリサイズ（例：224×224）
        
        transforms.RandomHorizontalFlip(),  # 画像を左右反転（50%の確率）して水平方向の汎化能力を強化
        
        transforms.RandomRotation(20),  # -20〜+20度の範囲でランダムに回転（回転による向きの頑健性を付加）
        
        transforms.ColorJitter(
            brightness=0.3,  # 明るさを ±30% の範囲でランダム調整
            contrast=0.3,    # コントラストを ±30% の範囲でランダム調整
            saturation=0.2   # 彩度を ±20% の範囲でランダム調整
        ),  # カラー変動により光条件への頑健性を向上
        
        transforms.RandomAffine(
            degrees=15,  # 最大 ±15度まで回転
            translate=(0.1, 0.1)  # 横・縦ともに最大10%の範囲で平行移動（画像サイズ比）
        ),  # アフィン変換により位置ずれや角度への頑健性を強化
        
        transforms.ToTensor(),  # PIL画像を Tensor に変換（[0, 255] → [0.0, 1.0]、CHW順に変換）
        
        transforms.RandomErasing(
            p=0.5,  # 50%の確率で適用
            scale=(0.02, 0.2),  # 消去される領域の面積は画像全体の2〜20%
            ratio=(0.3, 3.3),  # 消去領域のアスペクト比（縦横比）は 0.3〜3.3 の範囲で変動
            value='random'  # 消去領域をランダムな値で塗りつぶす（色やノイズが入る）
        )  # 局所的な欠損への耐性を付け、強い正則化効果を持つ
    ])


    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(str(data_dir), transform=transform_train)
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
    val_ds.dataset.transform = transform_val

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.classes
