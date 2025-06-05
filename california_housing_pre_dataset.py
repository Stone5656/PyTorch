import pandas as pd
import torch
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from utils.struct.split_data import MLData


def setup_california_housing_data() -> Tuple[MLData, StandardScaler]:
    """
    カリフォルニア住宅価格データを準備し、前処理済みのデータを格納したMLDataオブジェクトを返す。
    途中経過のサマリーも表示する。
    """
    # 1. 元データの読み込み
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    # 今回は単回帰分析のため、説明変数を一つに絞る
    X_all = X

    # 2. 分割前のデータ概要を表示
    show_original_data_summary(X, y, housing.feature_names)

    # 3. データの前処理と構造体への格納
    data, scaler = _preprocess_and_create_ml_data(X_all, y)

    # 4. 前処理後のデータ形状を表示
    show_preprocessed_data_summary(data)
    
    return data, scaler

def _preprocess_and_create_ml_data(X: np.ndarray, y: np.ndarray) -> Tuple[MLData, StandardScaler]:
    """データの前処理（分割、標準化、Tensor変換）を行い、MLDataオブジェクトを返す。"""
    # データを訓練用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PyTorchのTensorへ変換
    # yは(n_samples,)の形状なので、(n_samples, 1)にリシェイプしてTensorに渡す
    return MLData(
        X_train=torch.from_numpy(X_train_scaled.astype(np.float32)),
        y_train=torch.from_numpy(y_train.astype(np.float32)).view(-1, 1),
        X_test=torch.from_numpy(X_test_scaled.astype(np.float32)),
        y_test=torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)
    ), scaler

# --- 表示用の関数群 ---

def show_original_data_summary(X: np.ndarray, y: np.ndarray, feature_names: list):
    """分割前の元データの概要を表示する"""
    print("--- 1. 元データの概要 ---")
    df = pd.DataFrame(X, columns=feature_names)
    df["MedHouseVal"] = y
    print("■ 最初の5行:")
    print(df.head())
    print("\n■ 基本統計量:")
    print(df.describe())

def show_preprocessed_data_summary(data: MLData):
    """前処理後のデータの形状を表示する"""
    print("\n--- 2. 前処理後のデータセットの形状 ---")
    print(f"X_train shape: {data.X_train.shape}")
    print(f"y_train shape: {data.y_train.shape}")
    print(f"X_test shape:  {data.X_test.shape}")
    print(f"y_test shape:  {data.y_test.shape}")
