import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from utils.struct.split_data import MLData


from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# MLData クラス、_preprocess_and_create_ml_data, show_original_data_summary, show_preprocessed_data_summary
# は既に定義済みとします。

def setup_linear_regression_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[MLData, StandardScaler]:
    """
    任意の特徴量X、目的変数y、および特徴量名を与えて線形回帰の前処理を行う。

    Parameters:
    - X: 説明変数の配列
    - y: 目的変数の配列
    - feature_names: 特徴量名のリスト
    - test_size: テストデータの割合（デフォルト: 0.2）
    - random_state: 乱数シード（デフォルト: 42）

    Returns:
    - MLData構造体とStandardScalerインスタンスのタプル
    """

    # 1. 入力データの概要を表示
    show_original_data_summary(X, y, feature_names)

    # 2. 前処理と分割、構造体格納
    data, scaler = _preprocess_and_create_ml_data(X, y, test_size=test_size, random_state=random_state)

    # 3. 処理後のデータ概要を表示
    show_preprocessed_data_summary(data)

    return data, scaler

def _preprocess_and_create_ml_data(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int) -> Tuple[MLData, StandardScaler]:
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
