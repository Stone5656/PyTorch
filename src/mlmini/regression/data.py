"""回帰データセット関連のユーティリティ。

California Housing データセットの取得・前処理を行います。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_regression_dataset(cache_directory: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """California Housing データセットを取得し、標準化して返す。

    Note:
        - 特徴量 X は StandardScaler で標準化します（y はそのまま）。
        - 返り値には学習用・検証用の分割済みデータを含みます。

    Args:
        cache_directory: キャッシュ用ディレクトリ（現時点では未使用・将来拡張用）。

    Returns:
        (X_train_scaled, X_valid_scaled, y_train, y_valid, feature_names, scaler)
    """
    dataset = fetch_california_housing()
    feature_matrix_raw: np.ndarray = dataset.data  # shape: (N, D)
    target_vector: np.ndarray = dataset.target     # shape: (N,)
    feature_names: List[str] = list(dataset.feature_names)

    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        feature_matrix_raw, target_vector, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_valid_scaled = scaler.transform(X_valid_raw)

    return X_train_scaled, X_valid_scaled, y_train, y_valid, feature_names, scaler
