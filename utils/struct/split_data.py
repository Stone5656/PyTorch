from dataclasses import dataclass
from typing import Any

# 1. 構造体を定義する
@dataclass
class MLData:
    """機械学習の訓練・テストデータを格納する構造体"""
    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any