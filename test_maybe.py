from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # ✅修正
from sklearn.model_selection import train_test_split
import torch

# --- モナドクラス ---

class Maybe:
    """
    値が `None` である可能性のある値に対して、
    安全に操作を連鎖させるためのモナドクラス。

    任意の操作を `.bind()` によって連続適用可能で、
    値が None の場合は処理をスキップする。
    """

    def __init__(self, value):
        """
        パラメータ
        ----------
        value : Any
            任意の初期値（Noneでも可）。
        """
        self.value = value

    def bind(self, func, *args, **kwargs):
        """
        値が None でなければ func を適用し、その結果を新しい Maybe にラップする。

        パラメータ
        ----------
        func : callable
            現在の値に適用する関数。

        戻り値
        -------
        Maybe
            適用後の値をラップした新しい Maybe オブジェクト。
        """
        if self.value is None:
            return Maybe(None)
        return Maybe(func(self.value, *args, **kwargs))
    
    def tap(self, side_effect_func, *args, **kwargs):
        """
        値が None でなければ副作用関数を呼び出す（返値は影響しない）。

        パラメータ
        ----------
        side_effect_func : callable
            ログ出力や表示など、値に影響を与えない処理。
        
        戻り値
        -------
        Maybe
            自身をそのまま返す。
        """
        if self.value is not None:
            side_effect_func(self.value, *args, **kwargs)
        return self

    def to_else(self, then_func, *, else_func):
        """
        値が存在すれば then_func を適用し、
        None であれば else_func を呼び出す。

        パラメータ
        ----------
        then_func : callable
            値が None でないときに呼び出す処理。引数として値が渡される。
        else_func : callable
            値が None のときに呼び出す処理。引数は渡されない。

        戻り値
        -------
        Any
            then_func または else_func の戻り値。
        """
        if self.value is not None:
            return then_func(self.value)
        else:
            return else_func()

    def __repr__(self):
        return f"Maybe({self.value})"

    def unwrap(self):
        """
        ラップされた値をそのまま返す。

        戻り値
        -------
        Any
            元の値（None含む）。
        """
        return self.value

    def expect(self, msg: str = "Error"):
        """
        値が None の場合にメッセージを表示し、None を返す。

        パラメータ
        ----------
        msg : str
            エラーメッセージ。

        戻り値
        -------
        Any
            元の値（None含む）。
        """
        if self.value is None:
            print(f"{msg}")
        return self.value

# --- データ構造 ---

@dataclass
class MLData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

@dataclass
class PipelineData:
    data: MLData
    feature_names: List[str]
    scaler: Optional[StandardScaler] = None

@dataclass
class SetupData:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]

# --- 関数定義 ---

def show_original_data(X: np.ndarray, feature_names: List[str]) -> None:
    print("--- Original ---")
    df = pd.DataFrame(X, columns=feature_names)
    print(df.head())

def split_data(X: np.ndarray, y: np.ndarray, test_size=0.3, random_state=42) -> MLData:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return MLData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

def scale_data(data: MLData, feature_names: List[str]) -> PipelineData:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(data.X_train)
    X_test_scaled = scaler.transform(data.X_test)

    return PipelineData(
        data=MLData(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=data.y_train,
            y_test=data.y_test
        ),
        scaler=scaler,
        feature_names=feature_names
    )

def to_tensor(data: MLData) -> MLData:
    return MLData(
        X_train=torch.tensor(data.X_train, dtype=torch.float32),
        X_test=torch.tensor(data.X_test, dtype=torch.float32),
        y_train=torch.tensor(data.y_train, dtype=torch.float32).reshape(-1, 1),
        y_test=torch.tensor(data.y_test, dtype=torch.float32).reshape(-1, 1),
    )

def show_processed_data(data: MLData) -> None:
    print("--- Processed ---")
    print(data.X_train.shape, data.y_train.shape)
