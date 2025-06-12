from pathlib import Path
from typing import Callable, Tuple
import os
import torch
import pickle
from sklearn.preprocessing import StandardScaler

from utils.path.root_abspath_setting import ROOT_DIR
from utils.struct.split_data import MLData

# --- 設定 ---
# 保存するキャッシュファイルの名前
root_path = Path(str(ROOT_DIR))
OUTPUT_PATH = root_path / 'out'

# --- 型の仮定 ---
# MLData は dataclass などで定義されていると仮定
# from your_module import MLData, setup_california_housing_data

def cache_preprocess_data_save(mldata, scaler, output_dir_path: Path=OUTPUT_PATH) -> None:
    cache_path = output_dir_path / "cache_preprocessed_data.pt"
    scaler_path = output_dir_path / "scaler.pkl"

    print(f"前処理したデータを '{cache_path}' に保存します...")
    torch.save({
        'X_train': mldata.X_train,
        'X_test': mldata.X_test,
        'y_train': mldata.y_train,
        'y_test': mldata.y_test
    }, cache_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)


def cache_preprocess_data_load(output_dir_path: Path=OUTPUT_PATH) -> Tuple:
    cache_path = output_dir_path / "cache_preprocessed_data.pt"
    scaler_path = output_dir_path / "scaler.pkl"

    print(f"'{cache_path}' が見つかりました。キャッシュからデータを読み込みます...")
    cached_data = torch.load(cache_path)

    mldata = MLData(
        X_train=cached_data['X_train'],
        y_train=cached_data['y_train'],
        X_test=cached_data['X_test'],
        y_test=cached_data['y_test'],
    )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return mldata, scaler


def cache_preprocessed_data_torch(
    preprocess_func: Callable[[], Tuple],  # 引数なしで (MLData, StandardScaler) を返す関数
    output_dir_path: Path = OUTPUT_PATH,
    *func_args,
    **func_kwargs,
) -> Tuple:
    output_dir_path.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir_path / "cache_preprocessed_data.pt"

    if os.path.exists(cache_path):
        return cache_preprocess_data_load(output_dir_path)
    else:
        print(f"'{cache_path}' が見つかりません。データを前処理します...")
        mldata, scaler = preprocess_func(*func_args, **func_kwargs)
        cache_preprocess_data_save(mldata, scaler, output_dir_path)
        return mldata, scaler
