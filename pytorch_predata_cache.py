from pathlib import Path
import pickle
from typing import Tuple
from california_housing_pre_dataset import setup_california_housing_data
import torch
from sklearn.preprocessing import StandardScaler
import os

from utils.path.root_abspath_setting import ROOT_DIR
from utils.struct.split_data import MLData

# --- 設定 ---
# 保存するキャッシュファイルの名前
root_path = Path(str(ROOT_DIR))
OUTPUT_PATH = root_path / 'out'

def cache_preprocessed_data_torch(output_dir_path:Path=OUTPUT_PATH) -> Tuple[MLData, StandardScaler]:
    cache_path = output_dir_path / "cache_preprocessed_data.pt"
    scaler_path = output_dir_path / 'scaler.pkl'

    Path(str(output_dir_path)).mkdir(parents=True, exist_ok=True)

    # --- データの準備 ---
    if os.path.exists(cache_path):
        # 1. キャッシュファイルが存在する場合：ファイルを読み込む
        print(f"'{cache_path}' が見つかりました。キャッシュからデータを読み込みます...")
        cached_data = torch.load(cache_path)
        mldata = MLData(
            X_train=cached_data['X_train'],
            y_train=cached_data['y_train'],
            X_test=cached_data['X_test'],
            y_test=cached_data['y_test'],
        )
        # 読み込み時
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    else:
        # 2. キャッシュファイルが存在しない場合：通常の前処理を実行
        print(f"'{cache_path}' が見つかりません。データを前処理します...")

        mldata, scaler = setup_california_housing_data()

        # 3. 処理したデータをファイルに保存する
        print(f"前処理したデータを '{cache_path}' に保存します...")
        torch.save({
            'X_train': mldata.X_train,
            'X_test': mldata.X_test,
            'y_train': mldata.y_train,
            'y_test': mldata.y_test
        }, cache_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    return mldata, scaler
