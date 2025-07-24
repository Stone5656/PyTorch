from pathlib import Path
import torch
import pickle

from test_maybe import MLData, PipelineData
from utils.path.root_abspath_setting import ROOT_DIR

# --- 設定 ---
root_path = Path(str(ROOT_DIR))
OUTPUT_PATH = root_path / 'out'


def cache_preprocess_data_save(
    data: PipelineData,
    output_cache_path: Path,
    output_scaler_path: Path,
    output_feature_path: Path
) -> None:
    """
    前処理済みデータ・スケーラー・特徴量名を指定されたパスに保存します。

    パラメータ
    ----------
    data : PipelineData
        保存対象の前処理済みデータ、スケーラー、特徴量名。
    output_cache_path : Path
        データ本体（Tensor）を保存するファイルパス。
    output_scaler_path : Path
        スケーラー（pickle）を保存するファイルパス。
    output_feature_path : Path
        特徴量名リストを保存するファイルパス。
    """
    mldata = data.data

    print(f"前処理データを '{output_cache_path}' に保存します...")
    torch.save({
        'X_train': mldata.X_train,
        'X_test': mldata.X_test,
        'y_train': mldata.y_train,
        'y_test': mldata.y_test
    }, output_cache_path)

    print(f"スケーラーを '{output_scaler_path}' に保存します...")
    with open(output_scaler_path, "wb") as f:
        pickle.dump(data.scaler, f)

    print(f"特徴量名を '{output_feature_path}' に保存します...")
    with open(output_feature_path, "wb") as f:
        pickle.dump(data.feature_names, f)


def cache_preprocess_data_load(
    input_cache_path: Path,
    input_scaler_path: Path,
    input_feature_path: Path
) -> PipelineData:
    """
    指定されたキャッシュファイルから前処理済みデータ、スケーラー、特徴量名を読み込みます。

    パラメータ
    ----------
    input_cache_path : Path
        データ本体（Tensor）を読み込むファイルパス。
    input_scaler_path : Path
        スケーラー（pickle）を読み込むファイルパス。
    input_feature_path : Path
        特徴量名リストを読み込むファイルパス。

    戻り値
    -------
    PipelineData
        データ、スケーラー、特徴量名を含む PipelineData オブジェクト。
    """
    print(f"'{input_cache_path}' からキャッシュを読み込みます...")
    cached_data = torch.load(input_cache_path)

    with open(input_scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(input_feature_path, "rb") as f:
        feature_names = pickle.load(f)

    mldata = MLData(
        X_train=cached_data['X_train'],
        y_train=cached_data['y_train'],
        X_test=cached_data['X_test'],
        y_test=cached_data['y_test'],
    )

    return PipelineData(
        data=mldata,
        scaler=scaler,
        feature_names=feature_names
    )
