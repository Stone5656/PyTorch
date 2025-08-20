"""回帰モデルの学習処理。

既存の linner.py の挙動を尊重しつつ、学習・保存・可視化の流れを関数化します。
出力ファイル名は従来と同一（weightN 直下）に保存します。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import joblib
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from mlmini.regression.data import load_regression_dataset
from mlmini.utils.file_io import allocate_next_weight_directory, ensure_output_directory
from mlmini.utils.utilities_common import set_global_random_seed, elapsed_timer
from mlmini.utils.visualization import (
    plot_ideal_line,
    plot_feature_importance,
    plot_prediction_grid,
)


def _save_artifacts(
    output_weight_directory: str,
    scaler,
    cache_object: Dict[str, object],
) -> None:
    """成果物を保存する（従来のファイル名に合わせる）。

    - scaler.pkl
    - cache_preprocessed_data.pt
    """
    scaler_path = os.path.join(output_weight_directory, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    cache_path = os.path.join(output_weight_directory, "cache_preprocessed_data.pt")
    torch.save(cache_object, cache_path)


def _save_figures(
    output_weight_directory: str,
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    feature_matrix_raw_for_grid: np.ndarray,
    feature_names: list[str],
) -> None:
    """可視化画像を保存する（従来のファイル名に合わせる）。"""
    ideal_line_path = os.path.join(output_weight_directory, "ideal_line.png")
    importance_weight_path = os.path.join(output_weight_directory, "importance_weight.png")
    prediction_grid_path = os.path.join(output_weight_directory, "prediction_grid.png")

    plot_ideal_line(true_values=true_values, predicted_values=predicted_values, output_path=ideal_line_path)
    # 係数は LinearRegression の coef_ を用いる
    # feature_names と配列長が一致することを期待
    # plot_feature_importance 内で回転・レイアウトを調整
    # 重要度は係数の大きさそのもの（スケーリング済み入力に対する係数）
    plot_feature_importance(weights=model_coefficients, feature_names=feature_names, output_path=importance_weight_path)  # type: ignore[name-defined]
    plot_prediction_grid(
        feature_matrix=feature_matrix_raw_for_grid,
        target_vector=true_values,
        predicted_vector=predicted_values,
        feature_names=feature_names,
        output_path=prediction_grid_path,
    )


def train_regression_model(output_base_directory: str = "./out", device: str = "cpu") -> str:
    """回帰モデルを学習して成果物（モデル、可視化、キャッシュ）を保存する。

    Args:
        output_base_directory: `out` ディレクトリのパス（内部で weightN を自動採番）。
        device: "cpu" または "cuda"（本実装では scikit-learn の LinearRegression を利用するため、値は実質的に無視されます）。

    Returns:
        作成された `out/weightN` ディレクトリの絶対パス。
    """
    set_global_random_seed(42)

    output_base_directory = ensure_output_directory(output_base_directory)
    output_weight_directory = allocate_next_weight_directory(output_base_directory)

    with elapsed_timer("load_dataset"):
        # 学習・検証用のスケーリング済みデータを取得
        X_train_scaled, X_valid_scaled, y_train, y_valid, feature_names, scaler = load_regression_dataset(
            cache_directory=os.path.join(output_weight_directory, "cache")
        )

    # 学習（scikit-learn の線形回帰を使用）
    with elapsed_timer("train_regression"):
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

    # 予測と評価（ここでは図に必要な値のみ）
    with elapsed_timer("predict_and_visualize"):
        y_valid_pred = model.predict(X_valid_scaled)

        # figures: ideal_line, importance_weight, prediction_grid
        # prediction_grid の x 軸は生の特徴量の方が直感的だが、ここでは簡易的に検証用のスケール前行列が必要になる。
        # load_regression_dataset はスケール前の検証行列を返していないため、
        # Grid 図は「スケール済み行列」をそのまま比較軸に使う設計でも良いが、視覚的直感を優先し、ここでは簡易に X_valid_scaled を使用する。
        # （後続コミットで raw を返すよう拡張可能）
        feature_matrix_for_grid = X_valid_scaled  # 簡易実装

        global model_coefficients  # plot_feature_importance で使用するため
        model_coefficients = np.asarray(model.coef_, dtype=float)

        _save_figures(
            output_weight_directory=output_weight_directory,
            true_values=y_valid,
            predicted_values=y_valid_pred,
            feature_matrix_raw_for_grid=feature_matrix_for_grid,
            feature_names=feature_names,
        )

    # キャッシュ・前処理の保存（従来ファイル名に合わせる）
    with elapsed_timer("save_artifacts"):
        cache_object = {
            "feature_names": feature_names,
            "model_coefficients": model_coefficients,
            "intercept": float(model.intercept_),
        }
        _save_artifacts(
            output_weight_directory=output_weight_directory,
            scaler=scaler,
            cache_object=cache_object,
        )

    print(f"saved to: {output_weight_directory}")
    return output_weight_directory


def cli_regression_train(args) -> None:  # type: ignore[no-untyped-def]
    """CLI から回帰モデルを学習するラッパ。

    Args:
        args: argparse.Namespace（--output-directory, --device を想定）。
    """
    output_base_directory = getattr(args, "output_directory", "./out")
    device = getattr(args, "device", "cpu")
    train_regression_model(output_base_directory=output_base_directory, device=device)


def train_main() -> None:
    """既存 linner.py から呼ばれるエントリポイント。

    Note:
        コミット3のラッパにより、環境変数 MLMINI_REDIRECT=1 のときにのみこの関数が実行されます。
        既存の引数名に合わせて `--out`, `--device` を受け取ります。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", dest="output_directory", default="./out")
    parser.add_argument("--device", dest="device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    train_regression_model(output_base_directory=args.output_directory, device=args.device)
