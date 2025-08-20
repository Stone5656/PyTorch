"""回帰モデルの推論処理（実装版）。

学習時に保存した以下の成果物を読み込んで、線形回帰に基づく予測を行います。
- scaler.pkl（StandardScaler）
- cache_preprocessed_data.pt（feature_names, model_coefficients, intercept を含む辞書）

CLI:
    mlmini regression-predict --model-directory ./out/weightN --json features.json
    mlmini regression-predict --model-directory ./out/weightN --json '{"features":{"MedInc":5.0,"Latitude":34.2,...}}'
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Mapping, Tuple

import joblib
import numpy as np
import torch


def _load_artifacts(model_directory: str) -> Tuple[List[str], np.ndarray, float, object]:
    """学習時の成果物を読み込む。

    Args:
        model_directory: out/weightN のディレクトリ。

    Returns:
        feature_names, model_coefficients, intercept, scaler のタプル。
    """
    scaler_path = os.path.join(model_directory, "scaler.pkl")
    cache_path = os.path.join(model_directory, "cache_preprocessed_data.pt")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"scaler.pkl が見つかりません: {scaler_path}")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"cache_preprocessed_data.pt が見つかりません: {cache_path}")

    scaler = joblib.load(scaler_path)
    cache_object = torch.load(cache_path, map_location="cpu")

    feature_names: List[str] = list(cache_object["feature_names"])
    model_coefficients = np.asarray(cache_object["model_coefficients"], dtype=float)
    intercept = float(cache_object["intercept"])

    if model_coefficients.shape[0] != len(feature_names):
        raise ValueError("係数の数と特徴量名の数が一致しません。")

    return feature_names, model_coefficients, intercept, scaler


def _features_dict_from_json_text(json_text_or_path: str) -> Dict[str, float]:
    """JSON文字列またはJSONファイルパスから特徴量辞書を得る。

    受理する形式:
        1) {"features": {"f1": 1.0, "f2": 2.0, ...}}
        2) {"f1": 1.0, "f2": 2.0, ...}

    Args:
        json_text_or_path: インラインJSON文字列、またはJSONファイルへのパス。

    Returns:
        特徴量の辞書。
    """
    text = None
    # ファイルとして読めるなら優先してファイル入力とみなす
    if os.path.exists(json_text_or_path) and os.path.isfile(json_text_or_path):
        with open(json_text_or_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = json_text_or_path

    try:
        data = json.loads(text)  # type: ignore[arg-type]
    except json.JSONDecodeError as e:
        raise ValueError(f"JSONの解析に失敗しました: {e}") from e

    if "features" in data and isinstance(data["features"], Mapping):
        features = data["features"]
    else:
        features = data

    if not isinstance(features, Mapping):
        raise TypeError("特徴量の形式が不正です。辞書（key=特徴量名, value=数値）を指定してください。")

    # 数値への変換（数値文字列も許容）
    converted: Dict[str, float] = {}
    for key, value in features.items():
        try:
            converted[key] = float(value)
        except Exception:
            raise TypeError(f"特徴量 '{key}' の値を float に変換できません: {value!r}")
    return converted


def _build_ordered_vector(features: Mapping[str, float], feature_names: List[str]) -> np.ndarray:
    """特徴量名の順序に合わせて、1行の特徴量ベクトルを構築する。

    Args:
        features: 入力された特徴量辞書。
        feature_names: 学習時の特徴量名（順序）。

    Returns:
        形状 (1, D) の numpy.ndarray（float）。
    """
    missing = [n for n in feature_names if n not in features]
    if missing:
        raise KeyError(f"必要な特徴量が不足しています: {missing}")

    # 余分な特徴量は無視（将来的な互換用）
    ordered = [float(features[name]) for name in feature_names]
    return np.asarray([ordered], dtype=float)


def load_regression_predictor(model_directory: str) -> Callable[[Dict[str, float]], float]:
    """保存済み回帰モデルを読み込み、辞書入力→スカラー出力の推論関数を返す。

    Args:
        model_directory: out/weightN ディレクトリのパス。

    Returns:
        `predict_one(features_dict) -> float` なコール可能オブジェクト。
    """
    feature_names, model_coefficients, intercept, scaler = _load_artifacts(model_directory)

    def predict_one(features: Dict[str, float]) -> float:
        """1サンプルの予測値を返す。"""
        vector = _build_ordered_vector(features, feature_names)
        vector_scaled = scaler.transform(vector)
        y = float(np.dot(vector_scaled[0], model_coefficients) + intercept)
        return y

    return predict_one


def cli_regression_predict(args) -> None:  # type: ignore[no-untyped-def]
    """CLI から回帰推論を行う。

    Args:
        args: argparse.Namespace（--model-directory, --json を想定）。
    """
    model_directory = getattr(args, "model_directory", None)
    json_arg = getattr(args, "json", None)

    if not model_directory or not json_arg:
        raise SystemExit("使用法: mlmini regression-predict --model-directory ./out/weightN --json <path-or-inline-json>")

    predictor = load_regression_predictor(model_directory)
    features = _features_dict_from_json_text(json_arg)
    y = predictor(features)
    print(json.dumps({"y": y}, ensure_ascii=False))
