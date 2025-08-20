"""コマンドラインインターフェース（サブコマンド接続版：プレースホルダ）。

このファイルは、回帰・分類の「学習」「推論」サブコマンドを提供します。
現時点では各サブコマンドは **プレースホルダ実装** に接続されており、
実際の学習・推論ロジックは後続のコミットで移植／実装されます。

- 回帰（regression）:
    - regression-train
    - regression-predict
- 分類（classification）:
    - classification-train
    - classification-predict
"""

from __future__ import annotations

import argparse
from typing import Callable

from src.mlmini.regression.train import cli_regression_train
from src.mlmini.regression.infer import cli_regression_predict
from src.mlmini.classification.train import cli_classification_train
from src.mlmini.classification.infer import cli_classification_predict


def build_argument_parser() -> argparse.ArgumentParser:
    """サブコマンドを備えた引数パーサを構築して返す。"""
    parser = argparse.ArgumentParser(
        prog="mlmini",
        description="最小構成の回帰／分類タスクを扱うCLI（現在はプレースホルダ動作）",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # --- regression-train ---
    parser_regression_train = subparsers.add_parser(
        "regression-train",
        help="回帰モデルを学習（プレースホルダ：実処理は後続コミットで有効化）",
        description="回帰モデルの学習を実行します（現在はプレースホルダ出力のみ）。",
    )
    parser_regression_train.add_argument(
        "--output-directory",
        default="./out",
        help="成果物を保存する基準ディレクトリ（デフォルト: ./out）",
    )
    parser_regression_train.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="計算デバイス（cpu/cuda）",
    )
    parser_regression_train.set_defaults(func=cli_regression_train)

    # --- regression-predict ---
    parser_regression_predict = subparsers.add_parser(
        "regression-predict",
        help="回帰モデルで推論（プレースホルダ：固定値を返す）",
        description="保存済み回帰モデルを読み込み、入力特徴量から1値を予測します（現在は固定出力）。",
    )
    parser_regression_predict.add_argument(
        "--model-directory",
        required=True,
        help="保存済みモデルのディレクトリへのパス",
    )
    parser_regression_predict.add_argument(
        "--json",
        required=True,
        help="特徴量JSONのファイルパス、またはインラインJSON文字列",
    )
    parser_regression_predict.set_defaults(func=cli_regression_predict)

    # --- classification-train ---
    parser_classification_train = subparsers.add_parser(
        "classification-train",
        help="分類モデルを学習（プレースホルダ：実処理は後続コミットで有効化）",
        description="分類モデルの学習を実行します（現在はプレースホルダ出力のみ）。",
    )
    parser_classification_train.add_argument(
        "--dataset-directory",
        required=True,
        help="ImageFolder 形式のデータセットディレクトリ",
    )
    parser_classification_train.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="学習エポック数（デフォルト: 5）",
    )
    parser_classification_train.add_argument(
        "--output-directory",
        default="./out",
        help="成果物を保存する基準ディレクトリ（デフォルト: ./out）",
    )
    parser_classification_train.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="計算デバイス（cpu/cuda）",
    )
    parser_classification_train.set_defaults(func=cli_classification_train)

    # --- classification-predict ---
    parser_classification_predict = subparsers.add_parser(
        "classification-predict",
        help="分類モデルで推論（プレースホルダ：固定ラベルを返す）",
        description="保存済み分類モデルを読み込み、画像1枚を分類します（現在は固定出力）。",
    )
    parser_classification_predict.add_argument(
        "--model-directory",
        required=True,
        help="保存済みモデルのディレクトリへのパス",
    )
    parser_classification_predict.add_argument(
        "--image-path",
        required=True,
        help="分類対象の画像ファイルへのパス",
    )
    parser_classification_predict.set_defaults(func=cli_classification_predict)

    return parser


def main() -> None:
    """エントリポイント。引数を解釈して対象サブコマンドを実行する。"""
    parser = build_argument_parser()
    arguments = parser.parse_args()
    # 各プレースホルダ関数（cli_...）に argparse.Namespace を渡す
    function: Callable = getattr(arguments, "func")
    function(arguments)


if __name__ == "__main__":
    main()
