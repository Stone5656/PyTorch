"""ファイル入出力ユーティリティ（スケルトン）。

ファイルシステム関連の処理をまとめます。
このコミットでは最小限のプレースホルダのみを定義し、
本格的な処理は後続のコミットで実装します。
"""

from __future__ import annotations
import json
import os
import pathlib
from typing import Any, Dict


def ensure_output_directory(path: str) -> str:
    """指定されたディレクトリを必ず作成して、その絶対パスを返す。"""
    absolute = os.path.abspath(path)
    os.makedirs(absolute, exist_ok=True)
    return absolute


def allocate_next_weight_directory(base_directory: str) -> str:
    """次の `weightN` ディレクトリを確保する。

    Args:
        base_directory: 新しい `weightN` ディレクトリを作成する基準ディレクトリ。

    Returns:
        作成されたディレクトリの絶対パス。
    """
    absolute_base = ensure_output_directory(base_directory)
    existing = {name for name in os.listdir(absolute_base) if name.startswith("weight")}
    index = 1
    while f"weight{index}" in existing:
        index += 1
    target = os.path.join(absolute_base, f"weight{index}")
    pathlib.Path(target).mkdir(parents=True, exist_ok=True)
    return os.path.abspath(target)


def save_json(path: str, data: Dict[str, Any]) -> None:
    """辞書をJSON形式で保存する。"""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """JSONファイルを読み込み辞書として返す。"""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
