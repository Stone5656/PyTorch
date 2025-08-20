"""共通ユーティリティ（スケルトン）。

シード固定、処理時間計測などの小さな汎用機能をまとめます。
"""

from __future__ import annotations
import contextlib
import random
import time
from typing import Iterator, Optional


def set_global_random_seed(seed: int = 42, enable_cuda: bool = True) -> None:
    """乱数シードを固定する（プレースホルダ）。"""
    random.seed(seed)
    # numpy や torch のシード設定は後続のコミットで追加予定
    _ = enable_cuda


@contextlib.contextmanager
def elapsed_timer(name: Optional[str] = None) -> Iterator[None]:
    """処理時間を計測するコンテキストマネージャ。"""
    start = time.time()
    try:
        yield
    finally:
        label = f"[{name}] " if name else ""
        duration = time.time() - start
        print(f"{label}{duration:.2f}s")
