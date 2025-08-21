"""Rich を使った色付きロギングの初期化ユーティリティ。"""
from __future__ import annotations
import logging, os
from typing import Optional
from rich.logging import RichHandler

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """指定名のロガーを色付き（Rich）で返す。二重ハンドラは避ける。"""
    logger = logging.getLogger(name)
    if level is None:
        level_name = os.getenv("MLMINI_LOGLEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=False, markup=True)
        # RichHandler は time/level を装飾して出すので message のみ
        fmt = logging.Formatter("%(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
