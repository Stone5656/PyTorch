import logging
import sys
from pathlib import Path

# ANSIカラー定義（ターミナルでの可読性向上用）
LOG_COLORS = {
    "DEBUG": "\033[36m",    # Cyan
    "INFO": "\033[32m",     # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",    # Red
    "CRITICAL": "\033[41m", # Red background
    "RESET": "\033[0m"
}

class ColorFormatter(logging.Formatter):
    """
    ANSIカラーコードを用いて、ログレベルごとに色分けされたログをターミナルに出力するFormatterクラス。

    Attributes:
    -----------
    format (str): 出力フォーマット文字列。
    datefmt (str): 日付フォーマット文字列。
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        ログレコードを色付き文字列に整形して返す。

        Parameters:
        -----------
        record : logging.LogRecord
            ログ出力時に渡されるレコードオブジェクト

        Returns:
        --------
        str
            整形済みログ文字列（色付き）
        """
        levelname = record.levelname
        color = LOG_COLORS.get(levelname, "")
        reset = LOG_COLORS["RESET"]
        record.levelname = f"[{color}{levelname}{reset}]"
        return super().format(record)

# ロガーインスタンスのキャッシュ辞書（複数回呼び出しても再生成しないため）
_logger_cache = {}

def get_logger(
    name: str = "app",
    log_file: Path = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    標準出力とファイル出力の両方に対応したカラーロガーを取得する関数。
    ロガーは名前ごとにキャッシュされ、再生成されない。

    Parameters:
    -----------
    name : str
        ロガーの名前（同名ロガーはキャッシュされる）
    log_file : Path | None
        ファイル出力を行う場合のログファイルパス（Noneでファイル出力なし）
    console_level : int
        標準出力のログレベル（例: logging.INFO）
    file_level : int
        ファイル出力のログレベル（例: logging.DEBUG）

    Returns:
    --------
    logging.Logger
        設定済みロガーインスタンス
    """
    if name in _logger_cache:
        return _logger_cache[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 親ロガーに伝播させない

    # コンソール出力用ハンドラ（カラー付き）
    formatter = ColorFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイル出力用ハンドラ（プレーンテキスト）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)

    _logger_cache[name] = logger
    return logger
