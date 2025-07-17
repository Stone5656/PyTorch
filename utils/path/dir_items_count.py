import os
import re


def count_matching_items(dir_path: str, pattern: str) -> int:
    """
    ディレクトリ内のすべてのエントリ（ファイルおよびディレクトリ）について、
    名前が正規表現パターンにマッチする個数を数えます。

    パラメータ
    ----------
    dir_path : str
        スキャン対象のディレクトリパス。
    pattern : str
        エントリ名に対して適用する正規表現パターン。

    戻り値
    -------
    int
        パターンに一致するファイル・ディレクトリの数。

    例外
    ------
    FileNotFoundError
        指定されたディレクトリが存在しない場合。
    NotADirectoryError
        指定されたパスがディレクトリでない場合。
    """
    regex = re.compile(pattern)
    with os.scandir(dir_path) as entries:
        return sum(1 for entry in entries if regex.search(entry.name))


def count_matching_files(dir_path: str, pattern: str) -> int:
    """
    ディレクトリ内のファイルについて、
    名前が正規表現パターンにマッチする個数を数えます。

    パラメータ
    ----------
    dir_path : str
        スキャン対象のディレクトリパス。
    pattern : str
        ファイル名に対して適用する正規表現パターン。

    戻り値
    -------
    int
        パターンに一致するファイルの数。

    例外
    ------
    FileNotFoundError
        指定されたディレクトリが存在しない場合。
    NotADirectoryError
        指定されたパスがディレクトリでない場合。
    """
    regex = re.compile(pattern)
    with os.scandir(dir_path) as entries:
        return sum(1 for entry in entries if entry.is_file() and regex.search(entry.name))


def count_matching_dirs(dir_path: str, pattern: str) -> int:
    """
    ディレクトリ内のサブディレクトリについて、
    名前が正規表現パターンにマッチする個数を数えます。

    パラメータ
    ----------
    dir_path : str
        スキャン対象のディレクトリパス。
    pattern : str
        ディレクトリ名に対して適用する正規表現パターン。

    戻り値
    -------
    int
        パターンに一致するサブディレクトリの数。

    例外
    ------
    FileNotFoundError
        指定されたディレクトリが存在しない場合。
    NotADirectoryError
        指定されたパスがディレクトリでない場合。
    """
    regex = re.compile(pattern)
    with os.scandir(dir_path) as entries:
        return sum(1 for entry in entries if entry.is_dir() and regex.search(entry.name))
