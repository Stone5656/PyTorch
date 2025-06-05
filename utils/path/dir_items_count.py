import os
import re


def count_matching_items(dir_path: str, pattern: str) -> int:
    regex = re.compile(pattern)
    with os.scandir(dir_path) as entries:
        return sum(1 for entry in entries if regex.search(entry.name))

def count_matching_files(dir_path: str, pattern: str) -> int:
    regex = re.compile(pattern)
    with os.scandir(dir_path) as entries:
        return sum(1 for entry in entries if entry.is_file() and regex.search(entry.name))

def count_matching_dirs(dir_path: str, pattern: str) -> int:
    regex = re.compile(pattern)
    with os.scandir(dir_path) as entries:
        return sum(1 for entry in entries if entry.is_dir() and regex.search(entry.name))
