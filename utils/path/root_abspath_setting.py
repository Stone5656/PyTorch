import os
from pathlib import Path

# このファイルの絶対パスを取得
# Path(__file__) -> このファイルのパスオブジェクト
# .resolve() -> シンボリックリンクなどを解決した絶対パスを取得
current_file_path = Path(__file__).resolve()

# このファイルからルートディレクトリまでの階層を遡る
# current_file_path.parents[0] -> .../utils/path
# current_file_path.parents[1] -> .../utils
# current_file_path.parents[2] -> .../root_dir
# したがって、2つ親のディレクトリがルートディレクトリになる
ROOT_DIR = current_file_path.parents[2]

# (オプション) os.environに設定する場合
# 一般的には定数として公開する方がクリーンですが、os.environに設定することも可能です。
# os.environ['ROOT_DIR'] = str(ROOT_DIR)

# 例：他のパスも定義しておくと便利
# DATA_DIR = ROOT_DIR / 'data'
# LOG_DIR = ROOT_DIR / 'logs'

# print(f"Project Root Directory set to: {ROOT_DIR}")