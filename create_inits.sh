#!/bin/sh
#
# create_inits.sh
#
# カレントディレクトリから再帰的にサブディレクトリを探索し、
# __init__.py が存在しない場合に空のファイルを作成します。
# Pythonのパッケージとして認識させたいディレクトリ構造を整える際に使用します。
#
# 無視するディレクトリ:
# - .git (Gitリポジトリ)
# - venv, .venv (Python仮想環境)
# - __pycache__ (Pythonキャッシュ)
# - *.egg-info (Pythonパッケージ情報)

echo "Searching for directories that need an __init__.py file..."

# findコマンドでディレクトリを探索し、パイプでwhileループに渡す
# -type d: ディレクトリのみを対象
# -name "..." -prune: 指定した名前のディレクトリを探索対象から除外
# -o -print: 除外されなかったパスを出力
find . -type d \( \
    -name ".git" -o \
    -name "venv" -o \
    -name ".venv" -o \
    -name "__pycache__" -o \
    -name "*.egg-info" \
  \) -prune -o -print | while IFS= read -r dir; do

  # ディレクトリ内に__init__.pyが存在しないかチェック
  # [ ! -f "path" ] はファイルが存在しない場合にtrueとなる
  if [ ! -f "$dir/__init__.py" ]; then
    # 存在しない場合、空の__init__.pyを作成
    echo "Creating: $dir/__init__.py"
    touch "$dir/__init__.py"
  fi
done

echo "Done."