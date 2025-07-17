from pathlib import Path
from typing import List
from matplotlib import pyplot as plt

from utils.path.root_abspath_setting import ROOT_DIR

# --- 設定 ---
# 保存するキャッシュファイルの名前
root_path = Path(str(ROOT_DIR))
OUTPUT_PATH = root_path / 'out'

def plot_weight_feature_importance(
    feature_names: List[str],
    weights,
    output_path=OUTPUT_PATH
):
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, weights)
    plt.xlabel("Weight")
    plt.title("Feature Importance (Linear Weights)")
    plt.tight_layout()
    plt.savefig(str(output_path / "importance_weight.png"), dpi=300)
    plt.show()
