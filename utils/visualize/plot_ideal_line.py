from pathlib import Path
from matplotlib import pyplot as plt

from utils.path.root_abspath_setting import ROOT_DIR

# --- 設定 ---
# 保存するキャッシュファイルの名前
root_path = Path(str(ROOT_DIR))
OUTPUT_PATH = root_path / 'out'

def plot_ideal_line(y_test, y_pred, output_path=OUTPUT_PATH):
    plt.figure()
    plt.scatter(y_test.numpy(), y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 理想線
    plt.xlabel("Actual House Value")
    plt.ylabel("Predicted House Value")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(str(output_path / "ideal_line.png"), dpi=300)
    plt.show()
