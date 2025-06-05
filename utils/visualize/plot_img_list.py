from pathlib import Path
from matplotlib import pyplot as plt

from utils.path.root_abspath_setting import ROOT_DIR

# --- 設定 ---
# 保存するキャッシュファイルの名前
root_path = Path(str(ROOT_DIR))
OUTPUT_PATH = root_path / 'out'

def plot_prediction_vs_actual(X_test, y_test, y_pred, feature_names, output_path=OUTPUT_PATH, cols=5):
    """
    特徴量ごとの予測と実測を1つの図にまとめて描画する。

    Parameters:
        X_test (Tensor): shape = [N, D]
        y_test (Tensor): shape = [N, 1] または [N]
        y_pred (Tensor): shape = [N, 1] または [N]
        feature_names (List[str]): 特徴量名
        cols (int): 1行に表示するグラフの数
    """
    num_features = X_test.shape[1]
    rows = (num_features + cols - 1) // cols  # 必要な行数
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()  # インデックスしやすく

    for i in range(num_features):
        ax = axes[i]
        ax.scatter(X_test[:, i].numpy(), y_test.numpy(), label='Actual', alpha=0.5)
        ax.scatter(X_test[:, i].numpy(), y_pred, label='Predicted', alpha=0.5)
        ax.set_xlabel(f"{feature_names[i]} (standardized)")
        ax.set_ylabel("House Value")
        ax.set_title(f"{feature_names[i]}")
        ax.legend()

    # 使わない余分なサブプロットを非表示
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Actual vs Predicted for Each Feature", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(output_path / "prediction_grid.png"), dpi=300)
    plt.show()
