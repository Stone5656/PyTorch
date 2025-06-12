import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

from california_housing_pre_dataset import setup_linear_regression_data
from pytorch_predata_cache import cache_preprocessed_data_torch
from utils.path.dir_items_count import count_matching_dirs
from utils.path.root_abspath_setting import ROOT_DIR
from utils.visualize.plot_barh_weight import plot_weight_feature_importance
from utils.visualize.plot_ideal_line import plot_ideal_line
from utils.visualize.plot_img_list import plot_prediction_vs_actual

# --- 設定 ---
# 保存するキャッシュファイルの名前
root_path = Path(str(ROOT_DIR))
OUTPUT_PATH = root_path / 'out'

weight_dir_count = count_matching_dirs(str(OUTPUT_PATH), r"^weight[0-9]+$")
OUTPUT_PATH_WEIGHT = OUTPUT_PATH / f'weight{weight_dir_count+1}'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

mldata, scaler = cache_preprocessed_data_torch(
    setup_linear_regression_data,
    OUTPUT_PATH_WEIGHT,
    X=X, y=y, feature_names=feature_names
)

X_train = mldata.X_train.to(device)
y_train = mldata.y_train.to(device)
X_test = mldata.X_test.to(device)
y_test = mldata.y_test.to(device)

model = nn.Linear(len(feature_names), 1).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)  # Tensor[?, 1]
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch(学習回数) {epoch}: Loss(損失) = {loss.item()}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test)

y_pred_np = y_pred.cpu().numpy()

mse = mean_squared_error(y_test.numpy(), y_pred_np)
mae = mean_absolute_error(y_test.numpy(), y_pred_np)
r2 = r2_score(y_test.numpy(), y_pred_np)

print(f"MSE(平均二乗誤差) : {mse:.4f}")
print(f"MAE(平均絶対誤差) : {mae:.4f}")
print(f"R²(決定係数) : {r2:.4f}")
print("X_train shape:", X_train.shape)  # → torch.Size([?, 8]) であることを確認
print("モデル構造:", model)

weights = model.weight.detach().numpy().flatten()

plot_prediction_vs_actual(X_test, y_test, y_pred_np, feature_names, OUTPUT_PATH_WEIGHT)
plot_ideal_line(y_test, y_pred_np, OUTPUT_PATH_WEIGHT)
plot_weight_feature_importance(feature_names, weights, OUTPUT_PATH_WEIGHT)
