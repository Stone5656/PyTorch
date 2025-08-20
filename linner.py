from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.path.dir_items_count import count_matching_dirs
from utils.visualize.plot_barh_weight import plot_weight_feature_importance
from utils.visualize.plot_ideal_line import plot_ideal_line
from utils.visualize.plot_img_list import plot_prediction_vs_actual
from test_maybe import Maybe, PipelineData, SetupData, scale_data, show_original_data, show_processed_data, split_data, to_tensor
from utils.predata.pytorch_predata_cache import cache_preprocess_data_load, cache_preprocess_data_save
from utils.path.root_abspath_setting import ROOT_DIR

# --- 設定 ---
root_path = Path(str(ROOT_DIR))
OUTPUT_PATH = root_path / 'out' / 'linner'

# 学習結果保存ディレクトリ作成
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

weight_dir_count = count_matching_dirs(str(OUTPUT_PATH), r"^weight[0-9]+$")
OUTPUT_PATH_WEIGHT = OUTPUT_PATH / f'weight{weight_dir_count+1}'

# 学習結果保存ディレクトリ作成
OUTPUT_PATH_WEIGHT.mkdir(parents=True, exist_ok=True)

# 🔽 キャッシュファイルのパスを定義
CACHE_PATH = OUTPUT_PATH / "cache_preprocessed_data.pt"
SCALER_PATH = OUTPUT_PATH / "scaler.pkl"
FEATURE_PATH = OUTPUT_PATH / "feature_names.skl"

cache_data: PipelineData = (
    Maybe(CACHE_PATH if CACHE_PATH.exists() and SCALER_PATH.exists() and FEATURE_PATH.exists() else None)
    .to_else(
        # キャッシュからロード
        lambda _: cache_preprocess_data_load(
            input_cache_path=CACHE_PATH,
            input_scaler_path=SCALER_PATH,
            input_feature_path=FEATURE_PATH
        ),
        else_func=lambda: (
            Maybe(fetch_california_housing())
            .bind(lambda housing: SetupData(
                X=housing.data,
                y=housing.target,
                feature_names=housing.feature_names
            ))
            .bind(lambda data: (
                Maybe(data.X)
                .tap(show_original_data, data.feature_names)
                .bind(split_data, data.y, 0.3, 42)
                .bind(lambda split_data: scale_data(split_data, data.feature_names))
                .expect("PipelineDataの生成に失敗しました")
            ))
            .to_else(
                lambda pipeline_data: (
                    Maybe(
                        PipelineData(
                            data=Maybe(pipeline_data.data)
                                .bind(to_tensor)
                                .tap(show_processed_data)
                                .expect("MLDataが見つかりません"),
                            feature_names=pipeline_data.feature_names,
                            scaler=pipeline_data.scaler
                        )
                    )
                    .tap(
                        lambda pdata: print(
                            f"[DEBUG] PipelineData:\n"
                            f"  - type(data): {type(pdata.data)}\n"
                            f"  - feature_names: {pdata.feature_names}\n"
                            f"  - type(scaler): {type(pdata.scaler)}"
                        )
                    )
                    .tap(
                        lambda pdata: cache_preprocess_data_save(
                            data=pdata,
                            output_cache_path=CACHE_PATH,
                            output_scaler_path=SCALER_PATH,
                            output_feature_path=FEATURE_PATH
                        )
                    )
                    .expect("PipelineData が生成できませんでした")
                ),
                else_func=lambda: exit("前処理に失敗しました")
            )
        )
    )
)

print(f"[DEBUG] type(cache_data): {type(cache_data)}")

mldata = cache_data.data
scaler = cache_data.scaler
feature_names = cache_data.feature_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
