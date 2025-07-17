# データ前処理およびキャッシュ機構

本プロジェクトでは、`California Housing` データセットを用いて、
機械学習に必要な前処理（分割・スケーリング・テンソル化）を行い、
その結果を **キャッシュファイルとして保存** し、次回以降の処理高速化を実現しています。

---

## 📦 使用データ

- ソース：`sklearn.datasets.fetch_california_housing`
- 特徴量：`housing.feature_names`
- 目的変数：住宅価格中央値 `housing.target`

---

## 🔄 前処理の流れ

### ステップ 1：SetupData の生成

```python
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

data = SetupData(X=X, y=y, feature_names=feature_names)
```

---

### ステップ 2：パイプライン処理（Maybeモナドを使用）

```python
pipeline_data = (
    Maybe(data.X)
    .tap(show_original_data, data.feature_names)  # 表示（任意）
    .bind(split_data, data.y, 0.3, 42)
    .bind(scale_data)
    .expect("PipelineDataが見つかりませんでした")
)
```

* `split_data`：訓練／検証用にデータを分割
* `scale_data`：StandardScalerなどで特徴量をスケーリング

---

### ステップ 3：テンソル化 + 表示

```python
pipeline_tensor_data = PipelineData(
    Maybe(pipeline_data.data)
    .bind(to_tensor)
    .tap(show_processed_data)
    .expect("MLDataが見つかりません"),
    pipeline_data.scaler
)
```

---

## 🧠 キャッシュ機構について

### 保存形式

| ファイル名                        | 内容                | フォーマット  |
| ---------------------------- | ----------------- | ------- |
| `cache_preprocessed_data.pt` | Tensor形式の訓練／検証データ | PyTorch |
| `scaler.pkl`                 | スケーラーオブジェクト       | Pickle  |

---

### 保存先の自動構成

```python
weight_dir_count = count_matching_dirs(str(OUTPUT_PATH), r"^weight[0-9]+$")
OUTPUT_PATH_WEIGHT = OUTPUT_PATH / f'weight{weight_dir_count+1}'
```

* `out/weight1`, `out/weight2` ... のように保存先ディレクトリを動的に割り当て

---

### ロード or セーブの条件分岐（カリー化しやすい形式）

```python
cache_data = (
    Maybe(OUTPUT_PATH)
    .bind(lambda path: (
        cache_preprocess_data_load(path)
        if (path / "cache_preprocessed_data.pt").exists()
        else (cache_preprocess_data_save(path, pipeline_tensor_data) or pipeline_tensor_data)
    ))
    .expect("キャッシュデータが見つかりませんでした")
)
```

* キャッシュが存在すればロード
* なければ保存し、以降再利用できるようにする

---

## 🔧 モデル学習に渡す形式

```python
X_train = mldata.X_train.to(device)
y_train = mldata.y_train.to(device)
X_test = mldata.X_test.to(device)
y_test = mldata.y_test.to(device)
```

* PyTorch `Device` に転送（`cuda` or `cpu`）
* `mldata = cache_data.data`, `scaler = cache_data.scaler` によって構成済み

---

## 📝 補足

* 本構造はスケーラブルなデータ前処理パイプラインを意図しています
* 「初回のみ保存、以降は高速ロード」を自動判定
* 前処理済データの再利用や実験管理に最適です

---

## 📁 関連ファイル

| ファイル                                             | 役割            |
| ------------------------------------------------ | ------------- |
| `cache_preprocess_data_save.py`                  | キャッシュ保存ロジック   |
| `cache_preprocess_data_load.py`                  | キャッシュ読み込みロジック |
| `split_data.py`, `scale_data.py`, `to_tensor.py` | パイプライン処理用関数群  |
