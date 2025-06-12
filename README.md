了解しました。以下は**コードブロックを外したバージョン**の `README.md` 内容です。マークダウン形式はそのまま維持し、インストール・実行方法や画像の説明も含めて整えています。

---

# California Housing Price Prediction - Linear Regression

本プロジェクトでは、カリフォルニア住宅価格データセットを用いた線形回帰分析を行い、住宅価格の予測モデルを構築しています。
予測モデルの可視化結果と、再実行の手順について解説します。

---

## 🔧 実行方法

以下の手順で環境構築と実行が可能です。

1. 必要なパッケージをインストールします：
```python
pip install -r requirements.txt
```
2. メインスクリプトを実行します：
```python
python liner.py
```
---

## 🧠 キャッシュの仕組みと再利用方法

`liner.py` の以下の設定部分では、出力先ディレクトリとして `out/weightN`（Nは連番）を自動生成しています。
```
./liner.py:16
# --- 設定 ---

root\_path = Path(str(ROOT\_DIR))
OUTPUT\_PATH = root\_path / 'out'

weight\_dir\_count = count\_matching\_dirs(str(OUTPUT\_PATH), r"^weight\[0-9]+\$")
OUTPUT\_PATH\_WEIGHT = OUTPUT\_PATH / f'weight{weight\_dir\_count+1}'
```
すでに存在する `out/weightN` ディレクトリからキャッシュ（`cache_preprocessed_data.pt`）とスケーラー（`scaler.pkl`）を別の出力先にコピーすれば、**再計算を行うことなく同じ結果が再現**できます。

---

## 📊 可視化結果の説明

### 1. `ideal_line.png`

![Actual vs Predicted](./docs/sample_images/ideal_line.png)

* **説明**：実際の住宅価格と予測値の散布図です。
* **赤の破線**：理想的な予測（予測＝実測）のラインを表しています。
* **解釈**：データが赤線に近いほど、予測精度が高いことを意味します。

---

### 2. `importance_weight.png`

![Feature Importance](./docs/sample_images/importance_weight.png)

* **説明**：線形回帰モデルで使用された各特徴量の重要度（重み）を示した棒グラフです。
* **正の重み**：住宅価格と正の相関を持つ要因（例：MedInc）。
* **負の重み**：住宅価格と負の相関を持つ要因（例：Longitude, Latitude）。

---

### 3. `prediction_grid.png`

![Feature-wise Actual vs Predicted](./docs/sample_images/prediction_grid.png)

* **説明**：各特徴量ごとの実測値と予測値の比較グラフです（青：実測、オレンジ：予測）。
* **解釈**：

  * MedInc（中央値収入）は住宅価格に強い影響を持っていることが視覚的に分かります。
  * 他の特徴量についても、どの程度予測に貢献しているかの傾向を確認できます。

---

## 📂 出力構成
```
out/
├── weight1/
│   ├── cache\_preprocessed\_data.pt
│   ├── scaler.pkl
│   ├── ideal\_line.png
│   ├── importance\_weight.png
│   └── prediction\_grid.png
```
新しい学習を行う場合は `weightN` を自動的にインクリメントして保存されます。

---

## 📌 注意

* 重回帰モデルを使用しているため、特徴量は複数使用されています。
* 特徴量や目的変数を変える場合は、`setup_linear_regression_data()` の引数を適切に変更してください。

---

## 📘 使用ライブラリ

代表的なライブラリ：

* scikit-learn
* matplotlib
* torch
* numpy
