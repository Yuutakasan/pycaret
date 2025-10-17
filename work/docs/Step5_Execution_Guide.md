# Step5 カテゴリ別compare_models()実行ガイド

## 📋 概要

このガイドでは、uplift/downlift分析結果に基づいてカテゴリを分類し、各グループに最適なモデルを選定する方法を説明します。**過学習検出機能**を含む包括的な分析を実施します。

---

## 🎯 実行目的

1. **予測難易度に基づくカテゴリ分類**
   - A: 高難易度（個別モデル必須）
   - B: 中難易度（カテゴリ別モデル推奨）
   - C: 低難易度（統合モデルでOK）

2. **各グループでcompare_models()実行**
   - 最適アルゴリズムの自動選定
   - 上位5モデルのベンチマーク

3. **過学習を包括的に検出**
   - Train/Test R²ギャップ分析
   - Learning Curve可視化
   - MAE/RMSE比較
   - 残差分析

---

## 📊 分析フロー

```
1. カテゴリ戦略CSV読み込み
   ↓
2. グループA: 個別モデル分析
   - 280:チケット・カード (uplift 4.7倍)
   - 220:化粧品 (uplift 4.1倍)
   ↓
3. グループB: カテゴリ別モデル分析
   - 140:カウンターＦＦ
   - 165:インスタント食品
   - 250:文具・玩具・趣味雑貨
   - 399:その他
   ↓
4. グループC: 統合モデル分析
   - 170:乾物・缶詰・調味料
   - 300:日本酒・焼酎
   - その他低変動カテゴリ
   ↓
5. 過学習検出と最終評価
```

---

## 🚀 実行手順

### ステップ1: 環境確認

```bash
# 必要ファイルの存在確認
ls output/category_modeling_strategy.csv  # ✅ 必須
ls output/06_final_enriched_20250701_20250930.csv  # ✅ 必須

# ディレクトリ作成
mkdir -p output/learning_curves
mkdir -p models
```

### ステップ2: Notebookを開く

```bash
cd /mnt/d/github/pycaret/work
jupyter notebook Step5_CategoryWise_Compare_with_Overfitting.ipynb
```

### ステップ3: セルを順番に実行

#### Cell 1-2: 環境セットアップ
- カテゴリ戦略CSV読み込み
- A/B/Cグループ分類確認

**期待される出力:**
```
✅ グループ分類完了:
  A（個別モデル必須）: 2カテゴリ
  B（カテゴリ別推奨）: 4カテゴリ
  C（統合モデルOK）: 4カテゴリ
```

#### Cell 3: データ読み込み
- 特徴量付与済みデータ読み込み
- カテゴリ列抽出

**期待される出力:**
```
📂 データ読み込み完了:
  総レコード数: 83,789行
  列数: 132列
✅ カテゴリ抽出完了: 24カテゴリ
```

#### Cell 4: 過学習検出関数定義
- `detect_overfitting()`: 4つの指標で過学習判定
- `plot_learning_curve()`: 学習曲線の可視化

**過学習判定基準:**
| 指標 | 軽度 (Mild) | 中程度 (Moderate) | 深刻 (Severe) |
|------|------------|------------------|--------------|
| R²ギャップ | 5-8% | 8-15% | 15%以上 |
| MAE増加率 | 10-20% | 20-30% | 30%以上 |
| 残差比率 | 1.2-1.5倍 | 1.5-2.0倍 | 2.0倍以上 |
| 予測範囲 | 70-80% | 50-70% | 50%未満 |

#### Cell 5: グループA分析（約5-10分）

**対象カテゴリ:**
- 280:チケット・カード
- 220:化粧品

**実行内容:**
1. カテゴリごとにデータ抽出
2. PyCaret setup（train_size=0.8, fold=5）
3. compare_models()で上位5モデル選定
4. ベストモデルで過学習検出
5. Learning Curve生成・保存

**期待される出力例:**
```
--- カテゴリ: 280:チケット・カード ---
データ数: 2,847行

📊 モデル比較結果（Top 5）:
              Model       R2      MAE     RMSE
0  Extra Trees Regressor  0.8234  1245.32  1890.45
1      LightGBM Regressor  0.8156  1289.67  1932.11
2     CatBoost Regressor  0.8098  1312.45  1965.78
3       XGBoost Regressor  0.7987  1398.23  2034.56
4  Random Forest Regressor  0.7845  1456.89  2098.34

🔬 過学習検出結果:
  Train R²: 0.9876
  Test R²: 0.8234
  R²ギャップ: 0.1642 (16.42%)
  過学習判定: はい
  深刻度: Severe
  理由:
    - Train/Test R²差分が大きい (16.42%)
    - 残差のばらつきが1.8倍に増加

✅ Learning Curve保存完了
```

#### Cell 6: グループB分析（約3-5分）

**実行内容:**
1. グループB全カテゴリを統合
2. カテゴリをダミー変数化
3. compare_models()実行
4. 過学習検出

**期待される出力:**
```
グループB統合データ: 25,346行, 4カテゴリ
有効データ: 25,183行, 147特徴量

📊 モデル比較結果（Top 5）:
              Model       R2      MAE     RMSE
0      LightGBM Regressor  0.6789  456.78  789.23
1     CatBoost Regressor  0.6712  478.34  812.45
...

🔬 過学習検出結果:
  Train R²: 0.7234
  Test R²: 0.6789
  R²ギャップ: 0.0445 (4.45%)
  過学習判定: いいえ

✅ グループB分析完了
```

#### Cell 7: グループC分析（約3-5分）

**実行内容:**
1. グループC全カテゴリを統合
2. 全アルゴリズムで比較
3. fold=10（安定性重視）
4. 過学習検出

#### Cell 8: 総合分析結果

**出力される情報:**
1. **全カテゴリ性能ランキング**
2. **グループ別サマリー**
3. **過学習カテゴリ一覧**
4. **最終推奨モデル戦略**
5. **CSV保存** → `output/category_compare_models_results.csv`

**期待される総合サマリー例:**
```
📈 グループ別パフォーマンスサマリー
           R2_Test           R2_Gap      過学習  カテゴリ
グループ  mean  std  min  max  mean  max  count  count
A        0.72  0.08  0.64  0.82  0.14  0.18    2      2
B        0.68  0.00  0.68  0.68  0.04  0.04    0      1
C        0.54  0.00  0.54  0.54  0.03  0.03    0      1

⚠️ 過学習が検出されたカテゴリ
        カテゴリ      ベストモデル  R2_Train  R2_Test  R2_Gap  深刻度
280:チケット・カード  ExtraTreesRegressor  0.9876   0.8234   0.1642  Severe
     220:化粧品  LightGBMRegressor  0.8923   0.7801   0.1122  Moderate
```

---

## 📈 Learning Curveの見方

### 正常なモデル（過学習なし）
```
1.0 ┤
    │     Train ████████████████
0.8 ┤           ████████████████
    │     Valid ████████████
0.6 ┤           ████████████
    │
0.4 ┤
    └───────────────────────────
    100    500   1000   2000  (samples)
```
- Train/Valid曲線が近接
- ギャップが5%未満

### 過学習モデル（Severe）
```
1.0 ┤ Train ████████████████████
    │       ████████████████████
0.8 ┤
    │ Valid ████████
0.6 ┤       ████████
    │
0.4 ┤
    └───────────────────────────
    100    500   1000   2000
```
- Train曲線が高い
- Valid曲線が低い
- ギャップが15%以上

---

## 🔬 過学習対策

### 検出された場合の対処法

#### 1. データ拡張
```python
# 時系列データの場合は期間延長
# 2ヶ月 → 6ヶ月 or 1年
data_extended = load_data('output/06_final_enriched_20250101_20251231.csv')
```

#### 2. 特徴量削減
```python
# PyCaret setupで閾値調整
s = setup(
    data,
    target='売上数量',
    feature_selection=True,
    feature_selection_threshold=0.9,  # 0.8 → 0.9に上げる
    remove_multicollinearity=True,
    multicollinearity_threshold=0.90  # 0.95 → 0.90に下げる
)
```

#### 3. 正則化強化
```python
# LightGBMの場合
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(
    reg_alpha=0.5,      # L1正則化（デフォルト0）
    reg_lambda=0.5,     # L2正則化（デフォルト0）
    min_child_samples=30,  # 葉ノードの最小サンプル数
    max_depth=5         # 深さ制限
)
```

#### 4. シンプルなモデルへ変更
```python
# 過学習しやすい: Extra Trees, Random Forest
# 過学習しにくい: Ridge, Lasso, ElasticNet

# 線形回帰を試す
best_models = compare_models(
    include=['ridge', 'lasso', 'en', 'lr', 'dt'],
    sort='R2'
)
```

#### 5. アンサンブル（ブレンディング）
```python
from pycaret.regression import blend_models

# 上位3モデルをブレンド
blended = blend_models(best_models[:3])
```

---

## 📊 出力ファイル一覧

### CSVファイル
```
output/
├── category_modeling_strategy.csv          # カテゴリ戦略（入力）
├── 06_final_enriched_20250701_20250930.csv # 特徴量データ（入力）
└── category_compare_models_results.csv     # 比較結果（出力）★
```

### Learning Curve画像
```
output/learning_curves/
├── learning_curve_A_280_チケット・カード.png
├── learning_curve_A_220_化粧品.png
├── learning_curve_B_unified.png
└── learning_curve_C_unified.png
```

### モデルファイル（オプション）
```
models/
├── 280_チケット・カード_model.pkl
└── 220_化粧品_model.pkl
```

---

## ✅ 成功基準

### グループAの成功基準
- ✅ Test R² > 0.70（70%以上）
- ✅ R²ギャップ < 0.10（10%未満）
- ✅ 過学習判定が「いいえ」または「Mild」

### グループBの成功基準
- ✅ Test R² > 0.60（60%以上）
- ✅ R²ギャップ < 0.08（8%未満）

### グループCの成功基準
- ✅ Test R² > 0.50（50%以上）
- ✅ R²ギャップ < 0.05（5%未満）

---

## 🎯 最終的な判断基準

### ✅ 採用推奨モデル
- Test R² > 目標値
- 過学習なし or Mild
- MAE/RMSEが実用レベル

### ⚠️ 要改善モデル
- Test R² > 目標値 BUT 過学習あり
- または Test R² < 目標値 BUT 過学習なし
→ **対策を実施して再評価**

### ❌ 再検討モデル
- Test R² < 目標値 AND 過学習あり
→ **データ拡張、特徴量見直し、アルゴリズム変更**

---

## 💡 Tips & ベストプラクティス

### 1. データ量の目安
- **個別モデル**: 最低500行（推奨1000行以上）
- **カテゴリ別**: 最低2000行（推奨5000行以上）
- **統合モデル**: 最低5000行（推奨10000行以上）

### 2. 実行時間の目安
- グループA（2カテゴリ）: 約10分
- グループB（統合）: 約5分
- グループC（統合）: 約5分
- **合計**: 約20-30分

### 3. メモリ使用量
- 推奨RAM: 8GB以上
- データ量83,789行の場合: 約4GB使用

### 4. GPU使用
LightGBM/XGBoost/CatBoostでGPU加速可能:
```python
# LightGBM
lgbm = LGBMRegressor(device='gpu')

# XGBoost
xgb = XGBRegressor(tree_method='gpu_hist')

# CatBoost
catboost = CatBoostRegressor(task_type='GPU')
```

---

## 🐛 トラブルシューティング

### エラー1: "Module 'pycaret' not found"
```bash
pip install pycaret
```

### エラー2: "カテゴリ列が見つかりません"
データに`フェイスくくり大分類`または`商品名`列が必要です。
```python
# 列名確認
print(data.columns.tolist())
```

### エラー3: "データ不足でスキップ"
カテゴリのデータが100行未満の場合はスキップされます。より長い期間のデータを使用してください。

### エラー4: メモリ不足
```python
# データをサンプリング
data_sampled = data.sample(frac=0.5, random_state=123)
```

---

## 📚 参考資料

- [PyCaret公式ドキュメント](https://pycaret.gitbook.io/docs/)
- [過学習の理論](https://www.ibm.com/topics/overfitting)
- [Learning Curve解釈ガイド](https://scikit-learn.org/stable/modules/learning_curve.html)

---

## 🎉 完了後の次ステップ

1. ✅ 過学習対策を実施（必要な場合）
2. ✅ 最適モデルでtune_model()実行（ハイパーパラメータ調整）
3. ✅ 最終モデルをfinalize_model()で確定
4. ✅ save_model()でモデル保存
5. ✅ 予測実行（Step 6）

おめでとうございます！これでカテゴリ別の最適モデル選定が完了です！🎊
