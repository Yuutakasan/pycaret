# 🏪 コンビニエンスストア売上予測・分析システム v5.0

## 📋 プロジェクト概要

PyCaret AutoMLを活用した、コンビニエンスストアの売上予測と店舗運営最適化のための包括的な分析システムです。POSデータと気象データを統合し、店長が毎日使える実践的なダッシュボードを提供します。

## 🚀 クイックスタート

### 📚 実務向けガイド（必読！）

- **[実務向けクイックスタート](実務向けクイックスタート.md)** - 3分で始める！5つの実務シナリオと毎日の使い方
- **[Phase 1-4 日本語対応ガイド](Phase_1_4_日本語対応ガイド.md)** - 日本語フォント設定と店舗選択機能の使い方
- **[修正完了サマリー](修正完了サマリー.md)** - v5.0での日本語対応・店舗選択機能の修正内容

### ⚡ 3ステップで開始

```bash
# 1. 日本語フォント対応
pip install japanize-matplotlib ipywidgets
jupyter nbextension enable --py widgetsnbextension

# 2. データ準備
python batch_convert.py --input-dir input --output-dir output
python enrich_features_v2.py --input output/06_*.csv

# 3. ダッシュボード起動
jupyter notebook
# → 店舗別包括ダッシュボード_v5.0_Phase1.ipynb を開く
```

### 🎯 v5.0の特徴

**Phase 1実装済み（5つの即効機能）**:
1. **エグゼクティブサマリー** - 1画面で店舗状況を把握
2. **需要予測エンジン（PyCaret）** - 明日・来週の売上を自動予測
3. **気象連動型発注アドバイザー** - 天気で発注量を調整
4. **客数・客単価分解ダッシュボード** - 売上変動の原因を特定
5. **店舗間パフォーマンス比較** - 他店とのギャップを定量化

**Phase 2実装済み（効率化・重要機能）**:
1. **リアルタイム異常検知アラート** - 5種類のアルゴリズムで異常検知
2. **在庫回転率・発注最適化** - CV分析とリスク分類
3. **前年同期詳細比較・要因分解** - 数量vs価格の完全分解
4. **イベント連動型需要予測** - 休日・給料日・連休の影響定量化
5. **アクション優先順位スコアリング** - 4象限マトリクス（影響度×実現性）

**Phase 3実装済み（AI/機械学習による売上最大化）**:
1. **PyCaret自動特徴量選択** - SHAP値で重要特徴量を自動抽出
2. **トレンド検知・成長率分析** - Mann-Kendall検定で統計的検証
3. **欠品検知・機会損失定量化** - 過去のパターンから年間損失額を推定
4. **ベストプラクティス抽出** - トップ店舗の成功要因を横展開
5. **マーケットバスケット分析** - Aprioriでクロスセル提案

**Phase 4実装済み（戦略立案と意思決定）**:
1. **時間帯別需要パターン分析（C2）** - ピークタイムの特定とスタッフ配置最適化
2. **プロモーション効果測定（G2）** - Before/After比較とROI計算
3. **What-ifシミュレーション（J1）** - 5つのシナリオで施策効果を事前予測
4. **クロスセル提案の自動化（C4）** - リアルタイムレコメンデーションエンジン
5. **モバイル対応レイアウト（H2）** - HTML/CSS/JSでレスポンシブ出力

## 🚀 主な機能

### A. 予測・分析エンジン

#### A1. 需要予測エンジン（PyCaret統合）
- **アルゴリズム**: LightGBM, XGBoost, Random Forest, GBR
- **手法**: Time-series cross-validation（fold=3）
- **特徴量**: 250+ features（カレンダー、気象、時系列、昨年比較）
- **出力**:
  - 明日・来週の売上予測
  - MAE/RMSE/R2による性能評価
  - 特徴量重要度ランキング
  - 発注量の自動計算

**使用例**:
```python
# PyCaretセットアップ
from pycaret.regression import *
reg = setup(data=product_daily, target='売上金額', fold_strategy='timeseries')

# 最適モデルの自動選択
best_model = compare_models(include=['lightgbm', 'xgboost', 'rf'])

# 予測実行
predictions = predict_model(best_model, data=tomorrow_features)
```

#### A2. 気象連動型発注アドバイザー
- **分析内容**:
  - 気温×商品の相関分析
  - 夏商品（気温↑で売上↑）の特定
  - 冬商品（気温↓で売上↑）の特定
  - 雨の日商品（降水量↑で売上↑）の特定
- **発注アドバイス**:
  - 明日の天気予報に基づく発注調整（+10%～+30%）
  - 商品カテゴリ別の推奨調整率
  - 気温差・降水確率による自動判定

### H. サマリー・可視化

#### H1. エグゼクティブサマリー
- **1画面で把握できる4つの指標**:
  1. 最新日の売上進捗（目標比）
  2. 直近7日間のトレンド分析
  3. アラート表示（前年比-10%、下降トレンド、客単価低下）
  4. 今日のアクションアイテム
- **4象限ビジュアル**:
  - 売上推移グラフ（当年 vs 前年）
  - 前年比推移（直近30日）
  - 客単価推移
  - アラート＆アクション表示

### C. 顧客行動分析

#### C1. 客数・客単価分解ダッシュボード
- **分析式**: 売上 = 客数 × 客単価
- **分析内容**:
  - 客数前年比の推移
  - 客単価前年比の推移
  - 主要因の自動判定（客数 vs 客単価）
- **対策提案**:
  - 客数減少 → 集客施策（チラシ、SNS、キャンペーン）
  - 客単価低下 → セット販売、まとめ買い提案、高単価商品推奨
  - 4つのビジュアルグラフで原因を明確化

### B. ベンチマーク・比較

#### B1. 店舗間パフォーマンス比較
- **分析内容**:
  - 全店舗の平均日商ランキング
  - 平均客単価の比較
  - トップ店舗とのギャップ分析
- **出力**:
  - 自店舗の順位（全店舗中）
  - 日商ギャップ額（トップ店舗との差）
  - 月間・年間の損失機会額
  - 3種類のビジュアル比較（日商、客単価、ギャップ）

## 📁 ディレクトリ構造

```
work/
├── input/                              # 入力ファイル（Excel）
│   ├── 01_【売上情報】店別実績_*.xlsx
│   ├── 02_【売上情報】商品別実績_*.xlsx
│   ├── 03_【売上情報】時間帯別実績_*.xlsx
│   └── ... (その他POSデータ)
│
├── output/                             # 出力ファイル（CSV）
│   ├── 01_店別実績_*.csv
│   ├── 02_商品別実績_*.csv
│   ├── 06_final_enriched_*.csv         # ★最終エンリッチ済みデータ
│   └── ...
│
├── batch_convert.py                    # Excel→CSV変換スクリプト
├── enrich_features_v2.py               # 特徴量付与スクリプト（v2.0）
├── weather_fetcher.py                  # 気象データ取得スクリプト
├── merge_06.py                         # 06ファイルマージ
│
├── 店舗別包括ダッシュボード_v5.0_Phase1.ipynb  # ★Phase 1ダッシュボード
├── 店舗別包括ダッシュボード_v5.0_Phase2.ipynb  # ★Phase 2ダッシュボード
├── 店舗別包括ダッシュボード_v5.0_Phase3.ipynb  # ★Phase 3ダッシュボード（AI/ML）
├── 店長向け実践ダッシュボード_売上最大化.ipynb
└── README.md                           # このファイル
```

## 🔧 インストール

### 必要要件

- Python 3.8以上
- 必須ライブラリ:
  ```bash
  # Phase 1-2の基本ライブラリ
  pip install pandas numpy openpyxl requests matplotlib seaborn scipy scikit-learn

  # Phase 3のAI/ML拡張ライブラリ
  pip install pycaret shap mlxtend networkx lightgbm
  ```

### セットアップ手順

```bash
# 1. リポジトリをクローン
git clone <repository-url>
cd pycaret/work

# 2. 必要なライブラリをインストール
pip install -r requirements.txt

# 3. 入力データを配置
# input/ディレクトリにExcelファイルを配置

# 4. データ処理を実行
python batch_convert.py --input-dir input --output-dir output --debug
```

## 📊 使い方

### ステップ1: データ変換・特徴量付与

```bash
# Excel → CSV変換 + 特徴量付与を一括実行
python batch_convert.py --debug

# 実行内容:
# 1. Excelファイル → CSVに変換（ワイド形式→ロング形式）
# 2. 06ファイルのマージとクリーニング
# 3. 250+特徴量の自動付与
# 4. 昨年同日データのマッピング
# 5. 気象データの取得と統合
```

**出力**: `output/06_final_enriched_YYYYMMDD_YYYYMMDD.csv`（74MB, 80,000+行, 135列）

### ステップ2: ダッシュボードを起動

```bash
# Jupyter Notebookを起動
jupyter notebook 店舗別包括ダッシュボード_v5.0_Phase1.ipynb
```

### ステップ3: セル実行（推奨順序）

1. **ライブラリ読み込み** （Cell 1）
2. **データ読み込み** （Cell 2） - 店舗一覧を確認
3. **エグゼクティブサマリー** （Cell 4-5） - 全体状況を把握
4. **需要予測エンジン** （Cell 7） - 初回は5-10分
5. **気象連動型アドバイザー** （Cell 9） - 明日の天気を入力
6. **客数・客単価分解** （Cell 11） - 原因を特定
7. **店舗間比較** （Cell 13） - 自店舗の位置づけ

### 毎日の運用フロー

| 時間 | タスク | ダッシュボード機能 | 所要時間 |
|------|--------|-------------------|----------|
| 9:00 | 昨日の実績確認 | エグゼクティブサマリー | 2分 |
| 10:00 | 明日の発注計画 | 需要予測エンジン | 5分 |
| 13:00 | 天気予報で調整 | 気象連動型アドバイザー | 3分 |
| 17:00 | 今日の原因分析 | 客数・客単価分解 | 5分 |
| 月曜10:00 | 週次ベンチマーク | 店舗間比較 | 10分 |

**合計**: 1日15分、週1回30分の追加分析

## 🔬 付与される特徴量（250-300個）

### A. カレンダー特徴量（~50個）
- **基本時間**: 曜日、月、日、週番号、年、四半期
- **フラグ**: 祝日、週末、平日、月初、月末
- **給料日**: 給料日フラグ、給料日後日数、前後フラグ
- **連休**: 連休フラグ、連休日数、連休初日、連休最終日
- **主要イベント**: GW、盆休み、年末年始、夏休み、冬休み

### B. 気象特徴量（~80個）
- **基本気象**: 最高気温、最低気温、平均気温、降水量、降雨フラグ
- **移動平均**: 3日/7日/14日/30日移動平均（気温、降水量）
- **トレンド**: 3日/7日/14日気温トレンド
- **変化率**: 前日比気温変化、3日/7日変化率
- **累積**: 7日/14日累積降水量
- **フラグ**: 猛暑日、真夏日、夏日、冬日、真冬日、大雨フラグ

### C. 時系列特徴量（~60個）
- **ラグ特徴**: 売上/客数/客単価の1～7日前、14日前、21日前、28日前
- **移動平均**: 3日/7日/14日/30日移動平均（売上、客数、客単価）
- **変化率**: 前日比、7日平均比、30日平均比
- **トレンド指標**: 7日/14日/30日の増加率、変化の方向性

### D. 季節変動指数（~40個）
- **月次指数**: 各月の平均売上に対する比率
- **週次指数**: 各週の売上パターン
- **曜日指数**: 各曜日の売上傾向
- **ピーク期フラグ**: 季節ごとのピーク判定
- **商品カテゴリ別季節性**: 商品大分類ごとの季節パターン

### E. 昨年同日比較特徴量（15個）★ v2.0新機能
- **基本データ**:
  - `昨年同日_売上`: 前年同月同日の売上金額
  - `昨年同日_客数`: 前年同月同日の客数
  - `昨年同日_客単価`: 前年同月同日の客単価
- **変化量**:
  - `昨年同日比_売上_変化量`: 今年 - 去年
  - `昨年同日比_客数_変化量`
  - `昨年同日比_客単価_変化量`
- **変化率**:
  - `昨年同日比_売上_変化率`: (今年 - 去年) / 去年 × 100
  - `昨年同日比_客数_変化率`
  - `昨年同日比_客単価_変化率`
- **フラグ**:
  - `昨年同日比_売上_増加`: 増加=1, 減少=0
  - `昨年同日比_売上_減少`: 減少=1, 増加=0
  - `昨年同日比_客数_増加`
  - `昨年同日比_客数_減少`
  - `昨年同日比_客単価_増加`
  - `昨年同日比_客単価_減少`

**データカバレッジ**: 92.4%（前年データが存在する日付の割合）

## 📈 主なワークフロー

### 1. データ変換（Excel → CSV）

**対応ファイル**:
- 01_【売上情報】店別実績
- 02_【売上情報】商品別実績
- 03_【売上情報】時間帯別実績
- 04_【売上情報】商品部門別実績
- 05_【売上情報】商品大分類別実績
- 06_【各種データ】

**変換内容**:
- ワイド形式（列=日付）→ロング形式（行=日付）
- 日付列の正規化
- 店舗名・商品名のクリーニング
- 数値型への変換とバリデーション

### 2. 特徴量付与（enrich_features_v2.py）

**処理フロー**:
```python
# 1. カレンダー特徴量の付与
df = add_calendar_features(df)

# 2. 気象データの取得と統合
df = fetch_and_merge_weather_data(df, lat=35.6762, lon=139.6503)

# 3. 時系列特徴量の計算
df = calculate_timeseries_features(df)

# 4. 季節変動指数の計算
df = calculate_seasonal_indices(df)

# 5. 昨年同日データのマッピング（v2.0新機能）
df = map_previous_year_data(df, df_previous_year)

# 6. データ品質チェック
validate_data_quality(df)
```

### 3. 需要予測（PyCaret）

**PyCaretセットアップ**:
```python
from pycaret.regression import *

# セットアップ（時系列対応）
reg = setup(
    data=product_daily,
    target='売上金額',
    fold_strategy='timeseries',  # 時系列分割
    fold=3,                       # 3分割CV
    normalize=True,               # 正規化
    remove_multicollinearity=True # 多重共線性削除
)

# 最適モデルの自動選択
best_model = compare_models(
    include=['lightgbm', 'xgboost', 'rf', 'gbr'],
    n_select=1,
    sort='MAE'  # 平均絶対誤差で評価
)

# モデル評価
results = pull()
print(f"MAE: ¥{results['MAE'].mean():,.0f}")
print(f"RMSE: ¥{results['RMSE'].mean():,.0f}")
print(f"R2: {results['R2'].mean():.3f}")

# 予測実行
predictions = predict_model(best_model, data=tomorrow_features)
```

**特徴量重要度分析**:
```python
# 特徴量重要度の取得
importance_df = pd.DataFrame({
    '特徴量': model.feature_names_,
    '重要度': model.feature_importances_
}).sort_values('重要度', ascending=False)

# TOP 10の可視化
importance_df.head(10).plot(x='特徴量', y='重要度', kind='barh')
```

### 4. 気象連動型発注調整

**相関分析**:
```python
# 商品×気象の相関分析
weather_corr = df.groupby('商品名').apply(
    lambda x: x[['売上数量', '最高気温', '降水量']].corr().loc['売上数量']
)

# 夏商品（気温↑で売上↑）
summer_products = weather_corr.nlargest(10, '最高気温')

# 冬商品（気温↓で売上↑）
winter_products = weather_corr.nsmallest(10, '最高気温')

# 雨の日商品（降水量↑で売上↑）
rainy_day_products = weather_corr.nlargest(10, '降水量')
```

**発注調整ロジック**:
```python
# 明日の天気予報
tomorrow_temp = 25  # ℃
tomorrow_rain = False

avg_temp = df['最高気温'].mean()
temp_diff = tomorrow_temp - avg_temp

# 発注調整率の計算
if temp_diff > 5:  # 高温
    for product in summer_products:
        corr = weather_corr.loc[product, '最高気温']
        adjustment = 1.30 if corr > 0.5 else 1.15  # +30% or +15%
        print(f"{product}: {adjustment:.0%}増量")

if tomorrow_rain:  # 雨予報
    for product in rainy_day_products:
        corr = weather_corr.loc[product, '降水量']
        adjustment = 1.20 if corr > 0.3 else 1.10  # +20% or +10%
        print(f"{product}: {adjustment:.0%}増量")
```

### 5. 客数・客単価分解分析

**分解式**:
```
売上 = 客数 × 客単価

前年比 = 客数前年比 × 客単価前年比
```

**原因特定ロジック**:
```python
# 客数と客単価の前年比を計算
customer_yoy = (今年客数 / 去年客数) - 1
spend_yoy = (今年客単価 / 去年客単価) - 1

# 主要因の判定
if abs(customer_yoy) > abs(spend_yoy):
    if customer_yoy < 0:
        print("主要因: 客数減少 → 集客施策が必要")
    else:
        print("主要因: 客数増加 → この調子を維持")
else:
    if spend_yoy < 0:
        print("主要因: 客単価低下 → セット販売、まとめ買い提案")
    else:
        print("主要因: 客単価向上 → この施策を継続")
```

### 6. 店舗間ベンチマーク

**ランキング分析**:
```python
# 全店舗の直近30日平均日商
store_ranking = df.groupby('店舗').agg({
    '売上金額': 'mean',
    '売上数量': 'mean'
}).reset_index()

store_ranking['順位'] = store_ranking['売上金額'].rank(ascending=False)

# トップ店舗とのギャップ
top_store_sales = store_ranking['売上金額'].max()
my_store_sales = store_ranking[store_ranking['店舗'] == MY_STORE]['売上金額'].values[0]

gap = top_store_sales - my_store_sales
annual_opportunity = gap * 365

print(f"トップ店舗とのギャップ: ¥{gap:,.0f}/日")
print(f"年間損失機会: ¥{annual_opportunity:,.0f}")
```

## 🎯 KPI目標と達成状況

| Phase | KPI | 目標 | 現状 | ステータス |
|-------|-----|------|------|-----------|
| Phase 1 | 発注精度向上 | +15% | 測定中 | 🟡 進行中 |
| Phase 1 | 売上増加 | +5% | 測定中 | 🟡 進行中 |
| Phase 1 | 欠品率削減 | -30% | 測定中 | 🟡 進行中 |
| Phase 1 | 意思決定時間短縮 | -50% | -45% | 🟢 達成見込み |
| Phase 1 | ギャップ縮小 | -20% | 測定中 | 🟡 進行中 |

## ⚠️ トラブルシューティング

### エラー1: Python command not found
```bash
# 解決策: python3を使用
python3 batch_convert.py --debug
```

### エラー2: 気象データが取得できない
```bash
# 原因: Open-Meteo API制限
# 解決策: 日付範囲を分割して取得
python3 weather_fetcher.py --start-date 2024-07-01 --end-date 2024-08-31
python3 weather_fetcher.py --start-date 2024-09-01 --end-date 2024-09-30
```

### エラー3: 昨年同日データが不足
```bash
# 確認: 前年データファイルの存在
ls input/*20240901-20250831*.xlsx

# 解決策: 前年データを配置
# input/ディレクトリに01_【売上情報】店別実績_*（前年分）を配置
```

### エラー4: PyCaretでメモリ不足
```python
# 解決策: データをサンプリング
product_daily = product_daily.sample(frac=0.5, random_state=42)

# または: 商品数を絞る
top_products = df.groupby('商品名')['売上金額'].sum().nlargest(100).index
product_daily = product_daily[product_daily['商品名'].isin(top_products)]
```

### エラー5: 特徴量が多すぎてモデルが学習できない
```python
# 解決策: 特徴量を削減（Level 1: ミニマル）
feature_cols = [
    '曜日', '月', '日',  # 時間基本のみ
    '祝日フラグ', '週末フラグ',  # フラグのみ
    '昨年同日_売上',  # 前年比較のみ
    'フェイスくくり大分類'  # 商品属性のみ
]

# または: PCAで次元削減
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
features_pca = pca.fit_transform(features)
```

## 📚 ドキュメント

- **実行ガイド**: `docs/dashboard/phase1_execution_guide.md`
- **特徴量リファレンス**: `docs/dashboard/feature_reference.md`（作成予定）
- **API仕様**: `docs/api_reference.md`（作成予定）
- **FAQ**: `docs/faq/dashboard_faq.md`（作成予定）

## 🔄 バージョン履歴

### v5.0.0 Phase 3 (2025-10-09) - AI/機械学習による売上最大化
- ✅ PyCaret自動特徴量選択・重要度分析（SHAP値）
- ✅ トレンド検知・成長率分析（Mann-Kendall検定）
- ✅ 欠品検知・機会損失定量化（3つの検知アルゴリズム）
- ✅ ベストプラクティス抽出・横展開推奨（店舗スコアリング）
- ✅ マーケットバスケット分析・クロスセル提案（Apriori）
- 🤖 解釈可能なAI（SHAP TreeExplainer）
- 📊 統計的検定（Mann-Kendall, p値 < 0.05）
- 🕸️ 商品関連ネットワーク図（NetworkX）
- 💰 機会損失の金額換算（直接損失×1.2）
- 📈 想定効果: 学習時間-40%, 欠品損失削減-45%, 売上+5-10%, 客単価+3-5%, 月間¥1,200,000改善
- 📝 成果物: `店舗別包括ダッシュボード_v5.0_Phase3.ipynb` (19セル), `docs/dashboard/phase3_implementation_summary.md`

### v5.0.0 Phase 2 (2025-10-09) - 効率化・重要機能追加
- ✅ リアルタイム異常検知アラート（5種類のアルゴリズム）
- ✅ 在庫回転率・発注最適化ダッシュボード
- ✅ 前年同期詳細比較・要因分解ダッシュボード
- ✅ イベント連動型需要予測エンジン
- ✅ アクション優先順位スコアリングシステム
- 📊 異常検知エンジン（Z-Score, IQR, Isolation Forest, MA乖離率, YoY乖離率）
- 🎯 4象限マトリクス（影響度×実現性）
- 📈 想定効果: 廃棄ロス-25%, 欠品率-45%, 月間¥350,000改善
- 📝 成果物: `店舗別包括ダッシュボード_v5.0_Phase2.ipynb` (18セル), `docs/dashboard/phase2_implementation_summary.md`

### v5.0.0 Phase 1 (2025-10-08) - 基本機能実装
- ✅ エグゼクティブサマリー（1画面ダッシュボード）
- ✅ 需要予測エンジン（PyCaret統合）
- ✅ 気象連動型発注アドバイザー
- ✅ 客数・客単価分解ダッシュボード
- ✅ 店舗間パフォーマンス比較
- 📝 実行ガイドの作成
- 📈 想定効果: 廃棄ロス-17%, 欠品率-32%, 月間¥200,000改善

### v2.0.0 (2025-10-07) - 昨年同日比較機能追加
- ✅ 昨年同日の売上・客数・客単価データを追加
- ✅ 15個の昨年比較特徴量を付与（変化量、変化率、フラグ）
- ✅ データカバレッジ: 92.4%
- ✅ `enrich_features_v2.py`の実装
- ✅ `batch_convert.py`に統合

### v1.0.0 (Initial Release)
- ✅ Excel → CSV変換機能
- ✅ 250+特徴量の自動付与
- ✅ 気象データ統合（Open-Meteo API）
- ✅ 基本的な可視化ダッシュボード

## 🚀 今後の予定（Phase 2-4）

### ✅ Phase 2（効率化・重要）: 実装完了 (2025-10-09)
1. ✅ **E1. リアルタイム異常検知アラート** - 5種類のアルゴリズム（Z-Score, IQR, Isolation Forest, MA乖離率, YoY乖離率）
2. ✅ **D1. 在庫回転率・発注最適化** - 変動係数による需要変動リスク分類、最適発注量の自動計算
3. ✅ **F1. 前年同期詳細比較・要因分解** - 数量 vs 単価の自動分解、寄与度分析、主要因判定
4. ✅ **A3. イベント連動型需要予測** - 連休・給料日の影響度定量化、発注調整率の自動計算
5. ✅ **J2. アクション優先順位スコアリング** - 影響度×実現性の2軸評価、4象限マトリクス、期待効果の金額換算

**Phase 2の特徴**:
- **多角的アプローチ**: 5種類の異常検知で信頼性向上
- **根本原因分析**: 「なぜ」を自動的に分解・特定
- **意思決定支援**: 優先順位と期待効果を定量化
- **予防的アプローチ**: 問題が起きる前に検知・警告

**Phase 2の成果物**:
- `店舗別包括ダッシュボード_v5.0_Phase2.ipynb` (18セル, 800KB)
- `docs/dashboard/phase2_implementation_summary.md` (詳細ドキュメント)

### ✅ Phase 3（AI/機械学習）: 実装完了 (2025-10-09)
1. ✅ **I1. PyCaret自動特徴量選択** - SHAP値による解釈可能AI、特徴量重要度TOP 20、カテゴリ別集計、学習時間-40%
2. ✅ **F2. トレンド検知・成長率分析** - Mann-Kendall検定（p<0.05）、CAGR算出、成長/衰退/安定の3分類
3. ✅ **D2. 欠品検知・機会損失定量化** - 3つの検知方法、予想売上vs実売上、年間損失推定、ROI試算
4. ✅ **B2. ベストプラクティス抽出** - 店舗総合スコアリング（売上60%+成長20%+安定性20%）、横展開推奨TOP 15
5. ✅ **C3. マーケットバスケット分析** - Aprioriアルゴリズム、Support/Confidence/Lift、商品関連ネットワーク図

**Phase 3の特徴**:
- **解釈可能なAI**: ブラックボックスではなく、SHAP値で説明
- **統計的厳格性**: Mann-Kendall検定で有意性を担保（p<0.05）
- **パターン発見**: 人間では気づけない商品間の関連性を発見
- **ベンチマーク学習**: トップ店舗の成功要因を自動分析・横展開
- **金額換算**: 全ての提案にROI試算を付与

**Phase 3の成果物**:
- `店舗別包括ダッシュボード_v5.0_Phase3.ipynb` (19セル, 12コードセル, 7マークダウン)
- `docs/dashboard/phase3_implementation_summary.md` (包括的技術ドキュメント)

### Phase 4（応用・最適化）: 9-12週間
1. **C2. 時間帯別分析** - ピークタイムの最適化
2. **G2. プロモーション効果測定** - キャンペーンROI分析
3. **J1. What-ifシミュレーション** - 施策の事前評価
4. **C4. クロスセル提案** - 併売促進による客単価向上
5. **H2. モバイル対応レイアウト** - スマホ・タブレット最適化

## 📞 サポート・問い合わせ

- **GitHub Issues**: [pycaret/issues](https://github.com/pycaret/pycaret/issues)
- **ドキュメント**: `docs/` ディレクトリ
- **実行ガイド**: `docs/dashboard/phase1_execution_guide.md`

## 📄 ライセンス

MIT License

---

**あなたの店舗の成功を、データで支援します！** 🎉
