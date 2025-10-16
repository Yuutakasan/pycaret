# 時系列分析モジュール (Time-Series Analysis Module)

## 概要

このモジュールは、PyCaret の時系列機能を使用した包括的な時系列分析を提供します。売上データ、トラフィックデータ、センサーデータなど、様々な時系列データの分析に対応しています。

## 主な機能

### 1. 月次トレンド分析 (YoY比較付き)
- 月次集計（合計、平均、最大、最小、標準偏差）
- MoM（Month over Month）変化率
- YoY（Year over Year）変化率
- トレンド方向の自動判定

### 2. 週次パフォーマンス追跡 (WoW変化付き)
- 週次集計と統計量
- WoW（Week over Week）変化率
- パフォーマンスカテゴリ分類（上昇/安定/下降）

### 3. 日次売上パターン (異常検知付き)
- 日次データ分析
- 曜日パターン分析
- 異常値の自動検知（Zスコア法）

### 4. 季節分解
- トレンド成分の抽出
- 季節性成分の抽出
- 残差成分の分析
- 加法/乗法モデルのサポート

### 5. 成長率計算
- 複数期間の成長率計算（1日、7日、30日、90日、365日）
- CAGR（年複合成長率）の計算

### 6. 移動平均
- 単純移動平均（SMA）
- 指数移動平均（EMA）
- ゴールデンクロス/デッドクロスの検出

### 7. 店舗レベルのフィルタリング
- 特定店舗でのフィルタリング
- 店舗別分析

## インストール

PyCaret とその依存関係がインストールされている必要があります：

```bash
pip install pycaret[full]
```

追加の依存関係：

```bash
pip install statsmodels scipy pyod
```

## 使用方法

### 基本的な使い方

```python
import pandas as pd
from src.analysis.time_series import create_analyzer

# データ読み込み
data = pd.read_csv('sales_data.csv')

# アナライザー作成
analyzer = create_analyzer(
    data,
    date_column='date',
    value_column='sales',
    store_column='store_id'
)

# 要約統計の取得
summary = analyzer.get_summary_statistics()
print(summary)
```

### 月次トレンド分析

```python
# YoY比較付きの月次トレンド分析
monthly_trend = analyzer.analyze_monthly_trend(yoy_comparison=True)
print(monthly_trend)
```

出力例：
```
                    合計        平均        最大        最小      標準偏差  MoM変化量  MoM変化率(%)  YoY変化量  YoY変化率(%)  トレンド
date
2023-01-31  31245.23  1008.23  1523.45   512.34    152.34       NaN          NaN        NaN          NaN     不明
2023-02-28  28456.78   890.52  1412.67   489.23    148.92  -2788.45        -8.93        NaN          NaN     下降
...
```

### 週次パフォーマンス追跡

```python
# WoW比較付きの週次パフォーマンス
weekly_perf = analyzer.analyze_weekly_performance(wow_comparison=True)
print(weekly_perf[['合計', 'WoW変化率(%)', 'パフォーマンス']])
```

### 異常検知

```python
# Zスコア法による異常検知
anomalies = analyzer.detect_anomalies(method='zscore', threshold=3.0)
print(f"検出された異常値: {anomalies.anomaly_count}件")
print(anomalies.anomalies)

# IQR法による異常検知
anomalies_iqr = analyzer.detect_anomalies(method='iqr', threshold=1.5)

# Isolation Forestによる異常検知
anomalies_if = analyzer.detect_anomalies(method='isolation_forest')
```

### 移動平均の計算

```python
# 7日間と30日間の移動平均
ma_data = analyzer.calculate_moving_averages(windows=[7, 30])
print(ma_data[['実績値', 'SMA_7日', 'SMA_30日', 'EMA_7日', 'クロス']])
```

### 季節分解

```python
# 週次季節性で分解
decomp = analyzer.seasonal_decomposition(model='additive', period=7)
decomp_df = decomp.to_dataframe()
print(decomp_df)
```

### 成長率分析

```python
# 複数期間の成長率を計算
growth_rates = analyzer.calculate_growth_rates(periods=[1, 7, 30, 90])
print(growth_rates)
```

### 店舗別分析

```python
# 特定店舗でフィルタリング
store_analyzer = analyzer.filter_by_store('店舗001')
store_monthly = store_analyzer.analyze_monthly_trend()
print(store_monthly)
```

### 分析結果のエクスポート

```python
# 全分析結果をExcelにエクスポート
analyzer.export_analysis('analysis_results.xlsx', include_all=True)
```

エクスポートされるシート：
- 要約統計
- 月次トレンド
- 週次パフォーマンス
- 日次パターン
- 移動平均
- 成長率
- 異常値
- 季節分解

## 設定オプション

### TimeSeriesConfig

```python
from src.analysis.time_series import TimeSeriesConfig, TimeSeriesAnalyzer

config = TimeSeriesConfig(
    date_column='date',           # 日付カラム名
    value_column='sales',         # 値カラム名
    store_column='store_id',      # 店舗カラム名（オプション）
    freq='D',                     # 周波数（'D': 日次, 'W': 週次, 'M': 月次）
    seasonal_period=7,            # 季節周期（週次=7）
    anomaly_threshold=3.0,        # 異常検知の閾値（標準偏差）
    ma_windows=[7, 30]           # 移動平均の窓サイズ
)

analyzer = TimeSeriesAnalyzer(data, config)
```

## クラスとメソッド

### TimeSeriesAnalyzer

主要なメソッド：

- `analyze_monthly_trend(yoy_comparison=True)` - 月次トレンド分析
- `analyze_weekly_performance(wow_comparison=True)` - 週次パフォーマンス分析
- `analyze_daily_patterns(detect_anomalies=True)` - 日次パターン分析
- `seasonal_decomposition(model='additive', period=None)` - 季節分解
- `calculate_growth_rates(periods=None)` - 成長率計算
- `calculate_moving_averages(windows=None)` - 移動平均計算
- `detect_anomalies(method='zscore', threshold=None)` - 異常検知
- `get_summary_statistics()` - 要約統計取得
- `filter_by_store(store_id)` - 店舗フィルタリング
- `export_analysis(output_path, include_all=True)` - 分析結果エクスポート

### データクラス

- `TimeSeriesConfig` - 分析設定
- `TrendAnalysisResult` - トレンド分析結果
- `SeasonalDecompositionResult` - 季節分解結果
- `AnomalyDetectionResult` - 異常検知結果

## 異常検知手法

### 1. Zスコア法（デフォルト）
```python
anomalies = analyzer.detect_anomalies(method='zscore', threshold=3.0)
```
標準偏差を基準とした異常値検出。閾値は標準偏差の倍数。

### 2. IQR法（四分位範囲）
```python
anomalies = analyzer.detect_anomalies(method='iqr', threshold=1.5)
```
四分位範囲を基準とした異常値検出。外れ値の検出に有効。

### 3. Isolation Forest
```python
anomalies = analyzer.detect_anomalies(method='isolation_forest')
```
機械学習ベースの異常値検出。複雑なパターンの検出に有効。

## パフォーマンスの最適化

大規模データセットの場合：

```python
# データの事前集約
aggregated_data = data.groupby('date').agg({
    'sales': 'sum'
}).reset_index()

# 必要な期間のみ抽出
recent_data = data[data['date'] >= '2024-01-01']

analyzer = create_analyzer(recent_data)
```

## トラブルシューティング

### データ不足エラー
```
ValueError: データが不足しています。最低 14 個のデータポイントが必要です。
```

**解決方法**: 季節分解には、少なくとも `2 * period` のデータポイントが必要です。

### 定常性の問題
非定常なデータの場合、差分を取ることで定常化できます：

```python
# 1次差分
data['sales_diff'] = data['sales'].diff()

# 季節差分
data['sales_seasonal_diff'] = data['sales'].diff(7)  # 週次季節性
```

## 実用例

### 売上データの包括的分析

```python
import pandas as pd
from src.analysis.time_series import create_analyzer

# データ読み込み
sales_data = pd.read_csv('daily_sales.csv')

# アナライザー作成
analyzer = create_analyzer(
    sales_data,
    date_column='sale_date',
    value_column='total_sales',
    store_column='store_code'
)

# 1. 全体トレンドの把握
summary = analyzer.get_summary_statistics()
print("定常性:", summary['定常性']['定常性'])

# 2. 月次パフォーマンスのモニタリング
monthly = analyzer.analyze_monthly_trend()
recent_months = monthly.tail(6)
print("直近6ヶ月のトレンド:")
print(recent_months[['合計', 'YoY変化率(%)', 'トレンド']])

# 3. 異常な売上日の特定
anomalies = analyzer.detect_anomalies(method='zscore', threshold=2.5)
if anomalies.anomaly_count > 0:
    print(f"\n異常値検出: {anomalies.anomaly_count}件")
    print(anomalies.anomalies[['日付', '値', 'スコア']])

# 4. 季節パターンの理解
decomp = analyzer.seasonal_decomposition(period=7)
print("\n季節性成分の平均:", decomp.seasonal.mean())

# 5. 全結果の保存
analyzer.export_analysis('comprehensive_analysis.xlsx')
```

## ベストプラクティス

1. **データの前処理**: 欠損値を適切に処理してからアナライザーに渡す
2. **適切な周期設定**: ビジネスサイクルに合わせた季節周期を設定
3. **異常検知の調整**: ビジネス特性に応じて閾値を調整
4. **定常性の確認**: 非定常データは差分を取る
5. **店舗別分析**: 全体トレンドと店舗別トレンドの両方を確認

## 参考資料

- [PyCaret Time Series Documentation](https://pycaret.gitbook.io/docs/get-started/modules)
- [statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
- [sktime Documentation](https://www.sktime.net/en/stable/)

## ライセンス

MIT License - PyCaret Project
