"""
時系列分析モジュールの使用例

このスクリプトは、PyCaret の時系列分析機能を使用した
包括的な時系列分析の実例を示します。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analysis.time_series import create_analyzer


def generate_sample_data() -> pd.DataFrame:
    """サンプルデータの生成"""
    np.random.seed(42)

    # 2年分の日次データを生成
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # 基本トレンド + 季節性 + ノイズ
    n_days = len(date_range)
    trend = np.linspace(1000, 1500, n_days)  # 上昇トレンド
    seasonality = 200 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # 年次季節性
    weekly_pattern = 100 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # 週次パターン
    noise = np.random.normal(0, 50, n_days)

    sales = trend + seasonality + weekly_pattern + noise

    # 異常値をいくつか追加
    anomaly_indices = np.random.choice(n_days, size=10, replace=False)
    sales[anomaly_indices] *= np.random.uniform(1.5, 2.0, size=len(anomaly_indices))

    # データフレームの作成
    data = pd.DataFrame({
        'date': date_range,
        'sales': sales,
        'store_id': np.random.choice(['店舗A', '店舗B', '店舗C'], n_days)
    })

    return data


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("時系列分析モジュール - 使用例")
    print("=" * 70)

    # 1. サンプルデータの生成
    print("\n1. サンプルデータの生成...")
    data = generate_sample_data()
    print(f"   データ件数: {len(data):,} 件")
    print(f"   期間: {data['date'].min()} ~ {data['date'].max()}")
    print(f"   店舗数: {data['store_id'].nunique()} 店舗")

    # 2. アナライザーの作成
    print("\n2. 時系列アナライザーの作成...")
    analyzer = create_analyzer(
        data,
        date_column='date',
        value_column='sales',
        store_column='store_id'
    )
    print("   ✓ アナライザーを初期化しました")

    # 3. 要約統計の取得
    print("\n3. 要約統計情報:")
    print("-" * 70)
    summary = analyzer.get_summary_statistics()

    print("\n   【基本統計】")
    for key, value in summary['基本統計'].items():
        print(f"   {key:12}: {value}")

    print("\n   【変化統計】")
    for key, value in summary['変化統計'].items():
        print(f"   {key:12}: {value}")

    print("\n   【定常性テスト】")
    for key, value in summary['定常性'].items():
        if key != '臨界値':
            print(f"   {key:12}: {value}")

    # 4. 月次トレンド分析
    print("\n4. 月次トレンド分析 (直近6ヶ月):")
    print("-" * 70)
    monthly_trend = analyzer.analyze_monthly_trend(yoy_comparison=True)
    print(monthly_trend.tail(6))

    # 5. 週次パフォーマンス
    print("\n5. 週次パフォーマンス (直近8週):")
    print("-" * 70)
    weekly_perf = analyzer.analyze_weekly_performance(wow_comparison=True)
    print(weekly_perf.tail(8)[['合計', 'WoW変化率(%)', 'パフォーマンス']])

    # 6. 異常検知
    print("\n6. 異常検知:")
    print("-" * 70)
    anomalies = analyzer.detect_anomalies(method='zscore', threshold=2.5)
    print(f"   検出された異常値: {anomalies.anomaly_count} 件")
    print(f"   異常値率: {anomalies.statistics['異常値率(%)']}%")

    if anomalies.anomaly_count > 0:
        print("\n   【異常値の例】")
        print(anomalies.anomalies.head())

    # 7. 移動平均
    print("\n7. 移動平均 (直近10日):")
    print("-" * 70)
    ma_data = analyzer.calculate_moving_averages(windows=[7, 30])
    print(ma_data.tail(10)[['実績値', 'SMA_7日', 'SMA_30日', 'クロス']])

    # 8. 成長率
    print("\n8. 成長率分析 (直近5日):")
    print("-" * 70)
    growth_rates = analyzer.calculate_growth_rates(periods=[1, 7, 30])
    print(growth_rates.tail(5)[['値', '1日成長率(%)', '7日成長率(%)', '30日成長率(%)']])

    # 9. 季節分解
    print("\n9. 季節分解:")
    print("-" * 70)
    try:
        decomp = analyzer.seasonal_decomposition(model='additive', period=7)
        decomp_df = decomp.to_dataframe()
        print(f"   分解完了: {len(decomp_df)} データポイント")
        print("\n   【分解結果サンプル】")
        print(decomp_df.tail())
    except Exception as e:
        print(f"   エラー: {e}")

    # 10. 店舗別分析
    print("\n10. 店舗別分析 (店舗A):")
    print("-" * 70)
    store_analyzer = analyzer.filter_by_store('店舗A')
    store_summary = store_analyzer.get_summary_statistics()
    print(f"   データ数: {store_summary['基本統計']['データ数']}")
    print(f"   平均売上: {store_summary['基本統計']['平均']:.2f}")
    print(f"   合計売上: {store_summary['基本統計']['合計']:,.2f}")

    # 11. 分析結果のエクスポート
    print("\n11. 分析結果のエクスポート:")
    print("-" * 70)
    output_file = "/mnt/d/github/pycaret/work/time_series_analysis_results.xlsx"

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        analyzer.export_analysis(output_file, include_all=True)
        print(f"   ✓ 分析結果を保存しました: {output_file}")
    except Exception as e:
        print(f"   エラー: {e}")

    print("\n" + "=" * 70)
    print("分析完了!")
    print("=" * 70)


if __name__ == '__main__':
    main()
