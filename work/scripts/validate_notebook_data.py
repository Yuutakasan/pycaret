#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ存在検証スクリプト

Phase 1-4ノートブックの各分析で使用するデータの存在を事前チェックします。
存在しないデータを使用する分析を検出して警告します。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def validate_data_availability(df, analysis_name="分析"):
    """
    データフレームの存在と必要カラムをチェック

    Parameters:
    -----------
    df : pandas.DataFrame
        検証対象のデータフレーム
    analysis_name : str
        分析名（エラーメッセージ用）

    Returns:
    --------
    dict : 検証結果 {'valid': bool, 'message': str, 'row_count': int}
    """
    if df is None or len(df) == 0:
        return {
            'valid': False,
            'message': f'❌ {analysis_name}: データが存在しません',
            'row_count': 0
        }

    return {
        'valid': True,
        'message': f'✅ {analysis_name}: データ利用可能 ({len(df):,}行)',
        'row_count': len(df)
    }


def check_required_columns(df, required_cols, analysis_name="分析"):
    """
    必須カラムの存在をチェック

    Parameters:
    -----------
    df : pandas.DataFrame
        検証対象のデータフレーム
    required_cols : list
        必須カラムのリスト
    analysis_name : str
        分析名

    Returns:
    --------
    dict : 検証結果
    """
    if df is None:
        return {
            'valid': False,
            'message': f'❌ {analysis_name}: データフレームがNone',
            'missing_cols': required_cols
        }

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        return {
            'valid': False,
            'message': f'❌ {analysis_name}: 必須カラム不足 - {missing_cols}',
            'missing_cols': missing_cols
        }

    return {
        'valid': True,
        'message': f'✅ {analysis_name}: 必須カラム存在確認',
        'missing_cols': []
    }


def check_hourly_data(df):
    """
    Phase 4の時間帯別分析用データをチェック
    """
    if '時刻' not in df.columns and '時間' not in df.columns:
        return {
            'valid': False,
            'message': '❌ 時間帯別データ: 時刻カラムが存在しません（時刻/時間）',
            'warning': '時間帯別分析は平均パターンで代替されます'
        }

    return {
        'valid': True,
        'message': '✅ 時間帯別データ: 利用可能',
        'warning': None
    }


def check_weather_data(df):
    """
    気象データの存在チェック
    """
    weather_cols = ['気温', '降水量', '天気', '天候']
    has_weather = any(col in df.columns for col in weather_cols)

    if not has_weather:
        return {
            'valid': False,
            'message': '❌ 気象データ: 存在しません',
            'warning': '気象連動分析は代替ロジックで実行されます'
        }

    return {
        'valid': True,
        'message': '✅ 気象データ: 利用可能',
        'warning': None
    }


def check_previous_year_data(df):
    """
    前年データの存在チェック
    """
    prev_year_cols = [col for col in df.columns if '昨年' in col or '前年' in col]

    if not prev_year_cols:
        return {
            'valid': False,
            'message': '❌ 前年データ: 存在しません',
            'warning': '前年比較分析はスキップされます'
        }

    # データの存在率チェック
    coverage = {}
    for col in prev_year_cols:
        non_null_pct = df[col].notna().sum() / len(df) * 100
        coverage[col] = non_null_pct

    avg_coverage = np.mean(list(coverage.values()))

    return {
        'valid': True,
        'message': f'✅ 前年データ: 利用可能（カバレッジ: {avg_coverage:.1f}%）',
        'warning': '⚠️ カバレッジが90%未満' if avg_coverage < 90 else None,
        'coverage': coverage
    }


def validate_phase1_data(df):
    """Phase 1の必須データをチェック"""
    print("\n" + "="*80)
    print("📊 Phase 1: 毎日の基本業務 - データ検証")
    print("="*80)

    results = []

    # 基本データ検証
    result = validate_data_availability(df, "Phase 1基本データ")
    results.append(result)
    print(result['message'])

    if not result['valid']:
        return results

    # 必須カラム検証
    required_cols = ['日付', '売上金額', '店舗']
    result = check_required_columns(df, required_cols, "Phase 1必須カラム")
    results.append(result)
    print(result['message'])

    # 気象データ検証
    result = check_weather_data(df)
    results.append(result)
    print(result['message'])
    if result['warning']:
        print(f"   {result['warning']}")

    # 前年データ検証
    result = check_previous_year_data(df)
    results.append(result)
    print(result['message'])
    if result['warning']:
        print(f"   {result['warning']}")

    return results


def validate_phase2_data(df):
    """Phase 2の必須データをチェック"""
    print("\n" + "="*80)
    print("📊 Phase 2: 問題の早期発見 - データ検証")
    print("="*80)

    results = []

    # 基本データ検証
    result = validate_data_availability(df, "Phase 2基本データ")
    results.append(result)
    print(result['message'])

    if not result['valid']:
        return results

    # 異常検知用カラム
    required_cols = ['日付', '売上金額', '商品名']
    result = check_required_columns(df, required_cols, "Phase 2異常検知")
    results.append(result)
    print(result['message'])

    # 在庫データ検証
    inventory_cols = ['在庫数', '発注数', '廃棄数']
    has_inventory = any(col in df.columns for col in inventory_cols)

    if not has_inventory:
        result = {
            'valid': False,
            'message': '❌ 在庫データ: 存在しません',
            'warning': '在庫最適化分析は基本的な計算のみ実行されます'
        }
        results.append(result)
        print(result['message'])
        print(f"   {result['warning']}")
    else:
        result = {
            'valid': True,
            'message': '✅ 在庫データ: 利用可能',
            'warning': None
        }
        results.append(result)
        print(result['message'])

    return results


def validate_phase3_data(df):
    """Phase 3の必須データをチェック"""
    print("\n" + "="*80)
    print("📊 Phase 3: AI/機械学習で深掘り - データ検証")
    print("="*80)

    results = []

    # 基本データ検証
    result = validate_data_availability(df, "Phase 3基本データ")
    results.append(result)
    print(result['message'])

    if not result['valid']:
        return results

    # 特徴量カラム検証
    feature_cols = ['客数', '客単価', '曜日', '月']
    result = check_required_columns(df, feature_cols, "Phase 3特徴量")
    results.append(result)
    print(result['message'])

    # バスケット分析用データ
    if '商品名' in df.columns and 'トランザクションID' in df.columns:
        result = {
            'valid': True,
            'message': '✅ バスケット分析データ: 利用可能',
            'warning': None
        }
        results.append(result)
        print(result['message'])
    else:
        result = {
            'valid': False,
            'message': '❌ バスケット分析データ: トランザクションIDまたは商品名が不足',
            'warning': 'バスケット分析はスキップされます'
        }
        results.append(result)
        print(result['message'])
        print(f"   {result['warning']}")

    return results


def validate_phase4_data(df):
    """Phase 4の必須データをチェック"""
    print("\n" + "="*80)
    print("📊 Phase 4: 戦略立案と意思決定 - データ検証")
    print("="*80)

    results = []

    # 基本データ検証
    result = validate_data_availability(df, "Phase 4基本データ")
    results.append(result)
    print(result['message'])

    if not result['valid']:
        return results

    # 時間帯別データ検証
    result = check_hourly_data(df)
    results.append(result)
    print(result['message'])
    if result['warning']:
        print(f"   {result['warning']}")

    # プロモーションデータ検証
    promo_cols = ['プロモーション', 'キャンペーン', '施策']
    has_promo = any(col in df.columns for col in promo_cols)

    if not has_promo:
        result = {
            'valid': False,
            'message': '❌ プロモーションデータ: 存在しません',
            'warning': 'プロモーション効果測定は仮想データで実行されます'
        }
        results.append(result)
        print(result['message'])
        print(f"   {result['warning']}")
    else:
        result = {
            'valid': True,
            'message': '✅ プロモーションデータ: 利用可能',
            'warning': None
        }
        results.append(result)
        print(result['message'])

    return results


def main():
    """メイン処理"""
    print("\n" + "="*80)
    print("🔍 Phase 1-4 データ存在検証スクリプト")
    print("="*80)
    print(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")

    # データファイル検索
    data_dir = Path('/mnt/d/github/pycaret/work/output')
    csv_files = sorted(data_dir.glob('06_*.csv'))

    if not csv_files:
        print("\n❌ エラー: output/06_*.csv ファイルが見つかりません")
        print("   enrich_features_v2.py を実行してデータを準備してください")
        return

    # 最新データファイル読み込み
    latest_file = csv_files[-1]
    print(f"\n📂 データファイル: {latest_file.name}")

    try:
        df = pd.read_csv(latest_file, parse_dates=['日付'])
        print(f"✅ データ読み込み成功: {len(df):,}行 x {len(df.columns)}列")
    except Exception as e:
        print(f"❌ エラー: データ読み込み失敗 - {e}")
        return

    # 店舗一覧
    if '店舗' in df.columns:
        stores = sorted(df['店舗'].unique())
        print(f"\n🏪 店舗数: {len(stores)}店舗")
        for i, store in enumerate(stores[:5], 1):
            print(f"   {i}. {store}")
        if len(stores) > 5:
            print(f"   ... 他 {len(stores)-5}店舗")

    # 各Phaseのデータ検証
    all_results = {
        'Phase 1': validate_phase1_data(df),
        'Phase 2': validate_phase2_data(df),
        'Phase 3': validate_phase3_data(df),
        'Phase 4': validate_phase4_data(df)
    }

    # サマリー
    print("\n" + "="*80)
    print("📋 検証サマリー")
    print("="*80)

    for phase, results in all_results.items():
        valid_count = sum(1 for r in results if r['valid'])
        total_count = len(results)
        print(f"{phase}: {valid_count}/{total_count} 項目OK")

    # 警告まとめ
    warnings_found = []
    for phase, results in all_results.items():
        for result in results:
            if result.get('warning'):
                warnings_found.append(f"{phase}: {result['warning']}")

    if warnings_found:
        print("\n⚠️ 警告:")
        for warning in warnings_found:
            print(f"   {warning}")
    else:
        print("\n✅ すべてのチェックに合格しました")

    print("\n" + "="*80)
    print("💡 ヒント:")
    print("   - 警告がある場合、該当する分析は代替ロジックで実行されます")
    print("   - データが不足している場合、データ収集を改善してください")
    print("   - 各Phaseのノートブックは、データの有無を自動判定します")
    print("="*80)


if __name__ == '__main__':
    main()
