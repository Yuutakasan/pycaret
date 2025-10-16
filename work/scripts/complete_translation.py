"""
すべての残存英語を日本語に完全翻訳
"""

import json
import re
from pathlib import Path

# 包括的な英語→日本語翻訳マップ
COMPREHENSIVE_TRANSLATIONS = {
    # グラフタイトル
    'Feature Importance for Sales Prediction': '売上予測の特徴量重要度',
    'Feature Importance by Category': 'カテゴリ別特徴量重要度',
    'SHAP Feature Importance': 'SHAP特徴量重要度',
    'Customer Count YoY Growth (%)': '客数前年比成長率 (%)',
    'Sales Distribution by Time Period': '時間帯別売上分布',
    'Weekday x Hour Heatmap': '曜日×時間帯ヒートマップ',
    'Hour of Day': '時刻',
    'Coefficient of Variation': '変動係数',
    'Average Sales': '平均売上',
    'Weekday data not available': '曜日データなし',

    # KPIテキスト
    "KPIs (Latest):": "主要指標 (最新):",
    "Sales:": "売上:",
    "YoY:": "前年比:",
    "Avg Spend:": "客単価:",
    "Trend (7d):": "トレンド(7日):",

    # アクションテキスト
    "TODAY'S ACTIONS:": "本日のアクション:",
    "Check TOP10 inventory": "TOP10在庫確認",
    "Adjust orders by weather": "天候による発注調整",
    "Analyze YoY negative items": "前年比減少商品分析",
    "Compare with other stores": "他店舗との比較",

    # アラートレベル
    "'🔴 Critical'": "'🔴 緊急'",
    "'🟡 Warning'": "'🟡 警告'",
    "Critical（緊急）": "緊急",
    "Warning（警告）": "警告",

    # 検出方法
    "'IQR'": "'四分位範囲'",
    "'Isolation Forest'": "'孤立森林'",
    "'MA Deviation'": "'移動平均偏差'",
    "'YoY Deviation'": "'前年比偏差'",

    # 評価指標
    "'MAE'": "'平均絶対誤差'",
    "'RMSE'": "'二乗平均平方根誤差'",
    "'MAPE'": "'平均絶対パーセント誤差'",
    "'CAGR'": "'年平均成長率'",

    # その他
    "'YlOrRd'": "'YlOrRd'",  # カラーマップ名は保持
    "per Customer": "顧客あたり",
    "Last 30d": "過去30日間",
}

# 特殊なパターン処理用
PATTERN_REPLACEMENTS = [
    # results['MAE'] のような辞書キー
    (r"results\['MAE'\]", "results['平均絶対誤差']"),
    (r"results\['RMSE'\]", "results['二乗平均平方根誤差']"),
    (r"results\['MAPE'\]", "results['平均絶対パーセント誤差']"),

    # anomaly_results['alert_level'] == 'Critical'
    (r"== '🔴 Critical'", "== '🔴 緊急'"),
    (r"== '🟡 Warning'", "== '🟡 警告'"),

    # CAGRの変数名（文字列の場合のみ）
    (r"'CAGR':", "'年平均成長率':"),
    (r"nlargest\(10, 'CAGR'\)", "nlargest(10, '年平均成長率')"),
    (r"\['CAGR'\]", "['年平均成長率']"),

    # sort='MAE'
    (r"sort='MAE'", "sort='平均絶対誤差'"),

    # detection_method =
    (r"detection_method'] = 'IQR'", "detection_method'] = '四分位範囲'"),
    (r"detection_method'] = 'Isolation Forest'", "detection_method'] = '孤立森林'"),
    (r"detection_method'] = 'MA Deviation'", "detection_method'] = '移動平均偏差'"),
    (r"detection_method'] = 'YoY Deviation'", "detection_method'] = '前年比偏差'"),
]

def translate_all_english(notebook_path):
    """すべての英語を日本語に翻訳"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    fixed_count = 0

    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if not isinstance(source, list):
            continue

        new_source = []
        modified = False

        for line in source:
            new_line = line

            # 1. 直接的な文字列置換
            for eng, jpn in COMPREHENSIVE_TRANSLATIONS.items():
                if eng in new_line:
                    new_line = new_line.replace(eng, jpn)
                    modified = True

            # 2. パターンベースの置換
            for pattern, replacement in PATTERN_REPLACEMENTS:
                if re.search(pattern, new_line):
                    new_line = re.sub(pattern, replacement, new_line)
                    modified = True

            new_source.append(new_line)
            if new_line != line:
                fixed_count += 1

        if new_source != source:
            cell['source'] = new_source

    if fixed_count > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return fixed_count

    return 0


if __name__ == '__main__':
    notebooks = [
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase1.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase2.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase3.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase4.ipynb'
    ]

    print("\n" + "="*80)
    print("🌐 残存英語の完全翻訳".center(80))
    print("="*80)

    total_fixed = 0

    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")
        count = translate_all_english(nb_path)
        if count > 0:
            print(f"  ✅ {count}箇所を翻訳")
            total_fixed += count
        else:
            print(f"  ℹ️ 翻訳箇所なし")

    print("\n" + "="*80)
    print(f"✅ 合計 {total_fixed}箇所を翻訳完了".center(80))
    print("="*80)
