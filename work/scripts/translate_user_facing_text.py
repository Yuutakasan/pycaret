"""
ユーザー表示テキストのみを翻訳
（変数名、関数名、技術用語は除外）
"""

import json
import re
from pathlib import Path

# ユーザー表示テキストの翻訳（print文、グラフタイトル等）
USER_FACING_TRANSLATIONS = [
    # print文内の表示テキスト
    (r'print\(f"   IQR法:', r'print(f"   四分位範囲法:'),
    (r'print\(f"   Isolation Forest:', r'print(f"   孤立森林法:'),
    (r'print\(f"   MA Deviation:', r'print(f"   移動平均偏差:'),
    (r'print\(f"   YoY Deviation:', r'print(f"   前年比偏差:'),

    # アラート表示
    (r'return "🔴 Critical"', r'return "🔴 緊急"'),
    (r'return "🟡 Warning"', r'return "🟡 警告"'),
    (r'return "🟢 Normal"', r'return "🟢 正常"'),

    # グラフ内テキスト（既に修正済みだが念のため）
    (r"'Critical'", r"'緊急'"),
    (r"'Warning'", r"'警告'"),
    (r"'Normal'", r"'正常'"),

    # データフレーム列名（ユーザーに表示される場合）
    (r"anomalies\['detection_method'\] = 'IQR'", r"anomalies['detection_method'] = '四分位範囲'"),
    (r"anomalies\['detection_method'\] = 'Isolation Forest'", r"anomalies['detection_method'] = '孤立森林'"),
    (r"anomalies\['detection_method'\] = 'MA Deviation'", r"anomalies['detection_method'] = '移動平均偏差'"),
    (r"anomalies\['detection_method'\] = 'YoY Deviation'", r"anomalies['detection_method'] = '前年比偏差'"),

    # アラートレベルフィルタ
    (r"\['alert_level'\] == '🔴 Critical'", r"['alert_level'] == '🔴 緊急'"),
    (r"\['alert_level'\] == '🟡 Warning'", r"['alert_level'] == '🟡 警告'"),
    (r"\['alert_level'\] == '🟢 Normal'", r"['alert_level'] == '🟢 正常'"),
]

def translate_user_facing_text(notebook_path):
    """ユーザー表示テキストのみを翻訳"""

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

            # パターンベースの置換
            for pattern, replacement in USER_FACING_TRANSLATIONS:
                if re.search(pattern, new_line):
                    new_line = re.sub(pattern, replacement, new_line)
                    modified = True
                    fixed_count += 1

            new_source.append(new_line)

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
    print("🌐 ユーザー表示テキストの翻訳".center(80))
    print("="*80)

    total_fixed = 0

    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")
        count = translate_user_facing_text(nb_path)
        if count > 0:
            print(f"  ✅ {count}箇所を翻訳")
            total_fixed += count
        else:
            print(f"  ℹ️ 翻訳箇所なし")

    print("\n" + "="*80)
    print(f"✅ 合計 {total_fixed}箇所を翻訳完了".center(80))
    print("="*80)

    print("\n💡 注意: 以下は技術用語として英語のまま保持しています:")
    print("   - 変数名 (IQR, iso_forest, anomalies等)")
    print("   - 関数名 (IsolationForest等)")
    print("   - コメント内の技術説明")
    print("   これらはコードの動作に必要で、ユーザーには表示されません。")
