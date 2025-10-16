"""
ノートブック内の英語テキストを検出
"""

import json
import re
from pathlib import Path

def find_english_in_notebook(notebook_path):
    """ノートブック内の英語テキストを検出"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    english_patterns = [
        # グラフ関連の英語
        r"['\"](?!.*[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])[A-Z][a-zA-Z\s]+['\"]",
        # 一般的な英単語
        r"\b(Summary|Growth|Average|Total|Count|Rate|Trend|Forecast|Actual|Target|Daily|Weekly|Monthly|Hourly|Store|Product|Category|Time|Date|Value|Amount|Alert|Level|High|Medium|Low|Critical|Normal|Warning|Revenue|Customer|Item|Sold|KPI|Action)\b",
    ]

    findings = []

    for cell_idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source

        lines = code.split('\n')

        for line_idx, line in enumerate(lines):
            # コメント行はスキップ
            if line.strip().startswith('#'):
                continue

            for pattern in english_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    findings.append({
                        'cell': cell_idx + 1,
                        'line': line_idx + 1,
                        'text': match.group(0),
                        'full_line': line.strip()
                    })

    return findings

if __name__ == '__main__':
    notebooks = [
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase1.ipynb'),
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase2.ipynb'),
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase3.ipynb'),
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase4.ipynb')
    ]

    print("\n" + "="*80)
    print("🔍 英語テキスト検出".center(80))
    print("="*80)

    for nb_path in notebooks:
        if not nb_path.exists():
            continue

        print(f"\n📝 {nb_path.name}")
        findings = find_english_in_notebook(nb_path)

        if findings:
            print(f"  ⚠️ {len(findings)}箇所で英語を検出:")
            for f in findings[:10]:  # 最初の10件を表示
                print(f"    Cell {f['cell']}, Line {f['line']}: {f['text']}")
                print(f"      → {f['full_line'][:80]}")
        else:
            print(f"  ✅ 英語テキストなし")

    print("\n" + "="*80)
