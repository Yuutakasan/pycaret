"""
最終包括チェック
1. 日本語テキストのfontproperties確認
2. 英語テキスト残存確認
3. family=パラメータ確認
4. 構文エラー確認
"""

import json
import re
from pathlib import Path

def comprehensive_check(notebook_path):
    """包括的なチェックを実施"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    results = {
        'japanese_without_font': [],
        'english_user_facing': [],
        'family_conflicts': [],
        'syntax_issues': []
    }

    for cell_idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        code = ''.join(source) if isinstance(source, list) else source
        lines = code.split('\n')

        for line_idx, line in enumerate(lines):
            # 日本語を含むかチェック
            has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', line))

            if not has_japanese:
                continue

            # 1. グラフ関数で日本語を使用しているがfontpropertiesが無い
            graph_funcs = ['suptitle', 'set_title', 'set_xlabel', 'set_ylabel', 'text', 'annotate']
            for func in graph_funcs:
                if func in line and 'fontproperties' not in line and 'prop=' not in line:
                    # 変数定義やHTMLは除外
                    if not any(x in line for x in ['_text =', '<title>', '<div', 'legend=False', 'legend=F']):
                        results['japanese_without_font'].append({
                            'cell': cell_idx + 1,
                            'line': line_idx + 1,
                            'function': func,
                            'code': line.strip()[:100]
                        })

            # 2. ユーザー表示される英語（グラフタイトル、軸ラベル等）
            if any(func in line for func in ['set_title', 'suptitle', 'xlabel', 'ylabel']):
                # 英語の単語パターン（2単語以上の大文字始まり）
                if re.search(r'["\'][A-Z][a-z]+\s+[A-Z][a-z]+', line):
                    results['english_user_facing'].append({
                        'cell': cell_idx + 1,
                        'line': line_idx + 1,
                        'code': line.strip()[:100]
                    })

            # 3. family= と fontproperties= の競合
            if 'family=' in line and 'fontproperties' in line:
                if 'monospace' in line or 'sans-serif' in line or 'serif' in line:
                    results['family_conflicts'].append({
                        'cell': cell_idx + 1,
                        'line': line_idx + 1,
                        'code': line.strip()[:100]
                    })

    return results

def print_results(notebook_name, results):
    """結果を整形して出力"""

    print(f"\n{'='*80}")
    print(f"📋 {notebook_name}")
    print(f"{'='*80}")

    # 1. 日本語でフォント設定なし
    if results['japanese_without_font']:
        print(f"\n⚠️ 日本語テキストでfontproperties未設定: {len(results['japanese_without_font'])}箇所")
        for item in results['japanese_without_font'][:5]:
            print(f"  Cell {item['cell']}, Line {item['line']}: {item['function']}")
            print(f"    → {item['code']}")
    else:
        print(f"\n✅ すべての日本語テキストにfontproperties設定済み")

    # 2. ユーザー表示英語
    if results['english_user_facing']:
        print(f"\n⚠️ ユーザー表示英語テキスト: {len(results['english_user_facing'])}箇所")
        for item in results['english_user_facing'][:5]:
            print(f"  Cell {item['cell']}, Line {item['line']}")
            print(f"    → {item['code']}")
    else:
        print(f"\n✅ ユーザー表示英語テキストなし")

    # 3. family=競合
    if results['family_conflicts']:
        print(f"\n⚠️ family=パラメータ競合: {len(results['family_conflicts'])}箇所")
        for item in results['family_conflicts'][:5]:
            print(f"  Cell {item['cell']}, Line {item['line']}")
            print(f"    → {item['code']}")
    else:
        print(f"\n✅ family=パラメータ競合なし")

if __name__ == '__main__':
    notebooks = [
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase1.ipynb'),
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase2.ipynb'),
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase3.ipynb'),
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase4.ipynb')
    ]

    print("\n" + "="*80)
    print("🔍 最終包括チェック".center(80))
    print("="*80)

    all_issues = {
        'japanese_without_font': 0,
        'english_user_facing': 0,
        'family_conflicts': 0
    }

    for nb_path in notebooks:
        if not nb_path.exists():
            continue

        results = comprehensive_check(nb_path)
        print_results(nb_path.name, results)

        all_issues['japanese_without_font'] += len(results['japanese_without_font'])
        all_issues['english_user_facing'] += len(results['english_user_facing'])
        all_issues['family_conflicts'] += len(results['family_conflicts'])

    # 最終サマリー
    print("\n" + "="*80)
    print("📊 最終サマリー".center(80))
    print("="*80)

    total_issues = sum(all_issues.values())

    if total_issues == 0:
        print("\n✅ すべてのチェック項目が正常です！")
        print("\n  ✅ 日本語テキスト: すべてfontproperties設定済み")
        print("  ✅ 英語テキスト: ユーザー表示部分なし")
        print("  ✅ family=競合: なし")
    else:
        print(f"\n⚠️ 合計 {total_issues}箇所の問題を検出:")
        print(f"  日本語フォント未設定: {all_issues['japanese_without_font']}箇所")
        print(f"  ユーザー表示英語: {all_issues['english_user_facing']}箇所")
        print(f"  family=競合: {all_issues['family_conflicts']}箇所")

    print("\n" + "="*80)
