"""
family= と fontproperties= の競合を解決
family="monospace" を削除し、fontproperties=JP_FP のみを使用
"""

import json
import re
from pathlib import Path

def fix_family_conflict(notebook_path):
    """family=パラメータを削除してfontproperties=JP_FPのみ使用"""

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

        for line in source:
            new_line = line

            # 日本語を含み、family= と fontproperties= の両方がある場合
            has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', line))

            if has_japanese and 'fontproperties' in line:
                # family="monospace" または family='monospace' を削除
                if re.search(r'family=["\']monospace["\'],?\s*', line):
                    new_line = re.sub(r'family=["\']monospace["\'],?\s*', '', new_line)
                    fixed_count += 1
                    print(f"  修正: family=\"monospace\"を削除")

                # family="sans-serif" なども削除
                elif re.search(r'family=["\'][^"\']+["\'],?\s*', line):
                    new_line = re.sub(r'family=["\'][^"\']+["\'],?\s*', '', new_line)
                    fixed_count += 1
                    print(f"  修正: familyパラメータを削除")

            new_source.append(new_line)

        if new_source != source:
            cell['source'] = new_source

    if fixed_count > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return fixed_count

    return 0


def detect_all_text_elements(notebook_path):
    """すべてのテキスト要素を検出（日本語を含む）"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    elements = []

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

            # グラフ関連の関数を検出
            text_functions = [
                'suptitle', 'set_title', 'set_xlabel', 'set_ylabel',
                'text', 'xlabel', 'ylabel', 'title', 'annotate'
            ]

            for func in text_functions:
                if func in line:
                    has_fontprop = 'fontproperties' in line or 'prop=' in line
                    elements.append({
                        'cell': cell_idx + 1,
                        'line': line_idx + 1,
                        'function': func,
                        'has_fontprop': has_fontprop,
                        'text': line.strip()[:100]
                    })
                    break

    return elements


if __name__ == '__main__':
    notebooks = [
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase1.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase2.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase3.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase4.ipynb'
    ]

    print("\n" + "="*80)
    print("🔧 family=パラメータ競合の解決".center(80))
    print("="*80)

    total_fixed = 0

    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")
        count = fix_family_conflict(nb_path)
        if count > 0:
            print(f"  ✅ {count}箇所のfamily=を削除")
            total_fixed += count
        else:
            print(f"  ℹ️ 競合なし")

    print("\n" + "="*80)
    print(f"✅ 合計 {total_fixed}箇所の競合を解決".center(80))
    print("="*80)

    # 全テキスト要素の検証
    print("\n" + "="*80)
    print("🔍 全テキスト要素の最終検証".center(80))
    print("="*80)

    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")
        elements = detect_all_text_elements(nb_path)

        missing_font = [e for e in elements if not e['has_fontprop']]

        if missing_font:
            print(f"  ⚠️ {len(missing_font)}箇所でフォント設定なし:")
            for elem in missing_font[:5]:
                print(f"    Cell {elem['cell']}, Line {elem['line']}: {elem['function']}")
                print(f"      → {elem['text']}")
        else:
            print(f"  ✅ すべてのテキスト要素にフォント設定あり ({len(elements)}箇所)")

    print("\n" + "="*80)
