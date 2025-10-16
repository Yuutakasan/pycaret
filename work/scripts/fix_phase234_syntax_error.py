"""
Phase 2, 3, 4のノートブックで発生した構文エラーを修正

エラー: try:の前にMY_STORE = DEFAULT_STOREが挿入されている
修正: この誤挿入行を削除
"""

import json
import re
from pathlib import Path

def fix_syntax_errors_in_notebooks():
    """Phase 2, 3, 4のノートブックの構文エラーを修正"""

    notebooks = [
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase2.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase3.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase4.ipynb'
    ]

    for notebook_path in notebooks:
        path = Path(notebook_path)
        if not path.exists():
            print(f"⚠️ ファイルなし: {path.name}")
            continue

        print(f"\n📝 修正中: {path.name}")
        print("="*80)

        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        fixed_count = 0

        for cell in nb['cells']:
            if cell.get('cell_type') != 'code':
                continue

            source = cell.get('source', [])
            if not source:
                continue

            new_source = []
            i = 0
            while i < len(source):
                line = source[i]

                # パターン1: else:の直後にMY_STORE = DEFAULT_STORE + try:
                # パターン2: except:の直後にMY_STORE = DEFAULT_STORE + try:
                if i < len(source) - 1:
                    next_line = source[i + 1]

                    # MY_STORE = DEFAULT_STORE の後にtry:が続く場合、MY_STORE行を削除
                    if 'MY_STORE = DEFAULT_STORE' in line and i + 1 < len(source):
                        if 'try:' in next_line or 'else:' in next_line:
                            print(f"  ✅ 削除: MY_STORE = DEFAULT_STORE (行{i+1})")
                            fixed_count += 1
                            i += 1
                            continue

                new_source.append(line)
                i += 1

            if new_source != source:
                cell['source'] = new_source

        # 保存
        if fixed_count > 0:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, ensure_ascii=False, indent=1)
            print(f"  ✅ {fixed_count}箇所を修正")
        else:
            print(f"  ℹ️ 修正箇所なし")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔧 Phase 2, 3, 4構文エラー修正スクリプト".center(80))
    print("="*80)

    fix_syntax_errors_in_notebooks()

    print("\n" + "="*80)
    print("✅ 修正完了".center(80))
    print("="*80)
