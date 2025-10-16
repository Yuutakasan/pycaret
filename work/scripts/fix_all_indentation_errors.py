"""
全ノートブックのインデントエラーを包括的に修正

問題:
1. else:の後にインデントなしのMY_STORE = DEFAULT_STORE
2. 関数定義内のインデント不一致
3. apply()メソッド内のインデント不一致
"""

import json
import re
from pathlib import Path

def fix_all_indentation_issues(notebook_path):
    """全てのインデントエラーを修正"""

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
        i = 0

        while i < len(source):
            line = source[i]

            # パターン1: else:の直後にインデントなしのMY_STORE = DEFAULT_STORE
            if i < len(source) - 1:
                next_line = source[i + 1]

                # else:の後にインデントなしでMY_STORE = DEFAULT_STOREがある
                if line.strip() == 'else:' and next_line.strip() == 'MY_STORE = DEFAULT_STORE':
                    # else:の後に適切なpass文を追加
                    indent = len(line) - len(line.lstrip())
                    new_source.append(line)
                    new_source.append(' ' * (indent + 4) + 'pass\n')
                    print(f"  ✅ 修正: else: pass 追加")
                    i += 2  # MY_STORE行をスキップ
                    fixed_count += 1
                    continue

            # パターン2: 関数定義内の不正なインデント (MY_STORE = DEFAULT_STORE)
            # カテゴリ分類関数内の問題を修正
            if 'MY_STORE = DEFAULT_STORE' in line:
                # 前の行を確認
                if i > 0:
                    prev_line = source[i - 1]

                    # return文の後、またはelse:の後のMY_STORE = DEFAULT_STOREは削除
                    if 'return' in prev_line or 'else:' in prev_line:
                        print(f"  ✅ 削除: 不正なMY_STORE = DEFAULT_STORE (行{i+1})")
                        fixed_count += 1
                        i += 1
                        continue

            # パターン3: apply()内の関数定義のインデント問題
            # categorize_feature, determine_alert_level等の関数
            if i > 0 and 'def ' in line:
                prev_line = source[i - 1]

                # apply(lambda)やapply(関数)の直後の関数定義は削除
                if 'apply(' in prev_line:
                    # この関数定義は誤挿入の可能性が高い
                    # 関数の終わりまでスキップ
                    indent_level = len(line) - len(line.lstrip())
                    i += 1

                    # 関数の終わりまでスキップ
                    while i < len(source):
                        current_line = source[i]
                        current_indent = len(current_line) - len(current_line.lstrip())

                        # インデントが戻ったら関数終了
                        if current_line.strip() and current_indent <= indent_level:
                            break
                        i += 1

                    print(f"  ✅ 削除: apply()直後の不正な関数定義")
                    fixed_count += 1
                    continue

            new_source.append(line)
            i += 1

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
    print("🔧 全インデントエラー包括修正".center(80))
    print("="*80)

    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")
        count = fix_all_indentation_issues(nb_path)
        if count > 0:
            print(f"  ✅ {count}箇所修正")
        else:
            print(f"  ℹ️ 修正箇所なし")

    print("\n" + "="*80)
    print("✅ 完了".center(80))
    print("="*80)
