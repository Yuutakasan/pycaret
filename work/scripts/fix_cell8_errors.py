"""
Cell 8のelse文の中のMY_STORE = DEFAULT_STORE誤配置を修正
"""

import json
from pathlib import Path

def fix_cell8_indentation(notebook_path):
    """Cell 8のelse文内のインデントエラーを修正"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    fixed = False

    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if not isinstance(source, list):
            continue

        # validate_data_column関数を含むセルを検索
        if not any('validate_data_column' in line for line in source):
            continue

        new_source = []
        i = 0
        while i < len(source):
            line = source[i]

            # else:の直後にインデントなしのMY_STORE = DEFAULT_STOREがある場合
            if i < len(source) - 1:
                next_line = source[i + 1]

                if 'else:' in line and 'MY_STORE = DEFAULT_STORE' in next_line:
                    # else:ブロックにprint文を追加
                    new_source.append(line)
                    new_source.append('        print(f"❌ 必須カラム \'{col}\' - 不足")\n')
                    i += 2  # MY_STORE行をスキップ
                    fixed = True
                    print(f"  ✅ 修正: else: print(...)に変更")
                    continue

            new_source.append(line)
            i += 1

        if new_source != source:
            cell['source'] = new_source

    if fixed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return True
    return False


if __name__ == '__main__':
    notebooks = [
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase1.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase2.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase3.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase4.ipynb'
    ]

    print("\n" + "="*80)
    print("🔧 Cell 8 else文エラー修正".center(80))
    print("="*80)

    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")
        if fix_cell8_indentation(nb_path):
            print(f"  ✅ 修正完了")
        else:
            print(f"  ℹ️ 修正箇所なし")

    print("\n" + "="*80)
    print("✅ 完了".center(80))
    print("="*80)
