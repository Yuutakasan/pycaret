"""
widgets.Dropdownの途中に誤挿入されたtry-exceptブロックを削除
"""

import json
from pathlib import Path

def fix_dropdown_syntax(notebook_path):
    """widgets.Dropdownの構文エラーを修正"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    fixed_count = 0

    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if not isinstance(source, list):
            continue

        # widgets.Dropdownを含むセルを探す
        if not any('widgets.Dropdown' in line for line in source):
            continue

        new_source = []
        skip_until = -1

        for i, line in enumerate(source):
            # スキップ中
            if i < skip_until:
                continue

            # try: MY_STORE の開始を検出
            if i < len(source) - 3:
                if ('try:' in line and
                    'MY_STORE' in source[i+1] and
                    'except' in source[i+2] and
                    'MY_STORE = DEFAULT_STORE' in source[i+3]):

                    # この4行をスキップ
                    print(f"  ✅ 削除: try-except ブロック (行{i+1}-{i+4})")
                    fixed_count += 1
                    skip_until = i + 4
                    continue

            new_source.append(line)

        if new_source != source:
            cell['source'] = new_source

    if fixed_count > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return fixed_count
    return 0


if __name__ == '__main__':
    notebooks = [
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase2.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase3.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase4.ipynb'
    ]

    print("\n" + "="*80)
    print("🔧 widgets.Dropdown構文エラー修正".center(80))
    print("="*80)

    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")
        count = fix_dropdown_syntax(nb_path)
        if count > 0:
            print(f"  ✅ {count}箇所修正")
        else:
            print(f"  ℹ️ 修正箇所なし")

    print("\n" + "="*80)
    print("✅ 完了".center(80))
    print("="*80)
