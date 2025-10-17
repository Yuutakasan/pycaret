#!/usr/bin/env python3
"""Stage5にPyCaretのインポートを追加"""

import nbformat

notebook_file = 'Stage5_PyCaret需要予測.ipynb'

# PyCaretのインポート文
pycaret_imports = """# PyCaretのインポート
from pycaret.regression import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("✅ PyCaret回帰モジュールをインポートしました")"""

print("=" * 70)
print("Stage5にPyCaretのインポートを追加")
print("=" * 70)

try:
    with open(notebook_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 既存のインポートセル（pandas, numpy等がある場所）を探す
    import_cell_index = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'import pandas as pd' in cell.source:
            # 日本語設定の次のインポートセルを探す
            if i > 0 and '日本語設定完了' in nb.cells[i-1].source:
                import_cell_index = i
                break

    if import_cell_index is not None:
        # PyCaretのインポートがあるか確認
        if 'from pycaret.regression import' not in nb.cells[import_cell_index].source:
            # 既存のインポートの後に追加
            nb.cells[import_cell_index].source += '\n\n' + pycaret_imports
            print(f"✓ {notebook_file}: PyCaretのインポートを追加しました")
            print("\n追加したインポート:")
            print("  - from pycaret.regression import *")
            print("  - from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score")
        else:
            print(f"○ {notebook_file}: PyCaretのインポートは既に存在します")
    else:
        print(f"⚠ {notebook_file}: インポートセルが見つかりませんでした")
        print("\n利用可能なコードセル:")
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                preview = cell.source[:50].replace('\n', ' ')
                print(f"  セル {i}: {preview}...")

    # 保存
    with open(notebook_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("\n" + "=" * 70)
    print("✅ 修正完了")
    print("=" * 70)

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
