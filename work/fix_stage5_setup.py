#!/usr/bin/env python3
"""Stage5のPyCaretセットアップコードを修正"""

import nbformat
import re

notebook_file = 'Stage5_PyCaret需要予測.ipynb'

print("=" * 70)
print("Stage5のPyCaretセットアップコードを修正")
print("=" * 70)

try:
    with open(notebook_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    modified = False

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            # PyCaretセットアップのセルを探す
            if 'reg = setup(' in cell.source and 'silent=True' in cell.source:
                # 古いsetup呼び出しを新しい形式に置き換え
                old_source = cell.source

                # PyCaret 3.x用の正しいパラメータに変更
                new_source = old_source.replace('silent=True,', '')
                new_source = new_source.replace(', silent=True', '')

                # verboseとhtmlは残す
                cell.source = new_source

                print(f"✓ セル {i}: setup()のパラメータを修正")
                print(f"  除外: silent=True (PyCaret 3.xでは非対応)")
                modified = True

    if modified:
        # 保存
        with open(notebook_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("\n✅ 修正完了")
    else:
        print("○ 修正不要（既に正しい形式）")

    print("\n" + "=" * 70)
    print("PyCaret 3.x対応完了")
    print("=" * 70)

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
