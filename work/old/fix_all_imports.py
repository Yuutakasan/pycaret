#!/usr/bin/env python3
"""Stage1-5のすべてのノートブックに必要なインポートを追加"""

import nbformat

# 各ノートブックに必要な標準インポート
standard_imports = [
    'import pandas as pd',
    'import numpy as np',
    'import matplotlib.pyplot as plt',
    'import seaborn as sns',
    'from datetime import datetime, timedelta',
    'from pathlib import Path'
]

# ノートブック固有の追加インポート
notebook_specific = {
    'Stage1_現状把握分析.ipynb': [],
    'Stage2_商品ABC分析.ipynb': [],
    'Stage3_発注最適化分析.ipynb': ['from scipy import stats'],
    'Stage4_外部要因分析.ipynb': [],
    'Stage5_PyCaret需要予測.ipynb': []
}

notebooks = [
    'Stage1_現状把握分析.ipynb',
    'Stage2_商品ABC分析.ipynb',
    'Stage3_発注最適化分析.ipynb',
    'Stage4_外部要因分析.ipynb',
    'Stage5_PyCaret需要予測.ipynb'
]

print("=" * 70)
print("Stage1-5 すべてのノートブックのインポートを統一")
print("=" * 70)

for notebook_file in notebooks:
    try:
        with open(notebook_file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 必要なインポートリスト
        required_imports = standard_imports + notebook_specific.get(notebook_file, [])

        # 日本語設定セルを探す
        japanese_cell_index = None
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and '日本語設定完了' in cell.source:
                japanese_cell_index = i
                break

        if japanese_cell_index is not None:
            # 日本語設定セルの次のセルをチェック
            next_index = japanese_cell_index + 1

            if next_index < len(nb.cells) and nb.cells[next_index].cell_type == 'code':
                # 既存のインポートセルを更新
                existing_source = nb.cells[next_index].source

                # 不足しているインポートを確認
                missing_imports = []
                for imp in required_imports:
                    # より厳密なチェック（'import pandas' と 'import pandas as pd' を区別）
                    if imp not in existing_source:
                        missing_imports.append(imp)

                if missing_imports:
                    # 既存のコードを保持しつつ、不足分を先頭に追加
                    nb.cells[next_index].source = '\n'.join(missing_imports) + '\n\n' + existing_source
                    print(f"✓ {notebook_file}: {len(missing_imports)}個のインポートを追加")
                else:
                    print(f"○ {notebook_file}: すべてのインポートが存在")
            else:
                # インポートセルが存在しない場合、新規作成
                new_cell = nbformat.v4.new_code_cell(source='\n'.join(required_imports))
                nb.cells.insert(next_index, new_cell)
                print(f"✓ {notebook_file}: インポートセルを新規作成（位置: {next_index}）")
        else:
            print(f"⚠ {notebook_file}: 日本語設定セルが見つかりません")
            continue

        # 保存
        with open(notebook_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

    except Exception as e:
        print(f"✗ {notebook_file}: エラー - {e}")
        import traceback
        traceback.print_exc()

print("=" * 70)
print("✅ すべてのノートブックのインポート修正完了")
print("=" * 70)
print("\n追加したインポート:")
for imp in standard_imports:
    print(f"  - {imp}")
print("\nStage3のみ追加:")
print(f"  - from scipy import stats")
