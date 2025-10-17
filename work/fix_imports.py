#!/usr/bin/env python3
"""Stage3-5のノートブックに不足しているインポートを追加"""

import nbformat

# 各ノートブックに追加すべきインポート
notebook_imports = {
    'Stage3_発注最適化分析.ipynb': [
        'import numpy as np',
        'import matplotlib.pyplot as plt',
        'import seaborn as sns',
        'from scipy import stats',
        'from pathlib import Path'
    ],
    'Stage4_外部要因分析.ipynb': [
        'import numpy as np',
        'import matplotlib.pyplot as plt',
        'import seaborn as sns',
        'from pathlib import Path'
    ],
    'Stage5_PyCaret需要予測.ipynb': [
        'import numpy as np',
        'import matplotlib.pyplot as plt',
        'import seaborn as sns',
        'from pathlib import Path'
    ]
}

print("=" * 70)
print("Stage3-5 ノートブックのインポート修正")
print("=" * 70)

for notebook_file, required_imports in notebook_imports.items():
    try:
        with open(notebook_file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 日本語設定セルの次のセルを探す（通常はインポートセル）
        import_cell_index = None
        for i, cell in enumerate(nb.cells):
            # "環境準備"や"データ読み込み"の前のコードセルを探す
            if cell.cell_type == 'code' and i > 0:
                # 日本語設定セルの次
                if '日本語設定完了' in nb.cells[i-1].source:
                    import_cell_index = i
                    break

        if import_cell_index is not None:
            # 既存のインポートを確認
            existing_source = nb.cells[import_cell_index].source

            # 必要なインポートを追加
            new_imports = []
            for imp in required_imports:
                if imp not in existing_source:
                    new_imports.append(imp)

            if new_imports:
                # 既存のコードの前に追加
                nb.cells[import_cell_index].source = '\n'.join(new_imports) + '\n\n' + existing_source
                print(f"✓ {notebook_file}: {len(new_imports)}個のインポートを追加")
                for imp in new_imports:
                    print(f"    - {imp}")
            else:
                print(f"○ {notebook_file}: 必要なインポートは既に存在")
        else:
            # インポートセルが見つからない場合、日本語設定の後に新規作成
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code' and '日本語設定完了' in cell.source:
                    new_cell = nbformat.v4.new_code_cell(source='\n'.join(required_imports))
                    nb.cells.insert(i + 1, new_cell)
                    print(f"✓ {notebook_file}: インポートセルを新規作成（位置: {i+1}）")
                    break

        # 保存
        with open(notebook_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

    except Exception as e:
        print(f"✗ {notebook_file}: エラー - {e}")
        import traceback
        traceback.print_exc()

print("=" * 70)
print("✅ 修正完了")
print("=" * 70)
print("\n次のステップ:")
print("  1. Jupyter Labでノートブックを開く")
print("  2. カーネルを再起動（Kernel → Restart Kernel）")
print("  3. セルを実行")
