#!/usr/bin/env python3
"""Stage1-5のノートブックに日本語設定を追加するスクリプト"""

import nbformat

notebooks = [
    'Stage1_現状把握分析.ipynb',
    'Stage2_商品ABC分析.ipynb',
    'Stage3_発注最適化分析.ipynb',
    'Stage4_外部要因分析.ipynb',
    'Stage5_PyCaret需要予測.ipynb'
]

japanese_config = """# 日本語表示設定
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
try:
    import japanize_matplotlib
    print("✅ 日本語表示: 有効 (japanize_matplotlib)")
except ImportError:
    print("⚠️ japanize_matplotlib未インストール - デフォルトフォントで対応")
    import matplotlib.pyplot as plt
    import matplotlib
    # フォールバック設定
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False

# pandas表示設定
import pandas as pd
pd.set_option('display.unicode.east_asian_width', True)

print("✅ 日本語設定完了")"""

print("=" * 70)
print("Stage1-5 ノートブックの日本語設定更新")
print("=" * 70)

for notebook_file in notebooks:
    try:
        # ノートブック読み込み
        with open(notebook_file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 日本語設定セルを探す
        found = False
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and 'japanize_matplotlib' in cell.source:
                # 既存セルを更新
                nb.cells[i].source = japanese_config
                found = True
                print(f"✓ {notebook_file}: 日本語設定を更新 (位置: {i})")
                break

        if not found:
            # 最初のコードセルの前に挿入
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    new_cell = nbformat.v4.new_code_cell(source=japanese_config)
                    nb.cells.insert(i, new_cell)
                    print(f"✓ {notebook_file}: 日本語設定を追加 (位置: {i})")
                    found = True
                    break

        if not found:
            print(f"⚠ {notebook_file}: コードセルが見つかりませんでした")
            continue

        # 保存
        with open(notebook_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

    except Exception as e:
        print(f"✗ {notebook_file}: エラー - {e}")
        import traceback
        traceback.print_exc()

print("=" * 70)
print("✅ すべてのノートブックの処理が完了しました")
print("=" * 70)
