#!/usr/bin/env python3
"""Stage1-5のノートブックに最終的な日本語設定を追加"""

import nbformat

notebooks = [
    'Stage1_現状把握分析.ipynb',
    'Stage2_商品ABC分析.ipynb',
    'Stage3_発注最適化分析.ipynb',
    'Stage4_外部要因分析.ipynb',
    'Stage5_PyCaret需要予測.ipynb'
]

# 最適化された日本語設定
japanese_config = """# 日本語表示設定
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
try:
    import japanize_matplotlib
    print("✅ 日本語表示: japanize_matplotlib で設定完了")
except ImportError:
    # japanize-matplotlibがない場合の代替設定
    import matplotlib.pyplot as plt
    import matplotlib
    # 利用可能な日本語フォントを優先順で設定
    matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'IPAMincho', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("⚠️ japanize_matplotlib未インストール - 代替フォント設定で対応")

# pandas表示設定
import pandas as pd
pd.set_option('display.unicode.east_asian_width', True)

print("✅ 日本語設定完了")"""

print("=" * 70)
print("Stage1-5 ノートブックの日本語設定を最終更新")
print("=" * 70)

for notebook_file in notebooks:
    try:
        with open(notebook_file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 日本語設定セルを探して更新
        updated = False
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and ('japanize_matplotlib' in cell.source or '日本語表示設定' in cell.source):
                nb.cells[i].source = japanese_config
                print(f"✓ {notebook_file}: 日本語設定を最終更新 (位置: {i})")
                updated = True
                break

        if not updated:
            # 見つからない場合は最初のコードセルの前に追加
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    new_cell = nbformat.v4.new_code_cell(source=japanese_config)
                    nb.cells.insert(i, new_cell)
                    print(f"✓ {notebook_file}: 日本語設定を新規追加 (位置: {i})")
                    updated = True
                    break

        if updated:
            with open(notebook_file, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
        else:
            print(f"⚠ {notebook_file}: 更新箇所が見つかりませんでした")

    except Exception as e:
        print(f"✗ {notebook_file}: エラー - {e}")

print("=" * 70)
print("✅ 完了")
print("=" * 70)
