#!/usr/bin/env python3
"""統合ダッシュボード_fixed_final.ipynbにインポートを追加"""

import nbformat

# 完全なインポートセル
imports_cell = """# 日本語表示設定
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

print("✅ 日本語設定完了")

# 標準ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# グラフスタイル設定
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("📊 コンビニエンスストア経営ダッシュボード")
print("="*80)
print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
"""

print("=" * 70)
print("統合ダッシュボード_fixed_final.ipynbにインポートを追加")
print("=" * 70)

try:
    with open('統合ダッシュボード_fixed_final.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # セル1（環境準備）の次にインポートセルを追加
    # まず既存のインポートセルを削除
    cells_to_keep = []
    for i, cell in enumerate(nb.cells):
        # 日本語フォント設定セルは削除
        if cell.cell_type == 'code' and '日本語フォント最終固定' in cell.source:
            print(f"✓ セル {i}: 旧フォント設定セルを削除")
            continue
        cells_to_keep.append(cell)

    nb.cells = cells_to_keep

    # 環境準備セクションの後に新しいインポートセルを挿入
    inserted = False
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '環境準備' in cell.source:
            # 次の位置に挿入
            new_cell = nbformat.v4.new_code_cell(source=imports_cell)
            nb.cells.insert(i + 1, new_cell)
            print(f"✓ セル {i+1}: 完全なインポートセルを挿入")
            inserted = True
            break

    if not inserted:
        # 先頭に挿入
        new_cell = nbformat.v4.new_code_cell(source=imports_cell)
        nb.cells.insert(0, new_cell)
        print("✓ セル 0: 完全なインポートセルを先頭に挿入")

    # 保存
    with open('統合ダッシュボード_fixed_final.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("\n✅ 修正完了")
    print("\n追加したインポート:")
    print("  - warnings, pandas, numpy")
    print("  - matplotlib, seaborn")
    print("  - datetime, timedelta, Path")
    print("  - 日本語フォント設定")
    print("  - グラフスタイル設定")

    print("\n" + "=" * 70)
    print("統合ダッシュボード_fixed_final.ipynb修正完了")
    print("=" * 70)

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
