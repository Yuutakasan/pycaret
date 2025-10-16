#!/usr/bin/env python3
"""統合ダッシュボード.ipynbにインポートを追加"""

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
print("統合ダッシュボード.ipynbにインポートを追加")
print("=" * 70)

try:
    with open('統合ダッシュボード.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 最初のコードセルを完全に置き換え
    replaced = False
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            # 最初のコードセルをインポートセルに置き換え
            cell.source = imports_cell
            print(f"✓ セル {i}: 完全なインポートセルに置き換えました")
            replaced = True
            break

    if not replaced:
        print("⚠️ コードセルが見つかりませんでした")

    # 保存
    with open('統合ダッシュボード.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("\n✅ 修正完了")
    print("\n追加したインポート:")
    print("  ✓ warnings, pandas, numpy")
    print("  ✓ matplotlib, seaborn")
    print("  ✓ datetime, timedelta, Path")
    print("  ✓ 日本語フォント設定")
    print("  ✓ グラフスタイル設定")

    print("\n" + "=" * 70)
    print("統合ダッシュボード.ipynb修正完了")
    print("=" * 70)
    print("\n次のステップ:")
    print("  1. Jupyter Labで統合ダッシュボード.ipynbを開く")
    print("  2. カーネルを再起動")
    print("  3. 全セルを実行")

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
