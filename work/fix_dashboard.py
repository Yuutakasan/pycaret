#!/usr/bin/env python3
"""統合ダッシュボード.ipynbに日本語設定を追加"""

import nbformat

notebook_file = '統合ダッシュボード.ipynb'

# 完全な日本語設定とインポート
complete_setup = """# 日本語表示設定
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

# 表示設定
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("📊 コンビニエンスストア経営ダッシュボード")
print("="*80)
print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")"""

print("=" * 70)
print("統合ダッシュボードの日本語設定追加")
print("=" * 70)

try:
    with open(notebook_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 最初のコードセルを探して置き換え
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            # 既存のインポートセルを完全に置き換え
            if 'import pandas as pd' in cell.source:
                cell.source = complete_setup
                print(f"✓ セル {i}: 日本語設定とインポートを更新")
                break
    else:
        # 見つからない場合、環境準備セクションの後に追加
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'markdown' and '環境準備' in cell.source:
                new_cell = nbformat.v4.new_code_cell(source=complete_setup)
                nb.cells.insert(i + 1, new_cell)
                print(f"✓ セル {i+1}: 日本語設定とインポートを新規追加")
                break

    # 保存
    with open(notebook_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("\n✅ 修正完了")
    print("\n追加内容:")
    print("  - 日本語フォント設定（japanize_matplotlib + 代替設定）")
    print("  - 必要なライブラリのインポート")
    print("  - pandas表示設定")
    print("  - matplotlib/seaborn設定")

    print("\n" + "=" * 70)
    print("統合ダッシュボード日本語対応完了")
    print("=" * 70)

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
