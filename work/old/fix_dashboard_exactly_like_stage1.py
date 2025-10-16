#!/usr/bin/env python3
"""統合ダッシュボードにStage1と全く同じフォント設定を適用"""

import nbformat

# Stage1と全く同じフォント設定
stage1_font_setup = """# 日本語表示設定
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

# Stage1と同じインポート
stage1_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path"""

# グラフスタイル設定
graph_style = """# グラフスタイル設定
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("📊 コンビニエンスストア経営ダッシュボード")
print("="*80)
print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")"""

# 完全なセットアップ
complete_setup = f"""{stage1_font_setup}

{stage1_imports}

{graph_style}"""

print("=" * 70)
print("統合ダッシュボードにStage1と同じ設定を適用")
print("=" * 70)

try:
    with open('統合ダッシュボード.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 最初のコードセルを完全に置き換え
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            # 既存のセットアップセルを置き換え
            if 'import' in cell.source or '日本語' in cell.source:
                cell.source = complete_setup
                print(f"✓ セル {i}: Stage1と同じ設定に完全置換")
                break

    # 保存
    with open('統合ダッシュボード.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("\n✅ 修正完了")
    print("\n適用した設定:")
    print("  1. japanize_matplotlibの利用")
    print("  2. 代替フォント: Noto Sans CJK JP, IPAGothic, IPAMincho")
    print("  3. pandas表示設定")
    print("  4. matplotlib/seabornスタイル設定")
    print("\n※ Stage1と完全に同じ設定です")

    print("\n" + "=" * 70)
    print("統合ダッシュボード修正完了")
    print("=" * 70)

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
