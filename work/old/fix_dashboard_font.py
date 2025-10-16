#!/usr/bin/env python3
"""統合ダッシュボードのフォント設定を強化"""

import nbformat

notebook_file = '統合ダッシュボード.ipynb'

# 強化されたフォント設定
enhanced_font_setup = """# 日本語表示設定（強化版）
import warnings
warnings.filterwarnings('ignore')

# matplotlibの設定を最初に行う
import matplotlib
import matplotlib.pyplot as plt

# 日本語フォント設定（複数の方法を試行）
font_configured = False

# 方法1: japanize_matplotlibを試す
try:
    import japanize_matplotlib
    print("✅ 日本語表示: japanize_matplotlib で設定完了")
    font_configured = True
except ImportError:
    print("⚠️ japanize_matplotlib未インストール")

# 方法2: 手動でフォントを設定
if not font_configured:
    try:
        # 利用可能なフォントを確認
        import matplotlib.font_manager as fm

        # 日本語フォントを検索
        japanese_fonts = []
        for font in fm.fontManager.ttflist:
            if any(keyword in font.name for keyword in ['IPA', 'Noto', 'Gothic', 'Mincho']):
                japanese_fonts.append(font.name)

        if japanese_fonts:
            # 見つかった日本語フォントを設定
            japanese_fonts = list(set(japanese_fonts))
            matplotlib.rcParams['font.family'] = japanese_fonts[:3] + ['sans-serif']
            print(f"✅ 日本語フォント設定: {japanese_fonts[0]}")
            font_configured = True
        else:
            print("⚠️ 日本語フォントが見つかりません")
    except Exception as e:
        print(f"⚠️ フォント設定エラー: {e}")

# 方法3: デフォルト設定
if not font_configured:
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    print("⚠️ デフォルトフォントを使用（日本語表示に制限あり）")

# その他のmatplotlib設定
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 100

# pandas設定
import pandas as pd
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

print("✅ 日本語設定完了")

# 標準ライブラリのインポート
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# グラフスタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# フォント確認用テスト
fig, ax = plt.subplots(figsize=(6, 2))
ax.text(0.5, 0.5, '日本語表示テスト: 売上・利益・顧客',
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.show()
plt.close()

print("="*80)
print("📊 コンビニエンスストア経営ダッシュボード")
print("="*80)
print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"使用フォント: {matplotlib.rcParams['font.family']}")"""

print("=" * 70)
print("統合ダッシュボードのフォント設定強化")
print("=" * 70)

try:
    with open(notebook_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 最初のコードセルを探して置き換え
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and ('import pandas as pd' in cell.source or '日本語設定' in cell.source):
            cell.source = enhanced_font_setup
            print(f"✓ セル {i}: フォント設定を強化しました")
            break

    # 保存
    with open(notebook_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("\n✅ 修正完了")
    print("\n強化内容:")
    print("  1. 複数の方法でフォント設定を試行")
    print("  2. 利用可能な日本語フォントを自動検出")
    print("  3. フォント確認用のテストプロット")
    print("  4. 使用フォントの表示")

    print("\n" + "=" * 70)
    print("統合ダッシュボードフォント設定強化完了")
    print("=" * 70)

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
