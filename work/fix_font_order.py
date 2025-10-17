#!/usr/bin/env python3
"""フォント設定の順序を正しく修正"""

import nbformat

# 正しい順序でのフォント設定
correct_font_setup = """# 日本語フォント設定（正しい順序）
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from datetime import datetime, timedelta
from pathlib import Path

# ========================================
# ステップ1: matplotlibの基本設定（最初）
# ========================================
matplotlib.rcParams['axes.unicode_minus'] = False

# ========================================
# ステップ2: 日本語フォントの検出と設定
# ========================================
chosen_font = None

# japanize_matplotlibを試す
try:
    import japanize_matplotlib
    chosen_font = "IPAexGothic"
    print("✅ japanize_matplotlib 使用")
except ImportError:
    # 手動でフォントを検索
    candidates = ["IPAexGothic", "IPAPGothic", "Noto Sans CJK JP", "Noto Sans JP",
                  "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Meiryo", "MS Gothic"]
    avail = {f.name: f.fname for f in font_manager.fontManager.ttflist}

    for font_name in candidates:
        if font_name in avail or any(font_name.lower() in nm.lower() for nm in avail):
            chosen_font = font_name
            break

    if chosen_font:
        print(f"✅ 日本語フォント検出: {chosen_font}")
    else:
        print("⚠️ 日本語フォント未検出 - DejaVu Sans使用")

# フォントファミリーを設定
if chosen_font:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [chosen_font, 'DejaVu Sans']
else:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

print(f"📝 使用フォント: {matplotlib.rcParams['font.sans-serif'][0]}")

# ========================================
# ステップ3: seabornのインポートと設定
# ========================================
import seaborn as sns

# seabornのスタイル設定（フォントを維持）
sns.set_style("darkgrid")
sns.set_palette("husl")

# seaborn実行後にフォントを再設定（重要！）
if chosen_font:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [chosen_font, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

# ========================================
# ステップ4: pandas設定
# ========================================
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# ========================================
# ステップ5: matplotlibスタイル（最後）
# ========================================
plt.style.use('seaborn-v0_8-darkgrid')

# スタイル適用後に再度フォントを設定（最重要！）
if chosen_font:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [chosen_font, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

print("✅ 日本語設定完了")
print("="*80)
print("📊 コンビニエンスストア経営ダッシュボード")
print("="*80)
print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# フォントテスト
fig, ax = plt.subplots(figsize=(8, 2))
test_text = '日本語表示テスト: 売上・利益・顧客数・店舗'
ax.text(0.5, 0.5, test_text, ha='center', va='center', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.show()
print(f"テスト文字列: {test_text}")
"""

print("=" * 70)
print("フォント設定の順序を修正")
print("=" * 70)

try:
    with open('統合ダッシュボード.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 最初のコードセルを置き換え
    replaced = False
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and ('import' in cell.source or '日本語' in cell.source):
            cell.source = correct_font_setup
            print(f"✓ セル {i}: 正しい順序のフォント設定に置き換えました")
            replaced = True
            break

    if not replaced:
        print("⚠️ コードセルが見つかりませんでした")

    # 保存
    with open('統合ダッシュボード.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("\n✅ 修正完了")
    print("\n重要な変更点:")
    print("  1. matplotlib基本設定を最初に実行")
    print("  2. 日本語フォントを検出して設定")
    print("  3. seabornをインポート")
    print("  4. seaborn後にフォントを再設定")
    print("  5. plt.style.use()を実行")
    print("  6. スタイル適用後に再度フォントを設定（最重要）")
    print("  7. フォントテストを表示")

    print("\n" + "=" * 70)
    print("統合ダッシュボード.ipynb修正完了")
    print("=" * 70)
    print("\n💡 この順序が重要:")
    print("   matplotlib設定 → フォント設定 → seaborn → フォント再設定 → style.use → フォント再設定")

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
