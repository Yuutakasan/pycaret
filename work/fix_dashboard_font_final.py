#!/usr/bin/env python3
"""統合ダッシュボードのフォント問題を完全解決"""

import nbformat

# 強化されたフォント設定（フォントキャッシュ再構築含む）
final_font_setup = """# 日本語表示設定（完全版）
import warnings
warnings.filterwarnings('ignore')

# matplotlibのフォントキャッシュを強制的に再構築
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# フォントキャッシュを再構築
print("📝 フォントキャッシュを再構築中...")
fm._load_fontmanager(try_read_cache=False)

# 日本語フォント設定（複数の方法を試行）
font_set = False

# 方法1: japanize_matplotlibを使用
try:
    import japanize_matplotlib
    print("✅ 日本語表示: japanize_matplotlib")
    font_set = True
except ImportError:
    pass

# 方法2: システムの日本語フォントを手動設定
if not font_set:
    # 利用可能な日本語フォントを検索
    japanese_fonts = []
    for font in fm.fontManager.ttflist:
        if 'Gothic' in font.name or 'Mincho' in font.name or 'IPA' in font.name or 'Noto' in font.name:
            japanese_fonts.append(font.name)

    if japanese_fonts:
        # 重複を削除
        japanese_fonts = sorted(set(japanese_fonts))
        # フォントファミリーを設定
        matplotlib.rcParams['font.family'] = japanese_fonts[:5] + ['sans-serif']
        print(f"✅ 日本語フォント設定: {japanese_fonts[0]}")
        print(f"   利用可能フォント: {len(japanese_fonts)}個")
        font_set = True

# 方法3: フォールバック設定
if not font_set:
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    print("⚠️ デフォルトフォント使用")

# その他の重要な設定
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 100

# 現在の設定を表示
print(f"\\n現在のフォント設定: {matplotlib.rcParams['font.family'][:3]}")

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

# フォントテスト（最初のセルで確認）
print("\\n📊 フォントテスト:")
fig, ax = plt.subplots(figsize=(8, 2))
test_text = '日本語表示テスト: 売上・利益・顧客数・店舗'
ax.text(0.5, 0.5, test_text, ha='center', va='center', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.show()
print(f"表示テキスト: {test_text}")

print("="*80)
print("📊 コンビニエンスストア経営ダッシュボード")
print("="*80)
print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")"""

print("=" * 70)
print("統合ダッシュボードのフォント問題を完全解決")
print("=" * 70)

try:
    with open('統合ダッシュボード.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 最初のコードセルを完全に置き換え
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and i <= 5:  # 最初の数セルを確認
            if 'import' in cell.source or '日本語' in cell.source:
                cell.source = final_font_setup
                print(f"✓ セル {i}: フォント設定を完全版に置き換えました")
                break

    # 保存
    with open('統合ダッシュボード.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("\n✅ 修正完了")
    print("\n改善内容:")
    print("  1. フォントキャッシュの強制再構築")
    print("  2. 複数の方法でフォント設定を試行")
    print("  3. 利用可能な日本語フォントを自動検出")
    print("  4. フォント設定の確認表示")
    print("  5. フォントテストプロットの自動表示")

    print("\n" + "=" * 70)
    print("統合ダッシュボード完全修正完了")
    print("=" * 70)
    print("\n次のステップ:")
    print("  1. Jupyter Labでノートブックを開く")
    print("  2. カーネルを再起動（重要！）")
    print("  3. 最初のセルを実行してフォントテストを確認")
    print("  4. 日本語が正しく表示されることを確認")

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
