#!/usr/bin/env python3
"""日本語表示のテストスクリプト"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np

print("=" * 70)
print("日本語表示テスト")
print("=" * 70)

# 1. japanize-matplotlib のテスト
print("\n1. japanize-matplotlib のインポート:")
try:
    import japanize_matplotlib
    print("   ✅ japanize-matplotlib が利用可能")
except ImportError as e:
    print(f"   ❌ japanize-matplotlib が見つかりません: {e}")

# 2. 日本語フォントの設定
print("\n2. 日本語フォントの手動設定:")
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'IPA Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
print(f"   設定したフォント: {matplotlib.rcParams['font.family']}")

# 3. テストプロット
print("\n3. テストプロット作成中...")
fig, ax = plt.subplots(figsize=(10, 6))

# テストデータ
categories = ['売上', '利益', '顧客数', '商品数', '店舗数']
values = [100, 80, 120, 90, 110]

ax.bar(categories, values, color='steelblue', edgecolor='black')
ax.set_title('日本語表示テスト - コンビニデータ', fontsize=16, fontweight='bold')
ax.set_xlabel('カテゴリ', fontsize=12)
ax.set_ylabel('値', fontsize=12)
ax.grid(axis='y', alpha=0.3)

# 値をラベル表示
for i, v in enumerate(values):
    ax.text(i, v, f'{v}', ha='center', va='bottom', fontsize=11)

# 保存
output_file = 'output/japanese_display_test.png'
plt.tight_layout()
plt.savefig(output_file, dpi=100, bbox_inches='tight')
print(f"   ✅ プロットを保存: {output_file}")

# 4. Pandasの日本語表示テスト
print("\n4. Pandas DataFrame の日本語表示:")
df = pd.DataFrame({
    '店舗': ['新宿店', '渋谷店', '池袋店'],
    '売上': [150000, 180000, 160000],
    '客数': [250, 300, 270]
})
print(df.to_string(index=False))

print("\n" + "=" * 70)
print("✅ テスト完了")
print("=" * 70)
print("\n次のステップ:")
print("  1. output/japanese_display_test.png を確認")
print("  2. 日本語が正しく表示されていれば成功")
print("  3. 文字化けしている場合は、フォント設定を調整")
