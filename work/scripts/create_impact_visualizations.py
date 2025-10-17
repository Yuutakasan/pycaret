#!/usr/bin/env python3
"""
包括的売上インパクト分析の可視化
格納率80%以上の全特徴量について視覚的に表示
"""

import pandas as pd
from pathlib import Path

print("\n" + "="*80)
print("📊 包括的売上インパクト分析 - テキスト可視化")
print("="*80)

# データ読み込み
impact_df = pd.read_csv('output/comprehensive_sales_impact_analysis.csv', encoding='utf-8-sig')

# インパクト率の絶対値でソート
impact_df['インパクト率_絶対値'] = impact_df['インパクト率'].abs()
impact_df = impact_df.sort_values('インパクト率_絶対値', ascending=False)

print(f"\n✅ 分析完了: {len(impact_df)}個の特徴量")

# ========================================
# 1. Top 20 全体ランキング
# ========================================
print("\n" + "="*80)
print("🏆 売上インパクトランキング Top 20")
print("="*80)
print(f"{'順位':<4} {'特徴量':<35} {'カテゴリ':<12} {'インパクト率':<12} {'絶対値':<15} {'分析タイプ':<20}")
print("-"*120)

for idx, row in enumerate(impact_df.head(20).iterrows(), 1):
    _, data = row
    impact_sign = '+' if data['インパクト率'] > 0 else ''
    print(f"{idx:<4} {data['特徴量']:<35} {data['カテゴリ']:<12} "
          f"{impact_sign}{data['インパクト率']*100:>6.2f}% "
          f"{impact_sign}{data['インパクト絶対値']:>10,.0f}円   "
          f"{data['分析タイプ']:<20}")

# ========================================
# 2. 正のインパクト Top 10
# ========================================
print("\n" + "="*80)
print("✅ 売上増加要因 Top 10")
print("="*80)
positive = impact_df[impact_df['インパクト率'] > 0].head(10)

for idx, row in enumerate(positive.iterrows(), 1):
    _, data = row
    print(f"{idx:2d}. {data['特徴量']:<35} "
          f"+{data['インパクト率']*100:6.2f}% (+{data['インパクト絶対値']:,.0f}円) "
          f"[{data['分析タイプ']}] [{data['カテゴリ']}]")

# ========================================
# 3. 負のインパクト Top 10
# ========================================
print("\n" + "="*80)
print("⚠️ 売上減少要因 Top 10")
print("="*80)
negative = impact_df[impact_df['インパクト率'] < 0].sort_values('インパクト率').head(10)

for idx, row in enumerate(negative.iterrows(), 1):
    _, data = row
    print(f"{idx:2d}. {data['特徴量']:<35} "
          f"{data['インパクト率']*100:+6.2f}% ({data['インパクト絶対値']:+,.0f}円) "
          f"[{data['分析タイプ']}] [{data['カテゴリ']}]")

# ========================================
# 4. カテゴリ別サマリー
# ========================================
print("\n" + "="*80)
print("📊 カテゴリ別インパクトサマリー")
print("="*80)

category_summary = impact_df.groupby('カテゴリ').agg({
    'インパクト率_絶対値': ['mean', 'max', 'count']
}).round(4)
category_summary.columns = ['平均インパクト率', '最大インパクト率', '特徴量数']
category_summary = category_summary.sort_values('最大インパクト率', ascending=False)

print(f"{'カテゴリ':<12} {'平均インパクト':<15} {'最大インパクト':<15} {'特徴量数':<10}")
print("-"*60)
for cat, row in category_summary.iterrows():
    print(f"{cat:<12} {row['平均インパクト率']*100:>10.2f}%    "
          f"{row['最大インパクト率']*100:>10.2f}%    "
          f"{int(row['特徴量数']):>6}個")

# ========================================
# 5. 分析タイプ別分布
# ========================================
print("\n" + "="*80)
print("📊 分析タイプ別分布")
print("="*80)

analysis_type_counts = impact_df['分析タイプ'].value_counts()
total = len(impact_df)

print(f"{'分析タイプ':<30} {'件数':<10} {'割合':<10}")
print("-"*50)
for atype, count in analysis_type_counts.items():
    print(f"{atype:<30} {count:>6}個   {count/total*100:>6.1f}%")

# ========================================
# 6. インパクト率の分布統計
# ========================================
print("\n" + "="*80)
print("📈 インパクト率の統計")
print("="*80)

print(f"\n正のインパクト: {(impact_df['インパクト率'] > 0).sum()}個")
print(f"  平均: +{impact_df[impact_df['インパクト率'] > 0]['インパクト率'].mean()*100:.2f}%")
print(f"  最大: +{impact_df['インパクト率'].max()*100:.2f}%")

print(f"\n負のインパクト: {(impact_df['インパクト率'] < 0).sum()}個")
print(f"  平均: {impact_df[impact_df['インパクト率'] < 0]['インパクト率'].mean()*100:.2f}%")
print(f"  最小: {impact_df['インパクト率'].min()*100:.2f}%")

# ========================================
# 7. 推奨アクション
# ========================================
print("\n" + "="*80)
print("🎯 推奨アクション")
print("="*80)

print("""
【売上最大化施策】

1. 超高インパクト要因の活用（+500%以上）
   ✓ 季節変動指数の監視と活用
   ✓ 売上数量トレンドの予測
   ✓ カテゴリ別在庫最適化

2. 正のインパクト要因の最大化（+20-300%）
   ✓ 気温差拡大時の在庫確保
   ✓ 寒暖変化に応じたプロモーション
   ✓ 連休・曜日別の販売強化

3. 負のインパクト要因の緩和（-70%～-20%）
   ✓ 季節上昇期の特別施策
   ✓ 暖かくなった時の代替商品提案
   ✓ 平日・休日別の価格戦略

4. 予測モデルへの活用
   ✓ インパクト率上位20特徴量を重点使用
   ✓ 時系列・気温・フラグの相互作用項作成
   ✓ カテゴリ別に重要特徴量を選定
   ✓ 季節変動指数の特別扱い（+797%の影響力）

5. ダッシュボード統合
   ✓ リアルタイムインパクト監視
   ✓ カテゴリ別アラート設定
   ✓ 週次・月次レポート自動生成
""")

print("\n" + "="*80)
print("✅ 包括的売上インパクト分析完了")
print("="*80)
print(f"\n📁 生成ファイル:")
print(f"  - output/comprehensive_sales_impact_analysis.csv ({len(impact_df)}特徴量)")
print(f"  - output/sales_impact_top50.csv (Top 50)")
print(f"  - 店舗別包括ダッシュボード_v6.1_提案強化.ipynb (分析セル追加)")
print(f"\n💡 次のステップ:")
print(f"  1. ダッシュボードで最新セルを実行")
print(f"  2. 予測モデルに上位特徴量を組み込む")
print(f"  3. カテゴリ別施策の実施と効果測定")
print()
