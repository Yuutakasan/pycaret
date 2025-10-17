#!/usr/bin/env python3
"""
店舗別包括ダッシュボードに包括的売上インパクト分析セルを追加
"""

import json
from pathlib import Path

# 包括的売上インパクト分析セル
impact_cell_code = """# ========================================
# 📊 包括的売上インパクト分析（全特徴量）
# ========================================

print('\\n' + '='*80)
print('📊 包括的売上インパクト分析（格納率80%以上の全特徴量）')
print('='*80)

# 売上列の確認
sales_col = '売上金額' if '売上金額' in df.columns else '売上数量'
print(f'\\n📊 売上指標: {sales_col}')

# ========================================
# データ品質分析
# ========================================

# 各列の格納率計算
completeness = {}
for col in df.columns:
    non_null_count = df[col].notna().sum()
    completeness[col] = non_null_count / len(df)

# 格納率80%以上のカラム抽出
high_quality_cols = [col for col, rate in completeness.items() if rate >= 0.8]

print(f'✅ 格納率80%以上のカラム: {len(high_quality_cols)}個')

# 分析対象カラム（除外列を除く）
exclude_cols = ['店舗', '商品名', '日付', sales_col, 'date', 'store_id', 'sku_id',
               'category_l', 'category_m', 'category_s',
               'フェイスくくり大分類', 'フェイスくくり中分類', 'フェイスくくり小分類']

analysis_cols = [col for col in high_quality_cols if col not in exclude_cols]
print(f'   分析対象カラム: {len(analysis_cols)}個')

# ========================================
# 売上インパクト計算
# ========================================

import warnings
warnings.filterwarnings('ignore')

impact_results = []

for col in analysis_cols:
    try:
        col_data = df[col].dropna()

        if len(col_data) < 100:
            continue

        # 数値型の場合
        if pd.api.types.is_numeric_dtype(col_data):
            unique_vals = col_data.unique()

            # バイナリフラグ（0/1）
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                flag_on = df[df[col] == 1][sales_col].mean()
                flag_off = df[df[col] == 0][sales_col].mean()

                if flag_off > 0:
                    impact = (flag_on - flag_off) / flag_off
                    impact_abs = flag_on - flag_off

                    impact_results.append({
                        '特徴量': col,
                        '分析タイプ': 'バイナリフラグ',
                        '売上_ON': flag_on,
                        '売上_OFF': flag_off,
                        'インパクト率': impact,
                        'インパクト絶対値': impact_abs,
                        'サンプル数_ON': (df[col] == 1).sum(),
                        'サンプル数_OFF': (df[col] == 0).sum()
                    })

            # 連続値（相関分析）
            elif len(unique_vals) > 10:
                correlation = df[[col, sales_col]].corr().iloc[0, 1]

                if not np.isnan(correlation):
                    q75 = col_data.quantile(0.75)
                    q25 = col_data.quantile(0.25)

                    sales_high = df[df[col] >= q75][sales_col].mean()
                    sales_low = df[df[col] <= q25][sales_col].mean()

                    if sales_low > 0:
                        impact = (sales_high - sales_low) / sales_low
                        impact_abs = sales_high - sales_low

                        impact_results.append({
                            '特徴量': col,
                            '分析タイプ': '連続値（Q75 vs Q25）',
                            '売上_高': sales_high,
                            '売上_低': sales_low,
                            'インパクト率': impact,
                            'インパクト絶対値': impact_abs,
                            '相関係数': correlation
                        })

            # カテゴリカル数値（3-10種類）
            elif 3 <= len(unique_vals) <= 10:
                category_sales = df.groupby(col)[sales_col].agg(['mean', 'count'])

                if len(category_sales) >= 2:
                    max_sales = category_sales['mean'].max()
                    min_sales = category_sales['mean'].min()

                    if min_sales > 0:
                        impact = (max_sales - min_sales) / min_sales
                        impact_abs = max_sales - min_sales

                        impact_results.append({
                            '特徴量': col,
                            '分析タイプ': 'カテゴリカル',
                            '売上_最大': max_sales,
                            '売上_最小': min_sales,
                            'インパクト率': impact,
                            'インパクト絶対値': impact_abs,
                            'カテゴリ数': len(unique_vals)
                        })

    except Exception:
        continue

print(f'\\n✅ 分析完了: {len(impact_results)}個の特徴量を分析')

# ========================================
# 結果集計
# ========================================

if len(impact_results) > 0:
    impact_df = pd.DataFrame(impact_results)
    impact_df['インパクト率_絶対値'] = impact_df['インパクト率'].abs()
    impact_df = impact_df.sort_values('インパクト率_絶対値', ascending=False)

    print('\\n' + '='*80)
    print('🏆 売上インパクトランキング Top 20')
    print('='*80)

    for idx, row in impact_df.head(20).iterrows():
        impact_sign = '+' if row['インパクト率'] > 0 else ''
        print(f"{row['特徴量']:30s} {impact_sign}{row['インパクト率']:.2%} "\
              f"({impact_sign}{row['インパクト絶対値']:,.0f}円) [{row['分析タイプ']}]")

    # 正のインパクトTop 5
    positive_impact = impact_df[impact_df['インパクト率'] > 0].head(5)
    print('\\n' + '='*80)
    print('✅ 売上増加要因 Top 5')
    print('='*80)
    for idx, row in positive_impact.iterrows():
        print(f"  {row['特徴量']}: {row['インパクト率']:+.2%} ({row['インパクト絶対値']:+,.0f}円)")

    # 負のインパクトTop 5
    negative_impact = impact_df[impact_df['インパクト率'] < 0].sort_values('インパクト率').head(5)
    print('\\n' + '='*80)
    print('⚠️ 売上減少要因 Top 5')
    print('='*80)
    for idx, row in negative_impact.iterrows():
        print(f"  {row['特徴量']}: {row['インパクト率']:+.2%} ({row['インパクト絶対値']:+,.0f}円)")

    print('\\n' + '='*80)
    print('🎯 推奨アクション')
    print('='*80)
    print('''
【売上最大化施策】
1. 正のインパクト要因を最大化
   - 該当条件での在庫確保
   - プロモーション強化
   - 価格最適化

2. 負のインパクト要因を緩和
   - 売上減少時の特別施策
   - 代替商品の提案
   - 割引キャンペーン

3. 予測モデルへの活用
   - インパクト率上位の特徴量を重点使用
   - 特徴量エンジニアリングで相互作用項作成
   - カテゴリ別に重要特徴量を選定
    ''')

else:
    print('\\n⚠️ インパクト分析結果が得られませんでした')
"""

# ノートブック読み込み
notebook_path = Path('店舗別包括ダッシュボード_v6.1_提案強化.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 新規セル作成
new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'comprehensive_impact_analysis',
    'metadata': {},
    'outputs': [],
    'source': impact_cell_code.split('\n')
}

# 各行に改行を追加
new_cell['source'] = [line + '\n' if i < len(new_cell['source'])-1 else line
                     for i, line in enumerate(new_cell['source'])]

# 最後に追加
nb['cells'].append(new_cell)

# 保存
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ 包括的売上インパクト分析セルを追加しました")
print(f"✅ 保存完了: {notebook_path}")
print("\nセル内容:")
print("  - 格納率80%以上の全特徴量を分析")
print("  - バイナリフラグ、連続値、カテゴリカルの3種類の分析")
print("  - 売上増加要因 Top 5")
print("  - 売上減少要因 Top 5")
print("  - 推奨アクション")
