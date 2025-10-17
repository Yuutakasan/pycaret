#!/usr/bin/env python3
"""
包括的売上インパクト分析
格納率80%以上の全特徴量について売上影響度を算出
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("📊 包括的売上インパクト分析")
print("="*80)

# データ読み込み
data_path = Path('output/06_final_enriched_20250701_20250930.csv')
if not data_path.exists():
    print(f"⚠️ データファイルが見つかりません: {data_path}")
    exit(1)

df = pd.read_csv(data_path, encoding='utf-8-sig')
print(f"\n✅ データ読み込み完了: {len(df):,}行 × {len(df.columns)}列")

# 売上列の確認
sales_col = '売上金額' if '売上金額' in df.columns else '売上数量'
print(f"📊 売上指標: {sales_col}")

# ========================================
# 1. データ品質分析（格納率80%以上）
# ========================================

print("\n" + "="*80)
print("🔍 ステップ1: データ品質分析")
print("="*80)

# 各列の格納率計算
completeness = {}
for col in df.columns:
    non_null_count = df[col].notna().sum()
    completeness[col] = non_null_count / len(df)

completeness_df = pd.DataFrame({
    '列名': list(completeness.keys()),
    '格納率': list(completeness.values()),
    'データ型': [str(df[col].dtype) for col in completeness.keys()],
    '非NULL数': [df[col].notna().sum() for col in completeness.keys()],
    'ユニーク数': [df[col].nunique() for col in completeness.keys()]
}).sort_values('格納率', ascending=False)

# 格納率80%以上のカラム抽出
high_quality_cols = completeness_df[completeness_df['格納率'] >= 0.8]['列名'].tolist()

print(f"\n✅ 格納率80%以上のカラム: {len(high_quality_cols)}個")
print(f"   (全{len(df.columns)}列中 {len(high_quality_cols)/len(df.columns):.1%})")

# カテゴリ別に分類
exclude_cols = ['店舗', '商品名', '日付', sales_col, 'date', 'store_id', 'sku_id',
               'category_l', 'category_m', 'category_s',
               'フェイスくくり大分類', 'フェイスくくり中分類', 'フェイスくくり小分類']

analysis_cols = [col for col in high_quality_cols if col not in exclude_cols]

print(f"   分析対象カラム: {len(analysis_cols)}個")

# カテゴリ分類
feature_categories = {
    '時系列': [],
    '天気': [],
    '気温': [],
    'フラグ': [],
    'イベント': [],
    '統計量': [],
    'その他': []
}

for col in analysis_cols:
    col_lower = col.lower()
    if any(x in col_lower for x in ['_t-', '_ma', 'lag', 'shift', 'トレンド', '変化率', '累積']):
        feature_categories['時系列'].append(col)
    elif any(x in col for x in ['天気', '天候', 'weather']):
        feature_categories['天気'].append(col)
    elif any(x in col for x in ['気温', '温度', 'temp', '気圧', '湿度', 'humidity']):
        feature_categories['気温'].append(col)
    elif 'フラグ' in col or col.endswith('_flag') or any(x in col for x in ['日曜', '月曜', '土曜', '週末', '平日', '祝日', '休日']):
        feature_categories['フラグ'].append(col)
    elif any(x in col for x in ['給料日', 'イベント', 'event', 'キャンペーン', '月初', '月末']):
        feature_categories['イベント'].append(col)
    elif any(x in col for x in ['平均', '最大', '最小', '合計', 'mean', 'max', 'min', 'sum', 'std']):
        feature_categories['統計量'].append(col)
    else:
        feature_categories['その他'].append(col)

print("\n📋 特徴量カテゴリ別集計:")
for category, cols in feature_categories.items():
    if cols:
        print(f"  {category}: {len(cols)}個")

# ========================================
# 2. 売上インパクト分析
# ========================================

print("\n" + "="*80)
print("💰 ステップ2: 売上インパクト分析")
print("="*80)

impact_results = []

for col in analysis_cols:
    try:
        # データ型確認
        col_data = df[col].dropna()

        if len(col_data) < 100:
            continue

        # 数値型の場合
        if pd.api.types.is_numeric_dtype(col_data):
            # バイナリフラグ（0/1）の場合
            unique_vals = col_data.unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                # フラグON/OFFでの売上比較
                flag_on = df[df[col] == 1][sales_col].mean()
                flag_off = df[df[col] == 0][sales_col].mean()

                if flag_off > 0:
                    impact = (flag_on - flag_off) / flag_off
                    impact_abs = flag_on - flag_off

                    impact_results.append({
                        '特徴量': col,
                        'カテゴリ': next((k for k, v in feature_categories.items() if col in v), 'その他'),
                        '分析タイプ': 'バイナリフラグ',
                        '売上_ON': flag_on,
                        '売上_OFF': flag_off,
                        'インパクト率': impact,
                        'インパクト絶対値': impact_abs,
                        'サンプル数_ON': (df[col] == 1).sum(),
                        'サンプル数_OFF': (df[col] == 0).sum(),
                        '格納率': completeness[col]
                    })

            # 連続値の場合（相関分析）
            elif len(unique_vals) > 10:
                correlation = df[[col, sales_col]].corr().iloc[0, 1]

                if not np.isnan(correlation):
                    # 上位25%と下位25%の売上比較
                    q75 = col_data.quantile(0.75)
                    q25 = col_data.quantile(0.25)

                    sales_high = df[df[col] >= q75][sales_col].mean()
                    sales_low = df[df[col] <= q25][sales_col].mean()

                    if sales_low > 0:
                        impact = (sales_high - sales_low) / sales_low
                        impact_abs = sales_high - sales_low

                        impact_results.append({
                            '特徴量': col,
                            'カテゴリ': next((k for k, v in feature_categories.items() if col in v), 'その他'),
                            '分析タイプ': '連続値（Q75 vs Q25）',
                            '売上_高': sales_high,
                            '売上_低': sales_low,
                            'インパクト率': impact,
                            'インパクト絶対値': impact_abs,
                            '相関係数': correlation,
                            'サンプル数_高': (df[col] >= q75).sum(),
                            'サンプル数_低': (df[col] <= q25).sum(),
                            '格納率': completeness[col]
                        })

            # カテゴリカル数値（3-10種類）の場合
            elif 3 <= len(unique_vals) <= 10:
                # 各カテゴリの売上平均
                category_sales = df.groupby(col)[sales_col].agg(['mean', 'count'])

                if len(category_sales) >= 2:
                    max_sales = category_sales['mean'].max()
                    min_sales = category_sales['mean'].min()

                    if min_sales > 0:
                        impact = (max_sales - min_sales) / min_sales
                        impact_abs = max_sales - min_sales

                        impact_results.append({
                            '特徴量': col,
                            'カテゴリ': next((k for k, v in feature_categories.items() if col in v), 'その他'),
                            '分析タイプ': 'カテゴリカル',
                            '売上_最大': max_sales,
                            '売上_最小': min_sales,
                            'インパクト率': impact,
                            'インパクト絶対値': impact_abs,
                            'カテゴリ数': len(unique_vals),
                            '格納率': completeness[col]
                        })

        # 文字列型の場合
        elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            unique_vals = col_data.unique()

            if 2 <= len(unique_vals) <= 20:
                category_sales = df.groupby(col)[sales_col].agg(['mean', 'count'])
                category_sales = category_sales[category_sales['count'] >= 10]  # 最低10サンプル

                if len(category_sales) >= 2:
                    max_sales = category_sales['mean'].max()
                    min_sales = category_sales['mean'].min()

                    if min_sales > 0:
                        impact = (max_sales - min_sales) / min_sales
                        impact_abs = max_sales - min_sales

                        impact_results.append({
                            '特徴量': col,
                            'カテゴリ': next((k for k, v in feature_categories.items() if col in v), 'その他'),
                            '分析タイプ': 'テキストカテゴリ',
                            '売上_最大': max_sales,
                            '売上_最小': min_sales,
                            'インパクト率': impact,
                            'インパクト絶対値': impact_abs,
                            'カテゴリ数': len(category_sales),
                            '格納率': completeness[col]
                        })

    except Exception as e:
        print(f"⚠️ {col}: エラー ({str(e)[:50]})")
        continue

print(f"\n✅ 分析完了: {len(impact_results)}個の特徴量を分析")

# ========================================
# 3. 結果集計とランキング
# ========================================

if len(impact_results) > 0:
    impact_df = pd.DataFrame(impact_results)

    # インパクト率の絶対値でソート
    impact_df['インパクト率_絶対値'] = impact_df['インパクト率'].abs()
    impact_df = impact_df.sort_values('インパクト率_絶対値', ascending=False)

    print("\n" + "="*80)
    print("🏆 売上インパクトランキング Top 30")
    print("="*80)

    # 表示用に整形
    display_df = impact_df.head(30).copy()
    display_df['インパクト率_表示'] = display_df['インパクト率'].apply(lambda x: f"{x:+.2%}")
    display_df['インパクト絶対値_表示'] = display_df['インパクト絶対値'].apply(lambda x: f"{x:+,.0f}円")

    print(display_df[['特徴量', 'カテゴリ', '分析タイプ', 'インパクト率_表示',
                      'インパクト絶対値_表示', '格納率']].to_string(index=False))

    # カテゴリ別サマリー
    print("\n" + "="*80)
    print("📊 カテゴリ別インパクトサマリー")
    print("="*80)

    category_summary = impact_df.groupby('カテゴリ').agg({
        'インパクト率_絶対値': ['mean', 'max', 'count']
    }).round(4)
    category_summary.columns = ['平均インパクト率', '最大インパクト率', '特徴量数']
    category_summary = category_summary.sort_values('最大インパクト率', ascending=False)

    print(category_summary)

    # CSV保存
    output_path = Path('output/comprehensive_sales_impact_analysis.csv')
    impact_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 詳細結果を保存: {output_path}")

    # Top 50だけ別途保存
    top50_path = Path('output/sales_impact_top50.csv')
    impact_df.head(50).to_csv(top50_path, index=False, encoding='utf-8-sig')
    print(f"✅ Top 50を保存: {top50_path}")

    # ========================================
    # 4. 洞察とアクションアイテム
    # ========================================

    print("\n" + "="*80)
    print("💡 主要な洞察")
    print("="*80)

    # 正のインパクトTop 5
    positive_impact = impact_df[impact_df['インパクト率'] > 0].head(5)
    print("\n✅ 売上増加要因 Top 5:")
    for idx, row in positive_impact.iterrows():
        print(f"  {row['特徴量']}: {row['インパクト率']:+.2%} ({row['インパクト絶対値']:+,.0f}円)")

    # 負のインパクトTop 5
    negative_impact = impact_df[impact_df['インパクト率'] < 0].sort_values('インパクト率').head(5)
    print("\n⚠️ 売上減少要因 Top 5:")
    for idx, row in negative_impact.iterrows():
        print(f"  {row['特徴量']}: {row['インパクト率']:+.2%} ({row['インパクト絶対値']:+,.0f}円)")

    print("\n" + "="*80)
    print("🎯 推奨アクション")
    print("="*80)

    print("""
【売上最大化施策】
1. 正のインパクト要因を最大化
   - 該当日・条件での在庫確保
   - プロモーション強化
   - 価格最適化

2. 負のインパクト要因を緩和
   - 売上減少日の特別施策
   - 代替商品の提案
   - 割引キャンペーン

3. 予測モデルへの活用
   - インパクト率上位の特徴量を重点的に使用
   - 特徴量エンジニアリングで相互作用項を作成
   - カテゴリ別に重要な特徴量を選定
    """)

else:
    print("\n⚠️ インパクト分析結果が得られませんでした")

print("\n" + "="*80)
print("✅ 包括的売上インパクト分析完了")
print("="*80)
