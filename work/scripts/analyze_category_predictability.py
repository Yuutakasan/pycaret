#!/usr/bin/env python3
"""
カテゴリ別予測難易度分析
uplift分析結果から、どのカテゴリを個別モデル化すべきか判定
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 日本語フォント設定
JP_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(JP_FONT_PATH).exists():
    JP_FP = font_manager.FontProperties(fname=JP_FONT_PATH)
else:
    JP_FP = font_manager.FontProperties(family='sans-serif')

# ========================================
# 1. Uplift分析データの構造化
# ========================================

uplift_data = {
    '降雨フラグ': [
        {'category': '160:インスタントラーメン', 'uplift': 4.136364, 'sales_high': 3051.0, 'sales_low': 594.0},
        {'category': '260:雨具', 'uplift': 1.999050, 'sales_high': 4596.28, 'sales_low': 1532.58},
        {'category': '399:その他', 'uplift': 1.388992, 'sales_high': 7920.43, 'sales_low': 3315.38},
        {'category': '280:チケット・カード', 'uplift': 1.007411, 'sales_high': 23514.02, 'sales_low': 11713.60},
        {'category': '250:文具・玩具・趣味雑貨', 'uplift': 0.999357, 'sales_high': 3880.77, 'sales_low': 1941.01},
    ],
    '週末フラグ': [
        {'category': '140:カウンターＦＦ', 'uplift': 2.480293, 'sales_high': 7532.66, 'sales_low': 2164.38},
        {'category': '220:化粧品', 'uplift': 0.489004, 'sales_high': 2517.45, 'sales_low': 1690.70},
        {'category': '170:乾物・缶詰・調味料', 'uplift': 0.466581, 'sales_high': 1630.0, 'sales_low': 1111.43},
        {'category': '225:バス・洗面用品', 'uplift': 0.303930, 'sales_high': 639.85, 'sales_low': 490.71},
        {'category': 'xxx:不明', 'uplift': 0.195467, 'sales_high': 782.96, 'sales_low': 654.94},
    ],
    '猛暑日': [
        {'category': '140:カウンターＦＦ', 'uplift': 0.754881, 'sales_high': 6180.62, 'sales_low': 3521.96},
        {'category': '220:化粧品', 'uplift': 0.442036, 'sales_high': 2510.37, 'sales_low': 1740.85},
        {'category': '105:調理麺', 'uplift': 0.237584, 'sales_high': 2373.44, 'sales_low': 1917.80},
        {'category': '200:健康食品', 'uplift': 0.185294, 'sales_high': 2306.38, 'sales_low': 1945.83},
        {'category': 'xxx:不明', 'uplift': 0.145317, 'sales_high': 767.74, 'sales_low': 670.33},
    ],
    '真夏日': [
        {'category': '300:日本酒・焼酎', 'uplift': 0.800766, 'sales_high': 354.93, 'sales_low': 197.10},
        {'category': '165:インスタント食品', 'uplift': 0.624350, 'sales_high': 632.68, 'sales_low': 389.50},
        {'category': '220:化粧品', 'uplift': 0.606440, 'sales_high': 2135.14, 'sales_low': 1329.11},
        {'category': '250:文具・玩具・趣味雑貨', 'uplift': 0.111295, 'sales_high': 2882.04, 'sales_low': 2593.41},
        {'category': '210:医薬品・医薬部外品', 'uplift': 0.091694, 'sales_high': 954.87, 'sales_low': 874.67},
    ],
    '夏日': [
        {'category': '280:チケット・カード', 'uplift': 4.718997, 'sales_high': 17156.99, 'sales_low': 3000.0},
        {'category': '220:化粧品', 'uplift': 4.127442, 'sales_high': 1959.54, 'sales_low': 382.17},
        {'category': '165:インスタント食品', 'uplift': 2.413557, 'sales_high': 570.06, 'sales_low': 167.0},
        {'category': '125:日配品・生鮮品', 'uplift': 1.773127, 'sales_high': 487.61, 'sales_low': 175.83},
        {'category': '399:その他', 'uplift': 1.499044, 'sales_high': 5832.77, 'sales_low': 2334.0},
    ],
    '給料日': [
        {'category': '165:インスタント食品', 'uplift': 0.519843, 'sales_high': 847.50, 'sales_low': 557.62},
        {'category': '280:チケット・カード', 'uplift': 0.396517, 'sales_high': 23500.0, 'sales_low': 16827.58},
        {'category': '245:衣料品', 'uplift': 0.264528, 'sales_high': 2732.83, 'sales_low': 2161.15},
        {'category': '265:雑誌・コミック・新聞', 'uplift': 0.224627, 'sales_high': 3871.56, 'sales_low': 3161.42},
        {'category': '290:たばこ', 'uplift': 0.201961, 'sales_high': 75923.78, 'sales_low': 63166.59},
    ],
    '給料日直後': [
        {'category': '250:文具・玩具・趣味雑貨', 'uplift': 0.896767, 'sales_high': 4834.17, 'sales_low': 2548.63},
        {'category': '140:カウンターＦＦ', 'uplift': 0.392280, 'sales_high': 5321.69, 'sales_low': 3822.29},
        {'category': '300:日本酒・焼酎', 'uplift': 0.250000, 'sales_high': 383.25, 'sales_low': 306.60},
        {'category': '170:乾物・缶詰・調味料', 'uplift': 0.183801, 'sales_high': 1400.0, 'sales_low': 1182.63},
        {'category': '148:半生菓子', 'uplift': 0.144487, 'sales_high': 3226.13, 'sales_low': 2818.84},
    ],
    '月初3日': [
        {'category': '250:文具・玩具・趣味雑貨', 'uplift': 2.220215, 'sales_high': 7350.77, 'sales_low': 2282.70},
        {'category': '170:乾物・缶詰・調味料', 'uplift': 0.840491, 'sales_high': 2100.0, 'sales_low': 1141.0},
        {'category': '165:インスタント食品', 'uplift': 0.697058, 'sales_high': 903.0, 'sales_low': 532.10},
        {'category': '210:医薬品・医薬部外品', 'uplift': 0.439005, 'sales_high': 1302.35, 'sales_low': 905.04},
        {'category': '195:スナック菓子', 'uplift': 0.186067, 'sales_high': 2125.19, 'sales_low': 1791.79},
    ],
    '月末3日': [
        {'category': '165:インスタント食品', 'uplift': 0.866377, 'sales_high': 989.33, 'sales_low': 530.08},
        {'category': '235:家庭雑貨', 'uplift': 0.500000, 'sales_high': 747.0, 'sales_low': 498.0},
        {'category': 'xxx:不明', 'uplift': 0.439743, 'sales_high': 957.44, 'sales_low': 665.01},
        {'category': '305:ワイン・洋酒', 'uplift': 0.085019, 'sales_high': 5077.84, 'sales_low': 4679.95},
        {'category': '185:洋風菓子・駄菓子', 'uplift': 0.050039, 'sales_high': 5029.30, 'sales_low': 4789.63},
    ],
}

# データフレーム化
all_uplifts = []
for trigger, categories in uplift_data.items():
    for cat_data in categories:
        all_uplifts.append({
            'トリガー': trigger,
            'カテゴリ': cat_data['category'],
            'uplift': cat_data['uplift'],
            'sales_high': cat_data['sales_high'],
            'sales_low': cat_data['sales_low'],
            'volatility': (cat_data['sales_high'] - cat_data['sales_low']) / cat_data['sales_low']
        })

df_uplift = pd.DataFrame(all_uplifts)

print("\n" + "="*80)
print("📊 カテゴリ別Uplift分析結果")
print("="*80)
print(f"\n総カテゴリ数: {df_uplift['カテゴリ'].nunique()}")
print(f"トリガー数: {df_uplift['トリガー'].nunique()}")

# ========================================
# 2. カテゴリごとの予測難易度スコア算出
# ========================================

# カテゴリごとに集約
category_stats = df_uplift.groupby('カテゴリ').agg({
    'uplift': ['mean', 'max', 'std', 'count'],
    'volatility': ['mean', 'max'],
    'sales_high': 'mean',
    'sales_low': 'mean'
}).reset_index()

category_stats.columns = ['カテゴリ', 'uplift_mean', 'uplift_max', 'uplift_std',
                          'trigger_count', 'volatility_mean', 'volatility_max',
                          'sales_high_mean', 'sales_low_mean']

# 予測難易度スコア (0-100): 高いほど予測が難しい
category_stats['難易度スコア'] = (
    category_stats['uplift_mean'] * 20 +          # 平均upliftの影響 (最大100)
    category_stats['uplift_std'] * 10 +           # upliftのばらつき
    category_stats['volatility_mean'] * 15 +      # 売上変動率
    category_stats['trigger_count'] * 3           # 影響を受けるトリガー数
).clip(0, 100)

# ========================================
# 3. カテゴリ分類（A/B/C）
# ========================================

# 難易度による分類
category_stats = category_stats.sort_values('難易度スコア', ascending=False).reset_index(drop=True)

# 閾値設定
HIGH_DIFFICULTY = 70  # A: 個別モデル必須
MEDIUM_DIFFICULTY = 40  # B: カテゴリ別モデル推奨
# C: 統合モデルでOK

category_stats['推奨モデル'] = pd.cut(
    category_stats['難易度スコア'],
    bins=[-np.inf, MEDIUM_DIFFICULTY, HIGH_DIFFICULTY, np.inf],
    labels=['C:統合モデル', 'B:カテゴリ別', 'A:個別モデル']
)

print("\n" + "="*80)
print("🎯 カテゴリ別予測難易度ランキング（Top 15）")
print("="*80)
print(category_stats[['カテゴリ', 'uplift_mean', 'uplift_max', 'volatility_mean',
                      'trigger_count', '難易度スコア', '推奨モデル']].head(15).to_string(index=False))

print("\n" + "="*80)
print("📋 モデリング戦略サマリー")
print("="*80)
model_strategy = category_stats.groupby('推奨モデル').agg({
    'カテゴリ': 'count',
    '難易度スコア': 'mean',
    'sales_high_mean': 'sum'
}).round(2)
model_strategy.columns = ['カテゴリ数', '平均難易度', '合計売上（高）']
print(model_strategy)

# ========================================
# 4. 個別モデル推奨カテゴリの詳細分析
# ========================================

high_difficulty_cats = category_stats[category_stats['推奨モデル'] == 'A:個別モデル']

print("\n" + "="*80)
print("⚠️ 個別モデル必須カテゴリ（難易度70+）")
print("="*80)
if len(high_difficulty_cats) > 0:
    print(high_difficulty_cats[['カテゴリ', 'uplift_mean', 'uplift_max',
                                'volatility_mean', '難易度スコア']].to_string(index=False))

    print("\n【個別モデル化が必要な理由】")
    for idx, row in high_difficulty_cats.iterrows():
        cat_name = row['カテゴリ']
        reasons = []

        if row['uplift_max'] > 2.0:
            reasons.append(f"特定条件で売上が{row['uplift_max']:.1f}倍に急増")
        if row['uplift_std'] > 1.0:
            reasons.append(f"upliftのばらつきが大きい (σ={row['uplift_std']:.2f})")
        if row['volatility_mean'] > 1.0:
            reasons.append(f"売上変動率が{row['volatility_mean']:.1%}と高い")
        if row['trigger_count'] >= 5:
            reasons.append(f"{int(row['trigger_count'])}個のトリガーに反応")

        print(f"\n  {cat_name}:")
        for reason in reasons:
            print(f"    - {reason}")
else:
    print("  該当なし（全カテゴリが統合/カテゴリ別モデルでOK）")

# ========================================
# 5. 実装推奨：compare_models()実行対象
# ========================================

print("\n" + "="*80)
print("🚀 実装推奨：compare_models()実行戦略")
print("="*80)

print("\n【戦略A】個別モデル化（難易度70+）")
print("  対象カテゴリ数:", len(high_difficulty_cats))
if len(high_difficulty_cats) > 0:
    print("  カテゴリリスト:")
    for cat in high_difficulty_cats['カテゴリ'].values:
        print(f"    - {cat}")
    print("\n  実装方法:")
    print("    1. カテゴリごとにデータ分割")
    print("    2. 各カテゴリでcompare_models(n_select=3)")
    print("    3. tune_model()で最適化")
    print("    4. 最終モデルを個別保存")

medium_difficulty_cats = category_stats[category_stats['推奨モデル'] == 'B:カテゴリ別']
print("\n【戦略B】カテゴリ別モデル（難易度40-70）")
print("  対象カテゴリ数:", len(medium_difficulty_cats))
if len(medium_difficulty_cats) > 0:
    print("  カテゴリリスト（Top 5）:")
    for cat in medium_difficulty_cats['カテゴリ'].head(5).values:
        print(f"    - {cat}")
    print("\n  実装方法:")
    print("    1. カテゴリグループでcompare_models()")
    print("    2. 共通モデルアーキテクチャで学習")

low_difficulty_cats = category_stats[category_stats['推奨モデル'] == 'C:統合モデル']
print("\n【戦略C】統合モデル（難易度40未満）")
print("  対象カテゴリ数:", len(low_difficulty_cats))
print("  実装方法:")
print("    1. 全データでcompare_models(turbo=False)")
print("    2. カテゴリをダミー変数化")
print("    3. 1つのモデルで全商品を予測")

# ========================================
# 6. 可視化
# ========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 6-1. 難易度分布
ax1 = axes[0, 0]
category_stats['難易度スコア'].hist(bins=20, ax=ax1, color='steelblue', edgecolor='black')
ax1.axvline(MEDIUM_DIFFICULTY, color='orange', linestyle='--', linewidth=2, label='B/C境界')
ax1.axvline(HIGH_DIFFICULTY, color='red', linestyle='--', linewidth=2, label='A/B境界')
ax1.set_xlabel('難易度スコア', fontproperties=JP_FP, fontsize=12)
ax1.set_ylabel('カテゴリ数', fontproperties=JP_FP, fontsize=12)
ax1.set_title('カテゴリ別予測難易度分布', fontproperties=JP_FP, fontsize=14, fontweight='bold')
ax1.legend(prop=JP_FP)
ax1.grid(True, alpha=0.3)

# 6-2. Uplift vs Volatility
ax2 = axes[0, 1]
colors = {'A:個別モデル': 'red', 'B:カテゴリ別': 'orange', 'C:統合モデル': 'green'}
for model_type, color in colors.items():
    subset = category_stats[category_stats['推奨モデル'] == model_type]
    ax2.scatter(subset['uplift_mean'], subset['volatility_mean'],
               c=color, label=model_type, s=100, alpha=0.6, edgecolors='black')

ax2.set_xlabel('平均Uplift', fontproperties=JP_FP, fontsize=12)
ax2.set_ylabel('平均Volatility', fontproperties=JP_FP, fontsize=12)
ax2.set_title('Uplift vs Volatility（モデル戦略別）', fontproperties=JP_FP, fontsize=14, fontweight='bold')
ax2.legend(prop=JP_FP)
ax2.grid(True, alpha=0.3)

# 6-3. トリガー数の影響
ax3 = axes[1, 0]
trigger_impact = category_stats.groupby('trigger_count').agg({
    '難易度スコア': 'mean',
    'カテゴリ': 'count'
}).reset_index()
ax3.bar(trigger_impact['trigger_count'], trigger_impact['難易度スコア'],
       color='coral', edgecolor='black', alpha=0.7)
ax3.set_xlabel('影響を受けるトリガー数', fontproperties=JP_FP, fontsize=12)
ax3.set_ylabel('平均難易度スコア', fontproperties=JP_FP, fontsize=12)
ax3.set_title('トリガー数と予測難易度の関係', fontproperties=JP_FP, fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 6-4. モデル戦略別カテゴリ数
ax4 = axes[1, 1]
strategy_counts = category_stats['推奨モデル'].value_counts()
colors_pie = ['red', 'orange', 'green']
ax4.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%',
       colors=colors_pie, startangle=90, textprops={'fontproperties': JP_FP, 'fontsize': 11})
ax4.set_title('推奨モデル戦略の分布', fontproperties=JP_FP, fontsize=14, fontweight='bold')

plt.tight_layout()
output_path = Path('output/category_predictability_analysis.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ グラフ保存完了: {output_path}")

# ========================================
# 7. CSVエクスポート
# ========================================

csv_path = Path('output/category_modeling_strategy.csv')
category_stats.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ 戦略CSV保存完了: {csv_path}")

# ========================================
# 8. 実装コード生成
# ========================================

print("\n" + "="*80)
print("💻 実装サンプルコード")
print("="*80)

print("""
# ========================================
# カテゴリ別モデリング実装例
# ========================================

import pandas as pd
from pycaret.regression import setup, compare_models, tune_model, finalize_model, save_model

# 戦略読み込み
strategy_df = pd.read_csv('output/category_modeling_strategy.csv')

# データ読み込み（全商品）
data = pd.read_csv('output/06_final_enriched_20250701_20250930.csv')

# カテゴリ列を追加（商品名から抽出）
data['カテゴリ'] = data['商品名'].str.extract(r'(\\d{3}:[^_]+)')[0]

# ========================================
# 戦略A: 個別モデル（難易度70+）
# ========================================

high_diff_cats = strategy_df[strategy_df['推奨モデル'] == 'A:個別モデル']['カテゴリ'].tolist()

models_individual = {}
for category in high_diff_cats:
    print(f'\\n🎯 個別モデル学習: {category}')

    # カテゴリデータ抽出
    cat_data = data[data['カテゴリ'] == category].copy()

    if len(cat_data) < 100:
        print(f'  ⚠️ データ不足 ({len(cat_data)}行) - スキップ')
        continue

    # PyCaret setup
    s = setup(cat_data, target='売上数量', session_id=123,
              fold=5, remove_multicollinearity=True,
              multicollinearity_threshold=0.95,
              normalize=True, feature_selection=True)

    # モデル比較（上位3モデル）
    best_models = compare_models(n_select=3, sort='R2', turbo=False,
                                include=['et', 'lightgbm', 'catboost', 'xgboost', 'rf'])

    # 最適化
    tuned = tune_model(best_models[0], n_iter=30, optimize='R2')

    # ファイナライズ
    final = finalize_model(tuned)

    # 保存
    model_path = f'models/{category.replace(":", "_")}_model'
    save_model(final, model_path)

    models_individual[category] = {
        'model': final,
        'path': model_path,
        'r2': final.score(cat_data.drop('売上数量', axis=1), cat_data['売上数量'])
    }

    print(f'  ✅ モデル保存完了: {model_path}')
    print(f'  📊 R²スコア: {models_individual[category]["r2"]:.4f}')

# ========================================
# 戦略B: カテゴリ別モデル（難易度40-70）
# ========================================

medium_diff_cats = strategy_df[strategy_df['推奨モデル'] == 'B:カテゴリ別']['カテゴリ'].tolist()

cat_group_data = data[data['カテゴリ'].isin(medium_diff_cats)].copy()

if len(cat_group_data) > 0:
    print(f'\\n🎯 カテゴリ別モデル学習 ({len(medium_diff_cats)}カテゴリ統合)')

    s = setup(cat_group_data, target='売上数量', session_id=123,
              categorical_features=['カテゴリ', '店舗'],
              fold=5, normalize=True)

    best = compare_models(n_select=3, sort='R2')
    tuned = tune_model(best[0], n_iter=20)
    final_medium = finalize_model(tuned)

    save_model(final_medium, 'models/category_group_model')
    print('  ✅ カテゴリ別モデル保存完了: models/category_group_model')

# ========================================
# 戦略C: 統合モデル（難易度40未満）
# ========================================

low_diff_cats = strategy_df[strategy_df['推奨モデル'] == 'C:統合モデル']['カテゴリ'].tolist()

unified_data = data[data['カテゴリ'].isin(low_diff_cats)].copy()

if len(unified_data) > 0:
    print(f'\\n🎯 統合モデル学習 ({len(low_diff_cats)}カテゴリ統合)')

    s = setup(unified_data, target='売上数量', session_id=123,
              categorical_features=['カテゴリ', '店舗'],
              fold=10, normalize=True)

    best = compare_models(turbo=False, sort='R2')
    tuned = tune_model(best, n_iter=50)
    final_unified = finalize_model(tuned)

    save_model(final_unified, 'models/unified_model')
    print('  ✅ 統合モデル保存完了: models/unified_model')

print('\\n✅ 全モデル学習完了！')
""")

print("\n" + "="*80)
print("📝 実行推奨順序")
print("="*80)
print("""
1. このスクリプトを実行して戦略CSVを生成
   → python3 analyze_category_predictability.py

2. 上記の実装コードを新規ノートブックにコピー
   → work/Step5_CategoryWise_Compare.ipynb

3. 個別モデル → カテゴリ別 → 統合モデルの順に学習

4. 予測時は戦略CSVを参照してモデル選択
""")
