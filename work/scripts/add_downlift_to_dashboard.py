#!/usr/bin/env python3
"""
店舗別包括ダッシュボード_v6.1_提案強化.ipynb にdownlift分析セルを追加
"""

import json
from pathlib import Path

# Downlift分析セルのコード
downlift_cell_code = """# ========================================
# 📉 Downlift分析：売上減少要因の特定
# ========================================

print('\\n' + '='*80)
print('📉 Downlift分析：トリガー別の売上減少カテゴリ')
print('='*80)

# 降雨時に売上が減少するカテゴリ（downlift）
downlift_results = {}

for trigger_col in ['降雨フラグ', '週末フラグ', '猛暑日', '真夏日', '夏日', '給料日', '給料日直後', '月初3日', '月末3日']:
    if trigger_col not in df.columns:
        continue

    # トリガーON/OFF時の売上比較（downlift: flag1 < flag0）
    comparison = df.groupby([trigger_col, 'category_l'], as_index=False)['売上金額'].mean()

    pivot = comparison.pivot_table(
        index='category_l',
        columns=trigger_col,
        values='売上金額',
        aggfunc='mean'
    )

    if 1.0 not in pivot.columns or 0.0 not in pivot.columns:
        continue

    # Downlift計算：(OFF - ON) / OFF （正の値 = 売上減少）
    pivot['downlift'] = (pivot[0.0] - pivot[1.0]) / pivot[0.0]

    # 売上減少が大きい順（downlift > 0.1 = 10%以上減少）
    significant_down = pivot[pivot['downlift'] > 0.1].copy()

    if len(significant_down) > 0:
        significant_down = significant_down.sort_values('downlift', ascending=False)
        significant_down = significant_down.rename(columns={
            0.0: 'sales_amt_flag0',
            1.0: 'sales_amt_flag1'
        })

        downlift_results[trigger_col] = significant_down[['sales_amt_flag1', 'sales_amt_flag0', 'downlift']].head(5)

# ========================================
# 結果表示
# ========================================

if downlift_results:
    print('\\n--- 提案（カテゴリ別の売上減少が大きい順）---')
    for trigger_key, result_df in downlift_results.items():
        if len(result_df) > 0:
            print(f'\\n[提案キー: {trigger_key}]')
            print(result_df.reset_index().to_string(index=False))

    print('\\n' + '='*80)
    print('💡 Downlift活用方法')
    print('='*80)
    print('''
【在庫最適化】
- 降雨時に売上が減るカテゴリ → 雨予報の日は発注を控える
- 週末に売上が減るカテゴリ → 平日に在庫を厚くする

【キャンペーン企画】
- 売上減少カテゴリに対して「雨の日割引」「週末特売」などで需要喚起
- 例：降雨時10%減のカテゴリ → 雨の日10%割引で需要を維持

【販売計画】
- Downlift率を考慮した売上予測（保守的見積もり）
- 例：給料日直後-15%、月末-20%など季節変動を反映
    ''')
else:
    print('\\n⚠️ 有意なdownlift（売上減少10%以上）は検出されませんでした')

# ========================================
# 📊 Uplift vs Downliftの統合ビュー
# ========================================

print('\\n' + '='*80)
print('📊 統合分析：Uplift + Downlift')
print('='*80)

# カテゴリごとの売上変動幅を計算
category_volatility = []

for category in df['category_l'].dropna().unique():
    cat_data = df[df['category_l'] == category]

    max_uplifts = []
    max_downlifts = []

    for trigger_col in ['降雨フラグ', '週末フラグ', '猛暑日', '真夏日', '夏日']:
        if trigger_col not in df.columns:
            continue

        comparison = cat_data.groupby(trigger_col)['売上金額'].mean()

        if 1.0 in comparison.index and 0.0 in comparison.index:
            uplift = (comparison[1.0] - comparison[0.0]) / comparison[0.0]
            downlift = (comparison[0.0] - comparison[1.0]) / comparison[0.0]

            max_uplifts.append(max(0, uplift))
            max_downlifts.append(max(0, downlift))

    if max_uplifts and max_downlifts:
        category_volatility.append({
            'カテゴリ': category,
            '最大Uplift': max(max_uplifts),
            '最大Downlift': max(max_downlifts),
            '売上変動幅': max(max_uplifts) + max(max_downlifts),
            '予測難易度': 'A:高難易度' if (max(max_uplifts) + max(max_downlifts)) > 1.5 else
                         'B:中難易度' if (max(max_uplifts) + max(max_downlifts)) > 0.5 else
                         'C:低難易度'
        })

volatility_df = pd.DataFrame(category_volatility).sort_values('売上変動幅', ascending=False)

print('\\n【カテゴリ別売上変動幅ランキング（Top 10）】')
print(volatility_df.head(10).to_string(index=False))

print('\\n💡 予測モデリング推奨戦略:')
print('  A:高難易度（変動幅1.5倍以上） → 個別モデル + 特徴量エンジニアリング強化')
print('  B:中難易度（変動幅0.5-1.5倍） → カテゴリ別モデル')
print('  C:低難易度（変動幅0.5倍未満） → 統合モデルでOK')

# CSV保存
output_path = Path('output/category_uplift_downlift_analysis.csv')
volatility_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f'\\n✅ 分析結果を保存: {output_path}')
"""

# ノートブック読み込み
notebook_path = Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v6.1_提案強化.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Uplift分析セルを探す
uplift_cell_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '提案キー' in source and 'uplift' in source and 'category_l' in source:
            uplift_cell_index = i
            print(f"✅ Uplift分析セルを検出: Cell {i}")
            break

if uplift_cell_index is not None:
    # Upliftセルの直後にDownliftセルを挿入
    new_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'id': 'downlift_analysis_cell',
        'metadata': {},
        'outputs': [],
        'source': downlift_cell_code.split('\n')
    }

    # 各行に改行を追加（最後の行以外）
    new_cell['source'] = [line + '\n' if i < len(new_cell['source'])-1 else line
                         for i, line in enumerate(new_cell['source'])]

    nb['cells'].insert(uplift_cell_index + 1, new_cell)

    # 保存
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"✅ Downlift分析セルを追加しました（Cell {uplift_cell_index + 1}）")
    print(f"✅ 保存完了: {notebook_path}")
else:
    print("⚠️ Uplift分析セルが見つかりませんでした")
    print("手動でDownliftセルを追加してください")

    # コードを別ファイルとして保存
    code_path = Path('scripts/downlift_analysis_code.py')
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(downlift_cell_code)
    print(f"📝 Downlift分析コードを保存: {code_path}")
