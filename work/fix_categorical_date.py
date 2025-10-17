#!/usr/bin/env python3
"""
Fix Categorical date min/max error in 特徴量AutoViz_PyCaret_v1.ipynb
"""

import json
from pathlib import Path

notebook_path = Path('/mnt/d/github/pycaret/work/特徴量AutoViz_PyCaret_v1.ipynb')

print('📖 Reading notebook...')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')

# Find and fix the problematic line
fixed = False
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # Check if this cell contains the problematic line
        if 'order_df["日付"].min()' in source and 'order_df["日付"].max()' in source:
            print(f'\n✅ Found problematic cell at index {cell_idx}')
            print('Original code snippet:')
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if '予測期間' in line and 'order_df["日付"]' in line:
                    print(f'  Line {i}: {line[:100]}...')

            # Fix the code - convert to datetime for min/max operations
            old_pattern = 'print(f\'  予測期間: {order_df["日付"].min()} ～ {order_df["日付"].max()}\')'
            new_code = '''# 日付列をdatetimeに変換してmin/maxを取得
        try:
            date_col = pd.to_datetime(order_df["日付"])
            print(f'  予測期間: {date_col.min()} ～ {date_col.max()}')
        except:
            # Categorical型の場合はユニーク値でソート
            dates = sorted(order_df["日付"].unique())
            if len(dates) > 0:
                print(f'  予測期間: {dates[0]} ～ {dates[-1]}')'''

            # Replace in source
            new_source = source.replace(old_pattern, new_code)

            # Update cell source
            cell['source'] = new_source.split('\n')
            if not cell['source'][-1].endswith('\n'):
                cell['source'][-1] += '\n'

            fixed = True
            print('\n✅ Fixed the code')
            print('New code:')
            print(new_code)
            break

if fixed:
    # Save the notebook
    print(f'\n💾 Saving notebook...')
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f'✅ Notebook saved: {notebook_path}')
    print('\n📝 Fix Summary:')
    print('  - Converted 日付 column to datetime before min/max operations')
    print('  - Added fallback for Categorical types using sorted unique values')
    print('  - Added try-except error handling')
else:
    print('\n⚠️ Could not find the problematic code pattern')

print('\n✅ Fix complete!')
