#!/usr/bin/env python3
"""
Notebookのcompare_models()をGPU対応に更新
"""

import json
from pathlib import Path
import re

notebook_path = Path('/mnt/d/github/pycaret/work/Step5_CategoryWise_Compare_with_Overfitting.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# compare_models()を含むセルを検索して更新
updated_cells = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    source = ''.join(cell['source'])

    # compare_models()呼び出しを検索
    if 'compare_models(' in source and 'include=' in source:
        # 既存のincludeパラメータをGPU_MODELSに置換
        # パターン1: include=['et', 'lightgbm', ...]
        pattern1 = r"include=\[[^\]]+\]"
        if re.search(pattern1, source):
            # GPU_MODELSに置換 + CPUモデルも追加
            new_source = re.sub(
                pattern1,
                "include=GPU_MODELS + ['et', 'rf', 'gbr', 'dt']",
                source
            )

            # コメント追加
            if 'GPU_MODELS' not in source:
                new_source = new_source.replace(
                    'compare_models(',
                    '# GPU高速化: XGBoost/CatBoost GPUを優先使用\n    compare_models('
                )

            cell['source'] = new_source.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line
                             for i, line in enumerate(cell['source'])]

            updated_cells += 1
            print(f"✅ Cell {i} を更新: compare_models()にGPU_MODELS追加")

# 保存
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ {updated_cells}個のセルを更新しました")
print(f"✅ 保存完了: {notebook_path}")

if updated_cells == 0:
    print("\n💡 手動で以下のように変更してください:")
    print("""
# 変更前:
compare_models(include=['et', 'lightgbm', 'catboost', 'xgboost'])

# 変更後:
compare_models(include=GPU_MODELS + ['et', 'rf'])
# GPU_MODELS = [xgb_gpu, cat_gpu] （Cell 2で定義済み）
""")
