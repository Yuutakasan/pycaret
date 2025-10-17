#!/usr/bin/env python3
"""
Step5_CategoryWise_Compare_with_Overfitting.ipynb にGPU有効化セルを追加
"""

import json
from pathlib import Path

# GPU有効化セル
gpu_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'gpu_acceleration',
    'metadata': {},
    'outputs': [],
    'source': [
        '# ========================================\n',
        '# 🚀 GPU高速化設定\n',
        '# ========================================\n',
        '\n',
        'import warnings\n',
        'warnings.filterwarnings(\'ignore\')\n',
        '\n',
        'print(\'\\n\' + \'=\'*80)\n',
        'print(\'🚀 GPU高速化設定\')\n',
        'print(\'=\'*80)\n',
        '\n',
        '# GPU対応モデルの準備\n',
        'from xgboost import XGBRegressor\n',
        'from catboost import CatBoostRegressor\n',
        'from lightgbm import LGBMRegressor\n',
        '\n',
        '# XGBoost GPU設定\n',
        'xgb_gpu = XGBRegressor(\n',
        '    tree_method=\'hist\',        # GPUには\'hist\'を使用\n',
        '    device=\'cuda\',             # CUDA有効化\n',
        '    n_estimators=1000,\n',
        '    learning_rate=0.05,\n',
        '    max_depth=6,\n',
        '    random_state=123,\n',
        '    n_jobs=-1\n',
        ')\n',
        '\n',
        '# CatBoost GPU設定\n',
        'cat_gpu = CatBoostRegressor(\n',
        '    task_type=\'GPU\',           # GPU有効化\n',
        '    devices=\'0\',               # GPU 0番を使用\n',
        '    iterations=1000,\n',
        '    learning_rate=0.05,\n',
        '    depth=6,\n',
        '    random_state=123,\n',
        '    verbose=False\n',
        ')\n',
        '\n',
        '# LightGBM GPU設定（利用可能な場合）\n',
        'try:\n',
        '    lgbm_gpu = LGBMRegressor(\n',
        '        device=\'gpu\',\n',
        '        gpu_platform_id=0,\n',
        '        gpu_device_id=0,\n',
        '        n_estimators=1000,\n',
        '        learning_rate=0.05,\n',
        '        num_leaves=31,\n',
        '        random_state=123,\n',
        '        n_jobs=-1,\n',
        '        verbose=-1\n',
        '    )\n',
        '    GPU_MODELS = [xgb_gpu, cat_gpu, lgbm_gpu]\n',
        '    print(\'✅ GPU対応モデル: XGBoost, CatBoost, LightGBM\')\n',
        'except Exception as e:\n',
        '    GPU_MODELS = [xgb_gpu, cat_gpu]\n',
        '    print(\'✅ GPU対応モデル: XGBoost, CatBoost\')\n',
        '    print(f\'⚠️ LightGBM GPU: 利用不可 ({str(e)})\')\n',
        '\n',
        '# GPU使用フラグ\n',
        'USE_GPU = True\n',
        '\n',
        'print(f\'\\n💡 使用方法:\')\n',
        'print(\'  compare_models(include=[xgb_gpu, cat_gpu])  # GPU高速化\')\n',
        'print(\'  または\')\n',
        'print(\'  compare_models(include=GPU_MODELS)\')\n',
        '\n',
        '# GPU情報表示\n',
        'try:\n',
        '    import torch\n',
        '    if torch.cuda.is_available():\n',
        '        print(f\'\\n🎮 GPU情報:\')\n',
        '        print(f\'  GPU数: {torch.cuda.device_count()}\')\n',
        '        print(f\'  GPU名: {torch.cuda.get_device_name(0)}\')\n',
        '        print(f\'  CUDAバージョン: {torch.version.cuda}\')\n',
        '        \n',
        '        # メモリ情報\n',
        '        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9\n',
        '        mem_allocated = torch.cuda.memory_allocated(0) / 1e9\n',
        '        print(f\'  総メモリ: {mem_total:.1f} GB\')\n',
        '        print(f\'  使用中: {mem_allocated:.1f} GB\')\n',
        '    else:\n',
        '        print(\'\\n⚠️ CUDA GPUが検出されませんでした\')\n',
        'except ImportError:\n',
        '    print(\'\\n⚠️ PyTorchがインストールされていません\')\n',
        '\n',
        'print(\'\\n✅ GPU高速化設定完了\')\n'
    ]
}

# ノートブック読み込み
notebook_path = Path('/mnt/d/github/pycaret/work/Step5_CategoryWise_Compare_with_Overfitting.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# setupセル（Cell 1）の後に挿入
nb['cells'].insert(2, gpu_cell)

# 保存
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ GPU高速化セルを追加しました（Cell 2）")
print("✅ 保存完了:", notebook_path)
print("\n使用方法:")
print("  Cell 2でGPU_MODELSを定義")
print("  Cell 5-11でcompare_models(include=GPU_MODELS)を使用")
