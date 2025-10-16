#!/usr/bin/env python3
"""
GPU環境確認スクリプト
PyCaret + RAPIDS環境の動作確認
"""

import subprocess
import sys

print("=" * 80)
print("🔍 GPU環境確認スクリプト")
print("=" * 80)

# 1. Python環境確認
print("\n1. Python環境:")
print(f"   Python version: {sys.version}")
print(f"   実行パス: {sys.executable}")

# 2. CUDA確認
print("\n2. CUDA環境:")
try:
    cuda_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
    print(f"   NVCC version: {cuda_version.splitlines()[3]}")
except:
    print("   ❌ NVCC not found")

try:
    nvidia_smi = subprocess.check_output(
        ["nvidia-smi"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    print("   ✅ nvidia-smi 実行可能")
    # GPUの簡易情報を表示
    for line in nvidia_smi.splitlines():
        if "NVIDIA" in line and "Driver" in line:
            print(f"   {line.strip()}")
except:
    print("   ❌ nvidia-smi 実行不可")

# 3. 必要なパッケージの確認
print("\n3. パッケージインポート確認:")

packages = {
    "numpy": "NumPy",
    "pandas": "Pandas",
    "sklearn": "Scikit-learn",
    "pycaret": "PyCaret",
    "cudf": "cuDF (RAPIDS)",
    "cuml": "cuML (RAPIDS)",
    "japanize_matplotlib": "日本語Matplotlib",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "explainerdashboard": "ExplainerDashboard",
    "shap": "SHAP",
}

import_results = {}

for module, name in packages.items():
    try:
        if module == "sklearn":
            import sklearn

            version = sklearn.__version__
        else:
            imported = __import__(module)
            version = (
                imported.__version__ if hasattr(imported, "__version__") else "unknown"
            )
        import_results[name] = f"✅ {version}"
        print(f"   {name}: ✅ (version: {version})")
    except ImportError as e:
        import_results[name] = f"❌ {str(e)}"
        print(f"   {name}: ❌")

# 4. GPU利用可能性確認
print("\n4. GPU利用可能性:")

# PyTorch GPU確認
try:
    import torch

    if torch.cuda.is_available():
        print(f"   PyTorch CUDA: ✅ (GPU数: {torch.cuda.device_count()})")
        print(f"   GPU名: {torch.cuda.get_device_name(0)}")
    else:
        print("   PyTorch CUDA: ❌")
except:
    print("   PyTorch: Not installed")

# cuML GPU確認
try:
    import cuml

    print("   cuML: ✅ GPU対応")

    # 簡単なテスト
    import cudf
    import numpy as np
    from cuml.ensemble import RandomForestRegressor as cuRF

    # データ生成
    n_samples = 1000
    n_features = 20
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.rand(n_samples).astype(np.float32)

    # cuDFデータフレームに変換
    X_cudf = cudf.DataFrame(X)
    y_cudf = cudf.Series(y)

    # モデル訓練
    model = cuRF(n_estimators=10)
    model.fit(X_cudf, y_cudf)

    print("   cuML RandomForest: ✅ 動作確認OK")
except Exception as e:
    print(f"   cuML テストエラー: {str(e)}")

# 5. PyCaret GPU設定確認
print("\n5. PyCaret GPU設定:")
try:
    # ダミーデータで設定確認
    import pandas as pd

    from pycaret.regression import setup
    from pycaret.utils import check_metric

    dummy_data = pd.DataFrame(
        {"x1": np.random.rand(100), "x2": np.random.rand(100), "y": np.random.rand(100)}
    )

    # GPU使用を試行
    try:
        s = setup(
            dummy_data,
            target="y",
            session_id=123,
            use_gpu=True,
            html=False,
            verbose=False,
            silent=True,
        )
        print("   PyCaret GPU設定: ✅ 有効")
    except:
        s = setup(
            dummy_data,
            target="y",
            session_id=123,
            use_gpu=False,
            html=False,
            verbose=False,
            silent=True,
        )
        print("   PyCaret GPU設定: ⚠️ GPUは使用不可、CPUモードで動作")

except Exception as e:
    print(f"   PyCaret設定エラー: {str(e)}")

# 6. 日本語フォント確認
print("\n6. 日本語フォント確認:")
try:
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # 利用可能なフォント確認
    fonts = [f.name for f in fm.fontManager.ttflist]
    jp_fonts = [f for f in fonts if "IPA" in f or "Noto" in f or "日本" in f]

    if jp_fonts:
        print(f"   日本語フォント: ✅ {len(jp_fonts)}個検出")
        print(f"   例: {', '.join(jp_fonts[:3])}")
    else:
        print("   日本語フォント: ❌ 見つかりません")

    # japanize_matplotlib動作確認
    import japanize_matplotlib

    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.title("日本語タイトルテスト")
    plt.xlabel("横軸")
    plt.ylabel("縦軸")
    plt.close()
    print("   japanize_matplotlib: ✅ 動作確認OK")

except Exception as e:
    print(f"   フォントエラー: {str(e)}")

# 7. メモリ情報
print("\n7. システムリソース:")
try:
    import psutil

    # CPU情報
    print(f"   CPU使用率: {psutil.cpu_percent()}%")
    print(f"   CPUコア数: {psutil.cpu_count()}")

    # メモリ情報
    memory = psutil.virtual_memory()
    print(f"   メモリ使用率: {memory.percent}%")
    print(f"   利用可能メモリ: {memory.available / (1024**3):.1f} GB")

    # GPU メモリ（nvidia-ml-py経由）
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(
                f"   GPU {i} メモリ: {info.used / (1024**3):.1f}/{info.total / (1024**3):.1f} GB"
            )
    except:
        pass

except:
    print("   リソース情報取得エラー")

print("\n" + "=" * 80)
print("環境確認完了！")

# サマリー
errors = [name for name, result in import_results.items() if "❌" in result]
if errors:
    print(f"\n⚠️ 以下のパッケージがインポートできません: {', '.join(errors)}")
    print("Dockerfileの再ビルドが必要かもしれません。")
else:
    print("\n✅ すべてのパッケージが正常にインポートできました！")
    print("PyCaret GPU環境の準備が整いました。")

print("=" * 80)
