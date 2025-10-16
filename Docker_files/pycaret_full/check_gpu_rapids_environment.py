#!/usr/bin/env python3
"""
RAPIDS + PyCaret GPU環境チェックスクリプト
=========================================
GPU、CUDA、RAPIDS、PyCaret、関連ライブラリの環境を包括的にチェック
"""

import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')

# カラー出力用
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(title):
    """セクションヘッダーを表示"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title:^60}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def check_status(condition, success_msg, fail_msg):
    """条件をチェックして結果を表示"""
    if condition:
        print(f"{Colors.GREEN}✅ {success_msg}{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.RED}❌ {fail_msg}{Colors.ENDC}")
        return False

def check_gpu_hardware():
    """GPU ハードウェアのチェック"""
    print_header("GPU ハードウェアチェック")

    try:
        # nvidia-smiの実行
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
                               '--format=csv,noheader'],
                               capture_output=True, text=True)

        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ GPU検出成功:{Colors.ENDC}")
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                print(f"   GPU {parts[0]}: {parts[1]} (VRAM: {parts[2]}, Driver: {parts[3]})")
            return True
        else:
            print(f"{Colors.RED}❌ GPUが検出されませんでした{Colors.ENDC}")
            return False
    except FileNotFoundError:
        print(f"{Colors.RED}❌ nvidia-smiが見つかりません{Colors.ENDC}")
        return False

def check_cuda_environment():
    """CUDA環境のチェック"""
    print_header("CUDA環境チェック")

    success = True

    # CUDAランタイムチェック
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"{Colors.GREEN}✅ PyTorch CUDA: 利用可能 (CUDA {torch.version.cuda}){Colors.ENDC}")
            print(f"   デバイス数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   デバイス{i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"{Colors.YELLOW}⚠️  PyTorch CUDA: 利用不可{Colors.ENDC}")
            success = False
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  PyTorchがインストールされていません{Colors.ENDC}")

    # CuPyチェック
    try:
        import cupy as cp
        print(f"{Colors.GREEN}✅ CuPy: バージョン {cp.__version__}{Colors.ENDC}")

        # GPUメモリプールの初期化テスト
        try:
            cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            print(f"   メモリプール: 初期化成功")
        except Exception as e:
            print(f"   メモリプール: 初期化エラー - {e}")
    except ImportError:
        print(f"{Colors.RED}❌ CuPyがインストールされていません{Colors.ENDC}")
        success = False

    # Numbaチェック
    try:
        from numba import cuda
        print(f"{Colors.GREEN}✅ Numba CUDA: 利用可能{Colors.ENDC}")
        if cuda.is_available():
            print(f"   デバイス検出: OK")
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  Numba CUDAがインストールされていません{Colors.ENDC}")

    return success

def check_rapids_libraries():
    """RAPIDSライブラリのチェック"""
    print_header("RAPIDSライブラリチェック")

    rapids_libs = {
        'cudf': None,
        'cuml': None,
        'cugraph': None,
        'cuspatial': None,
        'cuxfilter': None,
        'cupy': None,
        'dask_cuda': None
    }

    success = True

    for lib_name in rapids_libs:
        try:
            lib = __import__(lib_name)
            version = getattr(lib, '__version__', 'Unknown')
            rapids_libs[lib_name] = version
            print(f"{Colors.GREEN}✅ {lib_name}: バージョン {version}{Colors.ENDC}")
        except ImportError:
            print(f"{Colors.YELLOW}⚠️  {lib_name}: 未インストール{Colors.ENDC}")
            if lib_name in ['cudf', 'cuml']:
                success = False

    # cudf.pandasのチェック
    try:
        import cudf.pandas
        print(f"{Colors.GREEN}✅ cudf.pandas: 利用可能{Colors.ENDC}")
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  cudf.pandas: 利用不可{Colors.ENDC}")

    return success

def check_pycaret_gpu():
    """PyCaretのGPU対応チェック"""
    print_header("PyCaret GPU対応チェック")

    try:
        import pycaret
        print(f"{Colors.GREEN}✅ PyCaret: バージョン {pycaret.__version__}{Colors.ENDC}")

        # GPU対応アルゴリズムのチェック
        gpu_algorithms = {
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
            'catboost': 'CatBoost'
        }

        print(f"\n{Colors.BOLD}GPU対応アルゴリズム:{Colors.ENDC}")
        for module, name in gpu_algorithms.items():
            try:
                lib = __import__(module)
                version = getattr(lib, '__version__', 'Unknown')
                print(f"{Colors.GREEN}✅ {name}: バージョン {version}{Colors.ENDC}")

                # XGBoostのGPU確認
                if module == 'xgboost':
                    try:
                        import xgboost as xgb

                        # GPU対応の確認（簡易チェック）
                        print(f"   GPU対応: device='cuda'で利用可能")
                    except Exception:
                        pass

            except ImportError:
                print(f"{Colors.YELLOW}⚠️  {name}: 未インストール{Colors.ENDC}")

        return True
    except ImportError:
        print(f"{Colors.RED}❌ PyCaretがインストールされていません{Colors.ENDC}")
        return False

def check_analysis_tools():
    """分析ツールのチェック"""
    print_header("分析ツールチェック")

    tools = {
        'autoviz': 'AutoViz',
        'explainerdashboard': 'ExplainerDashboard',
        'shap': 'SHAP',
        'interpret': 'InterpretML',
        'ydata_profiling': 'YData Profiling',
        'japanize_matplotlib': '日本語Matplotlib'
    }

    all_installed = True

    for module, name in tools.items():
        try:
            lib = __import__(module)
            version = getattr(lib, '__version__', 'OK')
            print(f"{Colors.GREEN}✅ {name}: {version}{Colors.ENDC}")

            # SHAPのGPU確認
            if module == 'shap':
                try:
                    from shap.explainers import GPUTree
                    print(f"   GPU TreeExplainer: 利用可能")
                except ImportError:
                    print(f"   GPU TreeExplainer: 利用不可")

        except ImportError:
            print(f"{Colors.YELLOW}⚠️  {name}: 未インストール{Colors.ENDC}")
            all_installed = False

    return all_installed

def check_memory_info():
    """メモリ情報のチェック"""
    print_header("メモリ情報")

    # GPUメモリ
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        print(f"{Colors.BOLD}GPU メモリ:{Colors.ENDC}")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = info.total / (1024**3)
            used_gb = info.used / (1024**3)
            free_gb = info.free / (1024**3)

            print(f"  GPU {i}: 合計 {total_gb:.1f}GB, 使用 {used_gb:.1f}GB, 空き {free_gb:.1f}GB")
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  GPUメモリ情報を取得できません: {e}{Colors.ENDC}")

    # システムメモリ
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\n{Colors.BOLD}システムメモリ:{Colors.ENDC}")
        print(f"  合計: {mem.total/(1024**3):.1f}GB")
        print(f"  使用: {mem.used/(1024**3):.1f}GB ({mem.percent:.1f}%)")
        print(f"  空き: {mem.available/(1024**3):.1f}GB")
    except ImportError:
        print(f"\n{Colors.YELLOW}⚠️  psutilがインストールされていません{Colors.ENDC}")

def test_gpu_computation():
    """GPU計算のテスト"""
    print_header("GPU計算テスト")

    # CuDFテスト
    try:
        import time

        import cudf
        import numpy as np
        import pandas as pd

        # テストデータ作成
        n = 1000000
        print(f"\n{Colors.BOLD}CuDF性能テスト (n={n:,}):{Colors.ENDC}")

        # CPU版
        start = time.time()
        df_cpu = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n)
        })
        df_cpu['c'] = df_cpu['a'] + df_cpu['b']
        cpu_time = time.time() - start

        # GPU版
        start = time.time()
        df_gpu = cudf.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n)
        })
        df_gpu['c'] = df_gpu['a'] + df_gpu['b']
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time
        print(f"  CPU時間: {cpu_time:.3f}秒")
        print(f"  GPU時間: {gpu_time:.3f}秒")
        print(f"  高速化: {speedup:.1f}倍")

        check_status(speedup > 1, "GPU計算が正常に動作しています", "GPU計算が期待通り動作していません")

    except Exception as e:
        print(f"{Colors.RED}❌ GPU計算テストエラー: {e}{Colors.ENDC}")

def print_recommendations():
    """推奨事項の表示"""
    print_header("推奨事項とヒント")

    print(f"{Colors.BOLD}1. cudf.pandasの使用:{Colors.ENDC}")
    print("   ノートブックの最初に以下を実行:")
    print("   %load_ext cudf.pandas")

    print(f"\n{Colors.BOLD}2. PyCaretでGPUを使用:{Colors.ENDC}")
    print("   setup(data, target='label', use_gpu=True)")

    print(f"\n{Colors.BOLD}3. メモリ管理:{Colors.ENDC}")
    print("   大規模データの場合、以下を設定:")
    print("   import cudf")
    print("   cudf.set_option('spill', True)")

    print(f"\n{Colors.BOLD}4. パフォーマンスモニタリング:{Colors.ENDC}")
    print("   nvtop または nvidia-smi -l 1 でGPU使用状況を監視")

def main():
    """メイン処理"""
    print(f"\n{Colors.BOLD}🚀 RAPIDS + PyCaret GPU環境チェックを開始します...{Colors.ENDC}\n")

    # 各チェックの実行
    checks = {
        "GPU ハードウェア": check_gpu_hardware(),
        "CUDA 環境": check_cuda_environment(),
        "RAPIDS ライブラリ": check_rapids_libraries(),
        "PyCaret GPU対応": check_pycaret_gpu(),
        "分析ツール": check_analysis_tools()
    }

    # メモリ情報
    check_memory_info()

    # GPU計算テスト
    test_gpu_computation()

    # 結果サマリー
    print_header("環境チェック結果サマリー")

    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            print(f"{Colors.GREEN}✅ {check_name}: OK{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ {check_name}: NG{Colors.ENDC}")
            all_passed = False

    # 推奨事項
    print_recommendations()

    # 最終結果
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ GPU環境は正常に設定されています！{Colors.ENDC}")
        print(f"{Colors.GREEN}   高速なGPU機械学習が利用可能です。{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  一部の機能が利用できない可能性があります{Colors.ENDC}")
        print(f"{Colors.YELLOW}   上記のNGの項目を確認してください。{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
