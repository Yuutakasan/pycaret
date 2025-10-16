# 🚀 GPU環境セットアップガイド

## 📋 現在の状況

現在、以下のメッセージが表示されています：

```
ℹ️ PyTorch not installed (GPU detection skipped)
ℹ️ LightGBM: CPU mode
ℹ️ XGBoost: CPU mode
ℹ️ CatBoost not installed
ℹ️ GPU使用: 無効（CPUモードで実行）
```

これは **PyTorchがインストールされていない** ため、GPU検出ができていません。

---

## ✅ GPU環境を有効にする方法

### 前提条件

1. **NVIDIA GPUが搭載されている** こと（GeForce、RTX、Quadro、Teslaなど）
2. **NVIDIA GPUドライバ** がインストール済み
3. **CUDA Toolkit** がインストール済み（推奨: CUDA 11.8 または 12.x）

### ステップ1: CUDA対応PyTorchのインストール

```bash
# CUDA 11.8対応版（推奨）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# または CUDA 12.1対応版
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**インストール確認**:
```python
import torch
print(torch.cuda.is_available())  # Trueになれば成功
print(torch.cuda.get_device_name(0))  # GPU名が表示される
```

---

### ステップ2: cuDF（GPU Pandas）のインストール（オプション）

大規模データ処理を10〜50倍高速化します。

```bash
# CUDA 12.x の場合
pip install cudf-cu12

# CUDA 11.x の場合
pip install cudf-cu11
```

**インストール確認**:
```python
import cudf
print("cuDF installed successfully")
```

---

### ステップ3: GPU対応機械学習ライブラリのインストール

#### LightGBM（GPU版）

```bash
# GPUサポート付きでインストール
pip install lightgbm --install-option=--gpu

# または conda経由（推奨）
conda install -c conda-forge lightgbm
```

#### XGBoost（GPU自動対応）

```bash
pip install xgboost
# XGBoostはCUDAが検出されると自動的にGPU対応になります
```

#### CatBoost（GPU自動対応）

```bash
pip install catboost
# CatBoostもCUDAが検出されると自動的にGPU対応になります
```

---

### ステップ4: ノートブックを再実行

上記のインストール後、Jupyter Notebookを再起動して、Cell-1を再実行してください：

```bash
# Jupyter Notebookを一度停止
# Ctrl+C でカーネルを停止

# 再起動
jupyter notebook
```

ノートブックで「Kernel」→「Restart & Run All」を実行すると、以下のように表示されます：

```
✅ NVIDIA GPU検出: GeForce RTX 3090
   CUDA Version: 11.8
   GPU Memory: 24.0 GB
✅ LightGBM GPU対応: 可能
✅ XGBoost GPU対応: 可能
✅ CatBoost GPU対応: 可能
✅ cuDF (GPU Pandas) 対応: 可能
   💡 大規模データ処理が10～50倍高速化されます

🚀 GPU使用: 有効（AI学習 2～10倍 + データ処理 10～50倍 高速化）
   Device: cuda
```

---

## 🔧 トラブルシューティング

### 問題1: "CUDA not available" と表示される

**原因**: NVIDIA GPUドライバまたはCUDA Toolkitがインストールされていない

**解決方法**:

1. **NVIDIA GPUドライバのインストール**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install nvidia-driver-535

   # 確認
   nvidia-smi
   ```

2. **CUDA Toolkitのインストール**:
   ```bash
   # Ubuntu 22.04の場合（CUDA 12.1）
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda
   ```

3. **環境変数の設定**:
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

---

### 問題2: PyTorchインストールが失敗する

**原因**: Pythonバージョンが古い、またはパッケージコンフリクト

**解決方法**:

```bash
# 仮想環境を作成（推奨）
python -m venv gpu_env
source gpu_env/bin/activate  # Linux/Mac
# または
gpu_env\Scripts\activate  # Windows

# PyTorchを再インストール
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 問題3: LightGBM GPU版のインストールが失敗する

**原因**: OpenCL、Boost、CMakeなどの依存関係が不足

**解決方法**:

```bash
# 必要な依存関係をインストール
sudo apt-get install cmake libboost-dev libboost-system-dev libboost-filesystem-dev opencl-headers ocl-icd-opencl-dev

# LightGBMを再インストール
pip install lightgbm --install-option=--gpu

# または conda経由（より簡単）
conda install -c conda-forge lightgbm
```

---

### 問題4: cuDFのインストールが失敗する

**原因**: CUDA バージョンが一致していない

**解決方法**:

1. **CUDA バージョンを確認**:
   ```bash
   nvidia-smi
   # 右上に "CUDA Version: 12.1" などと表示される
   ```

2. **一致するバージョンをインストール**:
   ```bash
   # CUDA 12.x の場合
   pip install cudf-cu12

   # CUDA 11.x の場合
   pip install cudf-cu11
   ```

---

## 🎯 GPU非対応環境でも動作します

**重要**: GPU環境がなくても、ノートブックは **CPU モードで正常に動作** します。

- GPU検出に失敗した場合、自動的にCPUモードで実行されます
- すべての機能が利用可能（ただし実行時間は長くなります）
- GPU環境を用意できない場合は、そのままCPUモードで使用してください

**CPUモードでの実行時間目安**:
- PyCaret全モデル比較: **5〜10分**
- 予測実行（全商品×7日）: **2〜5分**

**GPUモードでの実行時間目安**:
- PyCaret全モデル比較: **1〜2分** （2〜5倍高速化）
- 予測実行（全商品×7日）: **0.5〜1分** （4〜5倍高速化）

---

## 📚 推奨GPU環境

### 最小構成
- **GPU**: NVIDIA GeForce GTX 1060 (6GB) 以上
- **VRAM**: 6GB以上
- **CUDA**: 11.8 以上

### 推奨構成
- **GPU**: NVIDIA RTX 3060 (12GB) / RTX 3070 / RTX 3080
- **VRAM**: 12GB以上
- **CUDA**: 12.1

### 最適構成（大規模データ）
- **GPU**: NVIDIA RTX 4090 (24GB) / A100 (40GB)
- **VRAM**: 24GB以上
- **CUDA**: 12.1

---

## 🔍 GPU検出の確認方法

ノートブックのCell-1を実行後、以下を確認してください：

### ✅ **GPU検出成功の場合**

```
✅ NVIDIA GPU検出: GeForce RTX 3090
   CUDA Version: 11.8
   GPU Memory: 24.0 GB
✅ LightGBM GPU対応: 可能
✅ XGBoost GPU対応: 可能
✅ CatBoost GPU対応: 可能
✅ cuDF (GPU Pandas) 対応: 可能

🚀 GPU使用: 有効（AI学習 2～10倍 + データ処理 10～50倍 高速化）
   Device: cuda
```

### ⚠️ **GPU検出失敗の場合（現在の状態）**

```
ℹ️ PyTorch not installed (GPU detection skipped)
ℹ️ LightGBM: CPU mode
ℹ️ XGBoost: CPU mode
ℹ️ CatBoost not installed

ℹ️ GPU使用: 無効（CPUモードで実行）
```

この場合は、上記の **ステップ1〜3** を実行してください。

---

## 💡 よくある質問

### Q1: GPU環境なしでも使えますか？
**A**: はい、CPUモードで完全に動作します。実行時間が長くなるだけです。

### Q2: Google ColabやKaggle Kernelで使えますか？
**A**: はい、Colabは無料でGPUが使えます（ランタイム→ランタイムのタイプを変更→GPU）。

### Q3: Macでは使えませんか？
**A**: Macには NVIDIA GPU が搭載されていないため、CPUモードのみです。

### Q4: cuDFは必須ですか？
**A**: いいえ、オプションです。大規模データ（100万行以上）でなければ、通常のpandasで十分です。

---

## 📞 サポート

GPU環境のセットアップでお困りの場合は、以下の情報を添えてご連絡ください：

1. **OS**: Ubuntu 22.04、Windows 11 など
2. **GPU型番**: `nvidia-smi` の出力
3. **CUDA バージョン**: `nvidia-smi` の右上に表示
4. **Pythonバージョン**: `python --version`
5. **エラーメッセージ**: 完全なエラーログ

---

**最終更新日**: 2025年10月15日
**バージョン**: v1.0
