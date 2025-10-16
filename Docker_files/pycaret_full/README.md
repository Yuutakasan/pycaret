提供されたDockerfileとdocker-compose.ymlを見ると、GPU環境が適切に構築されていない可能性があります。以下に、RAPIDSとPyCaret GPU対応の完全な環境構築ファイルを提供します。

## 1. Dockerfile（RAPIDS + PyCaret GPU対応版）## 2. docker-compose.yml（GPU対応設定）## 3. start_pycaret_gpu.sh（起動スクリプト）## 4. check_gpu_rapids_environment.py（環境チェックスクリプト）## セットアップと使用方法

### 1. 前提条件

- NVIDIA GPU（CUDA 12.8対応）
- NVIDIA GPUドライバー（525.60.13以降推奨）
- Docker（20.10.0以降）
- Docker Compose（v2推奨）
- nvidia-docker2（GPU対応）

### 2. セットアップ手順

```bash
# 1. すべてのファイルを同じディレクトリに配置
mkdir rapids-pycaret-gpu
cd rapids-pycaret-gpu

# 2. 上記の4つのファイルを作成・配置
# - Dockerfile
# - docker-compose.yml
# - start_pycaret_gpu.sh
# - check_gpu_rapids_environment.py

# 3. 実行権限を付与
chmod +x start_pycaret_gpu.sh

# 4. 環境を構築・起動（WSL/Windowsでworkが見えない場合の対処）
# 既定では ../../work を /home/rapids/work にバインドします。
# WSL2+Docker Desktop で D: のファイル共有が無効、または /mnt/d のバインドが空になる場合は、
# 以下いずれかで対処してください。
#
# A) Docker Desktop → Settings → Resources → File Sharing で該当ドライブ（例: D:）を許可
# B) リポジトリを WSL 側（/home/<user>/...）へ移動し、相対マウントを使用
# C) HOST_WORK_DIR 環境変数でホスト側の絶対パスを明示
#    - PowerShell:   $env:HOST_WORK_DIR = 'D:/github/pycaret/work'
#    - Bash(WSL):    export HOST_WORK_DIR=/mnt/d/github/pycaret/work  # 共有が有効な場合のみ
#
# その後、通常どおりビルド/起動します。
./start_pycaret_gpu.sh build  # 初回のみ（10-20分程度）
./start_pycaret_gpu.sh start  # コンテナ起動

# 古いファイルを使わずマウントのみで運用したい場合（応急コピー無効化）
# .env に設定: NO_FALLBACK_COPY=1
```

### 3. アクセス方法

起動後、以下のURLでJupyterLabにアクセス：
```
http://localhost:8888
```

### 4. GPU環境の確認

JupyterLabで新しいノートブックを作成し、以下を実行：

```python
# GPU環境チェック
!python /home/rapids/check_gpu_rapids_environment.py

# cudf.pandasの有効化
%load_ext cudf.pandas
print("✅ cudf.pandas有効化 - pandas APIがGPU加速されます")
```

### 5. コンテナ管理コマンド

```bash
# ステータス確認
./start_pycaret_gpu.sh status

# ログ確認
./start_pycaret_gpu.sh logs

# コンテナに入る
./start_pycaret_gpu.sh shell

# 再起動
./start_pycaret_gpu.sh restart

# 停止
./start_pycaret_gpu.sh stop
```

### 6. トラブルシューティング

**GPUが認識されない場合：**
```bash
# ホストでGPUを確認
nvidia-smi

# Docker GPUサポートを確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

**メモリ不足の場合：**
- docker-compose.ymlの`shm_size`を増やす（例：`16gb`）
- GPUメモリスピルを有効化：
  ```python
  import cudf
  cudf.set_option("spill", True)
  ```

**権限エラー/マウント空の場合：**
```bash
# A) Docker Desktop で対象ドライブの File Sharing を有効化
# B) HOST_WORK_DIR を設定（Windowsは D:/path 形式を推奨）
# C) それでも見えない場合の応急措置：work をコンテナ内にコピー
docker cp ../../work/. rapids_pycaret_notebook:/home/rapids/local_work
docker exec -u 0 rapids_pycaret_notebook chown -R rapids:conda /home/rapids/local_work
# JupyterLab から /home/rapids/local_work を参照してください
```

この環境により、提供されたコードが正常に動作し、GPUの高速化が有効になります。提供されたDockerfileとdocker-compose.ymlを見ると、GPU環境が適切に構築されていない可能性があります。以下に、RAPIDSとPyCaret GPU対応の完全な環境構築ファイルを提供します。

## 1. Dockerfile（RAPIDS + PyCaret GPU対応版）## 2. docker-compose.yml（GPU対応設定）## 3. start_pycaret_gpu.sh（起動スクリプト）## 4. check_gpu_rapids_environment.py（環境チェックスクリプト）## セットアップと使用方法

### 1. 前提条件

- NVIDIA GPU（CUDA 12.8対応）
- NVIDIA GPUドライバー（525.60.13以降推奨）
- Docker（20.10.0以降）
- Docker Compose（v2推奨）
- nvidia-docker2（GPU対応）

### 2. セットアップ手順

```bash
# 1. すべてのファイルを同じディレクトリに配置
mkdir rapids-pycaret-gpu
cd rapids-pycaret-gpu

# 2. 上記の4つのファイルを作成・配置
# - Dockerfile
# - docker-compose.yml
# - start_pycaret_gpu.sh
# - check_gpu_rapids_environment.py

# 3. 実行権限を付与
chmod +x start_pycaret_gpu.sh

# 4. 環境を構築・起動
./start_pycaret_gpu.sh build  # 初回のみ（10-20分程度）
./start_pycaret_gpu.sh start  # コンテナ起動
```

### 3. アクセス方法

起動後、以下のURLでJupyterLabにアクセス：
```
http://localhost:8888
```

### 4. GPU環境の確認

JupyterLabで新しいノートブックを作成し、以下を実行：

```python
# GPU環境チェック
!python /home/rapids/check_gpu_rapids_environment.py

# cudf.pandasの有効化
%load_ext cudf.pandas
print("✅ cudf.pandas有効化 - pandas APIがGPU加速されます")
```

### 5. コンテナ管理コマンド

```bash
# ステータス確認
./start_pycaret_gpu.sh status

# ログ確認
./start_pycaret_gpu.sh logs

# コンテナに入る
./start_pycaret_gpu.sh shell

# 再起動
./start_pycaret_gpu.sh restart

# 停止
./start_pycaret_gpu.sh stop
```

### 6. トラブルシューティング

**GPUが認識されない場合：**
```bash
# ホストでGPUを確認
nvidia-smi

# Docker GPUサポートを確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

**メモリ不足の場合：**
- docker-compose.ymlの`shm_size`を増やす（例：`16gb`）
- GPUメモリスピルを有効化：
  ```python
  import cudf
  cudf.set_option("spill", True)
  ```

**権限エラーの場合：**
```bash
# workディレクトリの権限を修正
sudo chown -R 1000:1000 work/
```

この環境により、提供されたコードが正常に動作し、GPUの高速化が有効になります。
