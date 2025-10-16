# 🐳 Docker使用ガイド - RAPIDS + PyCaret GPU環境

## 🚀 最速スタート（3ステップ）

### ステップ1: ディレクトリ移動

```bash
cd Docker_files/pycaret_full/
```

### ステップ2: ビルド＆起動

```bash
./build_and_run.sh
```

### ステップ3: ブラウザでアクセス

```
http://localhost:8888
```

**これだけです！** JupyterLabが開き、すべてのノートブックが利用可能になります。

---

## 📋 含まれるノートブック

Docker環境には以下がプリインストールされています：

1. **特徴量AutoViz_PyCaret_v1.ipynb**
   - フラグ別売上分析
   - 商品カテゴリ選択
   - PyCaret GPU加速
   - 7日間売上予測

2. **店舗別包括ダッシュボード_v6.1_提案強化.ipynb**
   - KPI/ABC/特徴量/提案/アラート

3. **font_setup.py**
   - 日本語フォント自動設定

4. **docs/**
   - GPU環境セットアップガイド
   - GPU対応と売上最大化分析 完了報告

---

## 🔧 基本的なDocker操作

### コンテナの起動

```bash
cd Docker_files/pycaret_full/
docker-compose up -d
```

**出力例**:
```
Creating network "pycaret_full_rapids-network" with driver "bridge"
Creating rapids_pycaret_notebook ... done
```

### コンテナの停止

```bash
docker-compose down
```

### コンテナの再起動

```bash
docker-compose restart
```

### ログの確認

```bash
# リアルタイム表示
docker-compose logs -f

# 最新100行
docker-compose logs --tail=100
```

---

## 🖥️ GPU確認

### ホストからGPU確認

```bash
nvidia-smi
```

**出力例**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 3090    Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    29W / 350W |    512MiB / 24576MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### コンテナ内からGPU確認

```bash
docker-compose exec rapids-pycaret-gpu nvidia-smi
```

### コンテナ内でPython確認

```bash
docker-compose exec rapids-pycaret-gpu python -c "import torch; print(torch.cuda.is_available())"
```

**True** が表示されればGPU検出成功です。

---

## 📂 ファイルの配置

### ホスト → コンテナ

DockerfileによりBUILD時に以下がコピーされます：

```
work/font_setup.py → /home/rapids/work/font_setup.py
work/特徴量AutoViz_PyCaret_v1.ipynb → /home/rapids/work/
work/店舗別包括ダッシュボード_v6.1_提案強化.ipynb → /home/rapids/work/
work/docs/ → /home/rapids/work/docs/
```

### データファイルの追加

CSVファイルなどを追加する場合：

**方法1: ホストのworkディレクトリに配置**

```bash
# ホスト側
cp your_data.csv /path/to/pycaret/work/

# コンテナ再起動（ボリュームマウントで自動反映）
docker-compose restart
```

**方法2: 実行中のコンテナにコピー**

```bash
docker cp your_data.csv rapids_pycaret_notebook:/home/rapids/work/
```

---

## 🔄 イメージの更新

### ノートブックを更新した場合

```bash
# 1. コンテナを停止
docker-compose down

# 2. イメージを再ビルド
DOCKER_BUILDKIT=1 docker-compose build

# 3. 再起動
docker-compose up -d
```

### 依存パッケージを追加した場合

Dockerfileを編集してから：

```bash
docker-compose down
DOCKER_BUILDKIT=1 docker-compose build --no-cache
docker-compose up -d
```

---

## 🐛 よくある問題と解決方法

### 問題1: "bind: address already in use"

**原因**: ポート8888が既に使用されている

**解決方法**:

```bash
# 使用中のプロセスを確認
sudo lsof -i:8888

# ポートを変更する場合（docker-compose.yml編集）
ports:
  - "9999:8888"  # 9999に変更
```

---

### 問題2: "ERROR: No matching distribution found for torch"

**原因**: Pythonバージョンが古い、またはビルドキャッシュが破損

**解決方法**:

```bash
# キャッシュをクリアして再ビルド
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

### 問題3: GPU が認識されない

**原因**: NVIDIA Dockerがインストールされていない

**解決方法**:

```bash
# NVIDIA Dockerをインストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 動作確認
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

### 問題4: "Permission denied" エラー

**原因**: ボリュームのパーミッション問題

**解決方法**:

```bash
# workディレクトリの権限を修正
cd /path/to/pycaret
sudo chown -R $USER:$USER work/

# コンテナ再起動
docker-compose restart
```

---

## 📊 パフォーマンス確認

JupyterLabで特徴量AutoViz_PyCaret_v1.ipynbを開き、Cell-1を実行すると：

### ✅ GPU検出成功

```
✅ NVIDIA GPU検出: GeForce RTX 3090
   CUDA Version: 12.8
   GPU Memory: 24.0 GB
✅ cuDF (GPU Pandas) 対応: 可能
   💡 大規模データ処理が10～50倍高速化されます
🚀 GPU使用: 有効（AI学習 2～10倍 + データ処理 10～50倍 高速化）
```

### ⚠️ GPU未検出（CPUモード）

CPUモードでも完全に動作します（実行時間は長くなります）。

---

## 💾 データの永続化

以下のディレクトリは **ボリュームマウント** されているため、コンテナを削除してもデータは保持されます：

- `/home/rapids/work` → ホストの `work/`
- キャッシュディレクトリ（cupy、numba）

**安全にコンテナを削除**:

```bash
docker-compose down  # データは work/ に残る
```

**データも含めて完全削除**:

```bash
docker-compose down -v  # ⚠️ ボリュームも削除される
```

---

## 🚀 本番環境でのDocker使用

### docker-compose.override.yml

開発環境と本番環境で設定を分けたい場合：

```yaml
# docker-compose.override.yml（本番環境用）
version: '3.8'
services:
  rapids-pycaret-gpu:
    environment:
      - JUPYTER_TOKEN=your_production_token  # 本番用トークン
    restart: always  # 自動再起動
    deploy:
      resources:
        limits:
          memory: 64G  # メモリ制限
```

起動：

```bash
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

---

## 📞 サポート

Docker環境で問題が発生した場合は、以下の情報を添えてご連絡ください：

```bash
# 環境情報を取得
docker version > docker_info.txt
docker-compose version >> docker_info.txt
nvidia-smi >> docker_info.txt
docker-compose logs >> docker_info.txt
```

---

**最終更新日**: 2025年10月15日
**バージョン**: v1.0
