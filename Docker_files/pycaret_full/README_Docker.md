# 🚀 RAPIDS + PyCaret GPU環境 - Dockerセットアップ

NVIDIA RAPIDS + PyCaret + AutoViz を使用したGPU対応の売上分析・予測環境です。

## 📋 含まれる機能

### 🤖 **AI売上予測ノートブック**
- **特徴量AutoViz_PyCaret_v1.ipynb**
  - フラグ別売上分析（降雨・週末・猛暑日等）
  - 商品カテゴリ選択機能
  - PyCaret全モデル比較（GPU加速）
  - 7日間売上予測
  - GPU/cuDF対応（10〜50倍高速化）

### 📊 **店舗ダッシュボード**
- **店舗別包括ダッシュボード_v6.1_提案強化.ipynb**
  - KPI分析、ABC分析、特徴量重要度
  - 売上最大化提案、アラート機能

---

## 🎯 クイックスタート

### 🚀 ビルド＆起動

```bash
cd Docker_files/pycaret_full/
./build_and_run.sh
```

ブラウザで http://localhost:8888 にアクセス

---

## 🔧 コンテナ管理

```bash
# 起動
docker-compose up -d

# 停止
docker-compose down

# ログ確認
docker-compose logs -f
```

詳細は [完全版README](README_Docker.md) を参照してください。
