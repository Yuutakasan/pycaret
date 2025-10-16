#!/bin/bash
###############################################################################
# RAPIDS + PyCaret GPU環境 - ビルド＆起動スクリプト
###############################################################################

set -e

echo "=========================================="
echo "🚀 RAPIDS + PyCaret GPU環境"
echo "=========================================="
echo ""

# カレントディレクトリを確認
if [ ! -f "Dockerfile" ]; then
    echo "❌ エラー: Dockerfileが見つかりません"
    echo "   このスクリプトは Docker_files/pycaret_full/ で実行してください"
    exit 1
fi

# GPU確認
echo "🖥️  GPU検出中..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  警告: nvidia-smi が見つかりません"
    echo "   GPUドライバがインストールされていない可能性があります"
    echo ""
fi

# ビルドオプション
echo "📦 Dockerイメージをビルドします..."
echo ""
read -p "キャッシュを使用しますか？ (y/n, デフォルト: y): " use_cache
use_cache=${use_cache:-y}

if [ "$use_cache" = "n" ] || [ "$use_cache" = "N" ]; then
    echo "🔄 キャッシュなしでビルド（時間がかかります）..."
    DOCKER_BUILDKIT=1 docker-compose build --no-cache
else
    echo "⚡ キャッシュを使用してビルド..."
    DOCKER_BUILDKIT=1 docker-compose build
fi

echo ""
echo "✅ ビルド完了"
echo ""

# 起動確認
read -p "コンテナを起動しますか？ (y/n, デフォルト: y): " start_container
start_container=${start_container:-y}

if [ "$start_container" = "y" ] || [ "$start_container" = "Y" ]; then
    echo ""
    echo "🚀 コンテナを起動します..."
    docker-compose up -d

    echo ""
    echo "⏳ コンテナの起動を待機中..."
    sleep 5

    echo ""
    echo "=========================================="
    echo "✅ 起動完了"
    echo "=========================================="
    echo ""
    echo "📓 JupyterLabにアクセス:"
    echo "   http://localhost:8888"
    echo ""
    echo "📂 利用可能なノートブック:"
    echo "   - 特徴量AutoViz_PyCaret_v1.ipynb (AI売上予測)"
    echo "   - 店舗別包括ダッシュボード_v6.1_提案強化.ipynb (店舗ダッシュボード)"
    echo ""
    echo "📖 ドキュメント:"
    echo "   - docs/GPU環境セットアップガイド.md"
    echo "   - docs/GPU対応と売上最大化分析_完了報告.md"
    echo ""
    echo "🔧 コンテナ管理:"
    echo "   停止: docker-compose down"
    echo "   ログ: docker-compose logs -f"
    echo "   再起動: docker-compose restart"
    echo ""
else
    echo ""
    echo "ℹ️  後で起動する場合は以下を実行してください:"
    echo "   docker-compose up -d"
    echo ""
fi
