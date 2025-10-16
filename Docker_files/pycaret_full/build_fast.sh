#!/bin/bash
# 高速ビルドスクリプト（BuildKit有効化）

echo "🚀 高速ビルドを開始します..."
echo "   BuildKitとキャッシュを使用して高速化"

# BuildKit有効化
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# ビルド実行
docker-compose build --parallel

echo "✅ ビルド完了！"