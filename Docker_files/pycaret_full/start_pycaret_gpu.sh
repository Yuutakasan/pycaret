#!/bin/bash
###############################################################################
# RAPIDS + PyCaret GPU環境 起動スクリプト
#
# 使い方:
#   ./start_pycaret_gpu.sh [build|start|stop|restart|logs|status|shell]
###############################################################################

set -e

# カラー設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# コンテナ名
CONTAINER_NAME="rapids_pycaret_notebook"
COMPOSE_FILE="docker-compose.yml"

# ロゴ表示
show_logo() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║         RAPIDS + PyCaret GPU Environment                  ║"
    echo "║         高速GPU機械学習環境                               ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# GPUチェック
check_gpu() {
    echo -e "${YELLOW}🔍 GPU環境をチェック中...${NC}"

    # NVIDIA GPUの存在確認
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}❌ エラー: nvidia-smiが見つかりません。NVIDIA GPUドライバーをインストールしてください。${NC}"
        exit 1
    fi

    # GPUステータス表示
    echo -e "${GREEN}✅ GPUが検出されました:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
}

# Docker環境チェック
check_docker() {
    echo -e "${YELLOW}🔍 Docker環境をチェック中...${NC}"

    # Dockerの存在確認
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ エラー: Dockerが見つかりません。Dockerをインストールしてください。${NC}"
        exit 1
    fi

    # Docker Composeの確認
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ エラー: Docker Composeが見つかりません。${NC}"
        exit 1
    fi

    # NVIDIA Docker Runtimeの確認
    if ! docker info 2>/dev/null | grep -q nvidia; then
        echo -e "${YELLOW}⚠️  警告: NVIDIA Docker Runtimeが設定されていない可能性があります。${NC}"
        echo -e "${YELLOW}   nvidia-docker2をインストールしてください。${NC}"
    fi

    echo -e "${GREEN}✅ Docker環境が正常です${NC}"
    echo ""
}

# ディレクトリ作成
create_directories() {
    echo -e "${YELLOW}📁 必要なディレクトリを作成中...${NC}"

    # 作業ディレクトリ
    mkdir -p work
    mkdir -p cache/cupy
    mkdir -p cache/numba
    mkdir -p jupyter_config

    # 権限設定（UID 1000 = rapids user）
    chmod 755 work cache cache/cupy cache/numba jupyter_config

    echo -e "${GREEN}✅ ディレクトリの作成が完了しました${NC}"
    echo ""
}

# ビルド関数
build_image() {
    echo -e "${YELLOW}🔨 Dockerイメージをビルド中...${NC}"
    echo -e "${YELLOW}   ※初回ビルドは10-20分かかる場合があります${NC}"

    # check_gpu_rapids_environment.pyが存在することを確認
    if [ ! -f "check_gpu_rapids_environment.py" ]; then
        echo -e "${RED}❌ エラー: check_gpu_rapids_environment.pyが見つかりません${NC}"
        echo -e "${YELLOW}   このファイルを同じディレクトリに配置してください${NC}"
        exit 1
    fi

    docker-compose build --no-cache

    echo -e "${GREEN}✅ イメージのビルドが完了しました${NC}"
}

# 起動関数
start_container() {
    echo -e "${YELLOW}🚀 コンテナを起動中...${NC}"

    docker-compose up -d

    # 起動待機
    echo -e "${YELLOW}⏳ JupyterLabの起動を待機中...${NC}"
    sleep 5

    # ステータス確認
    if docker ps | grep -q ${CONTAINER_NAME}; then
        echo -e "${GREEN}✅ コンテナが正常に起動しました${NC}"
        echo ""
        echo -e "${BLUE}📓 JupyterLabにアクセス:${NC}"
        echo -e "${GREEN}   http://localhost:8888${NC}"
        echo ""

        # ログから初期メッセージを表示
        docker logs ${CONTAINER_NAME} 2>&1 | head -n 20

        # マウント検査: /home/rapids/work が空なら（許可時のみ）応急コピーを実施
        echo ""
        echo -e "${YELLOW}🔎 workマウントを検査中...${NC}"
        if ! docker exec ${CONTAINER_NAME} bash -lc 'shopt -s nullglob dotglob; files=(/home/rapids/work/*); (( ${#files[@]} )) && echo HAS || echo EMPTY' | grep -q HAS; then
            if [ "${NO_FALLBACK_COPY:-0}" = "1" ]; then
                echo -e "${RED}❌ /home/rapids/work が空です（NO_FALLBACK_COPY=1 により応急コピーをスキップ）。${NC}"
                echo -e "${YELLOW}   対策: HOST_WORK_DIR を有効なパスに設定し、File Sharing 設定を見直してください。${NC}"
            else
                echo -e "${YELLOW}⚠️  /home/rapids/work が空です。ホストのworkをコンテナに一時コピーします${NC}"
                HOST_WORK_DIR_DEFAULT="../../work"
                HOST_WORK_DIR_EFFECTIVE="${HOST_WORK_DIR:-$HOST_WORK_DIR_DEFAULT}"
                if [ -d "$HOST_WORK_DIR_EFFECTIVE" ]; then
                    docker exec -u 0 ${CONTAINER_NAME} bash -lc 'mkdir -p /home/rapids/local_work'
                    docker cp "$HOST_WORK_DIR_EFFECTIVE"/. ${CONTAINER_NAME}:/home/rapids/local_work/
                    docker exec -u 0 ${CONTAINER_NAME} bash -lc 'chown -R rapids:conda /home/rapids/local_work'
                    echo -e "${GREEN}✅ 応急コピー完了。JupyterLab で /home/rapids/local_work を参照してください${NC}"
                    echo -e "${YELLOW}   恒久対策: HOST_WORK_DIR を設定するか、Docker DesktopのFile Sharingを有効にしてください${NC}"
                else
                    echo -e "${RED}❌ 応急コピー失敗: ホストのworkが見つかりません (${HOST_WORK_DIR_EFFECTIVE})${NC}"
                fi
            fi
        else
            echo -e "${GREEN}✅ /home/rapids/work にマウントが確認できました${NC}"
        fi
    else
        echo -e "${RED}❌ エラー: コンテナの起動に失敗しました${NC}"
        docker-compose logs
        exit 1
    fi
}

# 停止関数
stop_container() {
    echo -e "${YELLOW}🛑 コンテナを停止中...${NC}"
    docker-compose down
    echo -e "${GREEN}✅ コンテナが停止しました${NC}"
}

# ログ表示関数
show_logs() {
    echo -e "${YELLOW}📋 コンテナログを表示中...${NC}"
    docker-compose logs -f
}

# ステータス表示関数
show_status() {
    echo -e "${YELLOW}📊 コンテナステータス:${NC}"

    if docker ps | grep -q ${CONTAINER_NAME}; then
        echo -e "${GREEN}✅ コンテナは稼働中です${NC}"
        echo ""

        # リソース使用状況
        echo -e "${BLUE}リソース使用状況:${NC}"
        docker stats --no-stream ${CONTAINER_NAME}

        # GPU使用状況
        echo ""
        echo -e "${BLUE}GPU使用状況:${NC}"
        docker exec ${CONTAINER_NAME} nvidia-smi

    else
        echo -e "${RED}❌ コンテナは停止しています${NC}"
    fi
}

# シェルアクセス関数
shell_access() {
    echo -e "${YELLOW}🔧 コンテナにシェルアクセス中...${NC}"
    docker exec -it ${CONTAINER_NAME} /bin/bash
}

# メイン処理
main() {
    show_logo

    case "$1" in
        build)
            check_gpu
            check_docker
            create_directories
            build_image
            ;;
        start)
            check_gpu
            check_docker
            create_directories
            start_container
            ;;
        stop)
            stop_container
            ;;
        restart)
            stop_container
            sleep 2
            start_container
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        shell)
            shell_access
            ;;
        *)
            echo "使い方: $0 {build|start|stop|restart|logs|status|shell}"
            echo ""
            echo "コマンド:"
            echo "  build    - Dockerイメージをビルド"
            echo "  start    - コンテナを起動"
            echo "  stop     - コンテナを停止"
            echo "  restart  - コンテナを再起動"
            echo "  logs     - ログを表示"
            echo "  status   - ステータスを表示"
            echo "  shell    - コンテナにシェルアクセス"
            exit 1
    esac
}

# スクリプト実行
main "$@"
