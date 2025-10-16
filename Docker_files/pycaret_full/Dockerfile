###############################################################################
# RAPIDS + PyCaret GPU環境 - 超高速ビルド版（最適化済み）
# BuildKitの並列ビルドとキャッシュマウント + レイヤー分離で高速化
###############################################################################

# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/rapidsai/base:25.06-cuda12.8-py3.11

# 環境変数を最初に設定（キャッシュ効率化）
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=ja_JP.UTF-8 \
    LC_ALL=ja_JP.UTF-8 \
    RAPIDS_NO_INITIALIZE=1 \
    CUDA_VISIBLE_DEVICES=all

# rootで日本語環境設定（aptキャッシュマウントで高速化）
USER root
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        locales fonts-ipafont-gothic fonts-ipafont-mincho \
        fonts-noto-cjk fontconfig vim nano htop nvtop && \
    echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen && \
    fc-cache -fv

# rapidsユーザーに切り替え
USER rapids
WORKDIR /home/rapids

# Step 1: mamba パッケージのインストール（変更頻度低・キャッシュ効率高）
RUN --mount=type=cache,target=/opt/conda/pkgs,uid=1000,gid=1000 \
    mamba install -y -c conda-forge \
        pycaret=3.3.2 \
        autoviz=0.1.905 \
        explainerdashboard \
        pynvml \
        nvitop \
        catboost && \
    mamba clean -yaf

# Step 2: PyTorch GPU版インストール（大容量・独立レイヤー）
RUN --mount=type=cache,target=/home/rapids/.cache/pip,uid=1000,gid=1000 \
    pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 3: GPU対応LightGBM（CUDA対応ビルド）
RUN --mount=type=cache,target=/home/rapids/.cache/pip,uid=1000,gid=1000 \
    pip install --no-cache-dir lightgbm --config-settings=cmake.define.USE_CUDA=ON

# Step 3.5: GPU対応XGBoost（CUDA対応ビルド）
RUN --mount=type=cache,target=/home/rapids/.cache/pip,uid=1000,gid=1000 \
    pip install --no-cache-dir 'xgboost>=2.0.0'

# Step 4: その他のPyPIパッケージ（変更頻度高）
RUN --mount=type=cache,target=/home/rapids/.cache/pip,uid=1000,gid=1000 \
    pip install --no-cache-dir \
        japanize-matplotlib \
        'shap[gpu]' \
        ydata-profiling \
        interpret \
        jupyter-resource-usage \
        jupyterlab-execute-time \
        jupyterlab-system-monitor && \
    rm -rf ~/.cache/matplotlib

# Step 4: JupyterLab拡張（軽量化：lab buildのみ）
RUN --mount=type=cache,target=/home/rapids/.cache,uid=1000,gid=1000 \
    jupyter lab build

# 環境チェックスクリプトをコピー
COPY --chown=rapids:rapids check_gpu_rapids_environment.py /home/rapids/

# 注意: ノートブックとドキュメントはボリュームマウントで提供されます
# docker-compose.yml の volumes セクションを参照してください

# エントリーポイントスクリプト
RUN echo '#!/bin/bash\n\
echo "🚀 RAPIDS + PyCaret GPU環境を起動中..."\n\
echo "📊 環境チェック中..."\n\
python /home/rapids/check_gpu_rapids_environment.py\n\
echo ""\n\
echo "📓 JupyterLabを起動します..."\n\
echo "📂 ノートブック:"\n\
echo "  - 特徴量AutoViz_PyCaret_v1.ipynb (AI売上予測)"\n\
echo "  - 店舗別包括ダッシュボード_v6.1_提案強化.ipynb (店舗ダッシュボード)"\n\
echo "📖 ドキュメント: docs/"\n\
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root' \
> /home/rapids/start_jupyter.sh && chmod +x /home/rapids/start_jupyter.sh

VOLUME /home/rapids/work
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
