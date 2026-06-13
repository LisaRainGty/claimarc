#!/usr/bin/env bash
# 在 matpool 服务器上安装依赖（阿里 pip 镜像；模型走 ModelScope）。
# 用法（本机执行）：bash scripts/setup_server.sh
set -e
REMOTE=claimarc-gpu
REMOTE_ROOT='~/claimarc'

ssh "$REMOTE" "bash -lc '
set -e
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ >/dev/null 2>&1 || true
cd $REMOTE_ROOT
echo \"=== pip install ===\"
pip install -q -U pandas xlrd openpyxl tqdm Pillow scikit-learn modelscope 2>&1 | tail -3
# OCR / 嵌入（较大，失败不阻断文本阶段）
pip install -q paddleocr 2>&1 | tail -2 || echo \"[warn] paddleocr 安装失败，可后续单独装\"
pip install -q paddlepaddle-gpu 2>&1 | tail -2 || pip install -q paddlepaddle 2>&1 | tail -2 || echo \"[warn] paddle 安装失败\"
pip install -q FlagEmbedding 2>&1 | tail -2 || pip install -q sentence-transformers 2>&1 | tail -2 || echo \"[warn] 嵌入库安装失败\"
echo \"=== python check ===\"
python3 -c \"import pandas,xlrd,sklearn,PIL; print(\\\"core ok\\\", pandas.__version__)\"
'"
echo "[setup] done"
