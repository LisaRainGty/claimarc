#!/usr/bin/env bash
# 把代码/文档/数据上传到当前 GPU 服务器。
# 用法：bash scripts/sync_to_server.sh [code|raw|final|all]
#
# 可用环境变量覆盖：
#   CLAIMARC_REMOTE=root@host
#   CLAIMARC_REMOTE_PORT=29752
#   CLAIMARC_REMOTE_KEY=~/.ssh/claimarc_matpool_ed25519
#   CLAIMARC_REMOTE_ROOT=/mnt/gty/claimarc_active
set -euo pipefail

REMOTE="${CLAIMARC_REMOTE:-root@hz-t3.matpool.com}"
REMOTE_PORT="${CLAIMARC_REMOTE_PORT:-29752}"
REMOTE_KEY="${CLAIMARC_REMOTE_KEY:-$HOME/.ssh/claimarc_matpool_ed25519}"
REMOTE_ROOT="${CLAIMARC_REMOTE_ROOT:-/mnt/gty/claimarc_active}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHAT="${1:-code}"

SSH_CMD="ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no"
if [[ -f "$REMOTE_KEY" ]]; then
  SSH_CMD="$SSH_CMD -i $REMOTE_KEY"
fi

$SSH_CMD "$REMOTE" "mkdir -p '$REMOTE_ROOT/data/raw' '$REMOTE_ROOT/data/index' '$REMOTE_ROOT/data/final'"

if [[ "$WHAT" == "code" || "$WHAT" == "all" ]]; then
  echo "=== sync code/docs (src, scripts, docs, env.sh, requirements) ==="
  rsync -az --delete --exclude='._*' -e "$SSH_CMD" \
    "$LOCAL_ROOT/src/" "$REMOTE:$REMOTE_ROOT/src/"
  rsync -az --delete --exclude='._*' -e "$SSH_CMD" \
    "$LOCAL_ROOT/scripts/" "$REMOTE:$REMOTE_ROOT/scripts/"
  rsync -az --delete --exclude='._*' -e "$SSH_CMD" \
    "$LOCAL_ROOT/docs/" "$REMOTE:$REMOTE_ROOT/docs/"
  rsync -az --exclude='._*' -e "$SSH_CMD" \
    "$LOCAL_ROOT/env.sh" "$LOCAL_ROOT/requirements.txt" "$LOCAL_ROOT/requirements_lock.txt" "$REMOTE:$REMOTE_ROOT/"
  # 服务器端 env：CLAIMARC_ROOT 指向远程
  $SSH_CMD "$REMOTE" "sed -i 's#^export CLAIMARC_ROOT=.*#export CLAIMARC_ROOT=$REMOTE_ROOT#' '$REMOTE_ROOT/env.sh'"
fi

if [[ "$WHAT" == "final" || "$WHAT" == "all" ]]; then
  echo "=== sync final datasets/results ==="
  rsync -az --exclude='._*' -e "$SSH_CMD" \
    "$LOCAL_ROOT/data/final/" "$REMOTE:$REMOTE_ROOT/data/final/"
fi

if [[ "$WHAT" == "raw" || "$WHAT" == "all" ]]; then
  echo "=== sync index ==="
  rsync -az --exclude='._*' -e "$SSH_CMD" \
    "$LOCAL_ROOT/data/index/" "$REMOTE:$REMOTE_ROOT/data/index/"
  echo "=== sync raw/comment + raw/srt_cut ==="
  rsync -az --exclude='._*' -e "$SSH_CMD" \
    "$LOCAL_ROOT/data/raw/comment/"  "$REMOTE:$REMOTE_ROOT/data/raw/comment/"
  rsync -az --exclude='._*' -e "$SSH_CMD" \
    "$LOCAL_ROOT/data/raw/srt_cut/"  "$REMOTE:$REMOTE_ROOT/data/raw/srt_cut/"
  echo "=== sync raw/product_images (大, 3.4G) ==="
  rsync -az --exclude='._*' --info=progress2 -e "$SSH_CMD" \
    "$LOCAL_ROOT/data/raw/product_images/" "$REMOTE:$REMOTE_ROOT/data/raw/product_images/"
fi
echo "[sync->server] $WHAT done"
