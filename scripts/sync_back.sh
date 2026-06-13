#!/usr/bin/env bash
# 把服务器上的过程数据/结果数据拉回本机（保证服务器停租也不丢）。
# 用法：bash scripts/sync_back.sh           # 拉 processed + final + cache
#       bash scripts/sync_back.sh loop 300  # 每 300 秒自动拉一次
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

SSH_CMD="ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no"
if [[ -f "$REMOTE_KEY" ]]; then
  SSH_CMD="$SSH_CMD -i $REMOTE_KEY"
fi

pull() {
  mkdir -p "$LOCAL_ROOT/data/processed" "$LOCAL_ROOT/data/final" "$LOCAL_ROOT/data/cache"
  rsync -az --exclude='._*' -e "$SSH_CMD" "$REMOTE:$REMOTE_ROOT/data/processed/" "$LOCAL_ROOT/data/processed/" 2>/dev/null || true
  rsync -az --exclude='._*' -e "$SSH_CMD" "$REMOTE:$REMOTE_ROOT/data/final/"     "$LOCAL_ROOT/data/final/"     2>/dev/null || true
  # cache（LLM 调用缓存）也拉回，换机器可续跑省钱
  rsync -az --exclude='._*' -e "$SSH_CMD" "$REMOTE:$REMOTE_ROOT/data/cache/"     "$LOCAL_ROOT/data/cache/"     2>/dev/null || true
  echo "[sync<-server] $(date '+%H:%M:%S') pulled processed/final/cache"
}

if [[ "${1:-}" == "loop" ]]; then
  INT="${2:-300}"
  while true; do pull; sleep "$INT"; done
else
  pull
fi
