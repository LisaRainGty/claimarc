#!/usr/bin/env bash
# 等待当前 A1（food）跑完后，自动串跑 food 品类的 A2→final，逐阶段记日志。
# 在服务器上后台运行：nohup bash scripts/run_food_pipeline.sh > food_pipeline.log 2>&1 &
set -u
cd ~/claimarc && source env.sh && cd src
CAT=food_and_beverages

echo "[chain] waiting for A1 to finish..."
while pgrep -f a1_extract_aspects >/dev/null 2>&1; do sleep 20; done
echo "[chain] A1 finished at $(date '+%H:%M:%S')"

run() {
  local st="$1"
  echo ""
  echo "############## STAGE $st  ($(date '+%H:%M:%S')) ##############"
  python -u -m run_pipeline --stage "$st" --category "$CAT"
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[chain] !!! STAGE $st FAILED rc=$rc (continue)"; fi
}

for st in A2 A3 B0 B1 B2B3 B4B5 C1 C2 C3 C4 C5 labels final; do
  run "$st"
done
echo ""
echo "[chain] FOOD_PIPELINE_DONE at $(date '+%H:%M:%S')"
