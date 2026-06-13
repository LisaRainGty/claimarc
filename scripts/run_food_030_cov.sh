#!/usr/bin/env bash
# 阈值定档 0.30 + 提升证据覆盖（C3 全图 OCR 并行 + C4 扩代表图）。
# 复用：A1 抽取、C1 图像分流、已有 OCR 缓存（增量补全余下图）。
# 强制重算：A2->B、C2/C3/C4（清 evidence_ocr/vlm + claim_list）。
set -u
cd ~/claimarc && source env.sh && cd src
export CLAIMARC_A2_DISTANCE=0.30
CAT=food_and_beverages
P=/root/claimarc/data/processed

echo "[cov] 清除 CAS 依赖产物（保留 ocr_text 缓存与 image_index）"
rm -rf $P/stageB/claim_list
rm -f  $P/stageC/evidence_ocr.json $P/stageC/evidence_vlm.json $P/stageC/evidence_params.json

run() {
  echo ""
  echo "############## STAGE $1  ($(date '+%H:%M:%S')) ##############"
  python -u -m run_pipeline --stage "$1" --category "$CAT"
  [ $? -ne 0 ] && echo "[cov] !!! STAGE $1 FAILED"
}

for st in A2 A3 B0 B1 B2B3 B4B5 C1 C2 C3 C4 C5 labels final; do
  run "$st"
done
echo ""
echo "[cov] FOOD_030_COV_DONE at $(date '+%H:%M:%S')"
