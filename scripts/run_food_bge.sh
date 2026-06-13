#!/usr/bin/env bash
# 严格按 proposal 重做 food 属性集合：A0(BGE+LLM) → A1 → A2(BGE) → A3 → ... → final。
# 保留与属性无关的昂贵缓存：ocr_text/（原始OCR）、image_index.json（C1分流）、data/cache/（命中即复用）。
# 清除 CAS 依赖的产物以强制重算：claim_list/、evidence_ocr.json、evidence_vlm.json。
set -u
cd ~/claimarc && source env.sh && cd src
CAT=food_and_beverages
P=/root/claimarc/data/processed

echo "[bge] 备份旧 CAS+/dataset 以便对比"
mkdir -p $P/_prev_llm
cp $P/stageA/CAS+_${CAT}.json        $P/_prev_llm/ 2>/dev/null
cp $P/stageA/CAS_${CAT}.json         $P/_prev_llm/ 2>/dev/null
cp /root/claimarc/data/final/dataset.jsonl       $P/_prev_llm/dataset_prev.jsonl 2>/dev/null
cp /root/claimarc/data/final/table1_stats.json   $P/_prev_llm/table1_prev.json 2>/dev/null

echo "[bge] 清除 CAS 依赖产物（保留 ocr_text/image_index）"
rm -rf $P/stageB/claim_list
rm -f  $P/stageC/evidence_ocr.json $P/stageC/evidence_vlm.json

run() {
  local st="$1"
  echo ""
  echo "############## STAGE $st  ($(date '+%H:%M:%S')) ##############"
  python -u -m run_pipeline --stage "$st" --category "$CAT"
  local rc=$?
  [ $rc -ne 0 ] && echo "[bge] !!! STAGE $st FAILED rc=$rc"
}

for st in A0 A1 A2 A3 B0 B1 B2B3 B4B5 C1 C2 C3 C4 C5 labels final; do
  run "$st"
done
echo ""
echo "[bge] FOOD_BGE_DONE at $(date '+%H:%M:%S')"
