#!/usr/bin/env bash
# A2 评论侧聚类阈值敏感性扫描：固定 A0=0.20、A1 不变，仅变 A2 阈值，
# 每个阈值跑 A2->final，记录 CAS+ 属性数 / pair 数 / coverage / 正样本。
set -u
cd ~/claimarc && source env.sh && cd src
CAT=food_and_beverages
P=/root/claimarc/data/processed
SW=$P/_sweep
mkdir -p $SW
RES=$SW/results.tsv
echo -e "thr\tcas_attrs\treview\tpairs\tcov_ge1\tcov_pct\tpos\tpos_pct" > $RES

metrics() {  # args: thr
  /root/miniconda3/bin/python - "$1" <<'PY'
import json,sys
thr=sys.argv[1]
cas=json.load(open('/root/claimarc/data/processed/stageA/CAS+_food_and_beverages.json'))['attributes']
review=sum(1 for a in cas if a.get('source')=='review')
rows=[json.loads(l) for l in open('/root/claimarc/data/final/dataset.jsonl') if l.strip()]
n=len(rows)
ge1=sum(1 for r in rows if r.get('n_fact_sources',0)>=1)
pos=sum(1 for r in rows if r.get('y')==1)
print('%s\t%d\t%d\t%d\t%d\t%.1f\t%d\t%.1f'%(thr,len(cas),review,n,ge1,100*ge1/n,pos,100*pos/n))
PY
}

for THR in 0.20 0.30 0.35; do
  export CLAIMARC_A2_DISTANCE=$THR
  echo ""
  echo "########## A2 threshold = $THR  ($(date '+%H:%M:%S')) ##########"
  rm -rf $P/stageB/claim_list
  rm -f  $P/stageC/evidence_ocr.json $P/stageC/evidence_vlm.json
  for st in A2 A3 B0 B1 B2B3 B4B5 C1 C2 C3 C4 C5 labels final; do
    python -u -m run_pipeline --stage "$st" --category "$CAT" > /dev/null 2>>$SW/run_${THR}.err
  done
  cp /root/claimarc/data/final/dataset.jsonl $SW/dataset_${THR}.jsonl
  cp $P/stageA/CAS+_${CAT}.json $SW/CASplus_${THR}.json
  metrics "$THR" | tee -a $RES
done
echo ""
echo "[sweep] SWEEP_DONE at $(date '+%H:%M:%S')"
echo "===== results ====="
cat $RES
