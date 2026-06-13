# 新服务器恢复指南（在当前实验基础上续跑）

> 本地快照应至少包含：**全部代码/脚本/文档**（`src/`, `scripts/`, `docs/`）、
> **主数据集**（`data/final/dataset_verify_faithful*.jsonl`）、
> **关键实验产物**（尤其 `data/final/cleancl/` 下的 grouped-CV bundle、JSON 和日志）、
> **环境锁**（`requirements_lock.txt`）。换服务器后按下列步骤即可续跑。

本地工作区根目录：`/Volumes/My Passport/claimarc`

---

## 1. 配置 SSH 别名（可选，便于 rsync）

把新服务器写入 `~/.ssh/config`（替换 host/port/密码或密钥）：

```
Host claimarc-gpu
    HostName <新服务器IP>
    Port <端口>
    User root
```

> 旧服务器 MOTD 会破坏 rsync 流，新服务器若有横幅，请把 MOTD 限制为交互式 shell
> （在 `~/.bashrc` 顶部加 `case $- in *i*) ;; *) return;; esac` 之前再打印横幅）。

## 2. 上传代码 + 数据 + 检查点

```bash
cd "/Volumes/My Passport/claimarc"
# 推荐直接使用维护过的同步脚本；默认当前主机为 hz-t3.matpool.com:29752，
# 换新机器时用环境变量覆盖 CLAIMARC_REMOTE / CLAIMARC_REMOTE_PORT / CLAIMARC_REMOTE_ROOT。
bash scripts/sync_to_server.sh code
bash scripts/sync_to_server.sh final
```

## 3. 还原模型缓存（免重新下载，约 4.4G）

```bash
rsync -az "_server_snapshot/modelscope/" claimarc-gpu:~/.cache/modelscope/
```

> 包含：`AI-ModelScope/bge-large-zh-v1.5`、`dienstag/chinese-roberta-wwm-ext`、
> `tiansz/bert-base-chinese`、`Fengshenbang/Erlangshen-Roberta-110M-NLI`。
> 若跳过此步，`modelscope` 会在首次用到时自动联网下载（较慢）。

## 4. 重建 Python 环境

```bash
conda create -n myconda python=3.11 -y
conda activate myconda
# GPU 版 torch 按新主机 CUDA 版本安装；其余依赖一次装齐：
pip install -r ~/claimarc/requirements_lock.txt
```

关键包（已验证）：`torch 2.8.0 (cu128)`、`transformers 5.10.2`、`peft 0.19.1`、
`sentence-transformers 5.5.1`、`scikit-learn 1.9.0`、`scipy 1.17.1`、`modelscope 1.37.1`、`numpy 2.1.2`。

## 5. 环境变量

```bash
export PYTHONPATH=~/claimarc/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# LLM 类基线需要运行时临时设置；不要写入 env.sh 或仓库文件：
export MATPOOL_API_KEY=<你的key>
export MATPOOL_BASE_URL=https://token.matpool.com/v1
```

## 6. 验证环境就绪

```bash
cd ~/claimarc && export PYTHONPATH=src
python -c "import torch,transformers,peft,sentence_transformers,sklearn; print('ok', torch.cuda.is_available())"
python -c "import models.cv_eval, models.fusion_eval, models.train, models.llm_risk_baseline; print('imports ok')"
```

---

## 7. 续跑各实验（命令速查）

所有训练脚本**可断点续跑**（`os.path.exists` 跳过已完成的 `.pt`）。

```bash
cd ~/claimarc && export PYTHONPATH=src

# (A) 锁定的最优配置 = BGE + LoRA(48) + 双流 + RACL，canonical 单次训练：
python -m models.train --dataset ../data/final/dataset_verify_faithful.jsonl \
  --seed 0 --tag mymodel --warmup 3 --cl_epochs 6 --tau 0.07 --lambda_cl 0.5 \
  --loss asl --gamma_neg 4 --fusion_dropout 0.2 --lora_rank 48 --enc_train lora \
  --save_emb ../data/final/v2/mymodel_s0.pt

# (B) 架构/消融扫描编排（含 lora32/48/64、骨干对比、消融）：
python -m models.run_arch --seeds 0 1 2 --configs lora48 abl_nocl abl_gneg

# (C) 主对比 + 配对 bootstrap 显著性（固定划分，5 种子）：
python -m models.fusion_eval --dataset ../data/final/dataset_verify_faithful.jsonl \
  --cm $(ls ../data/final/v2/arch_lora48_s*.pt) --boundary \
  --baseline "roberta_cls=$(ls ../data/final/v2/roberta_cls_s*.pt|tr '\n' ','|sed 's/,$//');bge_lr=../data/final/v2/bge_lr_s0.pt" \
  --out ../data/final/v2/cmp_main.json

# (D) 严格评测：分组 5 折交叉验证（n=1694，最可靠）：
python -m models.cv_eval --dataset data/final/dataset_verify_faithful.jsonl \
  --folds 5 --cm_seeds 0 1 --baselines roberta_cls bge_lr bert_cls \
  --out data/final/v2/cv.json

# (F) direct LLM 判别基线 / teacher 候选（无需 GPU，需 MATPOOL_API_KEY）：
python -m models.llm_risk_baseline \
  --dataset data/final/dataset_verify_faithful_args.jsonl \
  --pred_out data/final/cleancl/llm_qwen_flash_args_full.jsonl \
  --eval_out data/final/cleancl/llm_qwen_flash_args_full_fs1.json \
  --fold_seed 1 \
  --bge_tmp data/final/cleancl/cv_tmp_args_small_e3_c10_det_fairbase_fs1

# (E) 跨域 RQ3 / 补充基线：
python -m models.crossdomain_v2 --dataset ../data/final/dataset_verify_faithful.jsonl \
  --holdouts apparel_and_underwear baby_kids_and_pets --seeds 0 1 2 --out ../data/final/v2/xdom.json
python -m models.baselines_extra --kind bge_lr --seed 0 --save_pred ../data/final/v2/bge_lr_s0.pt
```

> ⚠️ 已知问题：`crossdomain_v2`/`cv_eval` 循环内显存累积，最大品类上可能 OOM；
> 已在 `cv_eval` 加 `gc.collect()+empty_cache()`；若仍 OOM，重跑即可（已完成折会跳过）。

---

## 8. 当前结论速记（详见 `docs/Experiment_Results_v2.md`）

- 旧 CLAIMARC-v2 的 blend2/Platt 在严格 grouped CV 下不稳；不要再作为最终主方法。
- clean-RACL + argument-aware 分支能提供排序互补；`rankavg(args/no-args p_cls, fair BGE+LR)` 已在 `fold_seed=0/1/2` 对 AUPRC 稳定显著收益，但 Macro-F1/wF1 仍未稳定显著超过 fair BGE+LR。
- source-first evidence policy 是最新有用线索：`rankavg(sourcefirst_args_pcls, sourcefirst_BGE)` 在 `fold_seed=0/2` 同时显著提升 AUPRC/AUROC/Macro-F1，但 `fold_seed=1` 只复现 AUPRC。
- dual-head router 是当前最清楚的结构诊断：`src/models/cv_dual_head_router.py` 把 ranking score head 与 binary decision head 解耦，严格用 OOF `yhat` 做 Macro-F1 bootstrap。`cv_dual_head_router_srcfirst_a120_fs0/fs2_s0.json` 中已有三指标显著候选；`fs1` 仍失败。`drop_src0args` 变体让 fs1 的 AP/AUROC 可显著，但 Macro-F1 仍不显著。
- fs1 最新最强点估计仍是 atomic NLI posterior ranking + compact confidence headmix RACL decision：`cv_nli_dual_guard_srcargs_drop_fs1_s0_headmix_top6_5k.json` 中 `rankmix_nli25_hgb_bge` score + confidence headmix decision 得到 AP 0.5008 / AUROC 0.6425 / Macro-F1 0.6117；5k bootstrap 下 AP p=0.0016、AUROC p=0.0082、Macro-F1 p=0.0198。这是 fs1 hard split 首个 AP/AUROC/Macro-F1 三项同时显著超过 BGE+LR 的结果。
- fs1 的更简单 strict 候选已经更新：`cv_nli_dual_guard_srcargs_drop_fs1_s0_nlievidenceveto_top6_5k.json` 中 `rankmix_nli25` score + `scoreguard_clip_drop20_min30_srcbin_conf_bgefallback_src0_src2_3_lowabs` decision 得到 AP 0.4939 / AUROC 0.6420 / Macro-F1 0.6101，5k bootstrap p=0.0002/0.0056/0.0372。`rankmix_nli50_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs` 点估计更高（AP 0.5045 / AUROC 0.6433 / Macro-F1 0.6123），但 AUROC p=0.0508、Macro-F1 p=0.0956，不能作严格主候选。
- fs1 的当前统一协议雏形是 `cv_nli_dual_guard_srcargs_drop_fs1_s0_predef_lowabs_top6_5k.json` 中的 `predef_lowabs_r25_scorefallback_srcconf_bgefallback`：固定 `rankmix_nli25_hgb_bge` 排序头，在 source0 与 `source_count=2/3 + low/absent confidence` 上做 0.25 BGE score fallback，并在同一 lowabs mask 上做 `srcbin_conf` BGE decision fallback。结果 AP 0.4940 / AUROC 0.6424 / Macro-F1 0.6101，5k bootstrap p=0.0004/0.0030/0.0372；OOF 备份为 `oof_nli_dual_guard_srcargs_drop_fs1_s0_predef_lowabs.npz`。这比 headmix 点估计低，但更接近可预注册的 source/confidence fallback 协议。
- fs0/fs2 复核已经回收：`cv_nli_dual_guard_srcargs_drop_fs0_s0_scorefallback_quick_top8_5k.json` 中 `rankmix_nli25_hgb_bge_scorefallback_bge025_src0_src2_3_lowabs` 得到 AP 0.4915 / AUROC 0.6309 / Macro-F1 0.5959，5k bootstrap 下 AP p=0.0150、AUROC p=0.0458、Macro-F1 p=0.0202；`cv_nli_dual_guard_srcargs_drop_fs2_s0_bgefallback_top8_5k.json` 中 `rankmix_nli25` score + `scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3` decision 得到 AP 0.5128 / AUROC 0.6405 / Macro-F1 0.6040，5k bootstrap 下 AP p=0.0126、AUROC p=0.0270、Macro-F1 p=0.0028。`bgeedit_*` / `scoregroup_*` / `sourceveto_*` 是 bgefallback/scorefallback 前的诊断：能定位假阳性组，但不足以独立闭环。
- 第一版 `compact_router_valselect` 是负结果：fs0 quick 中复用同一 validation carve 既调 head 又选 head，会被 headmix 误导，Macro-F1 仅 0.5652；排除 headmix 后也只有 0.5854。新服务器续跑时不要直接扩大这个候选池，应先实现 inner split/nested selector 或完全预定义的 source/confidence fallback 规则。
- 第一版真正 `compact_router_nested_*` 已在 fs1 quick 跑通，但未闭环：`compact_router_nested_balanced` AP 0.4912 / AUROC 0.6371 / Macro-F1 0.5953，只比 BGE Macro-F1 0.5928 小幅正向，明显低于固定 headmix/scorefallback 候选。说明 validation 再切半后 head selection 信号不足；后续优先锁定完全预定义规则，而不是继续扩大 nested 候选池。
- `bgeadvfallback_*` 是负结果：按 validation group 判断 BGE 是否优于当前 head 再回退，fs1 quick 最好 Macro-F1 只有 0.5927；fs0/fs2 quick 还受到 NLI cache 不一致影响，不能与旧 5k 横比，但 adv 本身也没有接近有效候选。不要扩 5k。
- `bgerateguard_*` 也是负结果：只在 validation 中某 source/confidence 组过预测正类时回退 BGE，fs1 quick 最好 AP 0.5021 / AUROC 0.6431 / Macro-F1 0.5917，低于 BGE Macro-F1 0.5928。不要扩 5k。
- `nlievidenceveto` 为负结果：按 atomic NLI 聚合信号做单规则正类 veto，fs1 最好只有 AP 0.4939 / AUROC 0.6420 / Macro-F1 0.5984，明显弱于 base fallback。保留为诊断，不再扩网格。
- 复现警告：旧 fs0/fs2 的精确 NLI cache 当前没有在本地/远程快照中；用 `cache_nli_srcargs_a120.npz` 重跑会复现 BGE baseline，但不复现旧 NLI/rankmix 数值。fs0/fs2 论文证据应优先保留并引用已同步的 JSON/OOF；若换机重跑，必须先重建一致 cache。
- `predef_lowabs` 的 fs2 新缓存 quick 已跑，文件 `cv_nli_dual_guard_srcargs_drop_fs2_s0_predef_lowabs_newcache_quick.json` 只作诊断：最好 predef 为 AP 0.4939 / AUROC 0.6350 / Macro-F1 0.5909，对照 BGE+LR 为 AP 0.4940 / AUROC 0.6356 / Macro-F1 0.5841，只涨 Macro-F1、不保排序，不扩 5k。fs0 新缓存 quick 因缺少匹配 `fold_seed=0` 的 no-args 临时目录而标签不一致中止；不要混用旧 `cv_tmp_small_e3_c10`。
- 新增 `src/models/cv_nli_predef_lowabs.py` 作为 predef-only 评估器：只依赖 NLI cache、BGE fold 概率和 source/confidence 元数据，不加载 noargs/args 临时目录。它修复了 fs0 资产不匹配问题。新缓存 fs0 的 5k 文件为 `cv_nli_predef_lowabs_srcargs_drop_fs0_s0_newcache_top20_5k.json`，OOF 为 `oof_nli_predef_lowabs_srcargs_drop_fs0_s0_newcache_5k.npz`。其中 `rankmix_nli25_hgb_bge_scorefallback_bge100_src0` 得到 AP 0.4859 / AUROC 0.6338 / Macro-F1 0.5973，5k p=0.0094/0.0490/0.0176，是新缓存 fs0 的严格候选；但原 `predef_lowabs_r25_scorefallback_thr` AUROC p=0.1092，不严格。
- `predef_lowabs_valselect_macro/balanced` 是负结果：在 7 个固定协议里按 validation macro/balanced utility 选，fs0 quick 只有 AP 0.4839 / AUROC 0.6247 / Macro-F1 0.5909，弱于固定 `src0` scorefallback；不要把它扩 5k 或写成统一 router。
- 新增 `src/models/diagnose_fallback_mechanisms.py` 与输出 `data/final/cleancl/fallback_mechanism_diagnosis_20260608.json`：只读 OOF `.npz`，按 fold/source/confidence/category/BGE 不确定性统计候选相对 BGE 的 fixed/broken 翻转。诊断结论：fs0 的收益主要来自 BGE 不确定/lowabs 假阳性校正；fs1 的固定 predef 协议实质是保护 lowabs/source0，让 source-rich/medium-confidence 样本交给 NLI+BGE；fs2 是正类召回边界修复但会新增 FP。下一步写成 `protected BGE regions + RACL/NLI regions` 的少参数 hybrid，而不是扩大 validation selector。
- 新增 `src/models/diagnose_protected_hybrid_oof.py` 与输出 `protected_hybrid_oof_screen_20260608.json`、`protected_hybrid_forced_bootstrap_20260608.json`：OOF-level protected hybrid 筛查显示方向接近但未闭环。最接近统一的 `rank25_bge025_lowabs + protect_lowabs_scoreguard_srcbin_conf` 在 fs1/fs2 三项严格，但 fs0 旧缓存 Macro-F1 p=0.0566、fs0 新缓存 AUROC/Macro-F1 p=0.0800/0.1514。不要把它当最终主方法；若继续，应先加 fs0 FP 约束或极小 validation 开关。
- 追加 `protected_hybrid_fs0_newcache_failure_20260608.json` 与 `scorefallback_selfthr_forced_bootstrap_20260608.json`：fs0 新缓存统一候选的失败来自 `food_and_beverages`、fold1、medium confidence、`src2_3:medium` 等组新增 FP；fs0 strict 候选依赖 scorefallback 自带阈值（fs0 新缓存 `scorefallback_bge100_src0` 5k p=0.0096/0.0490/0.0176），而 fs1 的 scorefallback 自带阈值不保 Macro-F1。下一步统一协议应只在 validation 上做“scorefallback self-threshold vs protected decision fallback”的二值开关，并约束 medium/source-rich FP。
- `cv_nli_predef_lowabs.py` 已实现该二值开关并做 quick 复核：`switch_relaxed` 在 fs0 不误伤，`sf100src0_or_lowabs_srcconf_fp02_gain008` 等同 `scorefallback_bge100_src0`（AP 0.4861 / AUROC 0.6275 / Macro-F1 0.5948）；在 fs1 可从保守版 0.6001 提到 Macro-F1 0.6063，但仍低于固定 `predef_lowabs_r25_scorefallback_srcconf_bgefallback` 0.6089。暂不扩 5k；若继续，应加入 recall-loss guard 或反向选择“何时允许 self-threshold 覆盖 protected fallback”。
- `cv_nli_predef_lowabs.py` 已继续加入 reverse switch 与 fixed decoupled score calibration。新输出已备份：`cv_nli_predef_lowabs_srcargs_drop_fs1_s0_switch_reverse_quick.json`、`cv_nli_predef_lowabs_srcargs_drop_fs0_s0_switch_reverse_top20_5k.json`、`cv_nli_predef_lowabs_srcargs_drop_fs0_s0_decoupled_top12_5k.json` 及对应 OOF。结论：fs1 reverse quick 等同 fixed protected；fs0 reverse 5k AP/Macro-F1 显著但 AUROC p=0.188；decoupled score 将 fs0 AUROC p 拉到 0.0566-0.0672，仍未严格闭环。换机后不要把该线当最终主方法；下一步应做 score-level uncertainty/monotone calibration。
- `cv_nli_predef_lowabs.py` 最新已加入 source-first CM p_cls expert、固定 rank-weighted CM/NLI 融合、`sf025lowabs self-threshold vs lowabs_srcconf BGE-protected` 小 switch decision，以及 `source_count>=2 & confidence in {low, medium}` 的 source-sufficiency guard。相关文件已备份：`cv_nli_predef_lowabs_srcargs_drop_fs*_s0_nondropbge_cmpcls_quick.json/.npz`、`...cmpcls_decision_quick.json/.npz`、`...cmpcls_weighted_quick.json/.npz`、`oof_bootstrap_cmpcls_decoupled_20260608.json`、`oof_bootstrap_cmpcls_weighted_20260608.json`、`oof_bootstrap_cmpcls_weighted_switch_20260608.json`、`oof_bootstrap_cmpcls_weighted_guard_20260608.json`。当前最佳统一候选是 `rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect`：pooled AP 0.5017 / AUROC 0.6407 / Macro-F1 0.6059；相对 BGE 的增益 +0.0285/+0.0115/+0.0231，p=0.0000/0.0006/0.0000；相对旧 `rankavg_sourcefirst_cm_pcls_bge` 的增益 +0.0095/+0.0033/+0.0120，p=0.0010/0.0066/0.0016。注意：该组实验使用 drop-src0args 数据集，但 BGE/CM tmp 必须用非 drop source-first 目录 `cv_tmp_args_srcfirst_a120_small_e3_c10_fs*_s0`，这是为了与 source-first baseline 资产口径一致。
- 2026-06-08 追加 adaptive/taxonomy-aware 修复：`cv_nli_predef_lowabs.py` 已输出 `rankw_sourcefirst_cm040_nli060_score_src0ormedium_cmreinforce025_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect_src4pmedium_cmbgenli` 和 `rankw_sourcefirst_cm040_nli060_score_sportsgeneral_cm025_decision_sports_cm025`。相关文件：`cv_nli_predef_lowabs_srcargs_drop_fs*_s0_nondropbge_cmpcls_adaptive_quick.json/.npz`、`oof_bootstrap_cmpcls_adaptive_quick_20260608.json`。adaptive pooled AP/AUROC/Macro-F1 为 0.5049 / 0.6413 / 0.6071，对 BGE 和旧 CM+BGE 三项显著，但相对 current guarded 不显著；taxonomy-aware combo pooled 为 0.5031 / 0.6421 / 0.6086，且相对 current guarded 三项显著（p=0.0150/0.0012/0.0242）。该 taxonomy-aware 结果应视为商品类别可靠性适配的诊断候选/上界，若写主方法需要改成 fold 内 validation/nested taxonomy adapter。
- 2026-06-08 追加 validation-safe taxonomy adapter 诊断：新增 `src/models/diagnose_taxonomy_adapter_oof.py` 与输出 `taxonomy_adapter_oof_screen_20260608.json`。该脚本只读三划分 adaptive OOF 与 CV JSON `fold_meta`，在每个 outer fold 内用 validation 指标分别选择 score/decision head，再做 pooled bootstrap。结论为负：固定 taxonomy-aware combo 仍是 pooled 显著诊断上界，但最好的 validation-safe selector 只有 AP/AUROC/Macro-F1 0.5044 / 0.6405 / 0.6063，且相对 current guarded 不显著；不要把 taxonomy 补丁当最终主方法。
- 2026-06-08 追加 evidence-type adapter：新增 `src/models/diagnose_evidence_type_adapter_oof.py` 与输出 `evidence_type_adapter_oof_screen_20260608.json`、`oof_evidence_type_adapter_screen_20260608.npz`，并已把同一规则固化进 `cv_nli_predef_lowabs.py`，方法名为 `rankw_sourcefirst_cm040_nli060_score_src0orpomedium_cmreinforce025_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect_pomedium_cmbgenli`。规则：以 current guarded 为基底，在 `source_count==0 OR evidence_combo==PO & confidence==medium` 上采用 adaptive score，在 `PO:medium` 上采用 adaptive decision。pooled AP/AUROC/Macro-F1 为 0.5052 / 0.6412 / 0.6084；相对 BGE 与旧 CM+BGE 三项显著，相对 current guarded Macro-F1 显著（p=0.0453）但 AP/AUROC 不显著。fs1 smoke 已复核 method 生成成功，fs1 指标为 0.4978 / 0.6322 / 0.6099；fs1 vs BGE 的 AUROC/Macro-F1 仍不显著。
- 2026-06-08 追加 common OOF method audit：新增 `src/models/diagnose_common_oof_methods.py` 与输出 `common_oof_method_sweep_20260608.json`。它枚举 fs0/fs1/fs2 adaptive quick OOF 中共同存在的 227 个 evaluator 方法；最高 pooled AP 仍是 adaptive fixed 0.5049 / 0.6413 / 0.6071，最高 pooled AUROC/Macro-F1 仍是 taxonomy-aware fixed 0.5031 / 0.6421 / 0.6086。fs1 局部高分的 `cm025/nli075`、rankstable/valselect 方法会明显牺牲 pooled AP/AUROC；因此不要再优先横向搜索已保存 OOF head，下一轮应新增 score/representation 层面的证据关系信号。
- 2026-06-08 追加 relation adapter 诊断：新增 `src/models/diagnose_relation_oof_adapter.py` 与输出 `relation_oof_adapter_screen_20260608.json`、`oof_relation_adapter_screen_20260608.npz`。该脚本在 evidence-type OOF 上按 `pair_id` 分组 cross-fit LR/HGB 二层模型，避免 fs0/fs1/fs2 重复 pair 泄漏，测试现有概率与 source/confidence/evidence_combo 元数据是否足以学习“何时信任 RACL/NLI”。结论为负：`relation_lr_no_category` 为 0.4782 / 0.6294 / 0.5890，`relation_hgb_no_category` 为 0.4786 / 0.6242 / 0.5789；带 category/taxonomy 的诊断上界也只有 0.4950 / 0.6400 / 0.6001，弱于 evidence-type adapter。不要继续普通二层 stacker；下一步转向 per-source evidence pooling 或预注册 evidence-sufficiency rule。
- 2026-06-08 追加 NLI source-pooling micro-calibration 诊断：新增 `src/models/diagnose_nli_source_pooling_oof.py` 与输出 `nli_source_pooling_oof_screen_20260608.json`、`nli_source_pooling_oof_top1_3k_20260608.json`、`oof_nli_source_pooling_screen_20260608.npz`。pooled screen top 候选 `evtype_score_argref_neutral_rate35_a05_decision_evtype` 为 AP/AUROC/Macro-F1 0.5093 / 0.6414 / 0.6084；相对 current guarded 的 AP/Macro-F1 为正，但相对 evidence-type 自身的 AP 只有 sample p=0.0410、group p=0.0810。该规则已固化进 `cv_nli_predef_lowabs.py`，方法名含 `argrefneutral005`；补齐 fs0/fs1/fs2 fold-level smoke 后，pooled 输出 `oof_bootstrap_cmpcls_insuff_smoke_20260608.json`，argrefneutral005 为 0.5034 / 0.6404 / 0.6084，低于 evidence-type 0.5052 / 0.6412 / 0.6084；相对 evidence-type 的 dAP/dAUROC/dMacro 为 -0.0016 / -0.0009 / 0.0000，p=0.7253/0.9087/1.0000。结论：pooled screen 提升来自 pooled-rank 后验放大；该 micro-calibration 关闭。
- 2026-06-08 追加 evidence-type residual 与 evidence-sufficiency fallback 小规则筛查：新增 `src/models/diagnose_evtype_residuals.py`、`src/models/diagnose_evsuff_oof_rules.py` 与输出 `evtype_residual_diagnosis_20260608.json`、`evsuff_oof_rule_screen_20260608.json`、`evsuff_oof_rule_screen_rawblend_20260608.json`。注意：第一版 `evsuff_oof_rule_screen_20260608.json` 使用 mask-local rank，AP 被局部重标尺放大，不能引用；修正版 rawblend 只做原始 score 混合/替换。有效结论为负：`O:low -> BGE` 可到 0.5073 / 0.6424 / 0.6021，但 Macro-F1 明显低于 evidence-type；`src2_3:medium -> BGE` 为 0.5054 / 0.6414 / 0.6092，Macro 小涨不显著。规则层 source/evidence fallback 基本耗尽，下一步转向模型结构或训练目标。
- 2026-06-08/09 追加 source-policy multi-instance pooling 结构实验：`src/models/data.py`/`train.py`/`cv_eval.py` 支持 `--evidence_policy`、`--evidence_policy_mix` 与 `--dump_oof`，新增 `src/models/diagnose_source_policy_pooling.py` 与 `scripts/run_source_policy_experts_fs1.sh`。fs1/fs2/fs0 均已完成 `sourcefirst + noargs + ocr_only + params_only + BGE` pooling。三划分 pooled repeated-CV (`n=5082`) 中，`rankavg_all_score_bge_lr_src0_neg_guard` 为 0.4978 / 0.6345 / 0.5930，5k bootstrap vs BGE p=0.0000/0.0002/0.0062，vs noargs p=0.0000/0.0000/0.0104，vs sourcefirst AP/AUROC p=0.0000/0.0000，Macro p=0.116；`mean_all_score_bge_lr_src0_neg_guard` 为 0.4918 / 0.6301 / 0.5955，vs BGE p=0.0000/0.0042/0.0004，vs noargs p=0.0000/0.0000/0.0018，vs sourcefirst p=0.0016/0.0000/0.0502。该结构稳定胜 BGE/noargs/sourcefirst 的排序指标，但仍弱于 evidence-type adapter 0.5052 / 0.6412 / 0.6084；定位为结构性消融和 evidence-view dropout 的依据。`evtype_sp_logitcore_replace_O` hybrid 到 0.5052 / 0.6412 / 0.6106，但 vs evidence-type Macro p=0.2332，不能作为主方法。
- 2026-06-09 追加 fs3 source-policy 独立复核：`no_args` 0.4865 / 0.6199 / 0.5670 / 0.5612，`ocr_only` 0.4713 / 0.6112 / 0.5601 / 0.5722，`params_only` 0.4755 / 0.6130 / 0.5734 / 0.5735；pooling 中 `rankavg_all` 为 0.4834 / 0.6299 / 0.5771 / 0.5954，3k targeted bootstrap vs BGE 的 AP/AUROC p=0.0183/0.0213，但 Macro p=0.2307。`source_masked_mean` Macro 0.5807，但 vs BGE Macro p=0.1770。结论：fs3 只复现 source-policy 的排序/可靠性加权信号，没有复现分类闭环；不优先扩 fs4，主线继续以 RACL prototype relation score + sufficiency protocol 为准。
- 2026-06-09 source-policy augmented relation head quick screen 已记录为 `sourcepolicy_relation_adapter_quick_20260609.json`。按 `pair_id` 分组 cross-fit，把 evidence-type、current/adaptive、BGE 与 source-policy 分数/判决都作为特征；最好 LR 为 0.4986 / 0.6440 / 0.6030，AUROC 略高但 AP/Macro-F1 弱于 evidence-type adapter，不优先重跑。
- 2026-06-09 multi-view consistency all-loss smoke 已完成并备份：`src/models/data.py`/`train.py`/`cv_eval.py` 支持 `--view_consistency_mix` 与 `--view_ce_weight/--view_logit_weight/--view_embed_weight`。fs1 主视图 `source_first`、aux=`no_args,ocr_only,params_only`、CE=0.10/logit=0.05/embed=0.05 的结果为负：PCLS 0.4758 / 0.6104 / 0.5727，selectiveRKC 0.4764 / 0.6119 / 0.5722，BGE 0.4657 / 0.6288 / 0.6000；PCLS vs BGE dAP +0.0090 (p=0.2400)、dAUROC -0.0184 (p=0.9635)、dMacro -0.0158 (p=0.9020)。不要扩展 all-loss view consistency；若继续，只测试 embedding-only consistency。
- 2026-06-09 embedding-only consistency follow-up 也已完成并备份：`cv_viewcons_srcfirst_aux_noargs_ocr_params_emb005_fs1_s0.json` / `oof_viewcons_srcfirst_aux_noargs_ocr_params_emb005_fs1_s0.npz`。结果仍为负：PCLS 0.4753 / 0.6100 / 0.5711，selectiveRKC 0.4759 / 0.6110 / 0.5717；PCLS vs BGE dAP +0.0086 (p=0.2545)、dAUROC -0.0188 (p=0.9650)、dMacro -0.0129 (p=0.8540)。不要继续 view consistency。
- 2026-06-09 source-sufficiency auxiliary representation smoke 已完成并备份：`train.py`/`cv_eval.py` 支持 `--source_aux_combo_weight`、`--source_aux_conf_weight`、`--source_aux_count_weight`，在 retrieval embedding 上预测 evidence metadata。fs1 0.01/0.01/0.01 为 0.4667 / 0.6115 / 0.5900，BGE 为 0.4657 / 0.6288 / 0.6000；PCLS vs BGE dAP +0.0007 (p=0.4650)、dAUROC -0.0173 (p=0.9105)、dMacro -0.0109 (p=0.7950)。它能改善 fold0/fold3/fold4 的 Macro-F1，但不能修 fold1 或排序 AP；不继续调权重。
- 2026-06-09 cross-encoder reranker baseline 已完成并备份：新增 `src/models/cv_reranker_feature.py`，`BAAI/bge-reranker-v2-m3` logits cache 为 `reranker_bge_v2m3_srcfirst_fs1_logits.npz`，输出 `cv_reranker_bge_v2m3_fs1.json` / `oof_reranker_bge_v2m3_fs1.npz`。结果显著弱于 BGE：direct 0.4077 / 0.5185 / 0.5206，LR 0.3984 / 0.5104 / 0.5138，BGE 0.4657 / 0.6288 / 0.6000；direct vs BGE dAP -0.0588 (p=0.9995)、dAUROC -0.1105 (p=1.0000)、dMacro -0.0760 (p=1.0000)。通用 reranker 不作为 teacher。
- 2026-06-08 追加训练期 evidence-type hard-negative 过滤/soft bonus：`src/models/data.py` 现在把 `evidence_combo` 与 `confidence` 放进 batch，`src/models/train.py` 和 `cv_eval.py` 新增 `--cl_neg_filter {same_evtype,same_evtype_conf,medium_evtype_conf}` 与 `--cl_neg_bonus/--cl_neg_bonus_filter`。fs1/drop-src0args 小模型第一轮硬过滤 `medium_evtype_conf` 输出为 `cv_args_srcfirst_a120_drop_src0args_evhn_medium_fs1_s0.json` 与 `cv_tmp_args_srcfirst_a120_drop_src0args_evhn_medium_fs1_s0/`；PCLS AP/AUROC/Macro-F1 为 0.4856 / 0.6173 / 0.5899，对 BGE p=0.2115/0.8470/0.5560。第二轮 soft bonus `cl_neg_bonus=0.05, medium_evtype_conf` 输出为 `cv_args_srcfirst_a120_drop_src0args_evhn_soft005_fs1_s0.json` 与 `cv_tmp_args_srcfirst_a120_drop_src0args_evhn_soft005_fs1_s0/`；PCLS 为 0.4849 / 0.6193 / 0.5861，对 BGE p=0.2245/0.8025/0.5655。结论为不足：evidence-type negative sampling 不能解决 fs1；该信号应转为 auxiliary relation score 或 OOF-level score/decision adapter。

重跑某个 fold_seed 的最新 CM p_cls + NLI weighted/adaptive quick：

```bash
cd ~/claimarc && export PYTHONPATH=src
FS=1
python src/models/cv_nli_predef_lowabs.py \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
  --cache data/final/cleancl/cache_nli_srcargs_a120.npz \
  --bge_tmp data/final/cleancl/cv_tmp_args_srcfirst_a120_small_e3_c10_fs${FS}_s0 \
  --cm_tmp data/final/cleancl/cv_tmp_args_srcfirst_a120_small_e3_c10_fs${FS}_s0 \
  --cm_seed 0 \
  --fold_seed ${FS} \
  --quick \
  --n_boot 0 \
  --out data/final/cleancl/cv_nli_predef_lowabs_srcargs_drop_fs${FS}_s0_nondropbge_cmpcls_adaptive_quick.json \
  --dump_oof data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs${FS}_s0_nondropbge_cmpcls_adaptive_quick.npz
```

重算三划分 pooled bootstrap（weighted guard 主候选）：

```bash
python src/models/bootstrap_oof_methods.py \
  --case fs0=data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs0_s0_nondropbge_cmpcls_weighted_quick.npz \
  --case fs1=data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs1_s0_nondropbge_cmpcls_weighted_quick.npz \
  --case fs2=data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs2_s0_nondropbge_cmpcls_weighted_quick.npz \
  --baseline bge_lr \
  --baseline rankavg_sourcefirst_cm_pcls_bge \
  --method rankavg_sourcefirst_cm_pcls_nli075_decision_nli075_lowabs \
  --method rankw_sourcefirst_cm033_nli067_decision_nli075_lowabs \
  --method rankw_sourcefirst_cm040_nli060_decision_nli075_lowabs \
  --method rankw_sourcefirst_cm040_nli060_decision_cmbge_nli075 \
  --method rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp02_gain008 \
  --method rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008 \
  --method rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --method rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_allconf_cmbgeprotect \
  --n_boot 3000 \
  --seed 20260608 \
  --out data/final/cleancl/oof_bootstrap_cmpcls_weighted_guard_20260608.json
```

重算 adaptive/taxonomy-aware pooled bootstrap：

```bash
python src/models/bootstrap_oof_methods.py \
  --case fs0=data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs0_s0_nondropbge_cmpcls_adaptive_quick.npz \
  --case fs1=data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs1_s0_nondropbge_cmpcls_adaptive_quick.npz \
  --case fs2=data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs2_s0_nondropbge_cmpcls_adaptive_quick.npz \
  --baseline bge_lr \
  --baseline rankavg_sourcefirst_cm_pcls_bge \
  --baseline rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --method rankw_sourcefirst_cm040_nli060_score_src0ormedium_cmreinforce025_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect_src4pmedium_cmbgenli \
  --method rankw_sourcefirst_cm040_nli060_score_sportsgeneral_cm025_decision_sports_cm025 \
  --n_boot 5000 \
  --seed 20260608 \
  --out data/final/cleancl/oof_bootstrap_cmpcls_adaptive_quick_20260608.json
```

重算 validation-safe taxonomy adapter OOF screen：

```bash
python src/models/diagnose_taxonomy_adapter_oof.py \
  --n_boot 3000 \
  --seed 20260608 \
  --out data/final/cleancl/taxonomy_adapter_oof_screen_20260608.json
```

重算 evidence-type adapter OOF screen：

```bash
python src/models/diagnose_evidence_type_adapter_oof.py \
  --n_boot 3000 \
  --seed 20260608 \
  --out data/final/cleancl/evidence_type_adapter_oof_screen_20260608.json \
  --dump_oof data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz
```

补 room-level group bootstrap（先按 `pair_id` 反接 `room_id`）：

```bash
python - <<'PY'
import json, numpy as np
from pathlib import Path
src = Path("data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz")
out = Path("data/final/cleancl/oof_evidence_type_adapter_screen_room_20260609.npz")
room_by_pair, attr_by_pair = {}, {}
with open("data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        pid = str(r.get("pair_id", ""))
        if pid:
            room_by_pair[pid] = str(r.get("room_id", ""))
            attr_by_pair[pid] = str(r.get("attribute_id", ""))
z = np.load(src, allow_pickle=True)
d = {k: z[k] for k in z.files}
pairs = [str(x) for x in z["pair_id"]]
d["room_id"] = np.asarray([room_by_pair.get(pid, "") for pid in pairs], dtype=object)
if "attribute_id" not in d:
    d["attribute_id"] = np.asarray([attr_by_pair.get(pid, "") for pid in pairs], dtype=object)
np.savez_compressed(out, **d)
PY

PYTHONPATH=src python -m models.bootstrap_oof_methods \
  --case evtype=data/final/cleancl/oof_evidence_type_adapter_screen_room_20260609.npz \
  --method evtype_adapt_score_src0_po_medium_decision_po_medium \
  --method rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --method rankavg_sourcefirst_cm_pcls_bge \
  --baseline bge_lr \
  --baseline rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key room_id \
  --only_group \
  --out data/final/cleancl/evidence_type_adapter_room_bootstrap_group_5k_20260609.json
```

已完成结果：`evtype_adapt_score_src0_po_medium_decision_po_medium` 为 0.5052 / 0.6412 / 0.6084 / 0.6355。room-level group bootstrap 相对 BGE 的 p(AP/AUROC/Macro) = 0.0006 / 0.0728 / 0.0000；相对上一版 guarded `rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect` 的 p = 0.1674 / 0.3916 / 0.0732。可强主张 AP/Macro-F1 胜 BGE；不能强主张胜上一版 guard。

在新增 fs3/fs4 上复核固定 evidence-type adapter（不重训，只读已保存 fold bundle 和 NLI cache）：

```bash
for FS in 3 4; do
  PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_nli_predef_lowabs \
    --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
    --cache data/final/cleancl/cache_nli_srcargs_a120.npz \
    --bge_tmp data/final/cleancl/cv_tmp_args_srcfirst_a120_drop_src0args_small_e3_c10_fs${FS}_s0 \
    --cm_tmp data/final/cleancl/cv_tmp_args_srcfirst_a120_drop_src0args_small_e3_c10_fs${FS}_s0 \
    --cm_seed 0 \
    --fold_seed ${FS} \
    --folds 5 \
    --quick \
    --n_boot 0 \
    --out data/final/cleancl/cv_nli_predef_lowabs_srcargs_drop_fs${FS}_s0_nondropbge_cmpcls_evtype_fixed_20260609.json \
    --dump_oof data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs${FS}_s0_nondropbge_cmpcls_evtype_fixed_20260609.npz
done
```

给 fs3/fs4 OOF 反接 `room_id` 并把长 evaluator 方法名复制为统一别名 `evtype_adapt_score_src0_po_medium_decision_po_medium`：

```bash
python - <<'PY'
import json, numpy as np
from pathlib import Path
EV_LONG = "rankw_sourcefirst_cm040_nli060_score_src0orpomedium_cmreinforce025_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect_pomedium_cmbgenli"
EV_ALIAS = "evtype_adapt_score_src0_po_medium_decision_po_medium"
room_by_pair = {}
with open("data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        room_by_pair[str(r.get("pair_id", ""))] = str(r.get("room_id", ""))
for fs in (3, 4):
    src = Path(f"data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs{fs}_s0_nondropbge_cmpcls_evtype_fixed_20260609.npz")
    out = Path(f"data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs{fs}_s0_nondropbge_cmpcls_evtype_fixed_room_alias_20260609.npz")
    z = np.load(src, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["room_id"] = np.asarray([room_by_pair.get(str(pid), "") for pid in z["pair_id"]], dtype=object)
    for suffix in ("__p", "__yhat"):
        d[EV_ALIAS + suffix] = z[EV_LONG + suffix]
    np.savez_compressed(out, **d)
PY
```

五划分 evidence-type / guarded family pooled bootstrap：

```bash
EV=evtype_adapt_score_src0_po_medium_decision_po_medium
GUARD=rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect

PYTHONPATH=src python -m models.bootstrap_oof_methods \
  --case fs012=data/final/cleancl/oof_evidence_type_adapter_screen_room_20260609.npz \
  --case fs3=data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs3_s0_nondropbge_cmpcls_evtype_fixed_room_alias_20260609.npz \
  --case fs4=data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs4_s0_nondropbge_cmpcls_evtype_fixed_room_alias_20260609.npz \
  --method $EV \
  --method $GUARD \
  --method rankavg_sourcefirst_cm_pcls_bge \
  --baseline bge_lr \
  --baseline $GUARD \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key room_id \
  --skip_case \
  --only_group \
  --out data/final/cleancl/evidence_type_adapter_fs0_fs4_room_bootstrap_5k_20260609.json
```

已完成结果：fs3 evidence-type 为 0.4914 / 0.6336 / 0.5819 / 0.6043，低于旧 guarded 0.4953 / 0.6339 / 0.5886 / 0.6098；fs4 evidence-type 为 0.5201 / 0.6549 / 0.5938 / 0.6267，低于旧 guarded 0.5169 / 0.6554 / 0.5971 / 0.6276 的 Macro-F1。五划分 pooled 中 evidence-type 为 0.5039 / 0.6424 / 0.6005 / 0.6277，相对 BGE room-level p(AP/AUROC/Macro)=0.0008 / 0.0500 / 0.0004；旧 guarded 为 0.5029 / 0.6422 / 0.6010 / 0.6281，相对 BGE p=0.0008 / 0.0366 / 0.0004。evidence-type 相对旧 guarded 不显著且 Macro/wF1 略低；最终主表应优先报告五划分 guarded family，evidence-type 作为三划分可解释 repair 和机制分析。

重算 common OOF method audit：

```bash
PYTHONPATH=src python src/models/diagnose_common_oof_methods.py \
  --n_boot 1000 \
  --out data/final/cleancl/common_oof_method_sweep_20260608.json
```

重算 relation adapter OOF screen：

```bash
PYTHONPATH=src python src/models/diagnose_relation_oof_adapter.py \
  --n_boot 1000 \
  --out data/final/cleancl/relation_oof_adapter_screen_20260608.json \
  --dump_oof data/final/cleancl/oof_relation_adapter_screen_20260608.npz
```

重算 NLI source-pooling micro-calibration OOF screen：

```bash
PYTHONPATH=src python src/models/diagnose_nli_source_pooling_oof.py \
  --n_boot 3000 \
  --top_k_bootstrap 1 \
  --out data/final/cleancl/nli_source_pooling_oof_top1_3k_20260608.json \
  --dump_oof ''
```

用 evaluator 直接生成某个 fold_seed 的 evidence-type method（示例 fs1）：

```bash
FS=1
python src/models/cv_nli_predef_lowabs.py \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
  --cache data/final/cleancl/cache_nli_srcargs_a120.npz \
  --bge_tmp data/final/cleancl/cv_tmp_args_srcfirst_a120_small_e3_c10_fs${FS}_s0 \
  --cm_tmp data/final/cleancl/cv_tmp_args_srcfirst_a120_small_e3_c10_fs${FS}_s0 \
  --cm_seed 0 \
  --fold_seed ${FS} \
  --quick \
  --n_boot 0 \
  --out data/final/cleancl/cv_nli_predef_lowabs_srcargs_drop_fs${FS}_s0_nondropbge_cmpcls_evtype_smoke.json \
  --dump_oof data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs${FS}_s0_nondropbge_cmpcls_evtype_smoke.npz
```

重算 evidence-type residual 与修正版 evidence-sufficiency fallback 小规则：

```bash
PYTHONPATH=src python src/models/diagnose_evtype_residuals.py \
  --min_n 25 \
  --out data/final/cleancl/evtype_residual_diagnosis_20260608.json

PYTHONPATH=src python src/models/diagnose_evsuff_oof_rules.py \
  --n_boot 500 \
  --out data/final/cleancl/evsuff_oof_rule_screen_rawblend_20260608.json
```

重跑 source-policy experts（默认 fs1 的 `ocr_only params_only args_only`；可用环境变量改 split / policy）：

```bash
cd /mnt/gty/claimarc_active
FS=1 POLICIES="ocr_only params_only args_only" bash scripts/run_source_policy_experts_fs1.sh
FS=2 POLICIES="ocr_only params_only" nohup bash scripts/run_source_policy_experts_fs1.sh \
  > logs/source_policy_fs2_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
FS=0 POLICIES="no_args ocr_only params_only" nohup bash scripts/run_source_policy_experts_fs1.sh \
  > logs/source_policy_fs0_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

# fs3 独立复核；默认跳过内置 bootstrap，先保 OOF。
FS=3 POLICIES="no_args ocr_only params_only" N_BOOT=0 DATA=data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl \
  bash scripts/run_source_policy_experts_fs1.sh
```

重算 fs1 source-policy pooling 与 top guard bootstrap：

```bash
PYTHONPATH=src python -m models.diagnose_source_policy_pooling \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl \
  --fold_seed 1 \
  --cm_seeds 0 \
  --spec \
    sourcefirst=data/final/cleancl/cv_tmp_args_srcfirst_a120_small_e3_c10_fs1_s0 \
    noargs=data/final/cleancl/cv_tmp_noargs_small_e3_c10_fs1_s0 \
    ocr=data/final/cleancl/cv_tmp_sourcepolicy_ocr_only_small_e3_c10_fs1_s0 \
    params=data/final/cleancl/cv_tmp_sourcepolicy_params_only_small_e3_c10_fs1_s0 \
  --bge_tmp data/final/cleancl/cv_tmp_noargs_small_e3_c10_fs1_s0 \
  --n_boot 0 \
  --out data/final/cleancl/source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs1_rows_20260608.json \
  --dump_oof data/final/cleancl/oof_source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs1_rows_20260608.npz
```

fs2 source-policy pooling 复核（已完成，输出 rows/OOF 后再本地做定向 bootstrap）：

```bash
PYTHONPATH=src python -m models.diagnose_source_policy_pooling \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl \
  --fold_seed 2 \
  --cm_seeds 0 \
  --spec \
    sourcefirst=data/final/cleancl/cv_tmp_args_srcfirst_a120_small_e3_c10_fs2_s0 \
    noargs=data/final/cleancl/cv_tmp_noargs_small_e3_c10_fs2_s0 \
    ocr=data/final/cleancl/cv_tmp_sourcepolicy_ocr_only_small_e3_c10_fs2_s0 \
    params=data/final/cleancl/cv_tmp_sourcepolicy_params_only_small_e3_c10_fs2_s0 \
  --bge_tmp data/final/cleancl/cv_tmp_noargs_small_e3_c10_fs2_s0 \
  --n_boot 0 \
  --out data/final/cleancl/source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs2_rows_20260608.json \
  --dump_oof data/final/cleancl/oof_source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs2_rows_20260608.npz
```

fs0 source-policy pooling 复核：

```bash
PYTHONPATH=src python -m models.diagnose_source_policy_pooling \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl \
  --fold_seed 0 \
  --cm_seeds 0 \
  --spec \
    sourcefirst=data/final/cleancl/cv_tmp_args_srcfirst_a120_small_e3_c10_fs0_s0 \
    noargs=data/final/cleancl/cv_tmp_sourcepolicy_no_args_small_e3_c10_fs0_s0 \
    ocr=data/final/cleancl/cv_tmp_sourcepolicy_ocr_only_small_e3_c10_fs0_s0 \
    params=data/final/cleancl/cv_tmp_sourcepolicy_params_only_small_e3_c10_fs0_s0 \
  --bge_tmp data/final/cleancl/cv_tmp_sourcepolicy_no_args_small_e3_c10_fs0_s0 \
  --n_boot 0 \
  --out data/final/cleancl/source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs0_rows_20260608.json \
  --dump_oof data/final/cleancl/oof_source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs0_rows_20260608.npz
```

fs3 source-policy pooling 复核（不含 sourcefirst/args-only；先跑 `n_boot=0`，再用 OOF 做定向 bootstrap）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.diagnose_source_policy_pooling \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl \
  --fold_seed 3 \
  --spec noargs=data/final/cleancl/cv_tmp_sourcepolicy_no_args_small_e3_c10_fs3_s0 \
         ocr=data/final/cleancl/cv_tmp_sourcepolicy_ocr_only_small_e3_c10_fs3_s0 \
         params=data/final/cleancl/cv_tmp_sourcepolicy_params_only_small_e3_c10_fs3_s0 \
  --bge_tmp data/final/cleancl/cv_tmp_sourcepolicy_no_args_small_e3_c10_fs3_s0 \
  --n_boot 0 \
  --out data/final/cleancl/source_policy_pooling_guard_noargs_ocr_params_fs3_rows_noboot_20260609.json \
  --dump_oof data/final/cleancl/oof_source_policy_pooling_guard_noargs_ocr_params_fs3_rows_noboot_20260609.npz
```

已完成结果：`rankavg_all` 为 0.4834 / 0.6299 / 0.5771 / 0.5954；3k sample bootstrap vs BGE 的 AP/AUROC p=0.0183/0.0213，Macro p=0.2307。`source_masked_mean` 为 0.4848 / 0.6179 / 0.5807 / 0.5862，vs noargs Macro/wF1 p=0.0443/0.0183，但 vs BGE 不显著。结论：保留为 source reliability 消融，不扩主线。

重算 source-policy 三划分 pooled bootstrap：

```bash
PYTHONPATH=src python -m models.bootstrap_oof_methods \
  --case fs0=data/final/cleancl/oof_source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs0_rows_20260608.npz \
  --case fs1=data/final/cleancl/oof_source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs1_rows_20260608.npz \
  --case fs2=data/final/cleancl/oof_source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs2_rows_20260608.npz \
  --baseline bge_lr \
  --baseline noargs \
  --baseline sourcefirst \
  --method rankavg_all_score_bge_lr_src0_neg_guard \
  --method mean_all_score_bge_lr_src0_neg_guard \
  --n_boot 5000 \
  --seed 20260608 \
  --out data/final/cleancl/repeated_source_policy_pooling_fs0_fs1_fs2_top_bootstrap_5k_20260608.json
```

新结构烟测入口：训练期 evidence-view dropout / consistency，默认关闭。示例：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl \
  --folds 5 \
  --fold_seed 1 \
  --cm_seeds 0 \
  --baselines bge_lr \
  --encoder_name BAAI/bge-small-zh-v1.5 \
  --n_fusion 1 \
  --lora_rank 8 \
  --warmup 1 \
  --cl_epochs 2 \
  --bs 8 \
  --accum 4 \
  --cl_c_min 0.10 \
  --cl_neg_c_min 0.10 \
  --evidence_policy source_first \
  --evidence_policy_mix source_first,no_args,ocr_only,params_only \
  --tmpdir data/final/cleancl/cv_tmp_evviewmix_srcfirst_noargs_ocr_params_fs1_s0 \
  --out data/final/cleancl/cv_evviewmix_srcfirst_noargs_ocr_params_fs1_s0.json \
  --dump_oof data/final/cleancl/oof_evviewmix_srcfirst_noargs_ocr_params_fs1_s0.npz
```

该 smoke 已完成并备份。结果为负：PCLS 0.4700 / 0.6128 / 0.5779，selectiveRKC 0.4689 / 0.6119 / 0.5840，BGE 0.4657 / 0.6288 / 0.6000；相对 BGE 的 dAP +0.0033 (p=0.4065)、dAUROC -0.0163 (p=0.9290)、dMacro -0.0037 (p=0.6090)。不要优先扩展简单 `evidence_policy_mix`；source-policy 收益目前更像测试时多 expert pooling / guard，而不是训练期 view dropout。

重跑 multi-view consistency all-loss smoke（已完成且为负；只作复现用）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl \
  --folds 5 \
  --fold_seed 1 \
  --cm_seeds 0 \
  --baselines bge_lr \
  --encoder_name BAAI/bge-small-zh-v1.5 \
  --n_fusion 1 \
  --lora_rank 8 \
  --warmup 1 \
  --cl_epochs 2 \
  --bs 8 \
  --accum 4 \
  --cl_c_min 0.10 \
  --cl_neg_c_min 0.10 \
  --evidence_policy source_first \
  --view_consistency_mix no_args,ocr_only,params_only \
  --view_ce_weight 0.10 \
  --view_logit_weight 0.05 \
  --view_embed_weight 0.05 \
  --tmpdir data/final/cleancl/cv_tmp_viewcons_srcfirst_aux_noargs_ocr_params_ce010_logit005_emb005_fs1_s0 \
  --out data/final/cleancl/cv_viewcons_srcfirst_aux_noargs_ocr_params_ce010_logit005_emb005_fs1_s0.json \
  --dump_oof data/final/cleancl/oof_viewcons_srcfirst_aux_noargs_ocr_params_ce010_logit005_emb005_fs1_s0.npz
```

重跑 source-sufficiency auxiliary representation smoke（已完成且不足；只作复现用）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl \
  --folds 5 \
  --fold_seed 1 \
  --cm_seeds 0 \
  --baselines bge_lr \
  --encoder_name BAAI/bge-small-zh-v1.5 \
  --n_fusion 1 \
  --lora_rank 8 \
  --warmup 1 \
  --cl_epochs 2 \
  --bs 8 \
  --accum 4 \
  --cl_c_min 0.10 \
  --cl_neg_c_min 0.10 \
  --evidence_policy source_first \
  --source_aux_combo_weight 0.01 \
  --source_aux_conf_weight 0.01 \
  --source_aux_count_weight 0.01 \
  --tmpdir data/final/cleancl/cv_tmp_sourceaux_combo_conf_count_w001_fs1_s0 \
  --out data/final/cleancl/cv_sourceaux_combo_conf_count_w001_fs1_s0.json \
  --dump_oof data/final/cleancl/oof_sourceaux_combo_conf_count_w001_fs1_s0.npz
```

重跑 BGE reranker v2-m3 cross-encoder feature baseline（已完成且显著弱于 BGE；只作复现用）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_reranker_feature \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl \
  --model_name BAAI/bge-reranker-v2-m3 \
  --folds 5 \
  --fold_seed 1 \
  --batch_size 8 \
  --max_length 512 \
  --score_cache data/final/cleancl/reranker_bge_v2m3_srcfirst_fs1_logits.npz \
  --compare_oof data/final/cleancl/oof_sourceaux_combo_conf_count_w001_fs1_s0.npz \
  --out data/final/cleancl/cv_reranker_bge_v2m3_fs1.json \
  --dump_oof data/final/cleancl/oof_reranker_bge_v2m3_fs1.npz
```

重跑 fold-safe set-level sufficiency LR meta-head（已完成且为负；CPU 即可）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_set_sufficiency_meta \
  --oof data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz \
  --n_boot 3000 \
  --out data/final/cleancl/set_sufficiency_meta_lr_20260609.json \
  --dump_oof data/final/cleancl/oof_set_sufficiency_meta_lr_20260609.npz
```

重跑 fold-safe evidence-type 离散选择器（CPU 即可；balanced 版已完成）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_evidence_type_selector \
  --oof data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz \
  --objective balanced \
  --n_boot 3000 \
  --out data/final/cleancl/evidence_type_selector_balanced_20260609.json \
  --dump_oof data/final/cleancl/oof_evidence_type_selector_balanced_20260609.npz
```

重跑 RACL prototype verifier（CPU 即可；读取已保存 CLAIMARC fold embeddings，不重训）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_racl_prototype_verifier \
  --n_boot 0 \
  --out data/final/cleancl/racl_prototype_verifier_noboot_20260609.json \
  --dump_oof data/final/cleancl/oof_racl_prototype_verifier_noboot_20260609.npz
```

重跑 prototype + evidence-type score rankblend 诊断：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.diagnose_racl_proto_rankblend \
  --oof data/final/cleancl/oof_racl_prototype_verifier_noboot_20260609.npz \
  --out data/final/cleancl/racl_proto_evtype_rankblend_screen_20260609.json \
  --dump_oof data/final/cleancl/oof_racl_proto_evtype_rankblend_screen_20260609.npz

PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
  --case blend=data/final/cleancl/oof_racl_proto_evtype_rankblend_screen_20260609.npz \
  --method evtype_rankblend_proto50_decision_evtype \
  --baseline bge_lr \
  --baseline rankavg_sourcefirst_cm_pcls_bge \
  --baseline rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --baseline evtype_adapt_score_src0_po_medium_decision_po_medium \
  --n_boot 3000 \
  --seed 20260609 \
  --out data/final/cleancl/racl_proto_evtype_rankblend_bootstrap_20260609.json
```

重跑固定协议化 prototype calibration（推荐用于复现实验表，而不是上面的诊断网格）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_racl_proto_evtype_protocol \
  --oof data/final/cleancl/oof_racl_prototype_verifier_noboot_20260609.npz \
  --out data/final/cleancl/racl_proto_evtype_protocol_20260609.json \
  --dump_oof data/final/cleancl/oof_racl_proto_evtype_protocol_20260609.npz

PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
  --case protocol=data/final/cleancl/oof_racl_proto_evtype_protocol_20260609.npz \
  --method evtype_proto_cal50_decision_evtype \
  --method evtype_proto_raw25_decision_evtype \
  --baseline bge_lr \
  --baseline rankavg_sourcefirst_cm_pcls_bge \
  --baseline rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --baseline evtype_adapt_score_src0_po_medium_decision_po_medium \
  --n_boot 5000 \
  --seed 20260609 \
  --out data/final/cleancl/racl_proto_evtype_protocol_bootstrap_20260609.json
```

已完成结果：`evtype_proto_cal50_decision_evtype` 为 0.5071 / 0.6430 / 0.6084 / 0.6355；`evtype_proto_raw25_decision_evtype` 为 0.5070 / 0.6438 / 0.6084 / 0.6355。二者相对 BGE 和旧 CM+BGE 三项显著；相对 evidence-type adapter 未显著。因此它是 ranking/screening 的协议化最高点估计，不是最终分类闭环。

固定协议化 prototype calibration 的 room-level group bootstrap：

```bash
python - <<'PY'
import json, numpy as np
from pathlib import Path
src = Path("data/final/cleancl/oof_racl_proto_evtype_protocol_20260609.npz")
out = Path("data/final/cleancl/oof_racl_proto_evtype_protocol_room_20260609.npz")
room_by_pair, attr_by_pair = {}, {}
with open("data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        pid = str(r.get("pair_id", ""))
        if pid:
            room_by_pair[pid] = str(r.get("room_id", ""))
            attr_by_pair[pid] = str(r.get("attribute_id", ""))
z = np.load(src, allow_pickle=True)
d = {k: z[k] for k in z.files}
pairs = [str(x) for x in z["pair_id"]]
d["room_id"] = np.asarray([room_by_pair.get(pid, "") for pid in pairs], dtype=object)
if "attribute_id" not in d:
    d["attribute_id"] = np.asarray([attr_by_pair.get(pid, "") for pid in pairs], dtype=object)
np.savez_compressed(out, **d)
PY

PYTHONPATH=src python -m models.bootstrap_oof_methods \
  --case protoev=data/final/cleancl/oof_racl_proto_evtype_protocol_room_20260609.npz \
  --method evtype_proto_cal50_decision_evtype \
  --method evtype_proto_raw25_decision_evtype \
  --method evtype_adapt_score_src0_po_medium_decision_po_medium \
  --method rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --baseline bge_lr \
  --baseline evtype_adapt_score_src0_po_medium_decision_po_medium \
  --baseline rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key room_id \
  --only_group \
  --out data/final/cleancl/racl_proto_evtype_protocol_room_bootstrap_group_5k_20260609.json
```

已完成结果：`evtype_proto_cal50_decision_evtype` room-level vs BGE p(AP/AUROC/Macro)=0.0004 / 0.0674 / 0.0000；`evtype_proto_raw25_decision_evtype` vs BGE p=0.0004 / 0.0518 / 0.0000。二者相对 pure evidence-type adapter 的 AP/AUROC/Macro 均不显著（cal50: 0.3514/0.2642/1.0000；raw25: 0.3692/0.1826/1.0000）。

重跑 prototype decision feature（source-poor sufficiency guard；CPU）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_racl_proto_decision_feature \
  --oof data/final/cleancl/oof_racl_proto_evtype_protocol_20260609.npz \
  --n_boot 0 \
  --out data/final/cleancl/racl_proto_decision_feature_macro_20260609.json \
  --dump_oof data/final/cleancl/oof_racl_proto_decision_feature_macro_20260609.npz

PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
  --case proto_decision=data/final/cleancl/oof_racl_proto_decision_feature_macro_20260609.npz \
  --method proto_decision_fixed_veto0.20_promote0.75_raw_src0_score_raw25 \
  --method proto_decision_cvselect_src0nested_macro_raw25 \
  --method proto_decision_cvselect_macro_raw25 \
  --baseline bge_lr \
  --baseline rankavg_sourcefirst_cm_pcls_bge \
  --baseline rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --baseline evtype_adapt_score_src0_po_medium_decision_po_medium \
  --baseline evtype_proto_raw25_decision_evtype \
  --n_boot 5000 \
  --seed 20260609 \
  --out data/final/cleancl/racl_proto_decision_feature_top2_5k_20260609.json

PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
  --case proto_decision=data/final/cleancl/oof_racl_proto_decision_feature_macro_20260609.npz \
  --method proto_decision_fixed_veto0.20_promote0.75_raw_src0_score_raw25 \
  --method proto_decision_cvselect_src0nested_macro_raw25 \
  --baseline evtype_adapt_score_src0_po_medium_decision_po_medium \
  --baseline evtype_proto_raw25_decision_evtype \
  --group_key pair_id \
  --skip_case \
  --only_group \
  --n_boot 5000 \
  --seed 20260609 \
  --out data/final/cleancl/racl_proto_decision_feature_src0nested_group_only_5k_20260609.json
```

已完成正式结果：fixed source0 rule `proto_decision_fixed_veto0.20_promote0.75_raw_src0_score_raw25` 为 0.5070 / 0.6438 / 0.6142 / 0.6454；相对 evidence-type adapter 的 Macro-F1 sample p=0.0012，pair-level group p=0.0090。source0-only nested selector `proto_decision_cvselect_src0nested_macro_raw25` 为 0.5070 / 0.6438 / 0.6142 / 0.6434；相对 evidence-type adapter 的 Macro-F1 sample p=0.0010，group p=0.0050。宽 selector 为 0.5070 / 0.6438 / 0.6126 / 0.6437，group p=0.0824，说明稳定闭环来自 source0-only prototype sufficiency guard。

新增 split 验证（fs3/fs4；先训练 grouped CV，跳过内置 bootstrap 以便先落盘）：

```bash
for FS in 3 4; do
  PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_eval \
    --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
    --folds 5 \
    --fold_seed ${FS} \
    --cm_seeds 0 \
    --baselines bge_lr \
    --encoder_name BAAI/bge-small-zh-v1.5 \
    --n_fusion 1 \
    --lora_rank 8 \
    --warmup 1 \
    --cl_epochs 2 \
    --bs 8 \
    --accum 4 \
    --cl_c_min 0.10 \
    --cl_neg_c_min 0.10 \
    --evidence_policy source_first \
    --tmpdir data/final/cleancl/cv_tmp_args_srcfirst_a120_drop_src0args_small_e3_c10_fs${FS}_s0 \
    --out data/final/cleancl/cv_args_srcfirst_a120_drop_src0args_small_e3_c10_fs${FS}_s0.json \
    --dump_oof data/final/cleancl/oof_args_srcfirst_a120_drop_src0args_small_e3_c10_fs${FS}_s0.npz \
    --n_boot 0
done
```

把单 case OOF 规范化后重跑 prototype verifier 和 BGE-base decision feature：

```bash
for FS in 3 4; do
  PYTHONPATH=src /root/miniconda3/bin/python -m models.normalize_cv_oof \
    --input fs${FS}=data/final/cleancl/oof_args_srcfirst_a120_drop_src0args_small_e3_c10_fs${FS}_s0.npz \
    --rename CLAIMARC_pcls=sourcefirst_cm_pcls_saved \
    --rename bge_lr=bge_lr \
    --out data/final/cleancl/oof_args_srcfirst_a120_drop_src0args_small_e3_c10_fs${FS}_s0_norm.npz

  PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_racl_prototype_verifier \
    --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
    --base_oof data/final/cleancl/oof_args_srcfirst_a120_drop_src0args_small_e3_c10_fs${FS}_s0_norm.npz \
    --case fs${FS}=data/final/cleancl/cv_tmp_args_srcfirst_a120_drop_src0args_small_e3_c10_fs${FS}_s0 \
    --n_boot 0 \
    --out data/final/cleancl/racl_prototype_verifier_fs${FS}_room_20260609.json \
    --dump_oof data/final/cleancl/oof_racl_prototype_verifier_fs${FS}_room_20260609.npz

  PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_racl_proto_decision_feature \
    --oof data/final/cleancl/oof_racl_prototype_verifier_fs${FS}_room_20260609.npz \
    --decision_method bge_lr \
    --score_method rankavg_bge_cm_proto_source_bin \
    --score_method rankavg_saved_cm_bge \
    --score_method proto_source_bin \
    --source0_score_method rankavg_bge_cm_proto_source_bin \
    --baseline bge_lr \
    --baseline rankavg_bge_cm_proto_source_bin \
    --baseline rankavg_saved_cm_bge \
    --baseline sourcefirst_cm_pcls_saved \
    --objective macro \
    --min_gain 0.001 \
    --n_boot 0 \
    --out data/final/cleancl/racl_proto_decision_feature_fs${FS}_bge_room_20260609.json \
    --dump_oof data/final/cleancl/oof_racl_proto_decision_feature_fs${FS}_bge_room_20260609.npz
done
```

fs3/fs4 pooled 显著性：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
  --case fs3=data/final/cleancl/oof_racl_proto_decision_feature_fs3_bge_room_20260609.npz \
  --case fs4=data/final/cleancl/oof_racl_proto_decision_feature_fs4_bge_20260609.npz \
  --method proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin \
  --method proto_decision_fixed_veto0.20_promote0.80_cal_src0_or_lowabs_score_rankavg_bge_cm_proto_source_bin \
  --baseline bge_lr \
  --baseline rankavg_bge_cm_proto_source_bin \
  --n_boot 5000 \
  --skip_case \
  --out data/final/cleancl/racl_proto_decision_feature_fs3_fs4_bge_bootstrap_5k_key_20260609.json

PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
  --case fs3=data/final/cleancl/oof_racl_proto_decision_feature_fs3_bge_room_20260609.npz \
  --case fs4=data/final/cleancl/oof_racl_proto_decision_feature_fs4_bge_20260609.npz \
  --method proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin \
  --method proto_decision_fixed_veto0.20_promote0.80_cal_src0_or_lowabs_score_rankavg_bge_cm_proto_source_bin \
  --baseline bge_lr \
  --baseline rankavg_bge_cm_proto_source_bin \
  --group_key room_id \
  --skip_case \
  --only_group \
  --n_boot 5000 \
  --out data/final/cleancl/racl_proto_decision_feature_fs3_fs4_bge_room_bootstrap_5k_key_20260609.json
```

已完成结果：fs3+fs4 pooled 的 `proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin` 为 0.5009 / 0.6448 / 0.5981 / 0.6350；相对 BGE 的 sample p=0.0146/0.0466/0.0012，room-level group p=0.0562/0.1100/0.0064。fixed cal/src0-or-lowabs rule 为 0.5009 / 0.6448 / 0.5917 / 0.6255；相对 BGE 的 sample p=0.0114/0.0486/0.0002，room-level p=0.0464/0.1062/0.0008。结论：新增 split 复现了 prototype ranking 和 Macro-F1 decision 增益，但 decision edit 相对 score-only prototype 的 Macro-F1 仍未显著，论文主表应拆开报告。

五划分 score-only prototype 稳健性（无需重训；pair-level group bootstrap）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
  --case fs012=data/final/cleancl/oof_racl_prototype_verifier_noboot_20260609.npz \
  --case fs3=data/final/cleancl/oof_racl_prototype_verifier_fs3_room_20260609.npz \
  --case fs4=data/final/cleancl/oof_racl_prototype_verifier_fs4_20260609.npz \
  --method rankavg_bge_cm_proto_source_bin \
  --baseline bge_lr \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key pair_id \
  --out data/final/cleancl/racl_prototype_score_fs0_fs4_pair_bootstrap_5k_20260609.json
```

已完成结果：fs0-fs4 pooled n=8470，`rankavg_bge_cm_proto_source_bin` 为 0.5032 / 0.6414 / 0.5893 / 0.6143；BGE 为 0.4780 / 0.6321 / 0.5834 / 0.6127。相对 BGE 的 sample p=0.0000/0.0024/0.1172，pair-level p=0.0016/0.0660/0.1842。结论：prototype score 的 AP 排序增益跨五划分复现，Macro-F1 仍需要 source/evidence sufficiency guard。

五划分统一 BGE-base prototype decision protocol（fs0-fs2 也改用 BGE decision base，与 fs3/fs4 对齐）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_racl_proto_decision_feature \
  --oof data/final/cleancl/oof_racl_prototype_verifier_noboot_20260609.npz \
  --decision_method bge_lr \
  --score_method rankavg_bge_cm_proto_source_bin \
  --source0_score_method rankavg_bge_cm_proto_source_bin \
  --baseline bge_lr \
  --baseline rankavg_bge_cm_proto_source_bin \
  --objective macro \
  --min_gain 0.001 \
  --n_boot 0 \
  --out data/final/cleancl/racl_proto_decision_feature_fs012_bgebase_20260609.json \
  --dump_oof data/final/cleancl/oof_racl_proto_decision_feature_fs012_bgebase_20260609.npz

PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
  --case fs012=data/final/cleancl/oof_racl_proto_decision_feature_fs012_bgebase_20260609.npz \
  --case fs3=data/final/cleancl/oof_racl_proto_decision_feature_fs3_bge_room_20260609.npz \
  --case fs4=data/final/cleancl/oof_racl_proto_decision_feature_fs4_bge_20260609.npz \
  --method proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin \
  --method proto_decision_fixed_veto0.20_promote0.80_cal_src0_or_lowabs_score_rankavg_bge_cm_proto_source_bin \
  --baseline bge_lr \
  --baseline rankavg_bge_cm_proto_source_bin \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key pair_id \
  --skip_case \
  --out data/final/cleancl/racl_proto_decision_feature_fs0_fs4_bgebase_pair_bootstrap_5k_20260609.json
```

已完成结果：fs0-fs4 pooled n=8470，cross-fit BGE-base prototype decision 为 0.5032 / 0.6414 / 0.5913 / 0.6244；fixed cal/src0-or-lowabs rule 为 0.5032 / 0.6414 / 0.5906 / 0.6233。cross-fit 相对 BGE 的 sample p=0.0000/0.0024/0.0020，pair-level p=0.0016/0.0660/0.0092；fixed rule 相对 BGE 的 sample p=0.0000/0.0060/0.0000，pair-level p=0.0006/0.0652/0.0002。相对 score-only prototype 的 Macro-F1 增益未显著。结论：统一 BGE-base protocol 可以显著胜 BGE 的 AP/Macro-F1，但不能声称 guard 显著胜 score-only prototype。

五划分统一 BGE-base prototype decision 的 room-level group bootstrap（先给 fs012 OOF 反接 `room_id`）：

```bash
python - <<'PY'
import json, numpy as np
from pathlib import Path
src = Path("data/final/cleancl/oof_racl_proto_decision_feature_fs012_bgebase_20260609.npz")
out = Path("data/final/cleancl/oof_racl_proto_decision_feature_fs012_bgebase_room_20260609.npz")
mp = {}
with open("data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        pid = r.get("pair_id") or f"{r.get('product_id')}__{r.get('attribute_id')}"
        mp[pid] = r.get("room_id", "")
z = np.load(src, allow_pickle=True)
d = {k: z[k] for k in z.files}
d["room_id"] = np.asarray([mp[x] for x in z["pair_id"].astype(str)], dtype=object)
np.savez_compressed(out, **d)
PY

PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
  --case fs012=data/final/cleancl/oof_racl_proto_decision_feature_fs012_bgebase_room_20260609.npz \
  --case fs3=data/final/cleancl/oof_racl_proto_decision_feature_fs3_bge_room_20260609.npz \
  --case fs4=data/final/cleancl/oof_racl_proto_decision_feature_fs4_bge_20260609.npz \
  --method proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin \
  --method proto_decision_fixed_veto0.20_promote0.80_cal_src0_or_lowabs_score_rankavg_bge_cm_proto_source_bin \
  --baseline bge_lr \
  --baseline rankavg_bge_cm_proto_source_bin \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key room_id \
  --skip_case \
  --only_group \
  --out data/final/cleancl/racl_proto_decision_feature_fs0_fs4_bgebase_room_bootstrap_5k_20260609.json
```

已完成结果：cross-fit decision 相对 BGE 的 room-level p(AP/AUROC/Macro) = 0.0062 / 0.1438 / 0.0094；fixed rule 为 0.0060 / 0.1400 / 0.0000。相对 score-only prototype 的 Macro-F1 仍不显著（cross-fit p=0.4050，fixed p=0.4562）。

五划分 guarded-score + fixed RACL source0 guard（上一版主协议候选；当前主候选在后文加入 conservative Qwen3 disagreement router）：

```bash
python scripts/build_guarded_proto_fs0_fs4_oof.py

GUARD=rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect
FIXG=proto_decision_fixed_veto0.20_promote0.75_raw_src0_score_${GUARD}

PYTHONPATH=src python -m models.cv_racl_proto_decision_feature \
  --oof data/final/cleancl/oof_guarded_proto_fs0_fs4_room_20260609.npz \
  --decision_method $GUARD \
  --score_method rankavg_bge_cm_proto_source_bin \
  --baseline bge_lr \
  --baseline $GUARD \
  --baseline evtype_adapt_score_src0_po_medium_decision_po_medium \
  --baseline rankavg_bge_cm_proto_source_bin \
  --n_boot 0 \
  --out data/final/cleancl/racl_proto_decision_feature_fs0_fs4_guardedbase_20260609.json \
  --dump_oof data/final/cleancl/oof_racl_proto_decision_feature_fs0_fs4_guardedbase_20260609.npz

PYTHONPATH=src python -m models.bootstrap_oof_methods \
  --case all=data/final/cleancl/oof_racl_proto_decision_feature_fs0_fs4_guardedbase_20260609.npz \
  --method $FIXG \
  --baseline bge_lr \
  --baseline $GUARD \
  --baseline evtype_adapt_score_src0_po_medium_decision_po_medium \
  --baseline rankavg_bge_cm_proto_source_bin \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key room_id \
  --skip_case \
  --only_group \
  --out data/final/cleancl/racl_proto_decision_feature_fs0_fs4_guardedscore_fixed_room_bootstrap_5k_20260609.json
```

已完成结果：`CM/NLI guarded + fixed RACL source0 guard` 保留 guarded score，binary decision 只在 `source_count==0` 上用 raw prototype rank 做 `veto<0.20/promote>0.75`，五划分 pooled 为 0.5029 / 0.6422 / 0.6053 / 0.6359。room-level group bootstrap 相对 BGE 的 p(AP/AUROC/Macro)=0.0006 / 0.0382 / 0.0002；相对原 guarded 的 p=1.0000 / 1.0000 / 0.0156；相对 evidence-type adapter 的 Macro-F1 p=0.0238。fs0-fs4 的 Macro-F1 相对 guarded 均为正向提升，是当前主表优先候选。

fs3/fs4 独立子集验证与翻转诊断：

```bash
python scripts/subset_oof_by_case.py \
  --oof data/final/cleancl/oof_racl_proto_decision_feature_fs0_fs4_guardedbase_20260609.npz \
  --case fs3 \
  --case fs4 \
  --out data/final/cleancl/oof_racl_proto_decision_feature_fs3_fs4_guardedbase_20260609.npz

PYTHONPATH=src python -m models.bootstrap_oof_methods \
  --case fs34=data/final/cleancl/oof_racl_proto_decision_feature_fs3_fs4_guardedbase_20260609.npz \
  --method $FIXG \
  --baseline bge_lr \
  --baseline $GUARD \
  --baseline evtype_adapt_score_src0_po_medium_decision_po_medium \
  --baseline rankavg_bge_cm_proto_source_bin \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key room_id \
  --skip_case \
  --only_group \
  --out data/final/cleancl/racl_proto_decision_feature_fs3_fs4_guardedscore_fixed_room_bootstrap_5k_20260609.json

PYTHONPATH=src python -m models.diagnose_guard_flips \
  --oof data/final/cleancl/oof_racl_proto_decision_feature_fs0_fs4_guardedbase_20260609.npz \
  --base $GUARD \
  --method $FIXG \
  --out data/final/cleancl/guarded_proto_source0_flip_diagnosis_fs0_fs4_20260609.json

PYTHONPATH=src python -m models.diagnose_guard_flips \
  --oof data/final/cleancl/oof_racl_proto_decision_feature_fs3_fs4_guardedbase_20260609.npz \
  --base $GUARD \
  --method $FIXG \
  --out data/final/cleancl/guarded_proto_source0_flip_diagnosis_fs3_fs4_20260609.json
```

已完成结果：fs3/fs4 pooled 主方法为 0.5051 / 0.6446 / 0.5952 / 0.6235；相对 BGE 的 sample p=0.0010/0.0142/0.0298，pair-level p=0.0026/0.0358/0.0334，room-level p=0.0102/0.0582/0.0444。相对原 guarded 的 Macro-F1 为正但不显著（sample/pair/room p=0.1866/0.2196/0.2282）。翻转诊断显示五划分只翻转 161/8470 条、净正确 +25；fs3/fs4 翻转 58/3388 条、净正确 +2。

现代 embedding baseline 加固（BGE-large repro 与 Qwen3-Embedding-0.6B）：

```bash
# Qwen3 如无 HuggingFace 连接，可优先走 ModelScope。
python - <<'PY'
from modelscope import snapshot_download
print(snapshot_download("Qwen/Qwen3-Embedding-0.6B"))
PY

PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_embedding_baseline \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
  --model_name /root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B \
  --method_name qwen3emb06b_lr \
  --fold_seed 0 --fold_seed 1 --fold_seed 2 --fold_seed 3 --fold_seed 4 \
  --folds 5 \
  --cache data/final/cleancl/cache_qwen3emb06b_srcargs_drop_20260609.npz \
  --batch_size 16 \
  --trust_remote_code \
  --out data/final/cleancl/cv_embedding_qwen3emb06b_srcargs_drop_fs0_fs4_20260609.json \
  --dump_oof data/final/cleancl/oof_embedding_qwen3emb06b_srcargs_drop_fs0_fs4_20260609.npz

python scripts/merge_oof_methods.py \
  --base data/final/cleancl/oof_guardedbase_plus_bge_large_repro_fs0_fs4_20260609.npz \
  --src data/final/cleancl/oof_embedding_qwen3emb06b_srcargs_drop_fs0_fs4_20260609.npz \
  --method qwen3emb06b_lr \
  --out data/final/cleancl/oof_guardedbase_plus_modern_emb_fs0_fs4_20260609.npz

PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
  --case all=data/final/cleancl/oof_guardedbase_plus_modern_emb_fs0_fs4_20260609.npz \
  --method $FIXG \
  --baseline qwen3emb06b_lr \
  --baseline bge_large_repro_lr \
  --baseline bge_lr \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key room_id \
  --skip_case \
  --only_group \
  --out data/final/cleancl/guarded_source0_vs_modern_emb_room_bootstrap_5k_20260609.json

# Qwen3 query-prompt check. This is weaker than no-prompt on this task,
# but useful as a prompt-aware modern embedding robustness baseline.
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_embedding_baseline \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
  --model_name /root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B \
  --method_name qwen3emb06b_query_lr \
  --fold_seed 0 --fold_seed 1 --fold_seed 2 --fold_seed 3 --fold_seed 4 \
  --folds 5 \
  --cache data/final/cleancl/cache_qwen3emb06b_query_srcargs_drop_20260609.npz \
  --batch_size 16 \
  --claim_prompt_name query \
  --trust_remote_code \
  --out data/final/cleancl/cv_embedding_qwen3emb06b_query_srcargs_drop_fs0_fs4_20260609.json \
  --dump_oof data/final/cleancl/oof_embedding_qwen3emb06b_query_srcargs_drop_fs0_fs4_20260609.npz

python scripts/merge_oof_methods.py \
  --base data/final/cleancl/oof_guardedbase_plus_modern_emb_fs0_fs4_20260609.npz \
  --src data/final/cleancl/oof_embedding_qwen3emb06b_query_srcargs_drop_fs0_fs4_20260609.npz \
  --method qwen3emb06b_query_lr \
  --out data/final/cleancl/oof_guardedbase_plus_modern_emb_prompt_fs0_fs4_20260609.npz

OOF=data/final/cleancl/oof_guardedbase_plus_modern_emb_prompt_fs0_fs4_20260609.npz
for SCOPE in sample pair room; do
  EXTRA=()
  OUT=data/final/cleancl/guarded_source0_vs_modern_emb_prompt_${SCOPE}_bootstrap_5k_20260609.json
  if [ "$SCOPE" = "pair" ]; then EXTRA=(--group_key pair_id --only_group); fi
  if [ "$SCOPE" = "room" ]; then EXTRA=(--group_key room_id --only_group); fi
  PYTHONPATH=src /root/miniconda3/bin/python -m models.bootstrap_oof_methods \
    --case all=$OOF \
    --method $FIXG \
    --baseline qwen3emb06b_query_lr \
    --baseline qwen3emb06b_lr \
    --baseline bge_large_repro_lr \
    --baseline bge_lr \
    --n_boot 5000 \
    --seed 20260609 \
    --skip_case \
    "${EXTRA[@]}" \
    --out "$OUT"
done
```

已完成结果：`qwen3emb06b_lr` 为 0.4845 / 0.6376 / 0.5900 / 0.6111，强于旧 BGE 的 AUROC/Macro-F1；当前主方法相对 Qwen3 no-prompt 的 sample p=0.0020/0.1444/0.0010，pair-level p=0.0466/0.2578/0.0152，room-level p=0.0718/0.3072/0.0088。也就是说，主方法显著胜现代 embedding baseline 的 Macro-F1；AP 正向但 room-level 边界；AUROC 不强称。Qwen3 query-prompt baseline 为 0.4813 / 0.6351 / 0.5835 / 0.6093，弱于 no-prompt；当前主方法相对 query-prompt 的 sample p=0.0002/0.0468/0.0000，pair-level p=0.0230/0.1576/0.0016，room-level p=0.0344/0.1856/0.0032。BGE-large repro baseline 为 0.4816 / 0.6315 / 0.5845 / 0.6206，当前主方法相对它的 room-level p=0.0092/0.0424/0.0000。

Fold-safe Qwen3 disagreement router（decision module / 上一版主候选）：

```bash
MAIN=proto_decision_fixed_veto0.20_promote0.75_raw_src0_score_rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect
SEL=router_select_macro_mind00040_nvmind00150_maxfr0100_qwen3emb06b_lr_on_${MAIN}
FIX=router_fixed_b030_q015_all_both_qwen3emb06b_lr_on_${MAIN}

PYTHONPATH=src python -m models.cv_oof_disagreement_router \
  --oof data/final/cleancl/oof_guardedbase_plus_modern_emb_prompt_fs0_fs4_20260609.npz \
  --base_method $MAIN \
  --aux_method qwen3emb06b_lr \
  --min_val_delta 0.004 \
  --non_veto_min_val_delta 0.015 \
  --max_val_flip_rate 0.10 \
  --baseline bge_lr \
  --baseline qwen3emb06b_query_lr \
  --out data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs0_fs4_20260609.json \
  --dump_oof data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs0_fs4_20260609.npz

OOF=data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs0_fs4_20260609.npz
for SCOPE in sample pair room; do
  EXTRA=()
  OUT=data/final/cleancl/oof_disagreement_router_qwen3_conservative_${SCOPE}_bootstrap_5k_20260609.json
  if [ "$SCOPE" = "pair" ]; then EXTRA=(--group_key pair_id --only_group); fi
  if [ "$SCOPE" = "room" ]; then EXTRA=(--group_key room_id --only_group); fi
  PYTHONPATH=src python -m models.bootstrap_oof_methods \
    --case all=$OOF \
    --method "$SEL" \
    --method "$FIX" \
    --baseline "$MAIN" \
    --baseline qwen3emb06b_lr \
    --baseline bge_lr \
    --n_boot 5000 \
    --seed 20260609 \
    --skip_case \
    "${EXTRA[@]}" \
    --out "$OUT"
done

PYTHONPATH=src python -m models.bootstrap_oof_methods \
  --case all=$OOF \
  --method "$SEL" \
  --baseline rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --baseline evtype_adapt_score_src0_po_medium_decision_po_medium \
  --baseline bge_large_repro_lr \
  --baseline qwen3emb06b_query_lr \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key room_id \
  --skip_case \
  --only_group \
  --out data/final/cleancl/oof_disagreement_router_qwen3_conservative_extra_room_bootstrap_5k_20260609.json

PYTHONPATH=src python scripts/subset_oof_by_case.py \
  --oof data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs0_fs4_20260609.npz \
  --case fs3 \
  --case fs4 \
  --out data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs3_fs4_20260609.npz

OOF=data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs3_fs4_20260609.npz
for SCOPE in sample pair room; do
  EXTRA=()
  OUT=data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs3_fs4_${SCOPE}_bootstrap_5k_20260609.json
  if [ "$SCOPE" = "pair" ]; then EXTRA=(--group_key pair_id --only_group); fi
  if [ "$SCOPE" = "room" ]; then EXTRA=(--group_key room_id --only_group); fi
  PYTHONPATH=src python -m models.bootstrap_oof_methods \
    --case fs34=$OOF \
    --method "$SEL" \
    --method "$FIX" \
    --baseline "$MAIN" \
    --baseline qwen3emb06b_lr \
    --baseline bge_lr \
    --n_boot 5000 \
    --seed 20260609 \
    --skip_case \
    "${EXTRA[@]}" \
    --out "$OUT"
done

PYTHONPATH=src python -m models.diagnose_guard_flips \
  --oof data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs0_fs4_20260609.npz \
  --base $MAIN \
  --method $SEL \
  --out data/final/cleancl/oof_disagreement_router_qwen3_conservative_flip_diagnosis_fs0_fs4_20260609.json

PYTHONPATH=src python -m models.diagnose_guard_flips \
  --oof data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs3_fs4_20260609.npz \
  --base $MAIN \
  --method $SEL \
  --out data/final/cleancl/oof_disagreement_router_qwen3_conservative_flip_diagnosis_fs3_fs4_20260609.json
```

已完成结果：保守 selected Qwen3 router 为 0.5029 / 0.6422 / 0.6109 / 0.6411，相对 fixed source0 guard 的 sample/pair/room Macro p=0.0026/0.0056/0.0112。相对原 CM/NLI guarded 的 room Macro p=0.0004，相对 evidence-type adapter 的 room Macro p=0.0004，相对 no-prompt Qwen3 的 room Macro p=0.0004。fs3/fs4 子集为 0.5051 / 0.6446 / 0.6076 / 0.6392，相对 fixed source0 guard 的 sample/pair/room Macro p=0.0004/0.0004/0.0028。全量翻转 270/8470 条，净正确 +62；fs3/fs4 翻转 152/3388 条，净正确 +38。

RACL raw25 score + conservative router decision（当前 top-line 点估计）：

```bash
ROUTER=router_select_macro_mind00040_nvmind00150_maxfr0100_qwen3emb06b_lr_on_proto_decision_fixed_veto0.20_promote0.75_raw_src0_score_rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect

PYTHONPATH=src python -m models.cv_racl_proto_evtype_protocol \
  --oof data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs0_fs4_20260609.npz \
  --decision_method "$ROUTER" \
  --decision_label router \
  --out data/final/cleancl/racl_proto_evtype_score_router_decision_fs0_fs4_20260609.json \
  --dump_oof data/final/cleancl/oof_racl_proto_evtype_score_router_decision_fs0_fs4_20260609.npz

OOF=data/final/cleancl/oof_racl_proto_evtype_score_router_decision_fs0_fs4_20260609.npz
RAW=evtype_proto_raw25_decision_router
CAL=evtype_proto_cal50_decision_router
EV=evtype_adapt_score_src0_po_medium_decision_po_medium
for SCOPE in sample pair room; do
  EXTRA=()
  OUT=data/final/cleancl/racl_proto_evtype_score_router_${SCOPE}_bootstrap_5k_20260609.json
  if [ "$SCOPE" = "pair" ]; then EXTRA=(--group_key pair_id --only_group); fi
  if [ "$SCOPE" = "room" ]; then EXTRA=(--group_key room_id --only_group); fi
  PYTHONPATH=src python -m models.bootstrap_oof_methods \
    --case all=$OOF \
    --method $RAW \
    --method $CAL \
    --baseline "$ROUTER" \
    --baseline $EV \
    --baseline qwen3emb06b_lr \
    --baseline bge_lr \
    --n_boot 5000 \
    --seed 20260609 \
    --skip_case \
    "${EXTRA[@]}" \
    --out "$OUT"
done

PYTHONPATH=src python -m models.bootstrap_oof_methods \
  --case all=$OOF \
  --method $RAW \
  --baseline "$ROUTER" \
  --baseline proto_decision_fixed_veto0.20_promote0.75_raw_src0_score_rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --baseline rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect \
  --baseline $EV \
  --baseline qwen3emb06b_lr \
  --baseline qwen3emb06b_query_lr \
  --baseline bge_large_repro_lr \
  --baseline bge_lr \
  --n_boot 5000 \
  --seed 20260609 \
  --group_key room_id \
  --skip_case \
  --only_group \
  --out data/final/cleancl/racl_proto_evtype_score_router_extra_room_bootstrap_5k_20260609.json

PYTHONPATH=src python scripts/subset_oof_by_case.py \
  --oof data/final/cleancl/oof_racl_proto_evtype_score_router_decision_fs0_fs4_20260609.npz \
  --case fs3 \
  --case fs4 \
  --out data/final/cleancl/oof_racl_proto_evtype_score_router_decision_fs3_fs4_20260609.npz

OOF=data/final/cleancl/oof_racl_proto_evtype_score_router_decision_fs3_fs4_20260609.npz
for SCOPE in sample pair room; do
  EXTRA=()
  OUT=data/final/cleancl/racl_proto_evtype_score_router_fs3_fs4_${SCOPE}_bootstrap_5k_20260609.json
  if [ "$SCOPE" = "pair" ]; then EXTRA=(--group_key pair_id --only_group); fi
  if [ "$SCOPE" = "room" ]; then EXTRA=(--group_key room_id --only_group); fi
  PYTHONPATH=src python -m models.bootstrap_oof_methods \
    --case all=$OOF \
    --method $RAW \
    --method $CAL \
    --baseline "$ROUTER" \
    --baseline $EV \
    --baseline qwen3emb06b_lr \
    --baseline bge_lr \
    --n_boot 5000 \
    --seed 20260609 \
    --skip_case \
    "${EXTRA[@]}" \
    --out "$OUT"
done
```

已完成结果：`evtype_proto_raw25_decision_router` 为 0.5084 / 0.6456 / 0.6109 / 0.6411。相对 BGE 的 sample/pair/room p(AP/AUROC/Macro)=0.0000/0.0000/0.0000，0.0000/0.0048/0.0000，0.0004/0.0316/0.0000；相对 no-prompt Qwen3 的 room p=0.0274/0.1942/0.0002；相对 BGE-large repro 的 room p=0.0034/0.0290/0.0000。相对旧 conservative router score 的 AP/AUROC sample p=0.0226/0.0070，但 pair/room 只是正向不显著（pair 0.1232/0.0952；room 0.1650/0.1558）。fs3/fs4 子集为 0.5114 / 0.6482 / 0.6076 / 0.6392，相对 BGE 的 room p=0.0036/0.0370/0.0002。结论写法：这是当前 top-line 点估计和 RACL ranking 机制；不要声称 raw25 在 room-level 显著击败旧 router score。

No-selector fixed Qwen3 veto robustness check：

```bash
MAIN=proto_decision_fixed_veto0.20_promote0.75_raw_src0_score_rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect
FIXV=router_fixed_b020_q010_all_veto_qwen3emb06b_lr_on_${MAIN}

PYTHONPATH=src python -m models.cv_oof_disagreement_router \
  --oof data/final/cleancl/oof_guardedbase_plus_modern_emb_prompt_fs0_fs4_20260609.npz \
  --base_method $MAIN \
  --aux_method qwen3emb06b_lr \
  --fixed_base_uncertain_max 0.20 \
  --fixed_aux_confident_min 0.10 \
  --fixed_source_mode all \
  --fixed_direction veto \
  --baseline bge_lr \
  --baseline qwen3emb06b_query_lr \
  --out data/final/cleancl/oof_disagreement_router_qwen3_fixedveto_fs0_fs4_20260609.json \
  --dump_oof data/final/cleancl/oof_disagreement_router_qwen3_fixedveto_fs0_fs4_20260609.npz

OOF=data/final/cleancl/oof_disagreement_router_qwen3_fixedveto_fs0_fs4_20260609.npz
for SCOPE in sample pair room; do
  EXTRA=()
  OUT=data/final/cleancl/oof_disagreement_router_qwen3_fixedveto_${SCOPE}_bootstrap_5k_20260609.json
  if [ "$SCOPE" = "pair" ]; then EXTRA=(--group_key pair_id --only_group); fi
  if [ "$SCOPE" = "room" ]; then EXTRA=(--group_key room_id --only_group); fi
  PYTHONPATH=src python -m models.bootstrap_oof_methods \
    --case all=$OOF \
    --method "$FIXV" \
    --baseline "$MAIN" \
    --baseline qwen3emb06b_lr \
    --baseline bge_lr \
    --n_boot 5000 \
    --seed 20260609 \
    --skip_case \
    "${EXTRA[@]}" \
    --out "$OUT"
done

PYTHONPATH=src python scripts/subset_oof_by_case.py \
  --oof data/final/cleancl/oof_disagreement_router_qwen3_fixedveto_fs0_fs4_20260609.npz \
  --case fs3 \
  --case fs4 \
  --out data/final/cleancl/oof_disagreement_router_qwen3_fixedveto_fs3_fs4_20260609.npz
```

已完成结果：fixed veto 为 0.5029 / 0.6422 / 0.6092 / 0.6383，相对 fixed source0 guard 的 sample/pair/room Macro p=0.0016/0.0106/0.0046；全量翻转 145/8470 条，全部 veto，净正确 +55。fs3/fs4 为 0.5051 / 0.6446 / 0.5975 / 0.6261，相对 fixed source0 guard 的 sample/pair/room Macro p=0.0890/0.1216/0.0576，方向为正但未显著；相对 BGE 的 room p=0.0100/0.0498/0.0200。

重跑训练期 RACL prototype auxiliary smoke（fs1/drop-src0args；GPU）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
  --folds 5 \
  --fold_seed 1 \
  --cm_seeds 0 \
  --baselines bge_lr \
  --encoder_name BAAI/bge-small-zh-v1.5 \
  --n_fusion 1 \
  --lora_rank 8 \
  --warmup 1 \
  --cl_epochs 2 \
  --bs 8 \
  --accum 4 \
  --cl_c_min 0.10 \
  --cl_neg_c_min 0.10 \
  --evidence_policy source_first \
  --proto_aux_weight 0.02 \
  --proto_aux_group source_bin \
  --proto_aux_tau 0.10 \
  --proto_aux_min_class 3 \
  --proto_aux_c_min 0.10 \
  --tmpdir data/final/cleancl/cv_tmp_protoaux_srcbin_w002_fs1_s0 \
  --out data/final/cleancl/cv_protoaux_srcbin_w002_fs1_s0.json \
  --dump_oof data/final/cleancl/oof_protoaux_srcbin_w002_fs1_s0.npz
```

在 prototype auxiliary 训练出的 fs1 embedding 上重跑 prototype verifier（CPU）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_racl_prototype_verifier \
  --case fs1=data/final/cleancl/cv_tmp_protoaux_srcbin_w002_fs1_s0 \
  --n_boot 0 \
  --out data/final/cleancl/racl_prototype_verifier_protoaux_srcbin_w002_fs1_20260609.json \
  --dump_oof data/final/cleancl/oof_racl_prototype_verifier_protoaux_srcbin_w002_fs1_20260609.npz
```

该 smoke 已完成：PCLS 为 0.4845 / 0.6196 / 0.5845 / 0.5992，BGE 为 0.4736 / 0.6288 / 0.5928 / 0.6336；PCLS vs BGE 的 dAP +0.0103 (p=0.2340)、dAUROC -0.0093 (p=0.7970)、dMacro -0.0001 (p=0.4985)。同一新 embedding 的 `rankavg_bge_cm_proto_source_bin` 为 0.5022 / 0.6414 / 0.5942 / 0.6065。结论：prototype geometry 作为 score source 有用，但简单 prototype CE auxiliary 不能解决 fs1 分类边界；不要优先扩大该权重网格。

重跑训练期 RACL prototype margin auxiliary smoke（fs1/drop-src0args；GPU）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
  --folds 5 \
  --fold_seed 1 \
  --cm_seeds 0 \
  --baselines bge_lr \
  --encoder_name BAAI/bge-small-zh-v1.5 \
  --n_fusion 1 \
  --lora_rank 8 \
  --warmup 1 \
  --cl_epochs 2 \
  --bs 8 \
  --accum 4 \
  --cl_c_min 0.10 \
  --cl_neg_c_min 0.10 \
  --evidence_policy source_first \
  --proto_aux_weight 0.02 \
  --proto_aux_group source_bin \
  --proto_aux_mode margin \
  --proto_aux_margin 0.15 \
  --proto_aux_tau 0.10 \
  --proto_aux_min_class 3 \
  --proto_aux_c_min 0.10 \
  --tmpdir data/final/cleancl/cv_tmp_protoaux_margin_srcbin_w002_m015_fs1_s0 \
  --out data/final/cleancl/cv_protoaux_margin_srcbin_w002_m015_fs1_s0.json \
  --dump_oof data/final/cleancl/oof_protoaux_margin_srcbin_w002_m015_fs1_s0.npz \
  --n_boot 0

PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_racl_prototype_verifier \
  --case fs1=data/final/cleancl/cv_tmp_protoaux_margin_srcbin_w002_m015_fs1_s0 \
  --n_boot 0 \
  --out data/final/cleancl/racl_prototype_verifier_protoaux_margin_srcbin_w002_m015_fs1_20260609.json \
  --dump_oof data/final/cleancl/oof_racl_prototype_verifier_protoaux_margin_srcbin_w002_m015_fs1_20260609.npz
```

该 smoke 已完成：PCLS 为 0.4852 / 0.6196 / 0.5832 / 0.6005；BGE 为 0.4736 / 0.6288 / 0.5928 / 0.6336。verifier 中 `rankavg_bge_cm_proto_source_bin` 为 0.5020 / 0.6418 / 0.5892 / 0.6082，`rankavg_bge_cm_proto_global` 为 0.5001 / 0.6372 / 0.5971 / 0.6173。3k bootstrap 相对 `bge_lr_saved`：CE source-bin p(AP/AUROC/Macro)=0.0043/0.0487/0.4757；margin source-bin p=0.0060/0.0407/0.6857；margin global p=0.0093/0.1290/0.3647。结论：margin 目标保留 AP/AUROC 诊断收益但没有改善 Macro-F1 闭环，不扩展到 fs3/fs4。

重跑训练期 evidence-type hard-negative 过滤（fs1/drop-src0args smoke）：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
  --folds 5 \
  --fold_seed 1 \
  --cm_seeds 0 \
  --baselines bge_lr \
  --encoder_name BAAI/bge-small-zh-v1.5 \
  --n_fusion 1 \
  --lora_rank 8 \
  --warmup 1 \
  --cl_epochs 2 \
  --bs 8 \
  --accum 4 \
  --cl_c_min 0.10 \
  --cl_neg_c_min 0.10 \
  --cl_neg_filter medium_evtype_conf \
  --tmpdir data/final/cleancl/cv_tmp_args_srcfirst_a120_drop_src0args_evhn_medium_fs1_s0 \
  --out data/final/cleancl/cv_args_srcfirst_a120_drop_src0args_evhn_medium_fs1_s0.json
```

重跑训练期 evidence-type hard-negative soft bonus：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m models.cv_eval \
  --dataset data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl \
  --folds 5 \
  --fold_seed 1 \
  --cm_seeds 0 \
  --baselines bge_lr \
  --encoder_name BAAI/bge-small-zh-v1.5 \
  --n_fusion 1 \
  --lora_rank 8 \
  --warmup 1 \
  --cl_epochs 2 \
  --bs 8 \
  --accum 4 \
  --cl_c_min 0.10 \
  --cl_neg_c_min 0.10 \
  --cl_neg_bonus 0.05 \
  --cl_neg_bonus_filter medium_evtype_conf \
  --tmpdir data/final/cleancl/cv_tmp_args_srcfirst_a120_drop_src0args_evhn_soft005_fs1_s0 \
  --out data/final/cleancl/cv_args_srcfirst_a120_drop_src0args_evhn_soft005_fs1_s0.json
```
- 训练期 source-domain CL reweight 第一轮为负：`source0_cl_scale=0.20, source_rich_cl_scale=1.50` 在 fs1 drop-src0args 上弱于不加权版本，不要优先重复该方向。
- 普通 BGE BCE 蒸馏、保守分歧蒸馏、teacher-guided RACL、阈值/先验校准、`agree_pos` 都已试过，尚未解决 repeated-CV 分类显著优势。
- 当前待办：Qwen-Flash direct LLM、BGE reranker v2-m3 direct/LR、简单 outer-train LR reliability head、BGE similarity evidence-set head、激进 source-domain CL reweight、简单 sourceveto、bgerateguard、nlievidenceveto、普通 relation stacker、NLI 单特征微校准、evidence-sufficiency narrow fallback、`predef_lowabs_valselect_*`、简单 evidence-view dropout、view consistency、source auxiliary representation、fold-safe set-sufficiency LR head、fold-safe evidence-type free selector、第一版 prototype CE auxiliary 和 prototype margin auxiliary 均不足以最终闭环；source-policy multi-instance pooling 已在 repeated-CV 上显著胜 BGE/noargs/sourcefirst 的排序指标，但 fs3 只复现 AP/AUROC、没有复现 Macro-F1，且整体弱于 evidence-type/prototype 主线。RACL prototype rankblend / BGE-base prototype decision protocol 是新的主线：五划分 pooled 中 AP/Macro-F1 相对 BGE 显著，guard 相对 score-only prototype 的 Macro-F1 增益未显著。下一步优先把 fixed prototype score/decision protocol 写成可预注册结构，并沿 evidence-type adapter 做极窄先验化 FP 约束，而不是扩大后处理 selector 或训练期 prototype auxiliary 网格。
