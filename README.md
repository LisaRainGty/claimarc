# CLAIMARC 项目数据与文档目录

> **C**laim-Aware Misleading-Advertising detector with **R**etrieval-augmented **C**ontrastive learning  
> 直播电商消费者感知虚假宣传风险的端到端预测与解释框架。

本目录是 CLAIMARC 研究项目的自包含工作区：包含 **proposal**、**主索引**与 **三份原始数据**（主播话术 SRT 切片、用户评论、商品图片）的完整硬拷贝，可独立打包/迁移，不再依赖 `/mnt/livestream` 与 `/mnt/gty/product_images` 的原始路径。

---

## 1. 目录结构

```
claimarc/
├── README.md                          ← 本文件
├── docs/
│   └── proposal.md                    ← 研究提案（§1–§8 全文）
└── data/
    ├── index/
    │   └── product_index.json         ← 主索引（628 商品 × 9 一级品类）
    ├── raw/                           ← 三份原始数据（与 §7.1 上游一一对应）
    │   ├── srt_cut/                   ← 主播话术 SRT 切片
    │   │   └── <L1>/<L2>/<room>/<session>/<clip>.{srt,txt}
    │   ├── comment/                   ← 用户评论 xls
    │   │   └── <L1>/<L2>/<room_pinyin>/飞瓜数据_..._<商品名>.xls
    │   └── product_images/            ← 商品主图 + 详情图
    │       └── <product_id>/{main.webp, detail_001.jpeg, ...}
    ├── processed/                     ← 预留：Stage A/B/C 中间产物
    │   └── (空，待 §7.1 数据流水线产出)
    └── final/                         ← 预留：Stage 末端 dataset.jsonl + 标签
        └── (空，待 §7.2 弱监督标签产出)
```

`processed/` 与 `final/` 当前为空，按 `proposal.md` §7.1 的三阶段流水线（A 评论侧 → B 主播话术 + 评论对齐 → C 三源商品事实抽取）逐步落产物，整体路径约定如下：

| 阶段 | 输入 | 产出 | 落地位置 |
|---|---|---|---|
| Stage A | `data/raw/comment/`、`data/index/product_index.json` | `comment_attributes.jsonl` | `data/processed/stageA/` |
| Stage B | Stage A 产出 + `data/raw/srt_cut/` | `pair_records.jsonl`（主播话术 ↔ 评论对齐） | `data/processed/stageB/` |
| Stage C | `data/raw/product_images/` + `data/index/product_index.json` 的 `产品参数` | `fact_records.jsonl`（PARAM/OCR/VLM 三源证据） | `data/processed/stageC/` |
| §7.2 弱监督 | Stage A/B 产出 | $(y, c)$ 标签与样本权重 | `data/final/dataset.jsonl` |

---

## 2. 数据规模快照（拷贝时点）

| 项 | 大小 | 文件数 | 来源 |
|---|---|---|---|
| `docs/proposal.md` | 139 KB | 1 | `/mnt/gty/research_project/docs/proposal.md` |
| `data/index/product_index.json` | 4.9 MB | 1 | `/mnt/gty/product_index.json` |
| `data/raw/srt_cut/` | 347 MB | 1,214 | `/mnt/livestream/srt_cut/` |
| `data/raw/comment/` | 468 MB | 2,721 | `/mnt/livestream/comment/` |
| `data/raw/product_images/` | 3.4 GB | 12,017 | `/mnt/gty/product_images/` |
| **合计** | **≈ 4.2 GB** | **≈ 16,000** | — |

文件计数已与原路径逐项对账一致（`srt_cut / comment / product_images` 全部 OK）。

---

## 3. 与 `proposal.md` 中路径的映射

`proposal.md` §8 中以 `/mnt/livestream/...` 与 `/mnt/gty/product_images/...` 出现的路径，在本项目内的等价路径如下：

| `proposal.md` 中的路径 | 在 claimarc/ 内对应路径 |
|---|---|
| `/mnt/gty/product_index.json` | `data/index/product_index.json` |
| `/mnt/gty/product_images/{pid}/...` | `data/raw/product_images/{pid}/...` |
| `/mnt/livestream/srt_cut/<L1>/<L2>/<room>/<session>/<clip>.{srt,txt}` | `data/raw/srt_cut/<L1>/<L2>/<room>/<session>/<clip>.{srt,txt}` |
| `/mnt/livestream/comment/<L1>/<L2>/<room_pinyin>/...xls` | `data/raw/comment/<L1>/<L2>/<room_pinyin>/...xls` |

> 后续脚本建议统一从环境变量 `CLAIMARC_ROOT=/mnt/gty/claimarc` 出发，再拼 `data/raw/...`，避免硬编码 `/mnt/livestream` 路径。

---

## 4. 数据来源时点

- 拷贝完成时间：2026-04-27 12:15（CST）
- 原始挂载：`/mnt`（fx 文件系统，剩余 148 GB）
- 拷贝方式：`cp -a`（保留权限、时间戳与软链接结构）
