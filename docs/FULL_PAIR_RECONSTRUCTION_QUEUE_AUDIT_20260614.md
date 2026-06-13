# Full Pair Reconstruction Queue Audit v1

This report audits the reconstruction queue before LLM/VLM relabeling. It does not remove rows.

## Summary

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- rows: `13769`
- output json: `data/final/repaired_v1/full_pair_reconstruction_queue_audit_v1_20260614.json`

## Core Counts

| field | top counts |
|---|---|
| priority | `{'P0': 6336, 'P1': 5202, 'P2': 2016, 'P3': 215}` |
| queue type | `{'full_claim_evidence_label_rebuild': 10394, 'claim_reextract_label_rebuild': 1918, 'evidence_refresh_label_rebuild': 1117, 'label_rebuild_existing_triplet': 340}` |
| claim state | `{'claim_missing': 10666, 'claim_present_review_needed': 1646, 'claim_present_specific': 1457}` |
| evidence state | `{'evidence_missing': 6961, 'evidence_single_source': 4550, 'evidence_multi_source': 2258}` |
| old label state | `{'label_negative_no_aligned_review': 12219, 'label_positive_claim_aligned_neg': 966, 'label_negative_claim_aligned_nonneg': 584}` |
| category | `{'apparel_and_underwear': 2551, 'general': 2499, 'baby_kids_and_pets': 2332, 'food_and_beverages': 1354, 'shoes_and_bags': 1273, 'beauty_and_personal_care': 1084, 'smart_home': 1083, 'digital_and_electronics': 712, 'sports_and_outdoor': 707, 'jewelry_and_collectibles': 174}` |

## Comment Provenance

- compact mention type counts: `{'attribute': 49554, 'service': 778}`
- rows with service comments in compact mentions: `395`
- rows where all compact mentions are service typed: `0`
- commercial-promise attributes: `531`
- comments with explicit claim cues: `927`

## Missingness Combos

| combo | count |
|---|---:|
| `claim_missing|evidence_missing|label_negative_no_aligned_review` | 5769 |
| `claim_missing|evidence_single_source|label_negative_no_aligned_review` | 3316 |
| `claim_missing|evidence_multi_source|label_negative_no_aligned_review` | 1581 |
| `claim_present_review_needed|evidence_single_source|label_negative_no_aligned_review` | 336 |
| `claim_present_review_needed|evidence_missing|label_negative_no_aligned_review` | 336 |
| `claim_present_specific|evidence_missing|label_negative_no_aligned_review` | 277 |
| `claim_present_specific|evidence_single_source|label_negative_no_aligned_review` | 274 |
| `claim_present_review_needed|evidence_single_source|label_positive_claim_aligned_neg` | 200 |
| `claim_present_specific|evidence_single_source|label_positive_claim_aligned_neg` | 188 |
| `claim_present_review_needed|evidence_missing|label_positive_claim_aligned_neg` | 186 |
| `claim_present_specific|evidence_missing|label_positive_claim_aligned_neg` | 186 |
| `claim_present_specific|evidence_multi_source|label_negative_no_aligned_review` | 170 |
| `claim_present_review_needed|evidence_multi_source|label_negative_no_aligned_review` | 160 |
| `claim_present_review_needed|evidence_single_source|label_negative_claim_aligned_nonneg` | 128 |
| `claim_present_review_needed|evidence_missing|label_negative_claim_aligned_nonneg` | 123 |
| `claim_present_specific|evidence_single_source|label_negative_claim_aligned_nonneg` | 108 |
| `claim_present_review_needed|evidence_multi_source|label_positive_claim_aligned_neg` | 105 |
| `claim_present_specific|evidence_multi_source|label_positive_claim_aligned_neg` | 101 |
| `claim_present_specific|evidence_missing|label_negative_claim_aligned_nonneg` | 84 |
| `claim_present_review_needed|evidence_multi_source|label_negative_claim_aligned_nonneg` | 72 |
| `claim_present_specific|evidence_multi_source|label_negative_claim_aligned_nonneg` | 69 |

## Important Interpretation

- Large missing-claim or missing-evidence groups are reconstruction targets, not negative labels.
- Old labels are audit-only until a recovered claim is compared with aligned consumer comments.
- Service-like comments remain in the queue, but they cannot trigger a positive label unless they align to the same commercial-promise claim.

## Example Flags

### service_comment_boundary

- `p3679197849955992032__DIGITAL_安装方式` old_y=1 claim=claim_present_review_needed evidence=evidence_single_source label=label_positive_claim_aligned_neg claim_preview="安装师傅会给你带挂架"
  - [attribute/neg] "装不了镶入式的电视"
  - [attribute/neg] "安装时说架子掉下电视机不负责"
- `p3768024980667891731__SPORT_保价承诺` old_y=1 claim=claim_present_review_needed evidence=evidence_missing label=label_positive_claim_aligned_neg claim_preview="保价未来一整年"
  - [service/neg] "直播间说的保价子虚乌有"
  - [service/neg] "15天保价虚假宣传"
- `p3790945142245032045__BABY_套餐详情` old_y=1 claim=claim_present_review_needed evidence=evidence_missing label=label_positive_claim_aligned_neg claim_preview="可以拿出来5根试吃"
  - [service/neg] "直播间说好的送几支猫条"
  - [attribute/neg] "给的试吃是鱼油"
- `p3722540703549620336__BEAUTY_使用方法` old_y=1 claim=claim_present_review_needed evidence=evidence_single_source label=label_positive_claim_aligned_neg claim_preview="每天只需要早上一分钟、一分钟,一边坏边照光就可以让你享受\n每天一边画一边照光\n五分钟的洗\n第一点，所谓简单，想互不识别人工贝的\n每次长安后面这个按钮，把它晾红光\n精华弹装好之后，我们每天在家里，你就这样，一边物化"
  - [attribute/neg] "结果根本喷不出来"
  - [attribute/neg] "换精华补充液频次太高"
- `p3703129246668030235__HOME_使用方法` old_y=1 claim=claim_present_review_needed evidence=evidence_missing label=label_positive_claim_aligned_neg claim_preview="你买回家机器原始掌这个样子,咱直接把钻头擦上去就行了,然后呢,凭着把钻头拔出来,擦上去就可以了"
  - [attribute/neg] "说明书再详细一些就好了"
  - [attribute/neg] "没有任何说明！"
- `p3717436738369618131__BEAUTY_使用方法` old_y=1 claim=claim_present_review_needed evidence=evidence_single_source label=label_positive_claim_aligned_neg claim_preview="每天都要以油养肤"
  - [service/neg] "直播间说了也不回复"
  - [attribute/neg] "还让我把手搓热"
- `p3483616667710416469__FOOD_套餐详情` old_y=1 claim=claim_present_review_needed evidence=evidence_multi_source label=label_positive_claim_aligned_neg claim_preview="每一盒当中都是有菜有肉有主食\n五盒中奥贵宝给大家外加赠一个四三粒不收钢的餐厂\n六盒装给大家外加赠一包我们的厨房湿巾\n会给大家也是加赠一个四三粒不收钢的餐厂"
  - [attribute/neg] "不是有写送叉子吗"
  - [attribute/neg] "宣传视频说有肉块 实际肉粒都看不到"
- `p3700674430352097415__FOOD_套餐详情` old_y=1 claim=claim_present_review_needed evidence=evidence_multi_source label=label_positive_claim_aligned_neg claim_preview="会给大家也是加赠一个四三粒不收钢的餐厂"
  - [service/neg] "直播间不是说送捞面+叉子吗？没有收到叉子"
  - [attribute/neg] "商品详情页写买十盒送四片午餐肉"

### explicit_claim_cue_comments

- `p3721888693162737924__SHOEBAG_鞋底工艺` old_y=1 claim=claim_present_review_needed evidence=evidence_missing label=label_positive_claim_aligned_neg claim_preview="我这实心橡胶要越穿越软"
  - [attribute/neg] "鞋底也没有主播说的那么软"
  - [attribute/neg] "鞋底子穿上不舒服 硬"
- `p3663574409718972261__HOME_清洁效果` old_y=1 claim=claim_present_review_needed evidence=evidence_single_source label=label_positive_claim_aligned_neg claim_preview="喷上去之后就跟镀了膜似的"
  - [attribute/neg] "未像视频宣传的那样清亮"
  - [attribute/neg] "都洗不干净"
- `p3768024980667891731__SPORT_保价承诺` old_y=1 claim=claim_present_review_needed evidence=evidence_missing label=label_positive_claim_aligned_neg claim_preview="保价未来一整年"
  - [service/neg] "直播间说的保价子虚乌有"
  - [service/neg] "15天保价虚假宣传"
- `p3716622350351990803__BEAUTY_使用方法` old_y=1 claim=claim_present_review_needed evidence=evidence_single_source label=label_positive_claim_aligned_neg claim_preview="用在你化妆的第一步"
  - [attribute/neg] "要用很多量才有主播说的水光效果"
  - [attribute/neg] "说是需要摇匀再用"
- `p3758583403453218941__HOME_是否开刃` old_y=1 claim=claim_present_review_needed evidence=evidence_single_source label=label_positive_claim_aligned_neg claim_preview="这把刀是包你不生锈的"
  - [attribute/neg] "没有像抖音上说的那么锋利"
  - [attribute/neg] "就像没开刃一样"
- `p3768024980667891731__SPORT_主要功能` old_y=1 claim=claim_present_review_needed evidence=evidence_single_source label=label_positive_claim_aligned_neg claim_preview="1号链接是做三房的"
  - [attribute/neg] "根本不像直播间说的什么“防污防水防油”"
  - [attribute/neg] "特别容易油"
- `p3768024980667891731__SPORT_工艺` old_y=1 claim=claim_present_review_needed evidence=evidence_single_source label=label_positive_claim_aligned_neg claim_preview="四格防专容工艺"
  - [attribute/neg] "说是刺绣然后是印花"
  - [attribute/neg] "做工一点都不"
- `p3790945142245032045__BABY_套餐详情` old_y=1 claim=claim_present_review_needed evidence=evidence_missing label=label_positive_claim_aligned_neg claim_preview="可以拿出来5根试吃"
  - [service/neg] "直播间说好的送几支猫条"
  - [attribute/neg] "给的试吃是鱼油"

### missing_claim_old_negative_with_neg_comments

- `p3616981688330152430__APPAREL_尺码` old_y=0 claim=claim_missing evidence=evidence_single_source label=label_negative_no_aligned_review claim_preview=""
  - [attribute/neg] "复购的袜子很明显比第一次买的要小"
  - [attribute/neg] "尺码很明显不对"
- `p3702755032744198404__APPAREL_镜片分类` old_y=0 claim=claim_missing evidence=evidence_single_source label=label_negative_no_aligned_review claim_preview=""
  - [attribute/neg] "说是防蓝光，上面连个蓝膜都看不到"
  - [attribute/neg] "不是防蓝光的！"
- `p3702755032744198404__APPAREL_镜片尺寸` old_y=0 claim=claim_missing evidence=evidence_single_source label=label_negative_no_aligned_review claim_preview=""
  - [attribute/neg] "平均几块玻璃¥1透明的都没镀膜"
  - [attribute/neg] "防蓝光镜片有蓝膜"
- `p3715560892084126145__APPAREL_主要功能` old_y=0 claim=claim_missing evidence=evidence_single_source label=label_negative_no_aligned_review claim_preview=""
  - [attribute/neg] "澳门一姐直播间说得那么夸张的好质量。"
  - [attribute/neg] "虚假宣传"
- `p3715560892084126145__APPAREL_价格` old_y=0 claim=claim_missing evidence=evidence_missing label=label_negative_no_aligned_review claim_preview=""
  - [attribute/neg] "不算太理想"
  - [attribute/neg] "性价比：不怎么样"
- `p3720538529466548264__APPAREL_是否加绒` old_y=0 claim=claim_missing evidence=evidence_missing label=label_negative_no_aligned_review claim_preview=""
  - [attribute/neg] "我收到的裤缝能看到黑色的"
  - [attribute/neg] "还加绒加厚款？"
- `p3731321152212172814__APPAREL_价格` old_y=0 claim=claim_missing evidence=evidence_missing label=label_negative_no_aligned_review claim_preview=""
  - [attribute/neg] "买了三天马上掉价"
  - [attribute/neg] "十多块买了个"
- `p3731321152212172814__APPAREL_重量` old_y=0 claim=claim_missing evidence=evidence_single_source label=label_negative_no_aligned_review claim_preview=""
  - [attribute/neg] "根本没有20000"
  - [attribute/neg] "重量：抖爸爸他妈也是傻逼"

### existing_triplet_label_rebuild

- `p3809578084848500939__BABY_填充物种类` old_y=1 claim=claim_present_specific evidence=evidence_multi_source label=label_positive_claim_aligned_neg claim_preview="这一件依旧是2024年的新国标白亚绒"
  - [attribute/neg] "帽子里感觉填充的不是羽绒"
  - [attribute/neg] "跑出来的不是绒朵，是羽毛片"
- `p3483616667710416469__FOOD_净含量` old_y=1 claim=claim_present_specific evidence=evidence_multi_source label=label_positive_claim_aligned_neg claim_preview="面的精重是110克,这样比面更多"
  - [attribute/neg] "到货后不是加量的"
  - [attribute/neg] "牛肉没有视频中说的那么多"
- `p3490652681037608601__FOOD_食品工艺` old_y=1 claim=claim_present_specific evidence=evidence_multi_source label=label_positive_claim_aligned_neg claim_preview="从育苗,种植,采收,到现在,咱们儿,你看到这桶酸辣粉,我们都是一体化的,自产自销的。\n十斤红薯才能抹出一斤斑的粉,得到了咱们家的粉饼呢。\n真正的缝冰给您做到了,没有科技,没有狠火的\n咱们才能磨出一斤斑粉"
  - [attribute/neg] "加了胶你又要宣传纯红薯"
  - [attribute/neg] "不是真的纯红薯粉"
- `p3753593497471550320__FOOD_净含量` old_y=1 claim=claim_present_specific evidence=evidence_multi_source label=label_positive_claim_aligned_neg claim_preview="每包280克了，减糖款甜度是奶茶的五分甜"
  - [attribute/neg] "水只加了一点，豆奶也只放了一点"
  - [attribute/neg] "280克的"
- `p3784789614443760279__FOOD_保质期` old_y=1 claim=claim_present_specific evidence=evidence_multi_source label=label_positive_claim_aligned_neg claim_preview="保质期是只有21天短保"
  - [attribute/neg] "跟日期一点都不符"
  - [attribute/neg] "保质期是21天"
- `p3709840760133255169__GEN_功效` old_y=1 claim=claim_present_specific evidence=evidence_multi_source label=label_positive_claim_aligned_neg claim_preview="被称为牙齿的抛光神器\n牙齿的吸神器\n被称为牙齿的抛光神器"
  - [attribute/neg] "不见得跟直播间说的那么好"
  - [attribute/neg] "效果也没有商家说的那么夸张"
- `p3759321650240290980__GEN_张数` old_y=1 claim=claim_present_specific evidence=evidence_multi_source label=label_positive_claim_aligned_neg claim_preview="一包里面是680抽\n加大家后的 一包多少抽 680抽"
  - [attribute/neg] "标的680张，在别人家拍标600张"
  - [attribute/neg] "一包才三百多张"
- `p3703683175160086673__SHOEBAG_帮面材质` old_y=1 claim=claim_present_specific evidence=evidence_multi_source label=label_positive_claim_aligned_neg claim_preview="全皮面,没有其他材质品质\n看到没?看到了!全皮面,没有其他材质品质\n全皮面\n没有其他材质品贴"
  - [attribute/neg] "不是真皮"
  - [attribute/neg] "合成革的，不是真皮"
