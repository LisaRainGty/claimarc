# Source Recovery Queue v3

- source dataset: `data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_p0p1adjudicated_v1.jsonl`
- queue: `data/final/repaired_v1/source_recovery_queue_v3.jsonl`
- rows: `559`
- queue_type: `{'direct_product_source0': 559}`
- priority: `{'P0': 259, 'P1': 115, 'P2': 185}`
- objectivity: `{'scope_review_uncertain_attribute': 345, 'objective_numeric': 89, 'objective_visual': 110, 'objective_material': 7, 'objective_boolean': 8}`
- labels: `{'1': 259, '0': 300}`

## Interpretation
This broad queue is intended to separate three failure modes: true missing
product evidence, attribute-scope leakage, and weak single-source product
evidence. P0/P1 rows should be verified first with product params, OCR, and
detail-image VLM grounding before changing labels or filtering rows.

## P0 Examples
| pair_id | label | scope | attribute | product |
|---|---:|---|---|---|
| `p3439082779739038362__GEN_包数` | 1 | scope_review_uncertain_attribute | 包数 | 【翊合境】四川爱媛38号果冻橙5斤装10-12个8-10个7-9个 |
| `p3439082779739038362__GEN_尺寸` | 1 | objective_numeric | 尺寸 | 【翊合境】四川爱媛38号果冻橙5斤装10-12个8-10个7-9个 |
| `p3439082779739038362__GEN_皮质特征` | 1 | scope_review_uncertain_attribute | 皮质特征 | 【翊合境】四川爱媛38号果冻橙5斤装10-12个8-10个7-9个 |
| `p3482085810386379885__BABY_价格优惠` | 1 | objective_numeric | 价格优惠 | 【直播2包】宜婴新梦想家纸尿裤婴儿学步拉拉裤宝宝超薄透气尿不湿 |
| `p3482085810386379885__BABY_套餐详情` | 1 | scope_review_uncertain_attribute | 套餐详情 | 【直播2包】宜婴新梦想家纸尿裤婴儿学步拉拉裤宝宝超薄透气尿不湿 |
| `p3483616667710416469__FOOD_价格变动` | 1 | objective_numeric | 价格变动 | 【五盒装】锋味派官方旗舰店谢霆锋意大利面速食5盒 |
| `p3487313915229168499__APPAREL_适合肤质` | 1 | scope_review_uncertain_attribute | 适合肤质 | 高顶帽子男女通用新款硬顶有型棒球帽秋季大头围大号显脸小鸭舌帽 |
| `p3490652681037608601__FOOD_是否临期` | 1 | scope_review_uncertain_attribute | 是否临期 | 【轻食季】贵州特产绿色纯红薯粉条500g自选粉丝、粉条、粉皮 |
| `p3490848458481503952__FOOD_套餐份量` | 1 | scope_review_uncertain_attribute | 套餐份量 | 【热辣一夏】陈薯夜宵季菌汤宽粉99g*6桶装/箱宽粉酸辣粉尤莱特 |
| `p3538902193669330044__SPORT_主要功能` | 1 | objective_visual | 主要功能 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3538902193669330044__SPORT_安装方式` | 1 | scope_review_uncertain_attribute | 安装方式 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3538902193669330044__SPORT_组装过程` | 1 | scope_review_uncertain_attribute | 组装过程 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3538902193669330044__SPORT_耐用性` | 1 | scope_review_uncertain_attribute | 耐用性 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3538902193669330044__SPORT_舒适度` | 1 | scope_review_uncertain_attribute | 舒适度 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3538902193669330044__SPORT_赠品信息` | 1 | scope_review_uncertain_attribute | 赠品信息 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3538902193669330044__SPORT_车体表面处理` | 1 | scope_review_uncertain_attribute | 车体表面处理 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3538902193669330044__SPORT_车架螺丝` | 1 | scope_review_uncertain_attribute | 车架螺丝 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3538902193669330044__SPORT_重量` | 1 | objective_numeric | 重量 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3538902193669330044__SPORT_风格` | 1 | objective_visual | 风格 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3538902193669330044__SPORT_骑行阻力` | 1 | scope_review_uncertain_attribute | 骑行阻力 | 永久官方旗舰店铝合金山地自行车29寸成人变速减震油碟公路车GT05 |
| `p3551114774253288539__APPAREL_价格波动` | 1 | objective_numeric | 价格波动 | 【爆款加长自动卷】lena负离子全自动旋转卷发棒32mm水波纹持久防烫 |
| `p3551114774253288539__APPAREL_工艺` | 1 | objective_visual | 工艺 | 【爆款加长自动卷】lena负离子全自动旋转卷发棒32mm水波纹持久防烫 |
| `p3580184154987456894__DIGITAL_易碎性` | 1 | scope_review_uncertain_attribute | 易碎性 | 【图拉斯原感膜】适用16promax钢化膜3D热弯弧边无尘舱钢化膜全覆盖 |
| `p3580184154987456894__DIGITAL_颜色` | 1 | objective_visual | 颜色 | 【图拉斯原感膜】适用16promax钢化膜3D热弯弧边无尘舱钢化膜全覆盖 |
| `p3629782961358001865__BABY_商品缺货` | 1 | scope_review_uncertain_attribute | 商品缺货 | 酷娃米儿童秋冬保暖羽绒服白鸭绒面包服童装冬款外套 |
