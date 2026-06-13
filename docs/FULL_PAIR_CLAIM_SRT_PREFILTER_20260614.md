# Full Pair SRT Claim Prefilter v1

This report summarizes deterministic SRT candidate retrieval for full-pair reconstruction.
It does not label rows or promote claims.

## Summary

- input queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- output: `data/final/repaired_v1/full_pair_claim_srt_prefilter_v1_20260614.jsonl`
- rows processed: `12312`
- skipped: `{'claim_state_filter': 1457}`
- prefilter state: `{'weak_srt_candidate': 6769, 'strong_srt_candidate': 3322, 'very_weak_srt_candidate': 2141, 'no_srt_candidate': 80}`
- queue type: `{'full_claim_evidence_label_rebuild': 10394, 'claim_reextract_label_rebuild': 1918}`
- claim state: `{'claim_present_review_needed': 1646, 'claim_missing': 10666}`
- category: `{'digital_and_electronics': 573, 'food_and_beverages': 1291, 'general': 2375, 'shoes_and_bags': 1129, 'smart_home': 996, 'sports_and_outdoor': 617, 'beauty_and_personal_care': 974, 'apparel_and_underwear': 2119, 'baby_kids_and_pets': 2131, 'jewelry_and_collectibles': 107}`

## Cross Tabs

- claim state by prefilter: `{'claim_present_review_needed|weak_srt_candidate': 886, 'claim_present_review_needed|strong_srt_candidate': 531, 'claim_present_review_needed|very_weak_srt_candidate': 226, 'claim_missing|weak_srt_candidate': 5883, 'claim_missing|strong_srt_candidate': 2791, 'claim_missing|very_weak_srt_candidate': 1915, 'claim_present_review_needed|no_srt_candidate': 3, 'claim_missing|no_srt_candidate': 77}`
- queue type by prefilter: `{'full_claim_evidence_label_rebuild|weak_srt_candidate': 5792, 'full_claim_evidence_label_rebuild|strong_srt_candidate': 2681, 'full_claim_evidence_label_rebuild|very_weak_srt_candidate': 1861, 'claim_reextract_label_rebuild|strong_srt_candidate': 641, 'claim_reextract_label_rebuild|weak_srt_candidate': 977, 'claim_reextract_label_rebuild|very_weak_srt_candidate': 280, 'claim_reextract_label_rebuild|no_srt_candidate': 20, 'full_claim_evidence_label_rebuild|no_srt_candidate': 60}`

## Interpretation

- `strong_srt_candidate` means raw SRT likely contains attribute-related claim material for later LLM review.
- `weak_srt_candidate` means SRT has lexical overlap but needs careful review.
- `no_srt_candidate` means either no SRT file was available or no useful lexical candidate was found.

## Top Examples

- `p3769484498568347935__GEN_净含量` state=strong_srt_candidate score=81 attr=净含量 text="那一般榴莲制品,因为它是去果壳,留果肉,一般都是用的是C级果,但是它家叫金枕榴莲,坚持用的是A级果,A++的果形,然后我们这个化开以后呢,会发现,它里面没有那么多的汤的含量,然后非常甜,非常这个呼嘴,有耳夹吗,耳夹,耳夹到时候我们给大家对一个这个品,看看谁家耳夹靠谱, 真的好有趣啊,这会打开我新世界了,然后你看它是保重的,就是每一袋是400克,大小不一,然后你看,就是把它化开以后它是这样的,就非常非常绵密感,就跟那个冰激凌似的,特别爽,很好吃,很好吃,我们自己平常吃的时候都是,就是不完全化开,然后,必须要勺,直接
- `p3720772695445602597__GEN_是否加绒` state=strong_srt_candidate score=73 attr=是否加绒 text="它里边的绒是很厚很厚的,我们给它切个进景,就这款的话呢,整体你也是穿上雪地靴了,第一个特别适合雪天出去玩,它会比较的有聚水性,再者的话呢,暖和呀,您看里边这绒,对吧,就藏不住的热情,是,嗯,再者的话,桶高也不错,桶高在7.8厘米左右,更高在4厘米左右,这个桶高基本上能够把大家的脚踝,脚脖制啊,你比较趴冷的位置给它包裹住,再者的话呢,哎,穿起来的话,你看很有雪天的氛围感,嗯,没错,而且就是它整体比较有 细腻,这种厚实的绒毛,尤其在严寒的冬天,显得非常非常地暖和,对,我穿一双黑色,大家可以看一下上脚的效果,您或者穿这
- `p3585819184510160371__FOOD_淀粉颜色` state=strong_srt_candidate score=70 attr=淀粉颜色 text="好粉,纯正好酸辣,年天,年价,年色素,来姐妹看过来。 粉冰里面呢,它就是只有这个红薯淀粉,饮用水和熊盐的。 像什么添加剂,防腐剂,玉米淀粉,木薯淀粉,通通卖油的。"
- `p3702755032744198404__APPAREL_颜色分类` state=strong_srt_candidate score=70 attr=颜色分类 text="哪个颜色显不白去买苍雪辉, 红色马上给大家看, 黑色头发去买灰,"
- `p3702935326797464036__APPAREL_颜色` state=strong_srt_candidate score=70 attr=颜色 text="可以买黑色或灰色, 这两个颜色都不错, 帽子的米色跟灰色,"
- `p3660066402737519648__BABY_数量` state=strong_srt_candidate score=69 attr=数量 text="请拍单,一周以上的小孩大人都去拍一号,15重的氨基酸洗发水,更温和更洁净,头发有油的,有氧的,抓来抓去,挠来挠去,小雪花,大雪花,满天飞的,胆都去拍一号,啊,恭喜,恭喜,小懒猫,啊,来,最后28单,正装修名,啊,拍不到,抢不到,我不加单了,我这边马上就要下播了,哦,对,一周以上的大人小孩都,婴妈妈,不入期都去拍一号,我们家当家花蛋,已经卖出去300多万销量了,啊,亲爱的,我们家大姐宝贝家具,婴妈妈,不入期都去拍一号,我们家当家 每个瓶子有儿童小精断,成分足够温和,足够安全,不赖眼睛,不刺激,不过敏,过敏,我包退的
- `p3724404517601673447__BABY_颜色分类` state=strong_srt_candidate score=69 attr=颜色分类 text="颜色有两个 一个是酒红色 一个是我直播间的黑色"
- `p3735201930100736082__GEN_价格` state=strong_srt_candidate score=69 attr=<价格> text="细腻,这种厚实的绒毛,尤其在严寒的冬天,显得非常非常地暖和,对,我穿一双黑色,大家可以看一下上脚的效果,您或者穿这种宅腿的裤子,或者就穿一个瑜伽裤,沙子裤,然后呢,或者就是这种运动休闲裤,直接盖在上面,很好看,对,这特别百单啊,哪款,大家说是非常非常厚的,啊,咱今天确实有三款呢,三款,目前啊,你要说看起来非常厚的,我觉得应该是这款,就看起来就厚,你看它这个毛圈啊,也会加厚一层, 啊,这个价格是不是也更贵啊,哎呀,这是我一眼挑装的最贵的鞋子,你的眼睛就是尺啊,我的眼睛就是尺啊,但这个统高也会高一点点啊,它的保暖性估
- `p3660066402737519648__BABY_价格` state=strong_srt_candidate score=67 attr=<价格> text="你用到这用到这都没关系,好的产品他自己会说话,你真心觉得OK了,你再留下来就这么简单,好吧,宝贝们,我们已经养护好了千千万万头皮环境,不怕养护不了你,二号链接也是两频哦,540宝贝加急,会跳舞宝贝加急,好的,所有姐妹们关注点一下,方便售后找得到我,我家不做一锤子买卖,一周以上的小孩大人都能用,年货福利价格不是天天有的呀,再过两三天快递要停了,早点拍早点发货, 老饭都知道,咱之前一瓶是69,那我今年年货福利两瓶69.9,保价一整年,买贵我包退,买下我包赔,两瓶都是洗,一瓶洗发水,一瓶沐浴露,萨萨小姐,在二号链接,一
- `p3702935326797464036__APPAREL_颜色分类` state=strong_srt_candidate score=66 attr=颜色分类 text="啊,米色是乐巴同款,然后这个是白净亭同款,推荐米色,卡齐色买的人比较少,我不买东西,单纯为了来,为了看直播来的好的,啊,那你就左上角付带赶紧去参与,一边听我讲解,一边付带去参与讲,说不定哪个姐妹就能中个奖,一号的帽子,三号的尾巾全部含羊毛,啊, 来看主播的左上角付带去参与,啊,老粉姐妹们,一边看我直播,啊,就当,图个热闹,对不对,然后呢,左上角有个付带,说不定你就能中个奖,中个帽子回去,是不是,男生可以戴啊,一号链接,卷边卷起来就行啊, 男女同款,一号链接,推荐灰色,花灰色,今年格雷系的一个穿搭,不管去配黑色,灰
- `p3727479136529285164__BEAUTY_颜色` state=strong_srt_candidate score=66 attr=颜色 text="是自然色加齐白色 我是一个黄黑皮 我用自然色加齐白色"
- `p3601244194531867599__BABY_价格` state=strong_srt_candidate score=65 attr=<价格> text="来再跟大家说一下宝贝们,我们家欣赏的宝宝,包括小月灵宝宝,你们可以先去把维生素AD和DH的话,给我们家宝宝去吃吃带带,等到月灵满到6个月,接近6个月的时候再去把1号链接钙元素给我们家宝宝日常去吃,日常去添加起来就行了,单独想拍钙,选择1号链接钙,想拍AD选择2号链接,想拍DH,选择我们家的3号链接宝贝,第三有的宝贝在我们家的17号链接,17号链接就我们家的维生素D3, 2号链接我们家维生素D3,你家宝贝想吃D3的话,每天早上固定吃个一粒就行了宝贝,第三的话,直播间价格已经是开过价那个价格的宝贝,你们想拍的话,放心
- `p3720772695445602597__GEN_颜色` state=strong_srt_candidate score=65 attr=颜色 text="如果您说我写过已经有个咖色了,那您可以试一下这个其他颜色,棕色也不错。 真的,真的,好舒服,对,是吧,魔法变了,是吧,我知道魔法可以让煎饼消失,但是我不知道魔法可以让雪出现,对吧,真不错,好有感觉啊,好有感觉,你待会儿穿,你穿一个雪地靴吧,更有感觉,我扶着你,不用不用,OK,三,穿一个黑色,嗯,它里边 它里边的绒是很厚很厚的,我们给它切个进景,就这款的话呢,整体你也是穿上雪地靴了,第一个特别适合雪天出去玩,它会比较的有聚水性,再者的话呢,暖和呀,您看里边这绒,对吧,就藏不住的热情,是,嗯,再者的话,桶高也不错,桶
- `p3723511462895943927__HOME_品牌` state=strong_srt_candidate score=64 attr=品牌 text="能出都出掉。哇塞。这边好多东西啊。这是电池是吧?电池。电池。你们要的话都给你们了啊。对了。共商超的。我这个品牌都认识。比什么服务电池还耐用好多倍啊。超耐用的。来。这是新款将心意义的。对的。一人20个。一人给20个。那你这几年电池都不用买了?5号的。对的。7号的。我常用的都有。一节工厂制作成本是4块。要的。来拿上来。没事啊。 给你20个多少钱?算一下啊。20节大几时了。40便宜吗?便宜。三块钱给你20颗。放。那三块钱。你这几年的电池都包了?遥控器玩具车都可以去使。是的。左上角发的电钻自己的。而且非常耐用啊。我们的话
- `p3683596134795837638__FOOD_功能` state=strong_srt_candidate score=63 attr=功能 text="或者说有内到外,散发传意味怪味,导致整个嘴巴越来越重啊,越来越难闻,大家都能够含上一颗,因为咱们给你做到,每100克里面,是含有9500毫克茶多酚,高浓度的茶多酚,我们是给大家去意味给你提神,但是宝贝,如果说单单只有茶多酚,它只能够给大家做清洁口腔的作用,是远远不够的,所以我们在这个基础上,还给大家做到,每一片里面,都是添加了超过20亿的活性生菌,帮助大家, 是从根源出发,我们是由内到外,去给你做分解,给你做改善,给你做清理,分解大家嘴巴异味,改善嘴巴臭味,并且,我们是清理大家嘴巴里面,不好的味道,给你做持久留香
- `p3718004155214856462__APPAREL_颜色` state=strong_srt_candidate score=62 attr=颜色 text="喜欢可以直接去挪, 黑白的颜色比较多, 去买灰一号链接,"
- `p3734640811833426037__DIGITAL_颜色` state=strong_srt_candidate score=62 attr=颜色 text="呃这两位宝贝啊念到名字的宝贝,我们这边都是给您发的顺风快递啊,您放心就可以了,放心就可以了,念到名字的都是发顺风啊, 发黄直接给您免费唤醒啊,咱家手机可不会发黄,不会变色啊,发黄变色给您免费唤醒啊,没问题, 呃这个就是黑的啊,塑胶跑道宝贝,这个就是六皇帝年黑四啊,我给您放在前面了啊,"
- `p3757291204451107454__BABY_颜色分类` state=strong_srt_candidate score=62 attr=颜色分类 text="粉色 就是紫色 这几个颜色都挺好看的"
- `p3730164951386554776__APPAREL_是否加绒` state=strong_srt_candidate score=61 attr=是否加绒 text="给大家再减30, 到手价格69, 我们加绒加厚的裤子现在买到基本上就是,"
- `p3768382098625396975__SPORT_面料材质` state=strong_srt_candidate score=61 attr=面料材质 text="再给大家,这个配什么裤子,我给你们搭配一个裤子吧,还要好几个姐妹想要这个空气层的莱卡面料的阔腿裤的,我给你们把这个经典款给你们上一下,这个裤子很强手的,我们家老粉基本上都人手一件,而且这个黑色经典中的经典,很多姐妹蹲了四五场,想要回购都没有蹲到的啊,就昨天盘点出来,大概还有个一百多件,卖完不会再有了, 来卡含量这个达到了38%,空气层的面料,质感垂感肉眼可见,没有体验过的姐妹一定要去感受一下,小营标都给你们带着的支持柜位上验货啊,这款ZEMO柜位上是卖到一百六七十美金的,就一百件出头,卖完不会再有了,你们自己去捡

## No-Candidate Examples

- `p3750430700810141764__APPAREL_香型` attr=香型 category=apparel_and_underwear srt_files=1
- `p3775112844819956137__DIGITAL_描述相符` attr=描述相符 category=digital_and_electronics srt_files=1
- `p3750430700810141764__APPAREL_产地` attr=产地 category=apparel_and_underwear srt_files=1
- `p3750430700810141764__APPAREL_是否进口` attr=是否进口 category=apparel_and_underwear srt_files=1
- `p3750430700810141764__APPAREL_是否为有机食品` attr=是否为有机食品 category=apparel_and_underwear srt_files=1
- `p3775112844819956137__DIGITAL_功能` attr=功能 category=digital_and_electronics srt_files=1
- `p3775112844819956137__DIGITAL_识别度` attr=识别度 category=digital_and_electronics srt_files=1
- `p3775112844819956137__DIGITAL_扫描方式` attr=扫描方式 category=digital_and_electronics srt_files=1
- `p3750430700810141764__APPAREL_主播宣传` attr=主播宣传 category=apparel_and_underwear srt_files=1
- `p3750430700810141764__APPAREL_喷雾方式` attr=喷雾方式 category=apparel_and_underwear srt_files=1
- `p3750430700810141764__APPAREL_成分含量` attr=成分含量 category=apparel_and_underwear srt_files=1
- `p3775112844819956137__DIGITAL_价格` attr=价格 category=digital_and_electronics srt_files=1
- `p3775112844819956137__DIGITAL_商品缺货状态` attr=商品缺货状态 category=digital_and_electronics srt_files=1
- `p3750430700810141764__APPAREL_适用对象` attr=适用对象 category=apparel_and_underwear srt_files=1
- `p3750430700810141764__APPAREL_酒精度数` attr=酒精度数 category=apparel_and_underwear srt_files=1
- `p3750430700810141764__APPAREL_专卖店` attr=专卖店 category=apparel_and_underwear srt_files=1
- `p3750430700810141764__APPAREL_工艺` attr=工艺 category=apparel_and_underwear srt_files=1
- `p3750430700810141764__APPAREL_挥发性` attr=挥发性 category=apparel_and_underwear srt_files=1
- `p3750430700810141764__APPAREL_是否临期` attr=是否临期 category=apparel_and_underwear srt_files=1
- `p3750430700810141764__APPAREL_保质期` attr=保质期 category=apparel_and_underwear srt_files=1
