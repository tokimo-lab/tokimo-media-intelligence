/// CLIP zero-shot classification categories.
///
/// Two-level taxonomy: categories (user-facing labels) → subcategories (CLIP prompts).
/// Classification matches against subcategories for precision, then rolls up to category for display.
///
/// Embedding cache is persisted to disk so that dev-server restarts don't require
/// re-computing ~1000 text embeddings (~7 s on CPU). The cache file includes a
/// taxonomy hash — any change to CATEGORIES auto-invalidates the cache.
use std::io::{Read, Write};
use std::sync::Mutex;

/// A category with its display name and subcategory prompts.
pub struct TagCategory {
    pub name: &'static str,
    pub icon: &'static str,
    pub subs: &'static [&'static str],
}

/// Result of classifying an image vector against the taxonomy.
#[derive(Debug, Clone)]
pub struct TagResult {
    pub category: &'static str,
    pub icon: &'static str,
    pub subcategory: &'static str,
    pub score: f32,
}

// Chinese-CLIP logit_scale ≈ 4.6, so exp(4.6) ≈ 100.
// This temperature makes softmax sharply peaked on the best match.
const LOGIT_SCALE: f32 = 100.0;
// Minimum softmax probability to include a tag
const MIN_PROB: f32 = 0.02;
const MAX_TAGS: usize = 5;

pub static CATEGORIES: &[TagCategory] = &[
    TagCategory {
        name: "美食",
        icon: "🍽️",
        subs: &[
            "中餐",
            "西餐",
            "日料",
            "韩餐",
            "泰餐",
            "烧烤",
            "火锅",
            "麻辣烫",
            "小吃",
            "甜点",
            "蛋糕",
            "面包",
            "饼干",
            "巧克力",
            "冰淇淋",
            "马卡龙",
            "面条",
            "米饭",
            "粥",
            "饺子",
            "包子",
            "馒头",
            "煎饼",
            "春卷",
            "寿司",
            "刺身",
            "拉面",
            "披萨",
            "汉堡",
            "三明治",
            "沙拉",
            "牛排",
            "意面",
            "海鲜",
            "鱼",
            "虾",
            "螃蟹",
            "龙虾",
            "贝类",
            "生蚝",
            "烤肉",
            "炸鸡",
            "烤鸭",
            "红烧肉",
            "糖醋排骨",
            "回锅肉",
            "炒菜",
            "凉菜",
            "汤",
            "炖菜",
            "火锅食材",
            "水果拼盘",
            "果盘",
            "甜品台",
            "早餐",
            "便当",
            "快餐",
            "外卖",
            "薯条",
            "热狗",
            "墨西哥卷饼",
            "咖喱",
        ],
    },
    TagCategory {
        name: "饮品",
        icon: "🥤",
        subs: &[
            "咖啡",
            "拿铁",
            "卡布奇诺",
            "美式咖啡",
            "手冲咖啡",
            "咖啡拉花",
            "茶",
            "奶茶",
            "抹茶",
            "功夫茶",
            "珍珠奶茶",
            "果汁",
            "冰沙",
            "柠檬水",
            "椰子水",
            "啤酒",
            "红酒",
            "白酒",
            "鸡尾酒",
            "威士忌",
            "香槟",
            "清酒",
            "牛奶",
            "豆浆",
            "酸奶",
            "气泡水",
            "可乐",
            "矿泉水",
        ],
    },
    TagCategory {
        name: "水果",
        icon: "🍎",
        subs: &[
            "苹果",
            "香蕉",
            "橙子",
            "柠檬",
            "葡萄",
            "草莓",
            "蓝莓",
            "樱桃",
            "桃子",
            "西瓜",
            "哈密瓜",
            "芒果",
            "菠萝",
            "猕猴桃",
            "火龙果",
            "荔枝",
            "龙眼",
            "柿子",
            "石榴",
            "椰子",
            "榴莲",
            "牛油果",
            "山竹",
            "百香果",
            "杨梅",
            "枇杷",
            "木瓜",
            "无花果",
        ],
    },
    TagCategory {
        name: "蔬菜",
        icon: "🥬",
        subs: &[
            "白菜",
            "西兰花",
            "胡萝卜",
            "土豆",
            "番茄",
            "黄瓜",
            "茄子",
            "辣椒",
            "洋葱",
            "大蒜",
            "生姜",
            "玉米",
            "豆角",
            "蘑菇",
            "南瓜",
            "冬瓜",
            "丝瓜",
            "莲藕",
            "竹笋",
            "菠菜",
            "生菜",
            "芦笋",
            "秋葵",
            "豆芽",
            "花生",
            "红薯",
            "山药",
        ],
    },
    TagCategory {
        name: "猫",
        icon: "🐱",
        subs: &[
            "猫",
            "小猫",
            "猫咪",
            "橘猫",
            "黑猫",
            "白猫",
            "布偶猫",
            "英短",
            "美短",
            "暹罗猫",
            "波斯猫",
            "加菲猫",
            "缅因猫",
            "无毛猫",
            "猫咪睡觉",
            "猫咪玩耍",
            "猫爬架",
        ],
    },
    TagCategory {
        name: "狗",
        icon: "🐕",
        subs: &[
            "狗",
            "小狗",
            "狗狗",
            "金毛",
            "拉布拉多",
            "哈士奇",
            "柴犬",
            "柯基",
            "泰迪",
            "博美",
            "法斗",
            "边牧",
            "萨摩耶",
            "德牧",
            "小型犬",
            "大型犬",
            "比熊",
            "贵宾犬",
            "秋田犬",
            "雪纳瑞",
        ],
    },
    TagCategory {
        name: "动物",
        icon: "🐾",
        subs: &[
            "鸟",
            "鹦鹉",
            "麻雀",
            "老鹰",
            "天鹅",
            "鸽子",
            "孔雀",
            "火烈鸟",
            "猫头鹰",
            "兔子",
            "仓鼠",
            "松鼠",
            "刺猬",
            "豚鼠",
            "乌龟",
            "蛇",
            "蜥蜴",
            "变色龙",
            "青蛙",
            "壁虎",
            "马",
            "牛",
            "羊",
            "猪",
            "鹿",
            "驴",
            "骆驼",
            "羊驼",
            "鸡",
            "鸭",
            "鹅",
            "熊猫",
            "老虎",
            "狮子",
            "大象",
            "长颈鹿",
            "斑马",
            "猴子",
            "猩猩",
            "北极熊",
            "棕熊",
            "狐狸",
            "狼",
            "豹子",
            "浣熊",
            "海豚",
            "鲸鱼",
            "海豹",
            "企鹅",
            "考拉",
            "袋鼠",
            "海龟",
            "蝴蝶",
            "蜻蜓",
            "蜜蜂",
            "瓢虫",
            "螳螂",
            "蜗牛",
            "金鱼",
            "锦鲤",
            "热带鱼",
            "水母",
            "珊瑚",
            "海星",
            "章鱼",
        ],
    },
    TagCategory {
        name: "花卉",
        icon: "🌸",
        subs: &[
            "花",
            "花束",
            "花瓶里的花",
            "干花",
            "玫瑰",
            "向日葵",
            "樱花",
            "梅花",
            "荷花",
            "莲花",
            "菊花",
            "兰花",
            "牡丹",
            "郁金香",
            "薰衣草",
            "桂花",
            "百合",
            "康乃馨",
            "茉莉",
            "桃花",
            "杜鹃",
            "紫藤",
            "野花",
            "油菜花田",
            "花海",
            "花园",
            "绣球花",
            "栀子花",
        ],
    },
    TagCategory {
        name: "植物",
        icon: "🌿",
        subs: &[
            "树",
            "大树",
            "松树",
            "柳树",
            "银杏",
            "枫树",
            "椰子树",
            "棕榈树",
            "樟树",
            "竹子",
            "竹林",
            "草地",
            "草坪",
            "苔藓",
            "森林",
            "树林",
            "丛林",
            "红树林",
            "盆栽",
            "绿植",
            "多肉植物",
            "仙人掌",
            "藤蔓",
            "绿萝",
            "吊兰",
            "落叶",
            "枯叶",
            "树叶",
            "红叶",
            "银杏叶",
        ],
    },
    TagCategory {
        name: "山景",
        icon: "🏔️",
        subs: &[
            "山",
            "高山",
            "雪山",
            "山脉",
            "山峰",
            "火山",
            "山谷",
            "峡谷",
            "悬崖",
            "岩石",
            "丹霞地貌",
            "山路",
            "登山",
            "徒步",
            "栈道",
            "山间云雾",
            "云海",
        ],
    },
    TagCategory {
        name: "水景",
        icon: "🌊",
        subs: &[
            "大海",
            "海浪",
            "海岸",
            "海滩",
            "沙滩",
            "礁石",
            "湖",
            "湖泊",
            "湖面倒影",
            "河",
            "河流",
            "小溪",
            "溪流",
            "瀑布",
            "泉水",
            "温泉",
            "池塘",
            "水库",
            "水坝",
            "水渠",
        ],
    },
    TagCategory {
        name: "天空",
        icon: "🌅",
        subs: &[
            "日出",
            "日落",
            "夕阳",
            "朝霞",
            "晚霞",
            "蓝天",
            "白云",
            "蓝天白云",
            "火烧云",
            "星空",
            "银河",
            "月亮",
            "满月",
            "星星",
            "流星",
            "极光",
            "彩虹",
        ],
    },
    TagCategory {
        name: "自然风光",
        icon: "🏞️",
        subs: &[
            "风景",
            "自然风景",
            "田园风光",
            "草原",
            "平原",
            "田野",
            "稻田",
            "麦田",
            "梯田",
            "茶园",
            "沙漠",
            "戈壁",
            "绿洲",
            "沙丘",
            "冰川",
            "冰山",
            "冻土",
            "洞穴",
            "溶洞",
            "钟乳石",
            "湿地",
            "沼泽",
        ],
    },
    TagCategory {
        name: "天气",
        icon: "🌤️",
        subs: &[
            "晴天", "多云", "阴天", "下雨", "暴雨", "雨后", "雨滴", "水滴", "下雪", "雪景", "雪地", "雪花", "霜",
            "大雾", "雾气", "薄雾", "闪电", "雷暴", "台风",
        ],
    },
    TagCategory {
        name: "城市",
        icon: "🏙️",
        subs: &[
            "城市",
            "城市天际线",
            "都市夜景",
            "高楼大厦",
            "摩天大楼",
            "写字楼",
            "街道",
            "马路",
            "十字路口",
            "人行道",
            "斑马线",
            "小巷",
            "胡同",
            "弄堂",
            "广场",
            "公园",
            "喷泉",
            "霓虹灯",
            "灯光",
            "夜市",
            "商业街",
            "步行街",
            "地标建筑",
            "地铁站",
            "天桥",
            "立交桥",
        ],
    },
    TagCategory {
        name: "建筑",
        icon: "🏛️",
        subs: &[
            "建筑",
            "现代建筑",
            "古建筑",
            "传统建筑",
            "教堂",
            "大教堂",
            "清真寺",
            "寺庙",
            "佛塔",
            "神社",
            "鸟居",
            "城堡",
            "宫殿",
            "故宫",
            "桥",
            "大桥",
            "石桥",
            "拱桥",
            "吊桥",
            "塔",
            "灯塔",
            "钟楼",
            "电视塔",
            "纪念碑",
            "雕像",
            "亭子",
            "园林",
            "牌坊",
            "工厂",
            "仓库",
            "水塔",
        ],
    },
    TagCategory {
        name: "室内",
        icon: "🏠",
        subs: &[
            "客厅",
            "卧室",
            "厨房",
            "浴室",
            "卫生间",
            "餐厅",
            "饭厅",
            "酒吧",
            "咖啡厅",
            "办公室",
            "工作台",
            "书桌",
            "会议室",
            "教室",
            "图书馆",
            "书房",
            "走廊",
            "楼梯",
            "阳台",
            "露台",
            "车库",
            "商场",
            "超市",
            "市场",
            "便利店",
            "酒店大堂",
            "酒店房间",
            "健身房",
            "游泳池",
            "电影院",
        ],
    },
    TagCategory {
        name: "家居",
        icon: "🛋️",
        subs: &[
            "沙发",
            "床",
            "桌子",
            "椅子",
            "凳子",
            "书架",
            "柜子",
            "衣柜",
            "鞋柜",
            "灯",
            "台灯",
            "吊灯",
            "落地灯",
            "窗帘",
            "地毯",
            "地板",
            "瓷砖",
            "花瓶",
            "相框",
            "挂画",
            "镜子",
            "时钟",
            "抱枕",
            "毯子",
            "蜡烛",
            "摆件",
        ],
    },
    TagCategory {
        name: "交通",
        icon: "🚗",
        subs: &[
            "汽车",
            "轿车",
            "跑车",
            "越野车",
            "卡车",
            "货车",
            "房车",
            "公交车",
            "大巴",
            "出租车",
            "火车",
            "高铁",
            "地铁",
            "有轨电车",
            "飞机",
            "直升机",
            "热气球",
            "滑翔伞",
            "轮船",
            "游艇",
            "帆船",
            "渡轮",
            "皮划艇",
            "自行车",
            "摩托车",
            "电动车",
            "滑板车",
            "平衡车",
            "高速公路",
            "立交桥",
            "隧道",
            "停车场",
            "加油站",
            "机场",
            "车站",
            "港口",
            "码头",
            "电动汽车",
            "特斯拉",
            "新能源车",
        ],
    },
    TagCategory {
        name: "人物",
        icon: "👤",
        subs: &[
            "自拍",
            "肖像",
            "人像",
            "证件照",
            "合影",
            "合照",
            "团体照",
            "婴儿",
            "宝宝",
            "小孩",
            "儿童",
            "少年",
            "老人",
            "长辈",
            "情侣",
            "新人",
            "闺蜜",
            "兄弟",
            "背影",
            "剪影",
            "侧脸",
            "半身照",
            "全身照",
            "人群",
            "路人",
            "孕妇",
            "亲子",
            "家庭照",
            "模特",
            "写真",
            "艺术照",
        ],
    },
    TagCategory {
        name: "社交活动",
        icon: "🎉",
        subs: &[
            "婚礼",
            "婚纱",
            "婚宴",
            "婚纱照",
            "生日",
            "生日蛋糕",
            "生日派对",
            "聚会",
            "派对",
            "聚餐",
            "年夜饭",
            "毕业典礼",
            "毕业照",
            "颁奖",
            "典礼",
            "仪式",
            "野餐",
            "露营",
            "烧烤聚会",
            "开业",
            "剪彩",
        ],
    },
    TagCategory {
        name: "节日",
        icon: "🎊",
        subs: &[
            "春节",
            "对联",
            "灯笼",
            "烟花",
            "鞭炮",
            "红包",
            "年夜饭",
            "中秋",
            "月饼",
            "赏月",
            "端午",
            "粽子",
            "龙舟",
            "元宵",
            "汤圆",
            "花灯",
            "圣诞",
            "圣诞树",
            "圣诞老人",
            "万圣",
            "南瓜灯",
            "万圣节装扮",
            "情人节",
            "母亲节",
            "父亲节",
            "国庆",
            "阅兵",
            "升旗",
            "清明",
            "重阳",
        ],
    },
    TagCategory {
        name: "运动",
        icon: "⚽",
        subs: &[
            "足球",
            "篮球",
            "排球",
            "网球",
            "乒乓球",
            "羽毛球",
            "高尔夫",
            "棒球",
            "橄榄球",
            "冰球",
            "游泳",
            "跳水",
            "冲浪",
            "帆板",
            "潜水",
            "跑步",
            "马拉松",
            "田径",
            "骑行",
            "自行车赛",
            "登山",
            "攀岩",
            "蹦极",
            "滑雪",
            "滑冰",
            "花样滑冰",
            "滑板",
            "武术",
            "拳击",
            "柔道",
            "跆拳道",
            "瑜伽",
            "健身",
            "举重",
            "体操",
            "普拉提",
            "赛车",
            "摩托车赛",
            "卡丁车",
            "台球",
            "保龄球",
            "飞镖",
            "钓鱼",
            "射箭",
            "跳绳",
            "跳伞",
        ],
    },
    TagCategory {
        name: "旅行",
        icon: "✈️",
        subs: &[
            "旅行",
            "旅游",
            "度假",
            "行李箱",
            "背包",
            "护照",
            "机票",
            "地图",
            "酒店",
            "民宿",
            "帐篷",
            "露营地",
            "海边度假",
            "海岛",
            "沙滩椅",
            "景区",
            "景点",
            "博物馆",
            "美术馆",
            "展览",
            "游乐园",
            "游乐场",
            "摩天轮",
            "过山车",
            "动物园",
            "水族馆",
            "植物园",
            "温泉旅馆",
            "古镇",
            "古城",
        ],
    },
    TagCategory {
        name: "艺术",
        icon: "🎨",
        subs: &[
            "绘画",
            "油画",
            "水彩画",
            "国画",
            "素描",
            "速写",
            "漫画",
            "书法",
            "毛笔字",
            "雕塑",
            "石雕",
            "木雕",
            "涂鸦",
            "街头艺术",
            "壁画",
            "陶瓷",
            "瓷器",
            "陶艺",
            "手工艺品",
            "编织",
            "刺绣",
            "版画",
            "插画",
        ],
    },
    TagCategory {
        name: "音乐",
        icon: "🎵",
        subs: &[
            "吉他",
            "钢琴",
            "小提琴",
            "大提琴",
            "鼓",
            "架子鼓",
            "萨克斯",
            "长笛",
            "二胡",
            "古筝",
            "琵琶",
            "竖琴",
            "麦克风",
            "耳机",
            "音箱",
            "演唱会",
            "音乐会",
            "音乐节",
            "DJ",
            "乐谱",
            "唱片",
            "黑胶唱片",
            "尤克里里",
            "口琴",
            "手风琴",
        ],
    },
    TagCategory {
        name: "文档",
        icon: "📄",
        subs: &[
            "名片",
            "证件",
            "身份证",
            "驾照",
            "护照",
            "收据",
            "发票",
            "账单",
            "快递单",
            "菜单",
            "价目表",
            "书",
            "书页",
            "书本",
            "课本",
            "绘本",
            "报纸",
            "杂志",
            "传单",
            "海报",
            "广告牌",
            "信件",
            "信封",
            "明信片",
            "贺卡",
            "便签",
            "手写笔记",
            "白板",
            "表格",
            "图表",
            "PPT",
            "合同",
        ],
    },
    TagCategory {
        name: "电子产品",
        icon: "📱",
        subs: &[
            "手机",
            "智能手机",
            "iPhone",
            "电脑",
            "笔记本电脑",
            "台式电脑",
            "显示器",
            "屏幕",
            "一体机",
            "平板电脑",
            "iPad",
            "相机",
            "单反相机",
            "镜头",
            "摄像机",
            "运动相机",
            "GoPro",
            "键盘",
            "机械键盘",
            "鼠标",
            "鼠标垫",
            "耳机",
            "头戴式耳机",
            "蓝牙耳机",
            "AirPods",
            "音箱",
            "蓝牙音箱",
            "智能音箱",
            "手表",
            "智能手表",
            "Apple Watch",
            "电视",
            "投影仪",
            "游戏机",
            "游戏手柄",
            "Switch",
            "PlayStation",
            "无人机",
            "机器人",
            "内存条",
            "显卡",
            "主板",
            "CPU",
            "硬盘",
            "SSD",
            "散热器",
            "电源",
            "路由器",
            "交换机",
            "网线",
            "网卡",
            "充电器",
            "数据线",
            "移动电源",
            "充电宝",
            "U盘",
            "存储卡",
            "读卡器",
            "打印机",
            "扫描仪",
            "NAS",
            "服务器",
            "机柜",
        ],
    },
    TagCategory {
        name: "服饰",
        icon: "👔",
        subs: &[
            "衣服",
            "T恤",
            "衬衫",
            "外套",
            "夹克",
            "大衣",
            "毛衣",
            "卫衣",
            "羽绒服",
            "裙子",
            "连衣裙",
            "短裙",
            "长裙",
            "百褶裙",
            "裤子",
            "牛仔裤",
            "短裤",
            "运动裤",
            "西裤",
            "西装",
            "礼服",
            "制服",
            "校服",
            "工装",
            "运动鞋",
            "高跟鞋",
            "靴子",
            "拖鞋",
            "凉鞋",
            "板鞋",
            "皮鞋",
            "马丁靴",
            "帽子",
            "棒球帽",
            "草帽",
            "贝雷帽",
            "毛线帽",
            "渔夫帽",
            "包",
            "手提包",
            "背包",
            "钱包",
            "单肩包",
            "腰包",
            "斜挎包",
            "托特包",
            "围巾",
            "领带",
            "手套",
            "袜子",
            "泳衣",
            "比基尼",
            "睡衣",
            "浴袍",
            "居家服",
            "汉服",
            "旗袍",
            "和服",
            "韩服",
        ],
    },
    TagCategory {
        name: "饰品",
        icon: "💍",
        subs: &[
            "项链",
            "手链",
            "手镯",
            "戒指",
            "耳环",
            "耳钉",
            "手表",
            "腕表",
            "眼镜",
            "太阳镜",
            "墨镜",
            "胸针",
            "发饰",
            "头饰",
            "发卡",
            "发带",
            "珠宝",
            "钻石",
            "宝石",
            "翡翠",
            "玉",
        ],
    },
    TagCategory {
        name: "玩具",
        icon: "🧸",
        subs: &[
            "玩具",
            "毛绒玩具",
            "泰迪熊",
            "玩偶",
            "芭比",
            "手办",
            "模型",
            "积木",
            "乐高",
            "拼图",
            "魔方",
            "玩具车",
            "遥控车",
            "遥控飞机",
            "棋盘",
            "象棋",
            "围棋",
            "国际象棋",
            "扑克牌",
            "桌游",
            "麻将",
            "气球",
            "风筝",
            "水枪",
        ],
    },
    TagCategory {
        name: "厨房",
        icon: "🍳",
        subs: &[
            "厨具",
            "锅",
            "平底锅",
            "砂锅",
            "炒锅",
            "碗",
            "盘子",
            "碟子",
            "杯子",
            "玻璃杯",
            "马克杯",
            "刀",
            "菜刀",
            "砧板",
            "筷子",
            "勺子",
            "叉子",
            "烤箱",
            "微波炉",
            "电饭煲",
            "榨汁机",
            "咖啡机",
            "冰箱",
            "洗碗机",
            "空气炸锅",
            "电磁炉",
        ],
    },
    TagCategory {
        name: "医疗健康",
        icon: "🏥",
        subs: &[
            "医院",
            "诊所",
            "药房",
            "医生",
            "护士",
            "手术室",
            "药品",
            "药片",
            "针管",
            "注射器",
            "口罩",
            "体温计",
            "血压计",
            "听诊器",
            "轮椅",
            "拐杖",
            "绷带",
            "牙刷",
            "牙膏",
            "体重秤",
        ],
    },
    TagCategory {
        name: "学习办公",
        icon: "📚",
        subs: &[
            "学校",
            "大学",
            "教室",
            "黑板",
            "讲台",
            "笔",
            "铅笔",
            "钢笔",
            "彩笔",
            "笔记本",
            "日记本",
            "文件夹",
            "计算器",
            "尺子",
            "圆规",
            "橡皮",
            "办公桌",
            "文件",
            "打印机",
            "复印机",
            "书包",
            "台灯",
            "地球仪",
        ],
    },
    TagCategory {
        name: "宗教文化",
        icon: "🙏",
        subs: &[
            "佛像",
            "菩萨",
            "观音",
            "弥勒",
            "十字架",
            "耶稣",
            "圣母",
            "经书",
            "佛经",
            "圣经",
            "古兰经",
            "香炉",
            "烛台",
            "祭坛",
            "神龛",
            "传统服饰",
            "汉服",
            "和服",
            "旗袍",
            "唐装",
        ],
    },
    TagCategory {
        name: "化妆护肤",
        icon: "💄",
        subs: &[
            "口红",
            "唇膏",
            "粉底",
            "眼影",
            "腮红",
            "眉笔",
            "香水",
            "护肤品",
            "面膜",
            "精华液",
            "化妆镜",
            "化妆刷",
            "美甲",
            "洗面奶",
            "防晒霜",
            "乳液",
        ],
    },
    TagCategory {
        name: "母婴",
        icon: "👶",
        subs: &[
            "奶瓶",
            "奶嘴",
            "尿布",
            "纸尿裤",
            "婴儿车",
            "婴儿床",
            "摇篮",
            "婴儿玩具",
            "安抚玩具",
            "婴儿服",
            "口水巾",
        ],
    },
    TagCategory {
        name: "宠物用品",
        icon: "🦴",
        subs: &[
            "猫粮",
            "狗粮",
            "宠物零食",
            "猫砂盆",
            "猫窝",
            "狗窝",
            "宠物玩具",
            "磨爪板",
            "宠物衣服",
            "牵引绳",
            "宠物笼",
        ],
    },
    TagCategory {
        name: "工具",
        icon: "🔧",
        subs: &[
            "螺丝刀",
            "扳手",
            "锤子",
            "钳子",
            "电钻",
            "电锯",
            "角磨机",
            "胶带",
            "绳子",
            "剪刀",
            "梯子",
            "工具箱",
        ],
    },
    TagCategory {
        name: "情趣",
        icon: "🔞",
        subs: &[
            "内衣",
            "蕾丝内衣",
            "情趣内衣",
            "丝袜",
            "吊带袜",
            "网袜",
            "泳装",
            "比基尼",
            "死库水",
            "旗袍",
            "和服",
            "女仆装",
            "JK制服",
            "护士制服",
            "兔女郎",
            "紧身衣",
            "透视装",
            "低胸装",
            "露背装",
            "Cosplay",
            "角色扮演",
            "写真",
            "私房照",
            "人体艺术",
            "人体摄影",
            "性感自拍",
            "性感姿势",
            "性感泳装",
            "床上",
            "浴室自拍",
            "镜前自拍",
            "束缚",
            "捆绑",
            "项圈",
            "手铐",
            "皮具",
            "成人玩具",
            "情趣用品",
        ],
    },
    TagCategory {
        name: "手工DIY",
        icon: "✂️",
        subs: &[
            "手工",
            "DIY",
            "缝纫",
            "针织",
            "钩针",
            "折纸",
            "剪纸",
            "纸艺",
            "皮革",
            "皮雕",
            "皮具制作",
            "木工",
            "木雕",
            "竹编",
            "蜡烛制作",
            "肥皂制作",
            "香薰",
            "十字绣",
            "拼贴",
            "贴纸",
        ],
    },
    TagCategory {
        name: "游戏",
        icon: "🎮",
        subs: &[
            "电子游戏",
            "手机游戏",
            "电脑游戏",
            "游戏截图",
            "游戏画面",
            "游戏角色",
            "桌游",
            "卡牌游戏",
            "棋盘游戏",
            "VR游戏",
            "街机",
            "电竞",
            "比赛直播",
            "游戏主播",
        ],
    },
    TagCategory {
        name: "夜生活",
        icon: "🌃",
        subs: &[
            "酒吧",
            "夜店",
            "KTV",
            "卡拉OK",
            "夜景",
            "霓虹",
            "灯红酒绿",
            "鸡尾酒",
            "酒杯",
            "干杯",
            "DJ打碟",
            "舞池",
            "舞蹈",
        ],
    },
    TagCategory {
        name: "影视",
        icon: "🎬",
        subs: &[
            "电影海报",
            "电影剧照",
            "电视剧",
            "动漫",
            "动画",
            "卡通",
            "综艺",
            "真人秀",
            "脱口秀",
            "纪录片",
            "短片",
            "影院",
            "荧幕",
            "拍摄现场",
            "片场",
        ],
    },
    TagCategory {
        name: "园艺",
        icon: "🌱",
        subs: &[
            "种植",
            "播种",
            "育苗",
            "嫁接",
            "花盆",
            "花坛",
            "花架",
            "浇水",
            "施肥",
            "修剪",
            "菜园",
            "果园",
            "温室",
            "大棚",
            "阳台花园",
            "屋顶花园",
            "庭院",
        ],
    },
    TagCategory {
        name: "天文",
        icon: "🔭",
        subs: &[
            "望远镜",
            "天文望远镜",
            "星座",
            "星云",
            "星团",
            "月食",
            "日食",
            "日环食",
            "行星",
            "土星",
            "木星",
            "火星",
            "流星雨",
            "彗星",
            "卫星",
            "太空",
            "宇航员",
            "火箭",
            "空间站",
        ],
    },
];

/// Cached category embeddings for classification.
struct EmbeddingCache {
    entries: Vec<EmbeddingEntry>,
}

struct EmbeddingEntry {
    category_idx: usize,
    sub_idx: usize,
    vec: Vec<f32>,
}

static CACHE: Mutex<Option<EmbeddingCache>> = Mutex::new(None);

/// Compute cosine similarity between two vectors.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Deterministic hash of the full taxonomy (category names + subcategory prompts).
/// Any edit to CATEGORIES changes this value, which invalidates the disk cache.
fn taxonomy_hash() -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for cat in CATEGORIES {
        cat.name.hash(&mut hasher);
        cat.icon.hash(&mut hasher);
        for sub in cat.subs {
            sub.hash(&mut hasher);
        }
    }
    hasher.finish()
}

const CACHE_MAGIC: &[u8; 8] = b"CLIPTAG\0";
const CACHE_VERSION: u32 = 1;

/// Try to load embeddings from a disk cache file.
/// Returns `None` if the file doesn't exist, is corrupt, or has a stale taxonomy hash.
fn try_load_disk_cache(path: &std::path::Path, expected_hash: u64) -> Option<Vec<EmbeddingEntry>> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;

    let mut cursor = 0usize;

    // magic (8) + version (4) + hash (8) + vec_dim (4) + num_entries (4) = 28 bytes header
    if buf.len() < 28 {
        return None;
    }
    if &buf[cursor..cursor + 8] != CACHE_MAGIC {
        return None;
    }
    cursor += 8;

    let version = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().ok()?);
    if version != CACHE_VERSION {
        return None;
    }
    cursor += 4;

    let stored_hash = u64::from_le_bytes(buf[cursor..cursor + 8].try_into().ok()?);
    if stored_hash != expected_hash {
        tracing::info!("CLIP cache taxonomy hash mismatch — rebuilding");
        return None;
    }
    cursor += 8;

    let vec_dim = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().ok()?) as usize;
    cursor += 4;

    let num_entries = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().ok()?) as usize;
    cursor += 4;

    // Validate total size: header + entries * (cat_idx(4) + sub_idx(4) + vec_dim*4)
    let entry_size = 4 + 4 + vec_dim * 4;
    if buf.len() < 28 + num_entries * entry_size {
        return None;
    }

    let mut entries = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let cat_idx = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().ok()?) as usize;
        cursor += 4;
        let sub_idx = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().ok()?) as usize;
        cursor += 4;

        let floats: Vec<f32> = buf[cursor..cursor + vec_dim * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        cursor += vec_dim * 4;

        entries.push(EmbeddingEntry {
            category_idx: cat_idx,
            sub_idx,
            vec: floats,
        });
    }

    Some(entries)
}

/// Persist embeddings to disk so the next process start skips inference.
fn save_disk_cache(path: &std::path::Path, hash: u64, entries: &[EmbeddingEntry], vec_dim: usize) {
    let write = || -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(path)?;
        file.write_all(CACHE_MAGIC)?;
        file.write_all(&CACHE_VERSION.to_le_bytes())?;
        file.write_all(&hash.to_le_bytes())?;
        file.write_all(&(vec_dim as u32).to_le_bytes())?;
        file.write_all(&(entries.len() as u32).to_le_bytes())?;
        for entry in entries {
            file.write_all(&(entry.category_idx as u32).to_le_bytes())?;
            file.write_all(&(entry.sub_idx as u32).to_le_bytes())?;
            for &f in &entry.vec {
                file.write_all(&f.to_le_bytes())?;
            }
        }
        Ok(())
    };
    if let Err(e) = write() {
        tracing::warn!("Failed to save CLIP embedding cache: {e}");
    }
}

/// Initialize the embedding cache.
///
/// 1. Try loading from disk (`{cache_dir}/clip/category-embeddings.bin`)
/// 2. If missing/stale, compute all embeddings via `embed_fn` and persist to disk
pub fn ensure_cache(
    embed_fn: &dyn Fn(&str) -> Result<Vec<f32>, String>,
    cache_dir: Option<&str>,
) -> Result<(), String> {
    let mut guard = CACHE.lock().map_err(|e| format!("Cache lock: {e}"))?;
    if guard.is_some() {
        return Ok(());
    }

    let hash = taxonomy_hash();
    let cache_path = cache_dir.map(|dir| {
        std::path::PathBuf::from(dir)
            .join("clip")
            .join("category-embeddings.bin")
    });

    // Try disk cache first
    if let Some(ref path) = cache_path
        && let Some(entries) = try_load_disk_cache(path, hash)
    {
        tracing::info!(
            "CLIP category embeddings loaded from disk cache: {} entries ({} categories)",
            entries.len(),
            CATEGORIES.len()
        );
        *guard = Some(EmbeddingCache { entries });
        return Ok(());
    }

    // Compute from scratch
    let start = std::time::Instant::now();
    let mut entries = Vec::new();
    for (cat_idx, cat) in CATEGORIES.iter().enumerate() {
        for (sub_idx, sub) in cat.subs.iter().enumerate() {
            let prompt = format!("一张{sub}的照片");
            let vec = embed_fn(&prompt)?;
            entries.push(EmbeddingEntry {
                category_idx: cat_idx,
                sub_idx,
                vec,
            });
        }
    }
    let elapsed = start.elapsed();
    tracing::info!(
        "CLIP category embeddings computed: {} entries across {} categories in {:.1}s",
        entries.len(),
        CATEGORIES.len(),
        elapsed.as_secs_f32()
    );

    // Persist to disk for next startup
    if let Some(ref path) = cache_path {
        let vec_dim = entries.first().map_or(512, |e| e.vec.len());
        save_disk_cache(path, hash, &entries, vec_dim);
        tracing::info!("CLIP embedding cache saved to {}", path.display());
    }

    *guard = Some(EmbeddingCache { entries });
    Ok(())
}

/// Classify an image vector against the taxonomy.
///
/// Uses softmax normalization over per-category max similarities.
/// This makes scoring relative: only categories significantly more similar
/// than others get high probability, regardless of absolute cosine values.
pub fn classify(
    image_vec: &[f32],
    embed_fn: &dyn Fn(&str) -> Result<Vec<f32>, String>,
    cache_dir: Option<&str>,
) -> Result<Vec<TagResult>, String> {
    ensure_cache(embed_fn, cache_dir)?;
    let guard = CACHE.lock().map_err(|e| format!("Cache lock: {e}"))?;
    let cache = guard.as_ref().expect("cache initialized above");

    // Step 1: For each category, find the best-matching subcategory (max cosine sim)
    let mut best_per_cat: Vec<Option<(usize, f32)>> = vec![None; CATEGORIES.len()];

    for entry in &cache.entries {
        let sim = cosine_sim(image_vec, &entry.vec);
        let slot = &mut best_per_cat[entry.category_idx];
        match slot {
            Some((_, prev)) if sim > *prev => {
                *slot = Some((entry.sub_idx, sim));
            }
            None => {
                *slot = Some((entry.sub_idx, sim));
            }
            _ => {}
        }
    }

    // Step 2: Softmax over per-category max similarities
    // logits = cosine_sim * LOGIT_SCALE
    let cat_scores: Vec<(usize, usize, f32)> = best_per_cat
        .iter()
        .enumerate()
        .filter_map(|(cat_idx, best)| best.map(|(sub_idx, sim)| (cat_idx, sub_idx, sim)))
        .collect();

    if cat_scores.is_empty() {
        return Ok(vec![]);
    }

    let logits: Vec<f32> = cat_scores.iter().map(|(_, _, sim)| sim * LOGIT_SCALE).collect();
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();

    // Step 3: Filter by probability threshold and collect results
    let mut results: Vec<TagResult> = cat_scores
        .iter()
        .zip(logits.iter())
        .filter_map(|((cat_idx, sub_idx, _sim), logit)| {
            let prob = (logit - max_logit).exp() / exp_sum;
            if prob >= MIN_PROB {
                let cat = &CATEGORIES[*cat_idx];
                Some(TagResult {
                    category: cat.name,
                    icon: cat.icon,
                    subcategory: cat.subs[*sub_idx],
                    score: prob,
                })
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(MAX_TAGS);
    Ok(results)
}
