"""格斗家身体参数。原型阶段只有两个实验体 + 一个标准陪练。

字段含义（都是相对 1.0 的倍率，除非注明单位）：
  reach       臂展，米（拳头能伸出的距离）
  mass        体重，公斤（影响步速、被击退距离、互推）；不算属性分，由量级限定
  power       出拳伤害倍率
  hand_speed  出拳快慢（前摇/后摇帧数除以它）
  foot_speed  步速
  cardio      体能上限和回复速度
  chin        抗击打（受到伤害除以它）
  react_ticks 反应时间，帧（1 帧 = 0.05 秒；看到的对手状态是这么多帧以前的）
  stance      站架：orthodox 正架（左手在前）/ southpaw 反架（右手在前）
  school      流派（schools.py）：定默认架势、天赋加成、训练时哪些技术练得勤
  temperament 性格 {胆量, 沉着, 斗志, 冲动: 0~1}，没写的按 0.5
  mind        心智 {观察力, 拳商: 0~1}，没写的按 0.5
  skills      天赋熟练度 {技术名: 0~1}，没写的按 0.5（只有拳击模拟器用）。
              这是出身带来的起点；训练中常用、用成功的技术会越练越熟，不用的慢慢退回天赋水平
"""
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class Body:
    id: str
    label: str
    reach: float
    mass: float
    power: float
    hand_speed: float
    foot_speed: float
    cardio: float
    chin: float
    react_ticks: int
    stance: str = "orthodox"
    skills: dict = field(default=None, compare=False, hash=False)
    school: str = None
    temperament: dict = field(default=None, compare=False, hash=False)
    mind: dict = field(default=None, compare=False, hash=False)

    def to_dict(self):
        return asdict(self)


# 实验体 A：长臂、快、反应好、看得准，用力量和抗击打换来的 —— 按外围型配（属性总分 = 预算）
LANKY = Body("A", "实验体A·长臂", reach=0.88, mass=71, power=0.80,
             hand_speed=1.10, foot_speed=1.10, cardio=0.98, chin=0.85, react_ticks=3,
             skills={"刺拳·头": 0.75, "刺拳·身": 0.65, "后手直拳·头": 0.6, "后手直拳·身": 0.6, "前手勾拳·头": 0.5,
                     "前手勾拳·身": 0.35, "后手勾拳·头": 0.35, "后手勾拳·身": 0.3, "前手上勾·头": 0.35, "前手上勾·身": 0.3,
                     "后手上勾·头": 0.3, "后手上勾·身": 0.3, "高位格挡": 0.6, "护体": 0.4, "拍挡": 0.6, "侧闪": 0.6,
                     "下潜": 0.5, "后仰": 0.65, "缠抱": 0.5, "轴转": 0.6, "虚晃": 0.65, "步法": 0.7},
             school="外围拳击", temperament={"胆量": 0.4, "沉着": 0.65, "斗志": 0.4, "冲动": 0.35},
             mind={"观察力": 0.55, "拳商": 0.5})

# 实验体 B：拳重、抗揍、体能好，用臂展、速度和反应换来的 —— 按压迫型配（属性总分 = 预算）
STOCKY = Body("B", "实验体B·重拳", reach=0.65, mass=72.5, power=1.25,
              hand_speed=1.00, foot_speed=0.95, cardio=1.10, chin=1.25, react_ticks=4,
              skills={"刺拳·头": 0.45, "刺拳·身": 0.4, "后手直拳·头": 0.55, "后手直拳·身": 0.55,
                      "前手勾拳·头": 0.65, "前手勾拳·身": 0.6, "后手勾拳·头": 0.6, "后手勾拳·身": 0.55,
                      "前手上勾·头": 0.6, "前手上勾·身": 0.55, "后手上勾·头": 0.55, "后手上勾·身": 0.55,
                      "高位格挡": 0.6, "护体": 0.55, "拍挡": 0.4, "侧闪": 0.45, "下潜": 0.55, "后仰": 0.35,
                      "缠抱": 0.5, "轴转": 0.4, "虚晃": 0.4, "步法": 0.45},
              school="压迫型", temperament={"胆量": 0.7, "沉着": 0.45, "斗志": 0.7, "冲动": 0.6},
              mind={"观察力": 0.45, "拳商": 0.5})

# 标准陪练：各项居中，打法写死（见 bots.py），用来当量尺
SPARRING = Body("S", "标准陪练", reach=0.75, mass=75, power=1.0,
                hand_speed=1.0, foot_speed=1.0, cardio=1.0, chin=1.0, react_ticks=4)

ALL = {b.id: b for b in (LANKY, STOCKY, SPARRING)}

# ------------------------------------------------------------------ 属性预算
# 每项属性按范围折算成 0~10 分（反应时间越短分越高，所以范围倒过来写），普通人每项 5 分。
# 同一量级的角色总分相同：想要什么长处，就得在别处让出来。体重不算分，由量级限定。
ATTR_RANGES = {
    "reach": ("臂展", 0.55, 0.95), "power": ("力量", 0.70, 1.40), "hand_speed": ("手速", 0.85, 1.25),
    "foot_speed": ("步速", 0.80, 1.25), "cardio": ("体能", 0.80, 1.25), "chin": ("抗击打", 0.75, 1.35),
    "react_ticks": ("反应", 5, 2),
}
MIND_RANGES = {"观察力": (0.0, 1.0), "拳商": (0.0, 1.0)}
BUDGET = 5.0 * (len(ATTR_RANGES) + len(MIND_RANGES))
WEIGHT_CLASS = ("中量级", 69.9, 72.6)       # 公斤，同一量级才能打


def attr_points(body):
    """{属性中文名: 分数}。"""
    out = {}
    for k, (name, lo, hi) in ATTR_RANGES.items():
        out[name] = 10.0 * (getattr(body, k) - lo) / (hi - lo)
    mind = getattr(body, "mind", None) or {}
    for name, (lo, hi) in MIND_RANGES.items():
        out[name] = 10.0 * (mind.get(name, 0.5) - lo) / (hi - lo)
    return out


def budget_total(body):
    return sum(attr_points(body).values())
