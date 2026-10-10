"""二维拳击模拟器（向量化，N 场同时跑）。规则见 docs/拳击技术规格.md。

胜负只有 KO、TKO、点数判定。体力、体能、晕眩、身体伤都是资源池。
确定性：给定初始站位和双方每帧动作，结果完全确定，回放只存动作。

坐标：俯视的方形擂台 [0, RING]²，th 是朝向（弧度）。
"""
import numpy as np

SIM_VERSION = "box2d-v9"

DT = 0.05
RING = 6.1
BODY_R = 0.30
SHOULDER = 0.15
MIN_GAP = 2 * BODY_R
WALK = 1.6
BACK_MULT, SIDE_MULT, GUARD_WALK = 0.85, 0.8, 0.5
HP_MAX = 100.0
STUN_KD, STUN_HURT, BODY_KD = 100.0, 60.0, 100.0
STUN_DECAY = 8.0 * DT
BODY_DECAY = 2.0 * DT
STAM_BASE, STAM_REGEN = 100.0, 2.5     # 体能回复：持续出拳量被体能卡住（高手每分钟三十来拳）
WALK_COST = 0.04                        # 走动每帧耗的体能：外围型来回走也会累
TIRED = 0.25                            # 体能低于 25%：手会掉（举架效果 ×0.6、没有自然站姿保护）、反应慢 1 帧
TIRED_GUARD = 0.6
TURN_RATE = np.radians(300)
ANG_STRAIGHT, ANG_HOOK, ANG_GUARD = np.radians(30), np.radians(45), np.radians(25)
ROUND_TICKS = 1200          # 一回合 60 秒
N_ROUNDS = 3
DOWN_TICKS = 60             # 读秒压缩成 3 秒
CLINCH_TICKS, CLINCH_MIN, CLINCH_REACH = 40, 10, 0.85
PIVOT_TICKS, PIVOT_STEP = 3, 0.16
BUSY_TURN = 0.15          # 出拳、收招、挨打时转身只有平时的 15%
PULL_DIST = 0.25
NEUTRAL_GAP = 2.0
UNANSWERED_STOP = 8
HOLDS_WARN = 4
STANCE_SIGN = {"orthodox": 1.0, "southpaw": -1.0}

# ------------------------------------------------------------------ 动作
(STAY, FWD, BACK, LEFT, RIGHT, PIVOT_L, PIVOT_R,
 JAB_H, JAB_B, CROSS_H, CROSS_B, LHOOK_H, LHOOK_B, RHOOK_H, RHOOK_B,
 LUPPER_H, LUPPER_B, RUPPER_H, RUPPER_B,
 GUARD_HIGH, GUARD_FWD, GUARD_BODY, PARRY, SLIP, DUCK, PULL, CLINCH,
 FEINT_JAB, FEINT_CROSS, CHANGE_POSTURE, SWITCH_STANCE) = range(31)
N_ACTIONS = 31
PUNCH0 = JAB_H
ACTION_NAMES = ["不动", "前进", "后退", "左移", "右移", "左轴转", "右轴转",
                "刺拳·头", "刺拳·身", "后手直拳·头", "后手直拳·身", "前手勾拳·头", "前手勾拳·身",
                "后手勾拳·头", "后手勾拳·身", "前手上勾·头", "前手上勾·身", "后手上勾·头", "后手上勾·身",
                "高位格挡", "举架压上", "护体", "拍挡", "侧闪", "下潜", "后仰", "缠抱", "虚晃刺拳", "虚晃后手", "换架势", "换站架"]

# 状态
FREE, STARTUP, RECOVERY, HITSTUN, BLOCKSTUN, DEFEND, CLINCHED, DOWN, MOVING = range(9)
N_PHASES = 9
PHASE_NAMES = ["", "起手", "收招", "被打中", "格挡硬直", "防守中", "缠抱", "倒地", "轴转"]

# 正在做的招（move）：0..11 是 12 种拳
M_CLINCH, M_PARRY, M_SLIP, M_DUCK, M_PULL, M_PIVL, M_PIVR, M_NONE, M_POSTURE, M_STANCE = range(12, 22)
N_MOVES = 22
OBS_MOVES = 20          # 观测里招式的独热只到 M_NONE，换架势、换站架显示成"轴转/移动中"之外的空招

# ------------------------------------------------------------------ 12 种拳
#               刺头  刺身  后头  后身  前勾头 前勾身 后勾头 后勾身 前上头 前上身 后上头 后上身
HAND = np.array([0,    0,    1,    1,    0,     0,     1,     1,     0,     0,     1,     1])   # 0 前手 1 后手
TYPE = np.array([0,    0,    0,    0,    1,     1,     1,     1,     2,     2,     2,     2])   # 0 直 1 勾 2 上勾
TGT = np.array([0,     1,    0,    1,    0,     1,     0,     1,     0,     1,     0,     1])   # 0 头 1 身
SU = np.array([3,      4,    5,    6,    4,     5,     6,     6,     4,     5,     5,     6])
REC = np.array([5,     6,    8,    9,    8,     8,     10,    10,    8,     8,     10,    10])
DMG = np.array([4.0,   4,    11,   10,   11,    10,    13,    12,    9,     9,     12,    11])
COST = np.array([3.0,  3,    7,    7,    7,     7,     9,     9,     6,     6,     8,     8])
RNG = np.array([1.00,  0.95, 0.98, 0.92, 0.70,  0.65,  0.72,  0.68,  0.55,  0.52,  0.58,  0.55])
STUNK = np.array([0.8, 0,    1.0,  0,    1.4,   0,     1.4,   0,     1.3,   0,     1.5,   0])
BODYK = np.array([0.0, 1.0,  0,    1.2,  0,     1.6,   0,     1.4,   0,     1.2,   0,     1.3])
HITSTUN_T = np.array([2, 3,  5,    5,    6,     5,     7,     6,     6,     5,     7,     6])    # 刺拳只是刺一下，不会把人定住
MIN_SU = np.array([2,    2,    3,    3,    3,     3,     4,     4,     3,     3,     4,     4])    # 起手再快也有下限（刺拳 100 毫秒）
BLOCKSTUN_T = np.array([2, 2, 4,   3,    4,     3,     5,     4,     4,     3,     5,     4])
KB = np.array([0.08,   0.05, 0.25, 0.15, 0.12,  0.08,  0.15,  0.10,  0.10,  0.06,  0.14,  0.08])
TWIST = np.array([0.0, 0,    0,    0,    0.25,  0.10,  0.30,  0.10,  0,     0,     0,     0])
SCORE = np.array([1.0, 0.7,  2.0,  1.5,  2.0,   1.5,   2.0,   1.5,   2.0,   1.5,   2.0,   1.5])  # 裁判计分
IS_POWER = np.array([0, 0,   1,    1,    1,     1,     1,     1,     1,     1,     1,     1])
CRAMP_FRAC = 0.55
# 勾拳、上勾拳是弯臂拳：臂展长只占一部分优势（直拳全吃，勾拳一半，上勾拳 40%），其余按普通臂展 0.75 米算
REACH_W = np.array([1.0, 0.5, 0.4])
DMG_SCALE = 0.55
# 自然站姿：没专门举架、也没在出拳或挨打时，手本来就在脸附近。头部直拳 ×0.7、头部勾拳 ×0.85；低手位没有这层保护
NATURAL_GUARD = np.ones((4, 6))
NATURAL_GUARD[[0, 1, 2], 0] = 0.7
NATURAL_GUARD[[0, 1, 2], 2] = 0.85            # 整体伤害倍率：让比赛能打进后面的回合，KO 靠干净的重拳和反击
BODY_STAM, BODY_METER = 3.0, 2.0      # 身体拳：每点伤害扣多少体能、累积多少身体伤
COUNTER_DMG, COUNTER_STUN = 1.3, 1.6

# 拳的类别 = TYPE*2 + TGT：0 头直 1 身直 2 头勾 3 身勾 4 头上勾 5 身上勾
GUARD_MULT = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],      # 无
    [0.15, 0.7, 0.35, 0.6, 0.75, 0.7],   # 高位格挡
    [0.8, 0.2, 1.0, 0.25, 1.0, 0.3],     # 护体
])
EVADE_MULT = np.ones((N_MOVES, 6))
EVADE_MULT[M_PARRY] = [0, 1, 1, 1, 1, 1]
EVADE_MULT[M_SLIP] = [0, 1, 1, 1, 1, 1]
EVADE_MULT[M_DUCK] = [0, 1, 0, 1, 1.4, 1.2]
DEF_ACTIVE = np.zeros(N_MOVES, int)
DEF_REC = np.zeros(N_MOVES, int)
DEF_COST = np.zeros(N_MOVES)
OFFBAL = np.zeros(N_MOVES, int)           # 躲开 / 拍开之后，进攻方多出来的收招帧
for m, act, rec, cost, ob in ((M_PARRY, 4, 4, 1, 4), (M_SLIP, 5, 4, 2, 2), (M_DUCK, 6, 5, 3, 2), (M_PULL, 5, 4, 2, 1)):
    DEF_ACTIVE[m], DEF_REC[m], DEF_COST[m], OFFBAL[m] = act, rec, cost, ob
DEF_OF_ACTION = {PARRY: M_PARRY, SLIP: M_SLIP, DUCK: M_DUCK, PULL: M_PULL}

# 结局
M_NOTYET, M_KO, M_TKO, M_DEC, M_DRAW = range(5)
METHOD_NAMES = ["", "KO", "TKO", "点数判定", "平局"]
# 三位裁判：[刺拳分, 重拳分, 被挡拳, 压迫帧]
JUDGES = np.array([[1.0, 1.5, 0.0, 0.0],      # 看重拳
                   [1.0, 1.0, 0.1, 0.0],      # 看手数
                   [1.0, 1.0, 0.0, 0.01]])    # 看压迫
REWARD_PER_POINT = 0.02
REWARD_KD = 0.3

# 精准：打在射程最边缘只是擦到
GLANCE_FRAC, GLANCE_MULT = 0.08, 0.6

# 落点与击倒：干净打中头部时先抽落点，冲击 = 伤害 × 落点系数 × 拳种系数，再和挨打者此刻的抗击打线比
Z_JAW, Z_SIDE, Z_FRONT = range(3)
ZONE_NAMES = ["下颌", "头侧", "额头面门"]
#                      下颌   头侧   额头面门   （起点，没有现成数据，以后用击倒率校准）
ZONE_BASE = np.array([[0.30, 0.05, 0.65],     # 直拳
                      [0.45, 0.40, 0.15],     # 勾拳
                      [0.60, 0.00, 0.40]])    # 上勾拳
ZONE_F = np.array([1.0, 0.6, 0.25])           # 打下颌头部转得最猛：KO 多数打在下颌（Hutchison 2014、Cournoyer 2019）
ROT_F = np.array([0.75, 1.0, 0.4])            # 直拳 / 勾拳 / 上勾拳：奥运拳手打假人，头部转动勾拳 > 直拳 > 上勾拳（Viano 2005）
JAW_POSTURE = np.array([1.0, 1.0, 0.6, 1.3])  # 防守方架势：低姿抱架收着下巴 ×0.6，低手位 ×1.3
JAW_TIRED = 1.3                               # 累垮了手掉下来，下巴更露
JAW_SKILL = 0.4                               # 进攻方这一拳的熟练度从 0.5 到 1.0，打中下颌的概率 ×1.0 → ×1.2
UNSEEN_F = 1.15                               # 没看见的拳：比防守方的反应快，或者他自己正在起手（证据弱，只给小加成）
KD_T0, KD_SLOPE, KD_FLOOR = 12.3, 0.12, 0.3   # 满状态的抗击打线（定在：满状态挨一记反击后手勾拳打中下颌，约 25% 倒地）；过线后概率涨多快；冲击不到线的三成不可能倒

# 心态（气势）：资源池，不判胜负。低了会反应变慢、紧张掉体能、出拳犹豫
COMP_BASE = 80.0
COMP_DRIFT = 0.002              # 每帧向基准回归的比例
COMP_HIT = {"head": -3.0, "power_head": -5.0, "counter": -3.0, "body": -2.0}
COMP_LAND = {"head": 2.0, "power_head": 3.0, "body": 1.5}
COMP_DEFEND, COMP_WHIFF_POWER, COMP_FAKED = 1.0, -1.5, -4.0
COMP_KD_DOWN, COMP_KD_SCORE = -25.0, 15.0
COMP_PINNED, COMP_PINNING = -0.05, 0.03     # 每帧：被压在围绳上 / 把对手压在围绳上
COMP_SLOW1, COMP_SLOW2, COMP_TENSE, COMP_FREEZE = 40.0, 20.0, 30.0, 15.0

# 读招记忆：对手出什么拳、怎么防我的拳，按事件做指数平均
TEND_ALPHA = 0.1
PUNCH_GROUP = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])     # 刺拳 / 后手直拳 / 勾拳 / 上勾拳
DEF_CAT = np.zeros(N_MOVES, int)                                  # 0 没防 1 高位 2 护体 3 拍挡 4 侧闪 5 下潜 6 后仰
DEF_CAT[M_PARRY], DEF_CAT[M_SLIP], DEF_CAT[M_DUCK], DEF_CAT[M_PULL] = 3, 4, 5, 6
FEINT_CODE = 12

# 架势：标准高位 / 长架（前手撑开顶住对手） / 低姿抱架（缩小目标） / 低手位（手放低、看得清、出手快）
P_STANDARD, P_LONG, P_CROUCH, P_LOW = range(4)
POSTURE_NAMES = ["标准高位", "长架", "低姿抱架", "低手位"]
N_POSTURES = 4
SWITCH_TICKS = 4
#            直拳射程 对手头部直拳 对手勾拳 对手上勾 高位格挡效果 步速 体能回复 下潜窗口 闪躲后仰窗口 刺拳前手勾提速 反应提速
POSTURE_FX = np.array([
    [1.00,   1.00,      1.00,   1.00,   1.0,       1.00, 1.00,   0,       0,          0,           0],
    [1.05,   0.85,      1.10,   1.00,   0.9,       1.00, 1.00,   0,       0,          0,           0],
    [0.95,   0.80,      1.00,   1.25,   1.0,       0.92, 0.95,   1,       0,          0,           0],
    [1.00,   1.10,      1.10,   1.10,   0.5,       1.00, 1.10,   0,       1,          1,           1],
])
FX_RANGE, FX_VS_STRAIGHT, FX_VS_HOOK, FX_VS_UPPER, FX_GUARD, FX_WALK, FX_REGEN, FX_DUCK, FX_SLIP, FX_QUICK, FX_REACT = range(11)
# 正反架对位（一正一反）：后手直拳顺着中线进来更远更重，前手勾拳更容易打到，刺拳和对方前手互相干扰
OPEN_CROSS_RANGE, OPEN_CROSS_DMG, OPEN_LHOOK_DMG, OPEN_JAB_DMG = 1.05, 1.10, 1.10, 0.85

# 状态（当天手感）：每场开打前定下，-1~1，0 是正常
FORM_SD = 0.3

# 性格：胆量、沉着、斗志、冲动（0~1，0.5 普通）。改变心态对事件的反应；冲动的人被反击、被骗会上头
TILT_TICKS = 60
TILT_POWER, TILT_GLANCE = 1.10, 1.5
TELEGRAPH_SKILL = 0.35      # 熟练度低于这个的拳会"露馅"：对手早一帧看到起手

# 心智：观察力、拳商（0~1，0.5 普通）
#   观察力：读招学得快、对对手体能心态看得准、看得出对手露馅、能看穿一部分虚晃
#   拳商：出昏招少（压力下更明显），训练时学得快
TELL_OBS = 0.4              # 观察力到这个才看得出对手露馅
LAPSE_MAX = 0.12            # 拳商 0 时每帧出昏招的概率；心态差时翻倍

# 战术计划：限制一部分选择，不碰奖励。被限制的动作每帧只有 25% 的时候可选
PLAN_NAMES = ["自由发挥", "外围控距", "压迫打身", "防守反击", "重拳抢攻", "稳拿分数"]
N_PLANS = len(PLAN_NAMES)
PLAN_AVAIL = 0.25
PLAN_MASK = np.ones((N_PLANS, N_ACTIONS), bool)
for _p, _acts in enumerate([
        [],
        [GUARD_FWD, CLINCH, LHOOK_H, LHOOK_B, RHOOK_H, RHOOK_B, LUPPER_H, LUPPER_B, RUPPER_H, RUPPER_B],
        [BACK, PULL, JAB_H, CROSS_H, FEINT_JAB, FEINT_CROSS],
        [FWD, GUARD_FWD, FEINT_JAB, FEINT_CROSS, CLINCH, RHOOK_B, RUPPER_B],
        [JAB_H, JAB_B, FEINT_JAB, FEINT_CROSS, PULL, BACK, CLINCH],
        [RHOOK_H, RHOOK_B, RUPPER_H, RUPPER_B, LUPPER_H, LUPPER_B, GUARD_FWD]]):
    PLAN_MASK[_p, _acts] = False

# 熟练度：每个角色对每个技术有一个 0~1 的熟练度（0.5 = 普通）。影响这个角色用这个技术的性能
TECH_NAMES = ACTION_NAMES[PUNCH0:PUNCH0 + 12] + ["高位格挡", "护体", "拍挡", "侧闪", "下潜", "后仰", "缠抱", "轴转", "虚晃", "步法", "换架"]
N_TECH = len(TECH_NAMES)
T_GUARD_HIGH, T_GUARD_BODY, T_PARRY, T_SLIP, T_DUCK, T_PULL, T_CLINCH, T_PIVOT, T_FEINT, T_FOOT, T_SWITCH = range(12, 23)
TECH_OF_DEF = {M_PARRY: T_PARRY, M_SLIP: T_SLIP, M_DUCK: T_DUCK, M_PULL: T_PULL}

OBS_BASE = 117 + 2 * N_TECH
OBS_DIM = OBS_BASE + 29
HIST = 8
T_MAX = N_ROUNDS * ROUND_TICKS


def _ticks(base, speed):
    return np.maximum(1, np.floor(base / speed + 0.5)).astype(np.int64)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ray_to_rope(pos, ang):
    """从 pos 沿 ang 方向到围绳（身体中心可达边界）的距离。"""
    ux, uy = np.cos(ang), np.sin(ang)
    lo, hi = BODY_R, RING - BODY_R
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(ux > 1e-9, (hi - pos[..., 0]) / ux, np.where(ux < -1e-9, (lo - pos[..., 0]) / ux, np.inf))
        ty = np.where(uy > 1e-9, (hi - pos[..., 1]) / uy, np.where(uy < -1e-9, (lo - pos[..., 1]) / uy, np.inf))
    return np.maximum(0.0, np.minimum(tx, ty))


def zone_probs(typ, def_posture, def_tired, skill, glance):
    """一记干净打中头部的拳落在 下颌 / 头侧 / 额头面门 的概率，形状 (..., 3)。"""
    base = ZONE_BASE[typ]
    jaw = base[..., 0] * JAW_POSTURE[def_posture] * np.where(def_tired, JAW_TIRED, 1.0) * (1.0 + JAW_SKILL * (skill - 0.5))
    jaw = np.where(glance, 0.0, np.clip(jaw, 0.0, 0.95))          # 射程边缘擦到的打不到下颌
    side = base[..., 1] * (1.0 - jaw) / (base[..., 1] + base[..., 2])
    return np.stack([jaw, side, 1.0 - jaw - side], -1)


def impact_of(dealt, typ, zone, unseen):
    """冲击：伤害（已经按天生下巴除过）× 落点系数 × 拳种系数，没看见的拳再加一点。"""
    return dealt * ZONE_F[zone] * ROT_F[typ] * np.where(unseen, UNSEEN_F, 1.0)


def kd_threshold(stun, hp):
    """抗击打线：最近挨的打越多（眩晕）、本场累计伤得越重（体力），线越低。"""
    return KD_T0 * (1.0 - 0.5 * np.clip(stun / STUN_KD, 0, 1)) * (0.6 + 0.4 * np.clip(hp / HP_MAX, 0, 1))


def kd_prob(impact, thr):
    r = impact / thr
    return np.where(r < KD_FLOOR, 0.0, 1.0 / (1.0 + np.exp(-(r - 1.0) / KD_SLOPE)))


def getup_prob(kd_total, hp, severity):
    """读秒能不能站起来：这一下越过抗击打线多少（severity 0~1）、本场累计伤害、这是第几次倒地。"""
    return np.clip(0.95 - 0.2 * (kd_total - 1) - 0.5 * (1 - np.clip(hp / HP_MAX, 0, 1)) - 0.4 * severity, 0.05, 0.95)


_G, _M1, _M2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(z):
    z = z + _G
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def luck(seed, tick, salt):
    """可重放的运气 [0,1)，形状 (n, 2)：只由这场的种子、第几帧、哪一方、用途决定，回放时一模一样。"""
    z = _mix(_mix(np.asarray(seed).astype(np.uint64)) ^ np.asarray(tick).astype(np.uint64))
    z = _mix(z[:, None] ^ (np.arange(2, dtype=np.uint64)[None, :] * np.uint64(1 << 20) + np.uint64(salt)))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def skill_vector(skills):
    """{技术名: 熟练度} → 数组，没写的按 0.5。"""
    v = np.full(N_TECH, 0.5)
    for k, val in (skills or {}).items():
        v[TECH_NAMES.index(k)] = val
    return v


def lapse(actions, mask, sim, rng):
    """拳商：每帧有一定概率出昏招（换成一个随机的可选动作），心态差时翻倍。
    放在选动作之后、进模拟器之前，所以记录下来的就是实际执行的动作，回放不受影响。"""
    eps = LAPSE_MAX * (1 - sim.mind[None, :, 1]) * (1 + (sim.comp < COMP_SLOW1))
    hit = rng.random(actions.shape) < eps
    if not hit.any():
        return actions, hit
    score = rng.random(actions.shape + (N_ACTIONS,))
    if mask is not None:
        score = np.where(mask, score, -1)
    return np.where(hit, score.argmax(-1), actions), hit


class Sim:
    def __init__(self, n, bodies, seed=0, auto_reset=True, n_rounds=N_ROUNDS, skills=None, form_sd=0.0):
        self.n, self.bodies, self.auto_reset, self.n_rounds = n, bodies, auto_reset, n_rounds
        self.t_max = n_rounds * ROUND_TICKS
        self.rng = np.random.default_rng(seed)
        b0, b1 = bodies
        pick = lambda k: np.array([getattr(b0, k), getattr(b1, k)], dtype=np.float64)
        self.reach, self.mass, self.power = pick("reach"), pick("mass"), pick("power")
        self.chin, self.cardio, self.foot = pick("chin"), pick("cardio"), pick("foot_speed")
        hand = pick("hand_speed")
        self.react = np.array([b0.react_ticks, b1.react_ticks])
        self.stance0 = np.array([STANCE_SIGN[getattr(b0, "stance", None) or "orthodox"],
                                 STANCE_SIGN[getattr(b1, "stance", None) or "orthodox"]])
        import schools as SC                    # 流派、性格的数据
        self.temper = np.stack([SC.temperament(b) for b in bodies])        # [选手, 胆量/沉着/斗志/冲动]
        self.mind = np.stack([SC.mind(b) for b in bodies])                  # [选手, 观察力/拳商]
        self.tend_alpha = TEND_ALPHA * (0.5 + self.mind[:, 0])
        self.posture0 = np.array([SC.posture(b) for b in bodies])
        self.comp_base = 60.0 + 40.0 * self.temper[:, 1]                    # 沉着决定心态的基准
        self.comp_drift = COMP_DRIFT * (0.5 + self.temper[:, 1])
        self.form_mean = np.zeros(2)
        self.form_sd = form_sd                  # 当天状态的波动；训练和正式比赛用 FORM_SD，测试用 0
        self.plan_fixed = None                  # None = 每场随机（训练用）；否则 (2,) 固定计划
        self.tend_prior = [None, None]          # 看过录像：开场时读招记忆里已经装着对手的习惯（训练场"看录像"）
        self.stam_max = STAM_BASE * self.cardio
        self.regen = STAM_REGEN * self.cardio * DT
        self.walk_base = WALK * self.foot * np.sqrt(75.0 / self.mass) * DT
        self.turn_base = TURN_RATE * self.foot * DT
        self.hand = hand
        w = REACH_W[TYPE][:, None]
        self.rng_tab = BODY_R + SHOULDER + RNG[:, None] * (w * self.reach[None, :] + (1 - w) * 0.75)
        self.cramp = BODY_R + SHOULDER + CRAMP_FRAC * self.reach
        if skills is None:
            skills = np.stack([SC.talent(b) for b in bodies])
        self.set_skills(skills)

        z = lambda *s, dt=np.float64: np.zeros((n,) + s, dt)
        self.pos, self.th = z(2, 2), z(2)
        self.hp, self.stun, self.body, self.stam, self.cap = z(2), z(2), z(2), z(2), z(2)
        self.phase, self.move, self.timer, self.elapsed = z(2, dt=np.int64), z(2, dt=np.int64), z(2, dt=np.int64), z(2, dt=np.int64)
        self.feint, self.missed = z(2, dt=bool), z(2, dt=bool)
        self.guard = z(2, dt=np.int64)
        self.kd_total, self.kd_round, self.unanswered, self.holds = z(2, dt=np.int64), z(2, dt=np.int64), z(2, dt=np.int64), z(2, dt=np.int64)
        self.round, self.t_round = z(dt=np.int64), z(dt=np.int64)
        self.tally = z(2, 4)            # 本回合：刺拳分、重拳分、被挡拳数、压迫帧
        self.judge = z(3, 2)            # 三位裁判的累计分
        self.over = z(dt=bool)
        self.comp = z(2)
        self.tend_punch, self.tend_head, self.tend_def, self.tend_feint = z(2, 4), z(2), z(2, 7), z(2)
        self.act_ring = np.full((HIST, n, 2), -1, np.int64)
        self.posture, self.stance, self.form, self.tilt, self.plan = z(2, dt=np.int64), z(2), z(2), z(2, dt=np.int64), z(2, dt=np.int64)
        self.init = z(6)                # 初始距离、y 偏移、开场朝向扰动、双方当天状态、这场的运气种子
        self.luck_seed, self.tick = z(dt=np.int64), z(dt=np.int64)
        self.su_cur, self.kd_sev = z(2, dt=np.int64), z(2)
        self.hist = np.zeros((HIST, n, 2, 17))
        self.ptr = 0
        self.reset(np.ones(n, bool))

    def set_skills(self, skills):
        """熟练度（2, N_TECH）→ 各项性能表。0.5 时所有倍率都是 1。"""
        sk = np.clip(np.asarray(skills, dtype=np.float64), 0.0, 1.0)
        self.skills = sk
        pk = sk[:, :12].T                                              # [拳, 选手]
        speed = 1.2 - 0.4 * pk                                         # 熟练：起手、收招更快
        self.su_tab = np.maximum(MIN_SU[:, None], np.floor(SU[:, None] / self.hand[None, :] * speed + 0.5)).astype(np.int64)
        self.rec_tab = np.maximum(1, np.floor(REC[:, None] / self.hand[None, :] * speed + 0.5)).astype(np.int64)
        self.dmg_mult = 0.8 + 0.4 * pk                                 # 更重
        self.cost_tab = COST[:, None] * (1.2 - 0.4 * pk)               # 更省体能
        self.glance_tab = GLANCE_FRAC * (1.5 - pk)                     # 更准，擦到的区间更窄
        self.guard_scale = 0.8 + 0.4 * sk[:, [T_GUARD_HIGH, T_GUARD_BODY]]   # [选手, 高位/护体]
        self.def_active = np.tile(DEF_ACTIVE, (2, 1))
        self.def_rec = np.tile(DEF_REC, (2, 1))
        self.def_cost = np.tile(DEF_COST, (2, 1))
        for m, t in TECH_OF_DEF.items():
            d = sk[:, t] - 0.5
            self.def_active[:, m] = np.maximum(2, DEF_ACTIVE[m] + np.round(d * 4)).astype(int)   # 时机窗口更宽
            self.def_rec[:, m] = np.maximum(2, DEF_REC[m] - np.round(d * 2)).astype(int)
            self.def_cost[:, m] = DEF_COST[m] * (1.2 - 0.4 * sk[:, t])
        self.clinch_reach = CLINCH_REACH + 0.2 * (sk[:, T_CLINCH] - 0.5)
        self.pivot_step = PIVOT_STEP * self.foot * (0.8 + 0.4 * sk[:, T_PIVOT])
        self.feint_cost = 1.2 - 0.4 * sk[:, T_FEINT]
        self.switch_ticks = np.maximum(2, np.round(SWITCH_TICKS * (1.2 - 0.4 * sk[:, T_SWITCH]))).astype(int)
        self.walk = self.walk_base * (0.9 + 0.2 * sk[:, T_FOOT])
        self.turn = self.turn_base * (0.9 + 0.2 * sk[:, T_FOOT])

    # ---------------------------------------------------------------- 开局
    def reset(self, mask, init=None):
        k = int(mask.sum())
        if k == 0:
            return
        if init is None:
            form = np.clip(self.form_mean[None, :] + self.rng.normal(0, 1, (k, 2)) * self.form_sd, -1, 1)
            init = np.concatenate([np.stack([2.0 + self.rng.uniform(-0.3, 0.3, k), self.rng.uniform(-0.4, 0.4, k),
                                             self.rng.uniform(-0.15, 0.15, k)], axis=1), form,
                                   self.rng.integers(0, 2 ** 31, k)[:, None].astype(np.float64)], axis=1)
        init = np.asarray(init, dtype=np.float64)
        if init.shape[1] < 6:                   # 旧格式：没有当天状态、没有运气种子
            init = np.concatenate([init, np.zeros((len(init), 6 - init.shape[1]))], axis=1)
        self.init[mask] = init
        self.luck_seed[mask] = init[:, 5].astype(np.int64)
        self.tick[mask] = 0
        self.su_cur[mask], self.kd_sev[mask] = 0, 0.0
        gap, yo, tw = init[:, 0], init[:, 1], init[:, 2]
        self.form[mask] = init[:, 3:5]
        self.posture[mask] = self.posture0
        self.stance[mask] = self.stance0
        self.tilt[mask] = 0
        if self.plan_fixed is None:
            self.plan[mask] = self.rng.choice(N_PLANS, size=(k, 2), p=[0.3] + [0.7 / (N_PLANS - 1)] * (N_PLANS - 1))
        else:
            self.plan[mask] = self.plan_fixed
        c = RING / 2
        self.pos[mask, 0] = np.stack([c - gap / 2, c + yo], 1)
        self.pos[mask, 1] = np.stack([c + gap / 2, c + yo], 1)
        self.th[mask] = np.stack([tw, np.pi - tw], 1)
        self.hp[mask], self.stun[mask], self.body[mask] = HP_MAX, 0.0, 0.0
        self.stam[mask] = self.stam_max
        self.cap[mask] = self.stam_max
        for a in (self.phase, self.timer, self.elapsed, self.guard, self.kd_total, self.kd_round, self.unanswered, self.holds):
            a[mask] = 0
        self.move[mask] = M_NONE
        self.feint[mask] = False
        self.missed[mask] = False
        self.round[mask], self.t_round[mask] = 0, 0
        self.tally[mask], self.judge[mask] = 0.0, 0.0
        self.over[mask] = False
        self.comp[mask] = np.clip(self.comp_base[None, :] + 5.0 * self.form[mask], 0, 100)
        self.tend_punch[mask], self.tend_head[mask], self.tend_feint[mask] = 0.25, 0.5, 0.0
        self.tend_def[mask] = 1.0 / 7
        for i, pr in enumerate(self.tend_prior):
            if pr is not None:
                self.tend_punch[mask, i] = pr["punch"]
                self.tend_head[mask, i] = pr["head"]
                self.tend_def[mask, i] = pr["def"]
                self.tend_feint[mask, i] = pr["feint"]
        self.act_ring[:, mask] = -1
        snap = self._snapshot()
        self.hist[:, mask] = snap[mask][None]

    def desync(self, rng):
        """训练开头把各场的进度错开，避免所有场同时开打同时结束。"""
        self.round[:] = rng.integers(0, self.n_rounds, self.n)
        self.t_round[:] = rng.integers(0, ROUND_TICKS, self.n)

    def _snapshot(self):
        return np.stack([self.pos[..., 0], self.pos[..., 1], self.th, self.hp, self.stun, self.body,
                         self.stam / self.stam_max, self.phase, self.move, self.elapsed, self.guard,
                         self.kd_total, self.comp, self.posture, self.stance, self.tilt > 0,
                         (self.phase == STARTUP) & self.feint], axis=-1)

    def react_now(self):
        """当前的反应延迟：心态差会反应变慢；低手位看得清；对手的拳露馅（熟练度低、上头）就早一帧看到。"""
        tele = self.telegraph()
        r = (self.react[None, :] + (self.comp < COMP_SLOW1) + (self.comp < COMP_SLOW2)
             + (self.stam < TIRED * self.stam_max[None, :])
             - POSTURE_FX[self.posture, FX_REACT].astype(int) - tele[:, ::-1] * (self.mind[None, :, 0] >= TELL_OBS))
        return np.maximum(1, r)

    def telegraph(self):
        mv = np.minimum(self.move, 11)
        sk = np.take_along_axis(np.broadcast_to(self.skills[None], (self.n, 2, N_TECH)), mv[..., None], axis=2)[..., 0]
        return ((self.phase == STARTUP) & (self.move < 12) & ((sk < TELEGRAPH_SKILL) | (self.tilt > 0))).astype(int)

    def plan_mask(self):
        """战术计划允许的动作（n, 2, N_ACTIONS）。被限制的动作每帧随机放行 25%。"""
        allow = PLAN_MASK[self.plan] | (self.rng.random((self.n, 2, N_ACTIONS)) < PLAN_AVAIL)
        allow[..., STAY] = True
        return allow

    # ---------------------------------------------------------------- 工具
    def _geom(self, pos, th):
        d = pos[:, 1] - pos[:, 0]
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        b01 = np.arctan2(d[:, 1], d[:, 0])
        bear = np.stack([b01, b01 + np.pi], 1)
        return d / dist[:, None], dist, wrap(bear - th)

    def _separate(self, pos, envs, gap):
        """把 envs 里的两人沿连线拉开到 gap（围绳挡住就整体平移）。"""
        if not envs.any():
            return pos
        u, _, _ = self._geom(pos, np.zeros((self.n, 2)))
        mid = pos.mean(axis=1)
        p0 = mid - u * gap / 2
        p1 = mid + u * gap / 2
        lo, hi = BODY_R, RING - BODY_R
        shift = np.maximum(0, lo - np.minimum(p0, p1)) - np.maximum(0, np.maximum(p0, p1) - hi)
        p0, p1 = p0 + shift, p1 + shift
        pos = pos.copy()
        pos[envs, 0], pos[envs, 1] = p0[envs], p1[envs]
        return np.clip(pos, lo, hi)

    def _collide(self, pos):
        u, dist, _ = self._geom(pos, np.zeros((self.n, 2)))
        ov = np.maximum(0.0, MIN_GAP - dist)
        m0, m1 = self.mass
        pos = pos.copy()
        pos[:, 0] -= u * (ov * m1 / (m0 + m1))[:, None]
        pos[:, 1] += u * (ov * m0 / (m0 + m1))[:, None]
        pos = np.clip(pos, BODY_R, RING - BODY_R)
        u, dist, _ = self._geom(pos, np.zeros((self.n, 2)))
        ov = np.maximum(0.0, MIN_GAP - dist)
        # 一方被围绳卡住时，另一方承担剩下的
        stuck0 = np.any((pos[:, 0] <= BODY_R + 1e-9) | (pos[:, 0] >= RING - BODY_R - 1e-9), axis=1)
        pos[:, 1] += np.where(stuck0, 1.0, 0.5)[:, None] * u * ov[:, None]
        pos[:, 0] -= np.where(stuck0, 0.0, 0.5)[:, None] * u * ov[:, None]
        return np.clip(pos, BODY_R, RING - BODY_R)

    # ---------------------------------------------------------------- 一帧
    def step(self, actions):
        a = np.asarray(actions, dtype=np.int64)
        n = self.n
        side = np.arange(2)[None, :]
        ph, mv, tm = self.phase.copy(), self.move.copy(), self.timer.copy()
        feint, missed, guard_prev = self.feint.copy(), self.missed.copy(), self.guard.copy()
        el = self.elapsed + 1
        pos, th = self.pos.copy(), self.th.copy()
        hp, stun, body = self.hp.copy(), self.stun.copy(), self.body.copy()
        stam, cap = self.stam.copy(), self.cap.copy()
        tally = self.tally.copy()
        kd_total, kd_round = self.kd_total.copy(), self.kd_round.copy()
        unans, holds = self.unanswered.copy(), self.holds.copy()
        comp = self.comp.copy()
        tend_def = self.tend_def.copy()
        su_cur, kd_sev, tick = self.su_cur.copy(), self.kd_sev.copy(), self.tick
        react_def = self.react_now()[:, ::-1]        # [:, i] = i 的对手此刻的反应延迟（判"没看见的拳"用）
        ev_rew = np.zeros((n, 2))
        done = np.zeros(n, bool)
        winner = np.full(n, -1)
        method = np.zeros(n, np.int64)

        # 1) 计时
        busy = ph != FREE
        tm = np.where(busy, tm - 1, tm)
        exp = busy & (tm <= 0)
        fire = exp & (ph == STARTUP) & (mv < 12) & ~feint
        feint_end = exp & (ph == STARTUP) & feint
        clinch_try = exp & (ph == STARTUP) & (mv == M_CLINCH)
        def_end = exp & (ph == DEFEND)
        simple_end = exp & np.isin(ph, (RECOVERY, HITSTUN, BLOCKSTUN, MOVING))
        clinch_end = (exp & (ph == CLINCHED)).any(1)
        down_end = exp & (ph == DOWN)

        post_done = simple_end & (mv == M_POSTURE)
        stance_done = simple_end & (mv == M_STANCE)
        posture = np.where(post_done, (self.posture + 1) % N_POSTURES, self.posture)
        stance = np.where(stance_done, -self.stance, self.stance)
        tilt = np.maximum(0, self.tilt - 1)
        open_st = stance[:, 0] != stance[:, 1]
        free_now = feint_end | simple_end
        ph = np.where(free_now, FREE, ph)
        mv = np.where(free_now, M_NONE, mv)
        feint &= ~feint_end
        ph = np.where(def_end, RECOVERY, ph)
        tm = np.where(def_end, self.def_rec[side, mv], tm)
        el = np.where(free_now | def_end, 0, el)

        # 裁判分开缠抱
        cl_brk = clinch_end[:, None] & (ph == CLINCHED)
        ph = np.where(cl_brk, FREE, ph)
        mv = np.where(cl_brk, M_NONE, mv)
        el = np.where(cl_brk, 0, el)
        pos = self._separate(pos, clinch_end, 1.2)

        # 读秒结束：站起来或者 KO
        if down_end.any():
            # 站不站得起来看这一下多重、本场累计伤害、第几次倒地，带运气；不再看血量硬线
            gets_up = down_end & (luck(self.luck_seed, tick, 3) < getup_prob(kd_total, hp, kd_sev)) & (kd_round < 3)
            ko = down_end & ~gets_up
            ph = np.where(gets_up, FREE, ph)
            mv = np.where(gets_up, M_NONE, mv)
            el = np.where(gets_up, 0, el)
            stun = np.where(gets_up, 40.0, stun)
            body = np.where(gets_up, np.minimum(body, 50.0), body)
            ko_env = ko.any(1)
            winner = np.where(ko_env, np.where(ko[:, 0] & ko[:, 1], -1, np.where(ko[:, 0], 1, 0)), winner)
            method = np.where(ko_env, M_KO, method)
            done |= ko_env

        # 2) 出拳结算（双方同时，可以互换）
        u, dist, off = self._geom(pos, th)
        p = np.where(fire, mv, 0)
        typ, tgt = TYPE[p], TGT[p]
        cls = typ * 2 + tgt
        ang_ok = np.abs(off) <= np.where(typ == 1, ANG_HOOK, ANG_STRAIGHT)
        ph_t, mv_t, stun_t, off_t = ph[:, ::-1], mv[:, ::-1], stun[:, ::-1], off[:, ::-1]
        pull_t = (ph_t == DEFEND) & (mv_t == M_PULL)
        arm = self.rng_tab[p, side] - BODY_R - SHOULDER
        arm = arm * np.where(typ == 0, POSTURE_FX[posture, FX_RANGE], 1.0)
        arm = arm * np.where(open_st[:, None] & ((p == 2) | (p == 3)), OPEN_CROSS_RANGE, 1.0)
        reach_lim = BODY_R + SHOULDER + arm
        head_add = np.where((tgt == 0) & pull_t, PULL_DIST, 0.0)
        in_rng = dist[:, None] + head_add <= reach_lim + 1e-9
        glance = dist[:, None] + head_add > reach_lim - (self.glance_tab[p, side] * np.where(tilt > 0, TILT_GLANCE, 1.0)
                                                          * RNG[p] * self.reach[None, :])
        live = fire & ang_ok & (ph_t != DOWN) & (ph_t != CLINCHED) & (ph != CLINCHED)
        reach_ok = live & in_rng
        is_def = ph_t == DEFEND
        gtype = np.where((ph_t == FREE) | (ph_t == BLOCKSTUN), guard_prev[:, ::-1], 0)
        gm = GUARD_MULT[gtype, cls]
        gs = self.guard_scale[::-1]                                   # 防守方的格挡熟练度
        tired_t = (self.stam < TIRED * self.stam_max[None, :])[:, ::-1]
        gsc = np.where(gtype == 1, gs[None, :, 0] * POSTURE_FX[posture[:, ::-1], FX_GUARD], np.where(gtype == 2, gs[None, :, 1], 1.0))
        gsc = gsc * np.where(tired_t, TIRED_GUARD, 1.0)
        gm = np.where(gm < 1.0, np.clip(1.0 - (1.0 - gm) * gsc, 0.02, 1.0), gm)
        natural = (gtype == 0) & np.isin(ph_t, (FREE, MOVING)) & ~tired_t
        gm = np.where(natural, NATURAL_GUARD[posture[:, ::-1], cls], gm)
        m = np.where(is_def, EVADE_MULT[np.minimum(mv_t, N_MOVES - 1), cls], gm)
        eff = np.where(np.abs(off_t) > ANG_GUARD, 0.5, 1.0) * np.where(stun_t > STUN_HURT, 0.5, 1.0)
        m = np.where(m < 1.0, 1.0 - (1.0 - m) * eff, m)
        evaded = reach_ok & is_def & (m <= 1e-6)
        pulled = live & ~in_rng & pull_t & (dist[:, None] <= reach_lim + 1e-9)
        evaded |= pulled
        landed = reach_ok & ~evaded
        blocked = landed & ~is_def & (gtype > 0) & (m < 0.5)
        clean = landed & ~blocked
        cramp = np.where((typ == 0) & (dist[:, None] < self.cramp[None, :]), 0.5, 1.0)
        counter = clean & ((ph_t == STARTUP) | ((ph_t == RECOVERY) & missed[:, ::-1] & (mv_t < 12)))
        sf = np.clip(stam / self.stam_max, 0, 1)
        base = (DMG[p] * DMG_SCALE * self.power[None, :] * (0.5 + 0.5 * sf) * cramp * np.where(counter, COUNTER_DMG, 1.0)
                * np.where(glance, GLANCE_MULT, 1.0) * self.dmg_mult[p, side])
        # 防守方的架势（只影响头部拳）、正反架对位、当天状态、上头
        post_t = posture[:, ::-1]
        vs = np.where(tgt == 0, np.select([typ == 0, typ == 1, typ == 2],
                                          [POSTURE_FX[post_t, FX_VS_STRAIGHT], POSTURE_FX[post_t, FX_VS_HOOK],
                                           POSTURE_FX[post_t, FX_VS_UPPER]], 1.0), 1.0)
        op = open_st[:, None]
        openm = np.where(op & ((p == 2) | (p == 3)), OPEN_CROSS_DMG, np.where(op & ((p == 4) | (p == 5)), OPEN_LHOOK_DMG,
                         np.where(op & (p <= 1), OPEN_JAB_DMG, 1.0)))
        base = base * vs * openm * (1.0 + 0.08 * self.form) * np.where((tilt > 0) & (IS_POWER[p] == 1), TILT_POWER, 1.0)
        dealt = np.where(landed, base * m / self.chin[::-1][None, :], 0.0)   # [:, i] = i 打出的
        head = tgt == 0
        # 落点：干净打中头部才抽。冲击和挨打者挨这一下之前的抗击打线比，越过得多越可能倒
        head_clean = clean & head
        sk_p = self.skills[side, p]
        zp = zone_probs(typ, posture[:, ::-1], tired_t, sk_p, glance)
        u_zone = luck(self.luck_seed, tick, 1)
        zone = np.where(head_clean, (u_zone >= zp[..., 0]).astype(np.int64) + (u_zone >= zp[..., 0] + zp[..., 1]), -1)
        unseen = (su_cur <= react_def + 1) | (ph_t == STARTUP)
        impact = np.where(head_clean, impact_of(dealt, typ, np.maximum(zone, 0), unseen), 0.0)
        thr = kd_threshold(self.stun[:, ::-1], self.hp[:, ::-1])
        flash = head_clean & (luck(self.luck_seed, tick, 2) < kd_prob(impact, thr))          # [:, i] = i 把对手打倒
        flash_sev = np.clip(impact / thr - 1.0, 0.0, 1.0)
        hp_loss = np.where(head, dealt, 0.7 * dealt)
        stun_add = np.where(head & clean, dealt * STUNK[p] * 1.4 * np.where(counter, COUNTER_STUN, 1.0),
                            np.where(head & blocked, dealt * STUNK[p] * 0.5, 0.0))
        body_add = np.where(~head & landed, dealt * BODYK[p] * BODY_METER * np.where(blocked, 0.3, 1.0), 0.0)
        stam_loss = np.where(~head & landed, dealt * BODY_STAM, 0.0) + np.where(blocked, DMG[p] * DMG_SCALE * 0.5, 0.0)
        cap_loss = np.where(~head & clean, dealt * 0.4, 0.0)
        hp = np.maximum(0.0, hp - hp_loss[:, ::-1])          # 体力只是累计伤害，到 0 也不结束比赛
        stun = stun + stun_add[:, ::-1]
        body = body + body_add[:, ::-1]
        stam = stam - stam_loss[:, ::-1]
        cap = cap - cap_loss[:, ::-1]

        # 计分（裁判只看打中的）
        # 裁判看"有效"击打：累了的软拳、擦到的、打在手臂上的，分都打折
        pts = np.where(clean, SCORE[p] * (0.5 + 0.5 * sf) * np.where(glance, 0.5, 1.0) * np.minimum(m, 1.0), 0.0)
        tally[:, :, 0] += np.where(IS_POWER[p] == 0, pts, 0.0)
        tally[:, :, 1] += np.where(IS_POWER[p] == 1, pts, 0.0)
        tally[:, :, 2] += blocked
        score_rew = (pts - pts[:, ::-1]) * REWARD_PER_POINT

        # 读招：我出拳的那一刻，对手是怎么防的
        dcat = np.where(is_def, DEF_CAT[np.minimum(mv_t, N_MOVES - 1)], gtype)
        al = self.tend_alpha[None, :, None]
        tend_def = np.where(live[..., None], (1 - al) * tend_def + al * np.eye(7)[dcat], tend_def)

        # 心态：打中涨，挨打掉，防住涨，重拳打空掉
        power_head = clean & (tgt == 0) & (IS_POWER[p] == 1)
        land_gain = np.where(power_head, COMP_LAND["power_head"], np.where(clean & (tgt == 0), COMP_LAND["head"],
                             np.where(clean, COMP_LAND["body"], 0.0)))
        hit_loss = np.where(power_head, COMP_HIT["power_head"], np.where(clean & (tgt == 0), COMP_HIT["head"],
                            np.where(clean, COMP_HIT["body"], 0.0))) + np.where(counter, COMP_HIT["counter"], 0.0)
        defended = (blocked | evaded).astype(float) * COMP_DEFEND
        whiff_power = fire & ~landed & ~evaded & (IS_POWER[p] == 1)
        courage, drive, temper = self.temper[None, :, 0], self.temper[None, :, 2], self.temper[None, :, 3]
        comp = (comp + land_gain * (0.5 + drive) + hit_loss[:, ::-1] * (1.5 - courage) + defended[:, ::-1]
                + np.where(whiff_power, COMP_WHIFF_POWER, 0.0))
        # 冲动：被反击就上头
        tilt = np.where(counter[:, ::-1] & (temper > 0.5), np.maximum(tilt, (TILT_TICKS * (temper - 0.5) * 2).astype(int)), tilt)

        # 进攻方进收招；被躲开、被拍开的多收几帧
        def_code = np.where(pulled, M_PULL, np.where(evaded, mv_t, M_NONE))
        offbal = np.where(evaded, OFFBAL[np.minimum(def_code, N_MOVES - 1)], 0)
        ph = np.where(fire, RECOVERY, ph)
        tm = np.where(fire, self.rec_tab[p, side] + offbal, tm)
        el = np.where(fire, 0, el)
        whiff = fire & ~landed
        missed = np.where(fire, whiff, missed)
        stam = stam - np.where(whiff, 0.5 * self.cost_tab[p, side], 0.0)

        # 挨打的人进硬直，打断自己的动作
        got_clean, got_blocked, att_p = clean[:, ::-1], blocked[:, ::-1], p[:, ::-1]
        ph = np.where(got_blocked, BLOCKSTUN, ph)
        tm = np.where(got_blocked, BLOCKSTUN_T[att_p], tm)
        ph = np.where(got_clean, HITSTUN, ph)
        tm = np.where(got_clean, HITSTUN_T[att_p], tm)
        hit_any = got_clean | got_blocked
        mv = np.where(hit_any, M_NONE, mv)
        feint &= ~hit_any
        el = np.where(hit_any, 0, el)

        # 击退；勾拳把对手的身体打歪
        kb = KB[p] * self.power[None, :] * (75.0 / self.mass[::-1][None, :]) * np.where(clean, 1.0, np.where(blocked, 0.25, 0.0))
        pos[:, 1] += u * kb[:, 0:1]
        pos[:, 0] -= u * kb[:, 1:2]
        twist = np.where(clean & (typ == 1), TWIST[p] * np.where(HAND[p] == 0, 1.0, -1.0) * stance, 0.0)
        th = th + twist[:, ::-1]

        # 3) 缠抱成功与否
        cl_ok = clinch_try & (dist[:, None] <= self.clinch_reach[None, :] + 0.05) & (ph[:, ::-1] != DOWN) & ~got_clean
        cl_env = cl_ok.any(1)
        holds += cl_ok
        cl_fail = clinch_try & ~cl_ok & ~got_clean
        ph = np.where(cl_fail, RECOVERY, ph)
        tm = np.where(cl_fail, 4, tm)
        both = cl_env[:, None] & np.ones((1, 2), bool)
        ph = np.where(both, CLINCHED, ph)
        tm = np.where(both, CLINCH_TICKS, tm)
        mv = np.where(both, M_NONE, mv)
        feint &= ~both
        el = np.where(both, 0, el)

        # 4) 击倒、TKO
        kd = ((stun >= STUN_KD) | (body >= BODY_KD) | flash[:, ::-1]) & (ph != DOWN) & ~done[:, None]
        if kd.any():
            kd_sev = np.where(kd, np.where(flash[:, ::-1], flash_sev[:, ::-1], 0.5), kd_sev)    # 累积到线的倒地按中等算
            ph = np.where(kd, DOWN, ph)
            tm = np.where(kd, DOWN_TICKS, tm)
            mv = np.where(kd, M_NONE, mv)
            feint &= ~kd
            el = np.where(kd, 0, el)
            kd_total += kd
            kd_round += kd
            ev_rew += (kd[:, ::-1].astype(float) - kd) * REWARD_KD
            comp = comp + kd * COMP_KD_DOWN * (1.5 - courage) + kd[:, ::-1] * COMP_KD_SCORE * (0.5 + drive)
            # 对手回中立角：解除对手的动作，并拉开距离
            opp = kd[:, ::-1] & ~kd
            ph = np.where(opp, FREE, ph)
            mv = np.where(opp, M_NONE, mv)
            pos = self._separate(pos, kd.any(1), NEUTRAL_GAP)
        trying = (is_def | (gtype > 0))[:, ::-1]                      # 挨打时还在举架或闪躲：不算"不设防"
        unans = np.where(got_clean & ~trying & (TGT[att_p] == 0) & (stun > STUN_HURT), unans + 1, unans)
        defended_now = got_blocked | evaded[:, ::-1] | (got_clean & trying)
        unans = np.where(clean | defended_now | (stun < STUN_HURT), 0, unans)
        # TKO 只由裁判判：受伤后连续挨打不还手、一回合倒三次。体力归零不再结束比赛（体力只是资源池）
        stop = ((unans >= UNANSWERED_STOP) | (kd_round >= 3)) & ~done[:, None]
        # 只用于统计：TKO 是怎么判的。2 受伤后连续挨打不还手 3 一回合倒三次（1 血条归零，v9 起没有了）
        tko_cause = np.where(stop.any(1), np.where((kd_round >= 3).any(1), 3, 2), 0)
        if stop.any():
            st_env = stop.any(1)
            winner = np.where(st_env, np.where(stop[:, 0] & stop[:, 1], -1, np.where(stop[:, 0], 1, 0)), winner)
            method = np.where(st_env, M_TKO, method)
            done |= st_env

        # 5) 执行动作
        free = ph == FREE
        pa = np.clip(a - PUNCH0, 0, 11)
        is_pa = (a >= PUNCH0) & (a <= RUPPER_B)
        mvp = np.minimum(mv, 11)
        can_combo = (ph == RECOVERY) & (mv < 12) & (tm * 2 <= self.rec_tab[mvp, side]) & (HAND[pa] != HAND[mvp])
        acted = free | (can_combo & is_pa)
        start_p = (free | can_combo) & is_pa & (stam >= self.cost_tab[pa, side])
        quick = np.isin(pa, (0, 1)) * POSTURE_FX[posture, FX_QUICK].astype(int)      # 低手位：刺拳从下面弹出去更快
        su = np.maximum(MIN_SU[pa], self.su_tab[pa, side] + (stam < 0.3 * self.stam_max) + (comp < COMP_FREEZE) - quick)
        ph = np.where(start_p, STARTUP, ph)
        mv = np.where(start_p, pa, mv)
        tm = np.where(start_p, su, tm)
        su_cur = np.where(start_p, su, su_cur)
        feint &= ~start_p
        stam = stam - np.where(start_p, self.cost_tab[pa, side], 0.0)
        el = np.where(start_p, 0, el)
        started = np.where(start_p, pa + 1, 0)

        do_feint = free & ((a == FEINT_JAB) | (a == FEINT_CROSS)) & (stam >= self.feint_cost[None, :])
        ph = np.where(do_feint, STARTUP, ph)
        mv = np.where(do_feint, np.where(a == FEINT_JAB, 0, 2), mv)
        tm = np.where(do_feint, 2, tm)
        feint |= do_feint
        stam = stam - np.where(do_feint, self.feint_cost[None, :], 0.0)
        el = np.where(do_feint, 0, el)

        gnew = np.where(free & ((a == GUARD_HIGH) | (a == GUARD_FWD)), 1, np.where(free & (a == GUARD_BODY), 2, 0))
        guard = np.where(ph == BLOCKSTUN, guard_prev, gnew)

        dcode = np.select([a == PARRY, a == SLIP, a == DUCK, a == PULL], [M_PARRY, M_SLIP, M_DUCK, M_PULL], M_NONE)
        back_room = ray_to_rope(pos, th + np.pi)
        do_def = free & (dcode != M_NONE) & (stam >= self.def_cost[side, dcode]) & ~((a == PULL) & (back_room < PULL_DIST))
        faked = do_def & (self.phase[:, ::-1] == STARTUP) & self.feint[:, ::-1]      # 被虚晃骗出了防守动作
        comp = comp + np.where(faked, COMP_FAKED * (1.5 - courage), 0.0)
        tilt = np.where(faked & (temper > 0.5), np.maximum(tilt, (TILT_TICKS * (temper - 0.5) * 2).astype(int)), tilt)
        ph = np.where(do_def, DEFEND, ph)
        mv = np.where(do_def, dcode, mv)
        win_add = np.where(dcode == M_DUCK, POSTURE_FX[posture, FX_DUCK],
                           np.where((dcode == M_SLIP) | (dcode == M_PULL), POSTURE_FX[posture, FX_SLIP], 0)).astype(int)
        tm = np.where(do_def, self.def_active[side, dcode] + win_add, tm)
        stam = stam - np.where(do_def, self.def_cost[side, dcode], 0.0)
        el = np.where(do_def, 0, el)

        _, dist2, _ = self._geom(pos, th)
        do_cl = free & (a == CLINCH) & (dist2[:, None] <= self.clinch_reach[None, :]) & (ph[:, ::-1] != DOWN)
        ph = np.where(do_cl, STARTUP, ph)
        mv = np.where(do_cl, M_CLINCH, mv)
        tm = np.where(do_cl, 3, tm)
        el = np.where(do_cl, 0, el)

        do_piv = free & ((a == PIVOT_L) | (a == PIVOT_R)) & (stam >= 1.5)
        ph = np.where(do_piv, MOVING, ph)
        mv = np.where(do_piv, np.where(a == PIVOT_L, M_PIVL, M_PIVR), mv)
        tm = np.where(do_piv, PIVOT_TICKS, tm)
        stam = stam - np.where(do_piv, 1.5, 0.0)
        el = np.where(do_piv, 0, el)

        do_sw = free & ((a == CHANGE_POSTURE) | (a == SWITCH_STANCE))
        ph = np.where(do_sw, MOVING, ph)
        mv = np.where(do_sw, np.where(a == CHANGE_POSTURE, M_POSTURE, M_STANCE), mv)
        tm = np.where(do_sw, self.switch_ticks[None, :], tm)
        el = np.where(do_sw, 0, el)

        brk = (ph == CLINCHED) & (a == BACK) & (el >= CLINCH_MIN)
        brk_env = brk.any(1)
        bb = brk_env[:, None] & (ph == CLINCHED)
        ph = np.where(bb, FREE, ph)
        mv = np.where(bb, M_NONE, mv)
        el = np.where(bb, 0, el)
        pos = self._separate(pos, brk_env, 1.1)
        acted = acted | (self.phase == CLINCHED)

        # 6) 移动
        fwd = np.stack([np.cos(th), np.sin(th)], -1)
        lft = np.stack([-np.sin(th), np.cos(th)], -1)
        walking = free & np.isin(a, (FWD, BACK, LEFT, RIGHT, GUARD_FWD))
        tired = 0.6 + 0.4 * np.clip(stam / self.stam_max, 0, 1)
        hurt = np.where(stun > STUN_HURT, 0.7, 1.0)
        spd = (self.walk[None, :] * tired * hurt * POSTURE_FX[posture, FX_WALK])[..., None]
        cf = np.select([a == FWD, a == BACK, a == GUARD_FWD], [1.0, -BACK_MULT, GUARD_WALK], 0.0)[..., None]
        cl = np.select([a == LEFT, a == RIGHT], [SIDE_MULT, -SIDE_MULT], 0.0)[..., None]
        pos = pos + np.where(walking[..., None], (fwd * cf + lft * cl) * spd, 0.0)
        stam = stam - walking * WALK_COST
        piv = (ph == MOVING) & ((mv == M_PIVL) | (mv == M_PIVR))
        pdir = np.where(mv == M_PIVL, 1.0, -1.0)[..., None]
        pos = pos + np.where(piv[..., None], lft * pdir * self.pivot_step[None, :, None], 0.0)
        in_cl = (ph[:, 0] == CLINCHED) & (ph[:, 1] == CLINCHED)
        pos = self._separate(pos, in_cl, MIN_GAP + 0.02)
        pos = self._collide(pos)
        pressing = walking & np.isin(a, (FWD, GUARD_FWD))
        _, dist3, off3 = self._geom(pos, th)
        tally[:, :, 3] += pressing & (dist3[:, None] < 2.0)

        # 7) 转身面向对手
        turnable = np.isin(ph, (FREE, MOVING, DEFEND, CLINCHED))
        rate = self.turn[None, :] * np.where(turnable, 1.0, BUSY_TURN) * (ph != DOWN)
        th = wrap(th + np.clip(off3, -rate, rate))

        # 8) 资源池恢复
        pinned = (ray_to_rope(pos, th + np.pi) < 0.4) & (dist3[:, None] < 1.2)
        retreat = walking & (a == BACK)
        comp = (comp + pinned * COMP_PINNED * (1.5 - courage) + pinned[:, ::-1] * COMP_PINNING * (0.5 + drive)
                - retreat * 0.04 * np.maximum(0.0, drive - 0.3))
        target = np.clip(self.comp_base[None, :] + 5.0 * self.form, 0, 100)
        comp = np.clip(comp + (target - comp) * self.comp_drift[None, :], 0.0, 100.0)
        resting = np.where(ph == STARTUP, 0.0, np.where((guard > 0) | (ph == DEFEND), 0.5, 1.0))
        tense = np.where(comp < COMP_TENSE, 0.8, 1.0)
        regen = self.regen[None, :] * POSTURE_FX[posture, FX_REGEN] * (1.0 + 0.1 * self.form)
        stam = np.clip(stam + regen * resting * tense * np.where(stun > STUN_HURT, 0.5, 1.0), 0.0, cap)
        cap = np.clip(cap, 0.3 * self.stam_max[None, :], self.stam_max[None, :])
        stun = np.maximum(0.0, stun - STUN_DECAY * np.where(in_cl[:, None], 2.0, 1.0))
        body = np.maximum(0.0, body - BODY_DECAY)

        # 9) 回合钟
        t_round = self.t_round + 1
        rnd = self.round.copy()
        judge = self.judge.copy()
        r_end = (t_round >= ROUND_TICKS) & ~done & ~self.over
        if r_end.any():
            s = np.einsum("jk,nik->nji", JUDGES, tally)            # (n, 3 裁判, 2 人)
            diff = s[:, :, 0] - s[:, :, 1]
            r0 = np.where(diff > 0.5, 10.0, np.where(diff < -0.5, 9.0, 10.0))
            r1 = np.where(diff < -0.5, 10.0, np.where(diff > 0.5, 9.0, 10.0))
            rs = np.stack([r0, r1], -1) - kd_round[:, None, :] - (holds > HOLDS_WARN)[:, None, :]
            judge = np.where(r_end[:, None, None], judge + rs, judge)
            last = r_end & (rnd >= self.n_rounds - 1)
            votes = np.sign(judge[:, :, 0] - judge[:, :, 1])
            w0, w1 = (votes > 0).sum(1) >= 2, (votes < 0).sum(1) >= 2
            winner = np.where(last, np.where(w0, 0, np.where(w1, 1, -1)), winner)
            method = np.where(last, np.where(w0 | w1, M_DEC, M_DRAW), method)
            done |= last
            # 回合间休息
            rest = r_end & ~last
            rr = rest[:, None]
            stam = np.where(rr, stam + 0.5 * (cap - stam), stam)
            stun = np.where(rr, stun * 0.3, stun)
            body = np.where(rr, body * 0.7, body)
            hp = np.where(rr, np.minimum(HP_MAX, hp + 3.0), hp)
            comp = np.where(rr, comp + (np.clip(self.comp_base[None, :] + 5.0 * self.form, 0, 100) - comp) * 0.5, comp)
            if self.plan_fixed is None:         # 训练时，回合间有时换战术
                swap = rr & (self.rng.random((n, 2)) < 0.3)
                self.plan = np.where(swap, self.rng.integers(0, N_PLANS, (n, 2)), self.plan)
            for arr in (kd_round, holds, unans):
                arr[rest] = 0
            tally[r_end] = 0.0
            ph = np.where(rr, FREE, ph)
            mv = np.where(rr, M_NONE, mv)
            tm = np.where(rr, 0, tm)
            el = np.where(rr, 0, el)
            feint &= ~rr
            rnd = np.where(rest, rnd + 1, rnd)
            t_round = np.where(rest, 0, t_round)
            if rest.any():
                c = RING / 2
                pos[rest, 0] = np.stack([np.full(rest.sum(), c - 1.0), np.full(rest.sum(), c)], 1)
                pos[rest, 1] = np.stack([np.full(rest.sum(), c + 1.0), np.full(rest.sum(), c)], 1)
                th[rest] = np.array([0.0, np.pi])

        done &= ~self.over
        win_sign = np.where(winner == 0, 1.0, np.where(winner == 1, -1.0, 0.0)) * done
        reward = score_rew + ev_rew + np.stack([win_sign, -win_sign], 1)

        (self.pos, self.th, self.hp, self.stun, self.body, self.stam, self.cap) = (pos, th, hp, stun, body, stam, cap)
        (self.phase, self.move, self.timer, self.elapsed, self.feint, self.missed, self.guard) = (ph, mv, tm, el, feint, missed, guard)
        (self.kd_total, self.kd_round, self.unanswered, self.holds) = (kd_total, kd_round, unans, holds)
        self.tally, self.judge, self.round, self.t_round = tally, judge, rnd, t_round
        self.over = self.over | done
        self.comp, self.tend_def = comp, tend_def
        self.su_cur, self.kd_sev, self.tick = su_cur, kd_sev, self.tick + 1
        self.posture, self.stance, self.tilt = posture, stance, tilt

        outcome = np.where(clean, 1, np.where(blocked, 2, np.where(evaded, 3, np.where(fire, 4, 0))))
        info = {"acted": acted, "started": started, "fired": np.where(fire, p + 1, 0), "outcome": outcome,
                "evaded_by": np.where(evaded, def_code, -1)[:, ::-1], "counter": counter, "dmg": hp_loss,
                "points": pts, "kd": kd, "faked": faked, "clinch": cl_ok, "pivot": do_piv,
                "blocked_by": np.where(blocked, gtype, 0)[:, ::-1], "def_start": np.where(do_def, dcode, -1), "winner": winner, "method": method, "done": done,
                "ko": done & ((method == M_KO) | (method == M_TKO)), "gap": dist3, "round": rnd,
                "judges": judge.copy(), "tko_cause": tko_cause,
                # 只用于统计，口径对齐 CompuBox：打在拳套和手臂上的不算命中。被自然站姿削弱的按比例折算
                # （手本来在脸边，有一部分拳会打在手套上），射程边缘擦到的不算
                "landed_cb": np.where(clean & ~glance, np.minimum(m, 1.0), 0.0),
                "zone": zone, "impact": impact, "flash_kd": flash}
        self.ptr = (self.ptr + 1) % HIST
        self.hist[self.ptr] = self._snapshot()
        # 读招：对手出了什么拳。真拳在打出那一刻才确认是真的，虚晃在收回那一刻才露馅；再按反应延迟"看见"
        self.act_ring[self.ptr] = np.where(fire, p, np.where(feint_end, FEINT_CODE, -1))
        for i in (0, 1):
            code = self.act_ring[(self.ptr - self.react[i]) % HIST, :, 1 - i]
            punch = (code >= 0) & (code < 12)
            seen = code >= 0
            g = PUNCH_GROUP[np.clip(code, 0, 11)]
            ta = self.tend_alpha[i]
            al = ta * punch[:, None]
            self.tend_punch[:, i] = (1 - al) * self.tend_punch[:, i] + al * np.eye(4)[g]
            self.tend_head[:, i] += ta * punch * ((TGT[np.clip(code, 0, 11)] == 0) - self.tend_head[:, i])
            self.tend_feint[:, i] += ta * seen * ((code == FEINT_CODE) - self.tend_feint[:, i])
        if self.auto_reset and done.any():
            self.reset(done)
        return reward, done, info

    # ---------------------------------------------------------------- 观测
    def observe(self):
        n = self.n
        out = np.zeros((n, 2, OBS_DIM), np.float32)
        eyeP, eyeM, eyeG = np.eye(N_PHASES), np.eye(OBS_MOVES), np.eye(3)
        sf = self.stam / self.stam_max
        mvp = np.minimum(self.move, 11)
        can_combo = (self.phase == RECOVERY) & (self.move < 12) & (self.timer * 2 <= self.rec_tab[mvp, np.arange(2)[None, :]])
        pts = self.tally[:, :, 0] + self.tally[:, :, 1]
        react = self.react_now()
        ar = np.arange(n)
        for i in (0, 1):
            j = 1 - i
            old = self.hist[(self.ptr - react[:, i]) % HIST, ar, j]
            o = out[:, i]
            o[:, 0] = self.hp[:, i] / HP_MAX
            o[:, 1] = self.stun[:, i] / 100
            o[:, 2] = self.body[:, i] / 100
            o[:, 3] = sf[:, i]
            o[:, 4] = self.cap[:, i] / self.stam_max[i]
            o[:, 5:14] = eyeP[self.phase[:, i]]
            o[:, 14:34] = eyeM[np.minimum(self.move[:, i], OBS_MOVES - 1)]
            o[:, 34] = self.timer[:, i] / 12
            o[:, 35] = self.feint[:, i]
            o[:, 36:39] = eyeG[self.guard[:, i]]
            o[:, 39] = can_combo[:, i]
            th = self.th[:, i]
            for k, da in enumerate((0.0, np.pi, np.pi / 2, -np.pi / 2)):
                o[:, 40 + k] = ray_to_rope(self.pos[:, i], th + da) / RING
            dx, dy = old[:, 0] - self.pos[:, i, 0], old[:, 1] - self.pos[:, i, 1]
            dist = np.maximum(np.hypot(dx, dy), 1e-6)
            rel = np.arctan2(dy, dx) - th
            o[:, 44], o[:, 45] = np.sin(rel), np.cos(rel)
            o[:, 46] = self.kd_total[:, i] / 3
            o[:, 47] = self.kd_round[:, i] / 3
            o[:, 48] = self.t_round / ROUND_TICKS
            o[:, 49] = self.round / max(1, self.n_rounds - 1)
            o[:, 50] = (pts[:, i] - pts[:, j]) / 10
            o[:, 51] = dist / 3
            for k, pid in enumerate((0, 4, 8)):
                o[:, 52 + k] = dist - self.rng_tab[pid, i]
                o[:, 55 + k] = dist - self.rng_tab[pid, j]
            me_rel = np.arctan2(-dy, -dx) - old[:, 2]
            o[:, 58], o[:, 59] = np.sin(me_rel), np.cos(me_rel)
            o[:, 60] = old[:, 3] / HP_MAX
            o[:, 61] = old[:, 4] / 100
            o[:, 62] = old[:, 5] / 100
            g = 0.05 + 0.15 * (1 - self.mind[i, 0])               # 观察力越好，对体能看得越准
            o[:, 63] = np.round(old[:, 6] / g) * g
            o[:, 64:73] = eyeP[old[:, 7].astype(int)]
            o[:, 73:93] = eyeM[np.minimum(old[:, 8].astype(int), OBS_MOVES - 1)]
            o[:, 93] = old[:, 9] / 12
            o[:, 94:97] = eyeG[old[:, 10].astype(int)]
            ox, oy = old[:, 0], old[:, 1]
            o[:, 97] = np.minimum.reduce([ox - BODY_R, RING - BODY_R - ox, oy - BODY_R, RING - BODY_R - oy]) / RING
            o[:, 98] = old[:, 11] / 3
            # 气势、心理
            o[:, 99] = self.comp[:, i] / 100
            gc = 0.1 + 0.2 * (1 - self.mind[i, 0])                # 对手的心态只看得出个大概（身体语言），观察力好看得细
            o[:, 100] = np.round(old[:, 12] / 100 / gc) * gc
            # 记分卡：前几回合三位裁判的累计分差（角落会告诉你领先还是落后）
            o[:, 101:104] = (self.judge[:, :, i] - self.judge[:, :, j]) / 10
            # 读招记忆
            o[:, 104:108] = self.tend_punch[:, i]
            o[:, 108] = self.tend_head[:, i]
            o[:, 109:116] = self.tend_def[:, i]
            o[:, 116] = self.tend_feint[:, i]
            # 熟练度：自己的，和对手的（看过录像，知道他哪招练得好）
            o[:, 117:117 + N_TECH] = self.skills[i]
            o[:, 117 + N_TECH:117 + 2 * N_TECH] = self.skills[j]
            # 架势、站架、状态、性格、上头、战术
            b = OBS_BASE
            eyeK = np.eye(N_POSTURES)
            o[:, b:b + 4] = eyeK[self.posture[:, i]]
            o[:, b + 4:b + 8] = eyeK[old[:, 13].astype(int)]
            o[:, b + 8] = self.stance[:, i]
            o[:, b + 9] = old[:, 14]
            o[:, b + 10] = self.stance[:, i] != old[:, 14]
            o[:, b + 11] = self.form[:, i]
            o[:, b + 12:b + 16] = self.temper[i]
            o[:, b + 16] = self.tilt[:, i] > 0
            o[:, b + 17] = old[:, 15]
            o[:, b + 18:b + 24] = np.eye(N_PLANS)[self.plan[:, i]]
            o[:, b + 24:b + 26] = self.mind[i]
            see_through = np.clip((self.mind[i, 0] - 0.5) * 1.5, 0, 0.75)    # 观察力好的人能看穿一部分虚晃
            o[:, b + 26] = old[:, 16] * (self.rng.random(n) < see_through)
            o[:, b + 27] = self.holds[:, i] / HOLDS_WARN                # 这回合抱了几次（裁判警告是公开的）
            o[:, b + 28] = self.holds[:, j] / HOLDS_WARN
        return out
