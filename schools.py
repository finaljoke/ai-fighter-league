"""流派、性格、战术计划的数据。机制在 boxing.py，这里只放"谁是什么样"。

角色 = 身体（bodies.py）+ 流派 + 天赋熟练度 + 性格。
"""
import numpy as np

import boxing as B

# 流派：默认架势、对天赋熟练度的加成、训练时哪些技术练得更勤（练习量倍率）
SCHOOLS = {
    "外围拳击": {"posture": B.P_LONG, "bonus": {"刺拳·头": 0.1, "步法": 0.1, "后仰": 0.1, "轴转": 0.05},
             "drill": {"刺拳·头": 1.5, "刺拳·身": 1.3, "步法": 1.5, "后仰": 1.3, "轴转": 1.3, "拍挡": 1.2}},
    "压迫型": {"posture": B.P_CROUCH, "bonus": {"前手勾拳·身": 0.1, "下潜": 0.1, "前手勾拳·头": 0.05},
            "drill": {"前手勾拳·头": 1.4, "前手勾拳·身": 1.5, "后手勾拳·身": 1.3, "下潜": 1.5, "前手上勾·头": 1.3, "高位格挡": 1.2}},
    "重炮手": {"posture": B.P_STANDARD, "bonus": {"后手直拳·头": 0.1, "后手勾拳·头": 0.1},
            "drill": {"后手直拳·头": 1.5, "后手勾拳·头": 1.5, "后手上勾·头": 1.3, "高位格挡": 1.2}},
    "反击型": {"posture": B.P_LOW, "bonus": {"侧闪": 0.1, "拍挡": 0.1},
            "drill": {"侧闪": 1.5, "拍挡": 1.5, "后仰": 1.3, "后手直拳·头": 1.2}},
    "全能型": {"posture": B.P_STANDARD, "bonus": {}, "drill": {}},
}

TEMPER_NAMES = ["胆量", "沉着", "斗志", "冲动"]
MIND_NAMES = ["观察力", "拳商"]


def school_of(body):
    return SCHOOLS.get(getattr(body, "school", None) or "", None)


def talent(body):
    """天赋熟练度 = 身体自带的 + 流派加成。"""
    v = B.skill_vector(getattr(body, "skills", None))
    sc = school_of(body)
    if sc:
        for k, add in sc["bonus"].items():
            v[B.TECH_NAMES.index(k)] += add
    return np.clip(v, 0.0, 1.0)


def drill(body):
    """训练时各技术的练习量倍率。"""
    v = np.ones(B.N_TECH)
    sc = school_of(body)
    if sc:
        for k, m in sc["drill"].items():
            v[B.TECH_NAMES.index(k)] = m
    return v


def posture(body):
    sc = school_of(body)
    return sc["posture"] if sc else B.P_STANDARD


def mind(body):
    m = getattr(body, "mind", None) or {}
    return np.array([m.get(k, 0.5) for k in MIND_NAMES])


def temperament(body):
    t = getattr(body, "temperament", None) or {}
    return np.array([t.get(k, 0.5) for k in TEMPER_NAMES])
