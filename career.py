"""赛季、训练场、赛事流程。规则见 docs/赛季与训练场规格.md。

一个角色（Fighter）在两场比赛之间过日子：每天做一件事（不能并行），消耗疲劳，
带伤练效果打折、恢复拖长。有的项目只练一项（专项），有的同时练几项（综合，每项效率更低），
团队每天按需要取舍。比赛夜走完称重 → 战术会议 → 比赛 → 体检 → 战绩。
对练是真的跑强化学习；看录像是真的把对手的习惯装进开场的读招记忆。
"""
import copy
import dataclasses

import numpy as np

import bodies as BD
import boxing as B
import boxing_league as BL
import schools as SC
from boxing_bots import BoxingSparring
from nets import Adam, Agent, Policy

# ------------------------------------------------------------------ 规则常数
NATURAL_RECOVERY = 10.0         # 每天疲劳自然恢复
INJURY_TRAIN_DISCOUNT = 0.6     # 带伤训练：效果 ×(1 − 0.6 × 当前伤势)
INJURY_TRAIN_DELAY = 0.5        # 带伤训练一天，恢复期多拖 0.5 天
ATTR_HEADROOM = 2.0             # 潜力上限 = 天赋 + 2 分（且 ≤ 10）
DETRAIN_DAYS, DETRAIN_RATE = 21, 0.03   # 三周不练有氧，体能每天掉 0.03 分
KO_SUSPENSION, TKO_SUSPENSION = 30, 21
KO_CHIN_LOSS = 0.3              # 被 KO 一次，抗击打掉 0.3 分（"下巴被打穿"）
SPAR_ITERS = 12                 # 一天对练跑多少轮强化学习
SPAR_ENVS, SPAR_T = 96, 128
OVERTRAIN_FATIGUE = 60.0
SPAR_REPEAT = 0.12             # 最近 7 天每对练一次，对练的分数再降多少
VARIETY = 0.05                 # 最近 7 天每做一次同一个项目，这个项目的分数再降多少
TECH_DRILL = 0.20

# 睡眠与饮食：白天的训练只产生"待兑现的收益"，晚上睡觉、吃饭才把它变成真实属性
CONVERT = 0.85                  # 睡好吃好的一晚能兑现多少待兑现收益
PEND_KEEP = 0.6                 # 没兑现的部分，每晚还剩多少（两三天就淡掉）
REST_BONUS = 35.0               # 休息日额外恢复的疲劳（乘睡眠质量）
DIETS = {                       # 能量充足度、蛋白质充足度、每天体重变化（公斤）
    "减重": (0.55, 0.85, -0.10),
    "维持": (1.00, 1.00, 0.00),
    "增肌": (1.00, 1.00, 0.06),
}
CATEGORY = {"力量": "muscle", "体能": "aerobic", "手速": "neural", "反应": "neural", "步速": "neural",
            "观察力": "mind", "拳商": "mind"}
CUT_PENALTY = 0.3               # 称重超重，每临时脱水掉 1 公斤，比赛状态 −0.3
CUT_LIMIT = 2.0                 # 超重 2 公斤以上掉不下来：称重失败               # 招式训练满效率时，选中招式一天涨 0.2 ×(1 − 熟练度)（几个招式平分）

# 训练项目：疲劳、类型、属性增量（属性分/天）、技术增量、选招式的效率、是否算有氧、受伤风险
ACTIVITIES = {
    "休息":          {"load": -35, "kind": "rest"},
    "跑步":          {"load": 25, "kind": "train", "attrs": {"体能": 0.08}, "aerobic": True},
    "器械力量":      {"load": 25, "kind": "train", "attrs": {"力量": 0.08}, "risk": 0.01},
    "速度训练":      {"load": 20, "kind": "train", "attrs": {"手速": 0.08}},
    "反应球":        {"load": 15, "kind": "train", "attrs": {"反应": 0.08}},
    "绳梯":          {"load": 15, "kind": "train", "attrs": {"步速": 0.08}},
    "招式训练":      {"load": 15, "kind": "train", "tech": 1.0},
    "跳绳":          {"load": 20, "kind": "train", "attrs": {"体能": 0.04, "步速": 0.03}, "skills": {"步法": 0.03}, "aerobic": True},
    "循环训练":      {"load": 30, "kind": "train", "attrs": {"体能": 0.045, "力量": 0.045}, "aerobic": True, "risk": 0.01},
    "打沙袋":        {"load": 25, "kind": "train", "attrs": {"力量": 0.035, "体能": 0.035}, "tech": 0.6, "aerobic": True},
    "打手靶":        {"load": 22, "kind": "train", "attrs": {"手速": 0.035, "反应": 0.035, "拳商": 0.01}, "tech": 0.8},
    "看录像":        {"load": 5, "kind": "film", "attrs": {"观察力": 0.02}},
    "实战复盘":      {"load": 5, "kind": "train", "attrs": {"拳商": 0.04}},
    "对练·教练":     {"load": 30, "kind": "spar", "attrs": {"体能": 0.02, "反应": 0.02, "拳商": 0.01}, "risk": 0.02, "aerobic": True},
    "对练·邀请拳手": {"load": 35, "kind": "spar", "attrs": {"体能": 0.02, "反应": 0.02, "拳商": 0.015}, "risk": 0.04, "aerobic": True},
    "战术会议":      {"load": 0, "kind": "strategy"},
}
FIELD_OF = {"力量": "power", "手速": "hand_speed", "步速": "foot_speed", "体能": "cardio", "抗击打": "chin", "反应": "react_ticks"}
TRAINABLE = ["力量", "手速", "步速", "体能", "反应", "观察力", "拳商"]
# 流派的训练重点（团队取舍时的权重）
SCHOOL_PRIO = {
    "外围拳击": {"手速": 1.0, "步速": 1.0, "反应": 1.0, "体能": 0.8, "力量": 0.3, "观察力": 0.6, "拳商": 0.6},
    "压迫型": {"体能": 1.0, "力量": 1.0, "反应": 0.5, "手速": 0.5, "步速": 0.4, "观察力": 0.4, "拳商": 0.5},
    "重炮手": {"力量": 1.0, "手速": 0.6, "体能": 0.6, "反应": 0.5, "步速": 0.3, "观察力": 0.4, "拳商": 0.5},
    "反击型": {"反应": 1.0, "手速": 0.8, "步速": 0.6, "体能": 0.6, "力量": 0.4, "观察力": 1.0, "拳商": 0.8},
}


def pts_to_value(name, pts):
    """属性分 → 原始数值（和 bodies.attr_points 互逆）。"""
    field = FIELD_OF[name]
    _, lo, hi = BD.ATTR_RANGES[field]
    v = lo + pts / 10.0 * (hi - lo)
    return int(round(v)) if field == "react_ticks" else float(v)


class Fighter:
    """一个角色的全部可变状态：属性分、熟练度、出招网络、疲劳、伤病、了解度、战绩、日志。"""

    def __init__(self, body, policy_params, skills, seed=0):
        self.base = body
        self.id = body.id
        self.pts0 = BD.attr_points(body)                     # 天赋
        self.pts = dict(self.pts0)                           # 现在
        self.talent = SC.talent(body)
        self.skills = np.array(skills, dtype=float)
        self.pi = [p.copy() for p in policy_params]
        self.vf = None
        self.fatigue = 0.0
        self.injury, self.injury_days, self.injury_left = 0.0, 0.0, 0.0
        self.suspension = 0
        self.knowledge = {}
        self.strategy_days = 0
        self.last_aerobic_day = 0
        self.day = 0
        self.record = []
        self.log = []
        self.camp_fatigue = []
        self.mass = float(body.mass)
        self.pend_attr = {k: 0.0 for k in TRAINABLE}
        self.pend_skill = np.zeros(len(self.skills))
        self.pend_know = {}
        self.last_sleep_q = 1.0
        self.cut_penalty = 0.0
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------- 派生量
    def body(self):
        """按现在的属性分生成模拟器用的 Body。"""
        kw = {FIELD_OF[k]: pts_to_value(k, self.pts[k]) for k in FIELD_OF}
        mind = {"观察力": self.pts["观察力"] / 10.0, "拳商": self.pts["拳商"] / 10.0}
        return dataclasses.replace(self.base, mind=mind, mass=self.mass, **kw)

    def injury_now(self):
        if self.injury_days <= 0:
            return 0.0
        return self.injury * max(0.0, self.injury_left) / self.injury_days

    def effect(self):
        iq = self.pts["拳商"] / 10.0
        return ((1 - INJURY_TRAIN_DISCOUNT * self.injury_now()) * (1 - 0.5 * self.fatigue / 100.0)
                * (0.6 + 0.8 * iq))

    def cap(self, name):
        return min(10.0, self.pts0[name] + ATTR_HEADROOM)

    def room(self, name):
        cap = self.cap(name)
        if cap <= self.pts0[name]:
            return 0.0
        return float(np.clip((cap - self.pts[name]) / (cap - self.pts0[name]), 0.0, 1.0))

    def grow(self, name, amount):
        """属性往上涨，越接近潜力上限涨得越慢。"""
        self.pts[name] = min(self.cap(name), self.pts[name] + amount * self.room(name))

    def policy(self):
        return Policy([p.copy() for p in self.pi])

    def form_mean(self):
        """称重时的状态期望：精神越足越好，带伤和练过头都会拉低。"""
        fresh = 1 - self.fatigue / 100.0
        over = max(0.0, (np.mean(self.camp_fatigue) if self.camp_fatigue else 0.0) - OVERTRAIN_FATIGUE) / 40.0
        return float(np.clip(0.5 * fresh - 0.8 * self.injury_now() - 0.6 * over - self.cut_penalty, -1, 1))

    def state(self):
        return {"day": self.day, "fatigue": round(self.fatigue, 1), "injury": round(self.injury_now(), 3),
                "injury_left": round(self.injury_left, 1), "suspension": self.suspension,
                "pts": {k: round(v, 3) for k, v in self.pts.items()}, "skills": self.skills.round(4).tolist(),
                "knowledge": {k: round(v, 3) for k, v in self.knowledge.items()}, "strategy_days": self.strategy_days,
                "mass": round(self.mass, 2), "pending": round(sum(self.pend_attr.values()) + float(self.pend_skill.sum()), 4)}


# ------------------------------------------------------------------ 一天 = 白天训练 + 晚上睡觉吃饭
def do_day(f, activity, techs=None, target=None, partner=None, seed=0, days_left=99, sleep=None, diet=None):
    """让 f 过一天：白天做 activity，晚上睡觉吃饭把收益兑现。返回这一天的日志。
    techs：选的招式；target：看录像研究的对手；partner：对练对象 (body, 出招对象, 熟练度, 名字)；
    sleep / diet：睡多少小时、怎么吃，不给就由团队按情况定。"""
    entry = train(f, activity, techs=techs, target=target, partner=partner, seed=seed)
    entry["night"] = night(f, days_left=days_left, sleep=sleep, diet=diet)
    entry["after"] = f.state()
    return entry


def train(f, activity, techs=None, target=None, partner=None, seed=0):
    """白天：做一个训练项目。收益先记成"待兑现"，晚上才变成真实属性。"""
    if activity not in ACTIVITIES:
        raise ValueError(f"没有这个训练项目：{activity}")
    if activity.startswith("对练") and f.suspension > 0:
        activity = "看录像" if target is not None else "实战复盘"   # 禁赛期不能实战对练
    spec = ACTIVITIES[activity]
    kind, load = spec["kind"], spec["load"]
    before = f.state()
    eff = f.effect()
    note = {}

    for name, rate in spec.get("attrs", {}).items():
        f.pend_attr[name] += rate * eff * f.room(name)
    for name, rate in spec.get("skills", {}).items():
        k = B.TECH_NAMES.index(name)
        f.pend_skill[k] += rate * eff * (1 - f.skills[k])
    if spec.get("tech"):
        techs = techs or [B.TECH_NAMES[int(np.argmax(f.skills[:12]))]]
        for name in techs:
            k = B.TECH_NAMES.index(name)
            f.pend_skill[k] += TECH_DRILL / len(techs) * spec["tech"] * eff * (1 - f.skills[k])
        note["techs"] = techs
    if spec.get("aerobic"):
        f.last_aerobic_day = f.day
    if kind == "film" and target is not None:
        obs = f.pts["观察力"] / 10.0
        k0 = f.knowledge.get(target.id, 0.0) + f.pend_know.get(target.id, 0.0)
        f.pend_know[target.id] = f.pend_know.get(target.id, 0.0) + 0.35 * (1 - k0) * eff * (0.5 + obs)
        note["target"] = target.id
    elif kind == "strategy":
        f.strategy_days += 1
    elif kind == "spar":
        if partner is None:
            partner = (BD.SPARRING, BoxingSparring(), SC.talent(BD.SPARRING), "陪练教练")
        skills0 = f.skills.copy()
        note.update(spar(f, partner, iters=max(1, int(round(SPAR_ITERS * eff / 0.9))), seed=seed))
        f.pend_skill += f.skills - skills0            # 对练练出来的熟练度也要睡一觉才算数
        f.skills = skills0
        if target is not None and partner[3] == target.id:
            k0 = f.knowledge.get(target.id, 0.0)
            f.pend_know[target.id] = f.pend_know.get(target.id, 0.0) + 0.15 * (1 - k0)

    risk = spec.get("risk", 0.0) * (0.3 + (f.fatigue / 100.0) ** 2 * 2)
    if risk and f.rng.random() < risk:
        sev = float(f.rng.uniform(0.1, 0.3))
        add_injury(f, sev, 2 + 20 * sev)
        note["injured"] = round(sev, 2)

    f.rested = kind == "rest"
    f.load_today = max(0, load)
    f.fatigue = float(np.clip(f.fatigue + max(0, load), 0, 100))
    if f.injury_left > 0 and not f.rested and load > 0:
        f.injury_left += INJURY_TRAIN_DELAY             # 带伤练，恢复期被拉长
        f.injury_days += INJURY_TRAIN_DELAY
    f.suspension = max(0, f.suspension - 1)
    if f.day - f.last_aerobic_day > DETRAIN_DAYS:
        f.pts["体能"] = max(0.0, f.pts["体能"] - DETRAIN_RATE)
    f.day += 1
    entry = {"day": f.day, "activity": activity, "effect": round(eff, 3), "note": note, "before": before}
    f.log.append(entry)
    return entry


def coach_routine(f, days_left):
    """团队定今晚睡多久、这几天怎么吃。体重往"量级上限 − 0.8 公斤"靠，临近比赛不能超重。"""
    poise = SC.temperament(f.base)[1]
    sleep = 7.5 + 1.0 * poise
    hi = BD.WEIGHT_CLASS[2]
    walk = hi - 0.8
    if f.mass > walk + 0.3 or (days_left <= 10 and f.mass > hi - 0.3):
        diet = "减重"
    elif f.mass < walk - 0.6 and days_left > 14 and f.room("力量") > 0.2:
        diet = "增肌"
    else:
        diet = "维持"
    return sleep, diet


def night(f, days_left=99, sleep=None, diet=None):
    """晚上：睡觉、吃饭。按睡眠质量和饮食把待兑现收益变成真实属性，同时恢复疲劳和伤病、改变体重。"""
    s0, d0 = coach_routine(f, days_left)
    sleep = s0 if sleep is None else sleep
    diet = d0 if diet is None else diet
    t = SC.temperament(f.base)
    hours = sleep
    hours -= max(0.0, f.fatigue - 70) / 30.0 * 1.5             # 练过头会失眠
    hours -= 1.5 * f.injury_now()                              # 伤痛
    if days_left <= 2:
        hours -= (1 - t[1]) * 2.0                              # 沉着低的人赛前睡不着
    hours += float(f.rng.normal(0, 0.3))
    hours = float(np.clip(hours, 0, 12))
    q = float(np.clip((hours - 4.5) / 3.5, 0, 1))
    energy, protein, dmass = DIETS[diet]
    conv = {"mind": CONVERT * q, "neural": CONVERT * q,
            "muscle": CONVERT * q * min(energy, protein) ** 2,      # 热量、蛋白质不够，肌肉几乎不长
            "aerobic": CONVERT * q * energy}
    gained = {}
    for name, pend in f.pend_attr.items():
        add = pend * conv[CATEGORY[name]]
        if add > 0:
            new = min(f.cap(name), f.pts[name] + add)
            gained[name] = round(new - f.pts[name], 4)
            f.pts[name] = new
        f.pend_attr[name] = (pend - add) * PEND_KEEP
    add = f.pend_skill * conv["mind"]
    f.skills = np.clip(f.skills + add, 0, 1)
    f.pend_skill = (f.pend_skill - add) * PEND_KEEP
    for k, pend in list(f.pend_know.items()):
        a = pend * conv["mind"]
        f.knowledge[k] = min(1.0, f.knowledge.get(k, 0.0) + a)
        f.pend_know[k] = (pend - a) * PEND_KEEP
    # 恢复
    f.fatigue = float(np.clip(f.fatigue - NATURAL_RECOVERY * (0.4 + 0.6 * q)
                              - (REST_BONUS * (0.5 + 0.5 * q) if getattr(f, "rested", False) else 0.0), 0, 100))
    if f.injury_left > 0:
        heal = (1 + (1 if getattr(f, "rested", False) else 0)) * (0.5 + 0.5 * q) * (0.7 + 0.3 * energy)
        f.injury_left -= heal
        if f.injury_left <= 0:
            f.injury, f.injury_days, f.injury_left = 0.0, 0.0, 0.0
    f.mass = float(f.mass + dmass - 0.01 * getattr(f, "load_today", 0) / 25.0)
    f.last_sleep_q = q
    f.camp_fatigue.append(f.fatigue)
    f.rested, f.load_today = False, 0
    return {"sleep": round(hours, 1), "sleep_q": round(q, 2), "diet": diet, "mass": round(f.mass, 2),
            "gained": gained, "skill_gain": round(float(add.sum()), 4)}


def weigh_in(f):
    """称重：超重就临时脱水掉秤，比赛状态下降；超太多掉不下来就是称重失败。"""
    hi = BD.WEIGHT_CLASS[2]
    over = max(0.0, f.mass - hi)
    failed = over > CUT_LIMIT
    f.cut_penalty = 1.0 if failed else CUT_PENALTY * over
    if over > 0 and not failed:
        f.mass = hi
    return {"over": round(over, 2), "failed": failed, "penalty": round(f.cut_penalty, 2)}


def add_injury(f, severity, days):
    """叠加伤病：取更重的伤势，恢复天数取更长的那个。"""
    cur = f.injury_now()
    f.injury = float(min(1.0, max(cur, severity) + 0.5 * min(cur, severity)))
    f.injury_left = max(f.injury_left, days)
    f.injury_days = max(f.injury_left, 1e-6)


# ------------------------------------------------------------------ 对练：真的跑强化学习
def spar(f, partner, iters, seed=0):
    """f 对 partner（身体、出招对象、熟练度、名字）训练 iters 轮。更新 f 的出招网络和熟练度。"""
    from train import gae, practice_counts, evolve_skills
    p_body, p_ctrl, p_skills, p_name = partner
    rng = np.random.default_rng(seed)
    agent = Agent(B.OBS_DIM, B.N_ACTIONS, rng)
    agent.pi.params[:] = [p.copy() for p in f.pi]
    agent.opt_pi = Adam(agent.pi.params, 3e-4 * (0.3 + 0.7 * f.last_sleep_q))   # 没睡好，对练学不进去
    if f.vf is not None:
        agent.vf.params[:] = [p.copy() for p in f.vf]
        agent.opt_vf = Adam(agent.vf.params, 1e-3)
    sim = B.Sim(SPAR_ENVS, (f.body(), p_body), seed=seed, skills=np.stack([f.skills, p_skills]), form_sd=B.FORM_SD)
    sim.desync(rng)
    drill = (SC.drill(f.base) * (0.6 + 0.8 * f.pts["拳商"] / 10.0))[None]
    obs = sim.observe()
    wins = losses = 0
    T, N = SPAR_T, SPAR_ENVS
    learn = np.stack([np.ones(N, bool), np.zeros(N, bool)], 1)
    for _ in range(iters):
        bo, ba, blp, bv, br, bd, bm, bacted, blapse = ([] for _ in range(9))
        usage, succ, lt = np.zeros((2, B.N_TECH)), np.zeros((2, B.N_TECH)), np.zeros(2)
        for _t in range(T):
            mask = sim.plan_mask()
            a0, lp, v = agent.act(obs[:, 0], rng, mask[:, 0])
            a1 = p_ctrl.act(obs[:, 1], rng, mask[:, 1])
            acts, lapsed = B.lapse(np.stack([a0, a1], 1), mask, sim, rng)
            bo.append(obs[:, 0]); ba.append(acts[:, 0]); blp.append(lp); bv.append(v); bm.append(mask[:, 0])
            rew, done, info = sim.step(acts)
            rew = rew - 0.5 * (done & (info["winner"] < 0))[:, None]
            br.append(rew[:, 0]); bd.append(done); bacted.append(info["acted"][:, 0]); blapse.append(lapsed[:, 0])
            practice_counts(info, acts, learn, sim, usage, succ, lt)
            wins += int(((info["winner"] == 0) & done).sum())
            losses += int(((info["winner"] == 1) & done).sum())
            obs = sim.observe()
        val_a = np.array(bv, np.float32)
        adv = gae(np.array(br, np.float32), val_a, np.array(bd, np.float32), agent.value(obs[:, 0]), 0.995, 0.95)
        flat = lambda x: np.asarray(x).reshape(T * N, *np.asarray(x).shape[2:])
        pm = (np.array(bacted) & ~np.array(blapse)).astype(np.float32)
        agent.update(flat(bo), flat(ba).astype(np.int64), flat(blp), flat(adv), flat(adv + val_a), flat(pm),
                     np.ones(T * N, np.float32), rng, ent_coef=0.006, amask=flat(bm))
        sk = evolve_skills(np.stack([f.skills, f.skills]), np.stack([f.talent, f.talent]), usage, succ, lt,
                           np.concatenate([drill, drill]))
        f.skills = sk[0]
    f.pi = [p.copy() for p in agent.pi.params]
    f.vf = [p.copy() for p in agent.vf.params]
    return {"partner": p_name, "iters": iters, "bouts_won": wins, "bouts_lost": losses}


# ------------------------------------------------------------------ 看录像：测出对手的习惯
def scout_profile(fighter, n=24, seed=0):
    """让对手和陪练教练打一回合，读陪练那一侧的读招记忆，就是"录像里看出来的习惯"。"""
    coach = BoxingSparring()
    sim = B.Sim(n, (fighter.body(), BD.SPARRING), seed=seed, auto_reset=False, n_rounds=1,
                skills=np.stack([fighter.skills, SC.talent(BD.SPARRING)]))
    sim.plan_fixed = np.array([0, 0])
    sim.reset(np.ones(n, bool))
    pol = fighter.policy()
    rng = np.random.default_rng(seed)
    for _ in range(B.ROUND_TICKS):
        o = sim.observe()
        m = sim.plan_mask()
        sim.step(np.stack([pol.act(o[:, 0], rng, m[:, 0]), coach.act(o[:, 1], rng, m[:, 1])], 1))
    return {"punch": sim.tend_punch[:, 1].mean(0), "head": float(sim.tend_head[:, 1].mean()),
            "def": sim.tend_def[:, 1].mean(0), "feint": float(sim.tend_feint[:, 1].mean())}


def prior_from(profile, k):
    """了解度 k：开场读招记忆 = k × 录像里的习惯 + (1 − k) × 一无所知。"""
    blank = {"punch": np.full(4, 0.25), "head": 0.5, "def": np.full(7, 1 / 7), "feint": 0.0}
    return {key: k * np.asarray(profile[key]) + (1 - k) * np.asarray(blank[key]) for key in blank}


# ------------------------------------------------------------------ 赛前、比赛、赛后
STRATEGY_SIMS = [0, 6, 12, 24]      # 开 0/1/2/3+ 天战术会议，每个计划模拟几场


def corner_plan(f, opp, side, seed):
    """战术会议：开会天数越多，排名用的模拟场数越多。0 天只能自由发挥、按默认顺序调整。"""
    n = STRATEGY_SIMS[min(f.strategy_days, len(STRATEGY_SIMS) - 1)]
    if n == 0:
        return [(p, 0.0) for p in range(B.N_PLANS)], 0
    bodies = (f.body(), opp.body()) if side == 0 else (opp.body(), f.body())
    pols = (f.policy(), opp.policy()) if side == 0 else (opp.policy(), f.policy())
    sk = np.stack([f.skills, opp.skills]) if side == 0 else np.stack([opp.skills, f.skills])
    return BL.choose_plan(bodies, pols, side, sk, n=n, seed=seed), n


def fight_night(fa, fb, seed=0, title=""):
    """一个比赛夜：称重 → 战术会议 → 比赛 → 宣布 → 体检。返回完整记录（含回放档）。"""
    card = {"title": title, "day": fa.day, "fighters": [fa.id, fb.id]}
    cuts = [weigh_in(f) for f in (fa, fb)]
    card["weigh_in"] = [{"id": f.id, "mass": round(f.mass, 2), "form_mean": round(f.form_mean(), 3), "cut": cut,
                         "fatigue": round(f.fatigue, 1), "injury": round(f.injury_now(), 3)} for f, cut in zip((fa, fb), cuts)]
    rank_a, na = corner_plan(fa, fb, 0, seed)
    rank_b, nb = corner_plan(fb, fa, 1, seed + 1)
    card["plans"] = [{"rank": rank_a, "sims": na}, {"rank": rank_b, "sims": nb}]
    priors = []
    for me, opp in ((fa, fb), (fb, fa)):
        k = me.knowledge.get(opp.id, 0.0)
        priors.append(prior_from(scout_profile(opp, seed=seed + 7), k) if k > 0 else None)
    card["knowledge"] = [round(fa.knowledge.get(fb.id, 0.0), 3), round(fb.knowledge.get(fa.id, 0.0), 3)]
    bt = BL.run_matches((fa.body(), fb.body()), (fa.policy(), fb.policy()), (fa.id, fb.id), 1, seed=seed,
                        skills=np.stack([fa.skills, fb.skills]), plans=(rank_a[0][0], rank_b[0][0]),
                        plan_rank=([p for p, _ in rank_a], [p for p, _ in rank_b]),
                        form_mean=(fa.form_mean(), fb.form_mean()), tend_prior=priors)
    rec = BL.replay_record(bt, 0, [{"name": fa.id}, {"name": fb.id}])
    card["replay"] = rec
    w, m = int(bt.winner[0]), int(bt.method[0])
    card["result"] = {"winner": [fa.id, fb.id][w] if w >= 0 else None, "method": B.METHOD_NAMES[m],
                      "judges": bt.judges[0].tolist(), "seconds": round(float(bt.end[0]) * B.DT, 1)}
    # 赛后体检
    frames = BL.resimulate(rec)
    last = frames[-1]["f"]
    card["medical"] = []
    for k, f in enumerate((fa, fb)):
        hp_lost = max(0.0, 100.0 - last[k][3])
        kds = int(sum(fr["f"][k][14] for fr in frames))
        stopped = (w == 1 - k) and m in (B.M_KO, B.M_TKO)
        card["medical"].append(medical(f, hp_lost, kds, stopped, m == B.M_KO))
        f.record.append({"day": f.day, "opp": (fb, fa)[k].id, "result": "胜" if w == k else ("负" if w == 1 - k else "平"),
                         "method": B.METHOD_NAMES[m]})
        f.camp_fatigue = []
        f.strategy_days = 0
        f.cut_penalty = 0.0
        f.mass += 2.0 if cuts[k]["over"] > 0 else 0.0          # 比赛后补水，体重回升
    return card


def medical(f, hp_lost, knockdowns, stopped, ko):
    """赛后体检：伤势、恢复天数、禁赛，被 KO 的下巴变脆。"""
    sev = min(1.0, hp_lost / 150.0 + 0.15 * knockdowns + (0.3 if stopped else 0.0))
    days = 2 + 20 * sev
    if sev > 0:
        add_injury(f, sev, days)
    susp = KO_SUSPENSION if (stopped and ko) else (TKO_SUSPENSION if stopped else 0)
    f.suspension = max(f.suspension, susp)
    if stopped and ko:
        f.pts["抗击打"] = max(0.0, f.pts["抗击打"] - KO_CHIN_LOSS)
    return {"id": f.id, "hp_lost": round(hp_lost, 1), "knockdowns": knockdowns, "severity": round(sev, 3),
            "recovery_days": round(days, 1), "suspension": susp}


# ------------------------------------------------------------------ 团队排程：每天给候选项目打分，取舍
def attr_need(f):
    """各属性的需要程度 = 流派重点 × 离上限还有多远。"""
    prio = SCHOOL_PRIO.get(f.base.school or "", {})
    return {a: prio.get(a, 0.5) * f.room(a) for a in TRAINABLE}


def score_activity(f, act, opp, days_left, need=None, partner_available=True):
    """一个项目今天值多少分。分数 = 各项需要 × 增量之和（+ 目标进度），减去疲劳代价。"""
    spec = ACTIVITIES[act]
    need = need if need is not None else attr_need(f)
    t = SC.temperament(f.base)
    drive, iq = t[2], f.pts["拳商"] / 10.0
    eff = f.effect()
    v = sum(need.get(a, 0) * r for a, r in spec.get("attrs", {}).items()) * 10 * eff
    v += sum(0.5 * r * 10 for r in spec.get("skills", {}).values()) * eff
    if spec.get("tech"):
        v += 0.35 * spec["tech"] * eff
    know_target = 0.5 + 0.35 * iq
    strat_target = 1 + int(round(2 * iq))
    if spec["kind"] == "film":
        v += max(0.0, know_target - f.knowledge.get(opp.id, 0.0)) * (1.5 + iq) if opp is not None else 0
    if spec["kind"] == "strategy":
        v = 3.0 if (days_left <= 1 + strat_target and f.strategy_days < strat_target) else -1.0
    if spec["kind"] == "spar":
        if f.suspension > 0 or (act == "对练·邀请拳手" and not partner_available):
            return -9.0
        v += (0.35 + 0.45 * drive) * eff                  # 对练是让打法进步的主要途径，好斗的更爱
        recent_spar = sum(1 for e in f.log[-7:] if e["activity"].startswith("对练"))
        v -= SPAR_REPEAT * recent_spar                    # 教练控制对练频率：一周打太多会伤、会钝
    if spec["kind"] == "rest":
        v = 2.0 * (f.fatigue / 100.0) ** 2 * (1.6 - 1.6 * drive) + 1.5 * f.injury_now()   # 好斗的人不爱歇
        if days_left == 1:
            v += 5.0                                      # 赛前一天减量
    if spec["kind"] not in ("rest", "strategy", "spar"):
        v -= VARIETY * sum(1 for e in f.log[-7:] if e["activity"] == act)      # 同一个项目连着练，边际收益下降
    cost = max(0, spec["load"]) / 100.0 * (0.4 + f.fatigue / 100.0) * (1.4 - 0.6 * drive)
    cost += max(0, spec["load"]) / 100.0 * 1.5 * f.injury_now()
    return v - cost


def coach_choose(f, opp, days_left, partner_available=True, need=None, noise=0.08):
    """团队今天的选择：给所有项目打分，挑最高的（带一点随机）。返回 (项目, 参数)。"""
    acts = list(ACTIVITIES)
    scores = np.array([score_activity(f, a, opp, days_left, need, partner_available) for a in acts])
    scores = scores + f.rng.normal(0, noise, len(scores))
    act = acts[int(np.argmax(scores))]
    kw = {}
    if ACTIVITIES[act].get("tech"):
        top = list(np.argsort(-f.skills[:12])[:2])
        weak = list(np.argsort(f.skills[:12] - f.talent[:12])[:1])
        kw["techs"] = [B.TECH_NAMES[i] for i in dict.fromkeys(top + weak)][:3]
    if ACTIVITIES[act]["kind"] == "film":
        kw["target"] = opp
    return act, kw


def run_camp(f, opp, days, partner=None, seed=0, on_day=None):
    """跑完一整个训练营。partner：邀请来的对练对象 (body, 出招对象, 熟练度, 名字)。"""
    out = []
    for d in range(days):
        act, kw = coach_choose(f, opp, days - d, partner_available=partner is not None)
        if act == "对练·邀请拳手":
            kw["partner"] = partner
            kw["target"] = opp
        entry = do_day(f, act, seed=seed + d, days_left=days - d, **kw)
        out.append(entry)
        if on_day:
            on_day(f, entry)
    return out


def clone(f):
    return copy.deepcopy(f)
