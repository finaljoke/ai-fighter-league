"""训练场与赛事流程验收：每条规则一个小场景。
py -X utf8 test_career.py
"""
import numpy as np

import boxing as B
import career as C
from bodies import LANKY, STOCKY
from nets import Agent

C.SPAR_ENVS, C.SPAR_T = 16, 64          # 测试里对练跑小一点
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))


def fresh(body=LANKY, seed=0):
    pi = Agent(B.OBS_DIM, B.N_ACTIONS, np.random.default_rng(seed)).pi.params
    import schools as SC
    return C.Fighter(body, pi, SC.talent(body), seed=seed)


def gain(act, attr, **kw):
    f = fresh()
    v0 = f.pts[attr]
    C.do_day(f, act, **kw)
    return f.pts[attr] - v0


def tech_gain(act, tech):
    f = fresh()
    k = B.TECH_NAMES.index(tech)
    s0 = f.skills[k]
    C.do_day(f, act, techs=[tech])
    return f.skills[k] - s0


# ------------------------------------------------------------------ 时间：一天一件事
f = fresh()
for act in ["跑步", "看录像", "休息", "打手靶"]:
    C.do_day(f, act, target=fresh(STOCKY))
check("一天只能做一件事：做了几项就过了几天", f.day == 4 and len(f.log) == 4 and [e["activity"] for e in f.log] == ["跑步", "看录像", "休息", "打手靶"],
      f"第 {f.day} 天")

# ------------------------------------------------------------------ 专项 vs 综合
run, rope, circ = gain("跑步", "体能"), gain("跳绳", "体能"), gain("循环训练", "体能")
circ_pow = gain("循环训练", "力量")
check("综合项目每项都不如专项：练体能，跑步 > 跳绳、循环训练", run > rope and run > circ, f"体能 跑步 +{run:.3f}，跳绳 +{rope:.3f}，循环 +{circ:.3f}")
check("综合项目的总量略多：循环训练的体能 + 力量 > 跑步的体能", circ + circ_pow > run, f"循环 体能+力量 +{circ + circ_pow:.3f} vs 跑步 +{run:.3f}")
spd, mitt_spd, mitt_rx = gain("速度训练", "手速"), gain("打手靶", "手速"), gain("打手靶", "反应")
check("打手靶同时练手速、反应和招式，但手速涨得不如专项速度训练", spd > mitt_spd > 0 and mitt_rx > 0 and tech_gain("打手靶", "刺拳·头") > 0,
      f"手速 速度训练 +{spd:.3f}，打手靶 +{mitt_spd:.3f}（反应 +{mitt_rx:.3f}）")
t_pure, t_mitt, t_bag = tech_gain("招式训练", "前手勾拳·头"), tech_gain("打手靶", "前手勾拳·头"), tech_gain("打沙袋", "前手勾拳·头")
check("练招式：专门的招式训练 > 打手靶 > 打沙袋", t_pure > t_mitt > t_bag > 0, f"前手勾拳熟练度 +{t_pure:.4f} / +{t_mitt:.4f} / +{t_bag:.4f}")

# ------------------------------------------------------------------ 疲劳、伤病
def run_gain_at(fatigue=0.0, injury=0.0):
    f = fresh()
    f.fatigue = fatigue
    if injury:
        C.add_injury(f, injury, 10)
    v0 = f.pts["体能"]
    C.do_day(f, "跑步")
    return f.pts["体能"] - v0
check("疲劳越高训练效果越差", run_gain_at(80) < 0.7 * run_gain_at(0), f"疲劳 0 +{run_gain_at(0):.3f}，疲劳 80 +{run_gain_at(80):.3f}")
check("带伤训练效果打折", run_gain_at(injury=0.5) < 0.8 * run_gain_at(), f"健康 +{run_gain_at():.3f}，带伤 +{run_gain_at(injury=0.5):.3f}")

def left_after(act):
    f = fresh()
    C.add_injury(f, 0.5, 10)
    C.do_day(f, act)
    return f.injury_left
check("带伤训练恢复期被拉长，休息恢复得快", left_after("跑步") > left_after("休息") and left_after("跑步") > 9,
      f"练一天后剩 {left_after('跑步'):.1f} 天，休息一天后剩 {left_after('休息'):.1f} 天")

f = fresh()
f.fatigue = 50
C.do_day(f, "休息")
check("休息降疲劳", f.fatigue <= 5, f"疲劳 50 → {f.fatigue:.0f}")

# ------------------------------------------------------------------ 上限、退化
f = fresh()
for _ in range(400):
    C.do_day(f, "器械力量")
    f.fatigue = 0
check("属性有潜力上限：天赋 + 2 分", abs(f.pts["力量"] - f.cap("力量")) < 0.05 and f.pts["力量"] <= f.cap("力量") + 1e-9,
      f"力量 {f.pts0['力量']:.2f} → {f.pts['力量']:.2f}（上限 {f.cap('力量'):.2f}）")

f = fresh()
v0 = f.pts["体能"]
for _ in range(30):
    C.do_day(f, "看录像")
check("三周不练有氧，体能会掉", f.pts["体能"] < v0, f"体能 {v0:.2f} → {f.pts['体能']:.2f}")

# ------------------------------------------------------------------ 看录像、战术会议
fa, fb = fresh(LANKY), fresh(STOCKY, seed=1)
for _ in range(3):
    C.do_day(fa, "看录像", target=fb)
k = fa.knowledge[fb.id]
prof = C.scout_profile(fb, n=8, seed=3)
pr = C.prior_from(prof, k)
sim = B.Sim(1, (fa.body(), fb.body()), auto_reset=False)
sim.tend_prior = [pr, None]
sim.reset(np.ones(1, bool))
check("看录像：了解度上升，开打时读招记忆里已经装着对手的习惯",
      k > 0.4 and np.allclose(sim.tend_def[0, 0], pr["def"]) and not np.allclose(pr["def"], 1 / 7),
      f"了解度 {k:.2f}；开场就知道他防守的习惯 {np.round(pr['def'], 2).tolist()}")

f = fresh()
sims = []
for d in range(4):
    sims.append(C.STRATEGY_SIMS[min(f.strategy_days, len(C.STRATEGY_SIMS) - 1)])
    C.do_day(f, "战术会议")
check("战术会议：开会天数越多，给战术排名用的模拟场数越多", sims == sorted(sims) and sims[0] == 0 and sims[-1] > sims[1], f"0~3 天：{sims} 场")

# ------------------------------------------------------------------ 称重状态
def form(fatigue=0, injury=0, camp=None):
    f = fresh()
    f.fatigue = fatigue
    if injury:
        C.add_injury(f, injury, 10)
    f.camp_fatigue = camp or []
    return f.form_mean()
check("称重状态：精神足的好；累的、带伤的、练过头的差",
      form() > form(fatigue=60) and form() > form(injury=0.5) and form(camp=[40] * 20) > form(camp=[85] * 20),
      f"精神足 {form():+.2f}，累 {form(fatigue=60):+.2f}，带伤 {form(injury=0.5):+.2f}，练过头 {form(camp=[85] * 20):+.2f}")

# ------------------------------------------------------------------ 体检；赛博肉体没有禁赛
f = fresh()
m = C.medical(f, hp_lost=60, knockdowns=2, stopped=True, ko=True)
check("赛后体检：伤势 = 掉血/150 + 倒地×0.15 + 被终止 0.3；被 KO 下巴变脆",
      abs(m["severity"] - min(1, 60 / 150 + 0.3 + 0.3)) < 1e-6 and f.pts["抗击打"] < f.pts0["抗击打"],
      f"伤势 {m['severity']}，恢复 {m['recovery_days']} 天，抗击打 {f.pts0['抗击打']:.2f}→{f.pts['抗击打']:.2f}")
e = C.do_day(f, "对练·教练")
check("赛博肉体没有禁赛：被 KO 第二天照样能对练，只是带伤练效果打折",
      e["activity"] == "对练·教练" and e["effect"] < 0.9, f"项目 {e['activity']}，效果 {e['effect']:.0%}")
hurt, ok_ = fresh(), fresh()
hurt.fatigue = ok_.fatigue = 50.0
C.add_injury(hurt, 0.5, 10)
check("带伤练更容易再伤：同样的对练，伤势 0.5 时受伤风险是没伤时的 2 倍",
      abs(C.injury_risk(hurt, "对练·邀请拳手") / C.injury_risk(ok_, "对练·邀请拳手") - 2.0) < 1e-6,
      f"没伤 {C.injury_risk(ok_, '对练·邀请拳手'):.3f}，带伤 {C.injury_risk(hurt, '对练·邀请拳手'):.3f}")

# ------------------------------------------------------------------ 高强度
lo, hi = fresh(), fresh()
e_lo, e_hi = C.train(lo, "器械力量"), C.train(hi, "器械力量", high=True)
gain = lambda x: x.pend_attr["力量"]
check("高强度：收益 ×1.5、疲劳 ×2 再加 20、受伤风险 ×3（一天仍然只做一件事）",
      abs(gain(hi) / gain(lo) - 1.5) < 1e-6 and abs(hi.fatigue - (2 * lo.fatigue + C.HIGH_LOAD_ADD)) < 1e-6
      and abs(C.injury_risk(fresh(), "器械力量", True) / C.injury_risk(fresh(), "器械力量") - 3.0) < 1e-6 and e_hi["high"],
      f"力量收益 {gain(lo):.3f}→{gain(hi):.3f}，疲劳 {lo.fatigue:.0f}→{hi.fatigue:.0f}")
check("高强度只有训练和对练能选：休息、看录像、战术会议选了也按常规",
      not C.train(fresh(), "休息", high=True)["high"] and not C.train(fresh(), "看录像", high=True)["high"])
check("本来没风险的项目（跑步）练高强度也会有一点受伤风险", C.injury_risk(fresh(), "跑步") == 0 and C.injury_risk(fresh(), "跑步", True) > 0)
bold, timid = fresh(STOCKY), fresh(LANKY)                  # B 胆量、斗志 0.7，A 只有 0.4
edge = lambda f: C.score_activity(f, "打沙袋", None, 20, high=True) - C.score_activity(f, "打沙袋", None, 20)
check("胆大、好斗的更爱高强度（团队排程时把受伤和疲劳看得轻）", edge(bold) > edge(timid), f"高强度多出的分：胆大 {edge(bold):+.2f}，胆小 {edge(timid):+.2f}")

# ------------------------------------------------------------------ 对练真的跑强化学习
f = fresh()
before = [p.copy() for p in f.pi]
e = C.do_day(f, "对练·教练", seed=5)
changed = any(not np.array_equal(a, b) for a, b in zip(before, f.pi))
check("对练是真的训练：出招网络被更新", changed and e["note"]["iters"] >= 1, f"跑了 {e['note']['iters']} 轮")

# ------------------------------------------------------------------ 团队取舍
f = fresh()
f.knowledge["B"] = 1.0
opp = fresh(STOCKY)
only_cardio = {a: 0.0 for a in C.TRAINABLE} | {"体能": 1.0}
both = {a: 0.0 for a in C.TRAINABLE} | {"体能": 1.0, "力量": 1.0}
s = lambda act, need: C.score_activity(f, act, opp, 20, need)
check("取舍：只缺体能就专练（跑步），体能和力量都缺就选综合（循环训练）",
      s("跑步", only_cardio) > max(s("跳绳", only_cardio), s("循环训练", only_cardio))
      and s("循环训练", both) > max(s("跑步", both), s("器械力量", both)),
      f"只缺体能：跑步 {s('跑步', only_cardio):.2f} / 循环 {s('循环训练', only_cardio):.2f}；都缺：循环 {s('循环训练', both):.2f} / 跑步 {s('跑步', both):.2f}")

import dataclasses
calm = fresh(dataclasses.replace(LANKY, temperament={"斗志": 0.2}))
fierce = fresh(dataclasses.replace(LANKY, temperament={"斗志": 0.9}))
for x in (calm, fierce):
    x.fatigue = 65
    x.knowledge["B"] = 1.0
pick = lambda x: C.coach_choose(x, opp, 20, noise=0.0)[0]
check("性格：同样累，斗志低的选择歇，好斗的接着练（容易练过头）", pick(calm) == "休息" and pick(fierce) != "休息",
      f"疲劳 65：斗志低选 {pick(calm)}，好斗选 {pick(fierce)}")


# ------------------------------------------------------------------ 睡眠与饮食
f = fresh()
v0 = f.pts["力量"]
C.train(f, "器械力量")
mid = f.pts["力量"]
C.night(f, sleep=8, diet="维持")
check("训练收益要过夜才兑现：练完当天属性不变，睡一觉才涨", mid == v0 and f.pts["力量"] > v0 and f.pend_attr["力量"] > 0,
      f"白天练完 {v0:.3f}→{mid:.3f}，睡一觉后 {f.pts['力量']:.3f}")

def tech_after(sleep, diet="维持", nights=1):
    f = fresh()
    k = B.TECH_NAMES.index("前手勾拳·头")
    s0 = f.skills[k]
    C.train(f, "招式训练", techs=["前手勾拳·头"])
    for _ in range(nights):
        C.night(f, sleep=sleep, diet=diet)
    return f.skills[k] - s0
check("不睡觉学不进去：同样练招式，睡 5 小时兑现的熟练度远少于睡 8 小时", tech_after(5.0) < 0.4 * tech_after(8.0),
      f"睡 8 小时 +{tech_after(8.0):.4f}，睡 5 小时 +{tech_after(5.0):.4f}")

def strength_after(diet):
    f = fresh()
    v0 = f.pts["力量"]
    for _ in range(5):
        C.train(f, "器械力量")
        C.night(f, sleep=8, diet=diet)
        f.fatigue = 0
    return f.pts["力量"] - v0
check("狂练不吃，力量不长：减重饮食下力量涨得远少于吃够", strength_after("减重") < 0.65 * strength_after("维持"),
      f"吃够 +{strength_after('维持'):.3f}，减重 +{strength_after('减重'):.3f}")
check("饮食主要影响肌肉：技巧类熟练度不太受减重影响", tech_after(8.0, "减重") > 0.95 * tech_after(8.0, "维持"),
      f"招式熟练度 吃够 +{tech_after(8.0, '维持'):.4f}，减重 +{tech_after(8.0, '减重'):.4f}")

f = fresh()
C.train(f, "跑步")
p0 = f.pend_attr["体能"]
for _ in range(4):
    C.night(f, sleep=3.0)
check("没兑现的收益会慢慢淡掉：连着几晚不睡，练的就白练了", f.pend_attr["体能"] < 0.2 * p0, f"待兑现 {p0:.3f} → {f.pend_attr['体能']:.4f}")

def recover(sleep):
    f = fresh()
    f.fatigue = 70
    C.add_injury(f, 0.4, 10)
    C.train(f, "休息")
    C.night(f, sleep=sleep)
    return f.fatigue, f.injury_left
(fa8, il8), (fa5, il5) = recover(8.0), recover(5.0)
check("睡眠决定恢复：睡得少，疲劳和伤都恢复得慢", fa5 > fa8 and il5 > il8, f"睡 8 小时：疲劳 {fa8:.0f}、伤还剩 {il8:.1f} 天；睡 5 小时：疲劳 {fa5:.0f}、伤还剩 {il5:.1f} 天")

def mass_after(diet, days=10):
    f = fresh()
    for _ in range(days):
        C.train(f, "招式训练")
        C.night(f, diet=diet)
    return f.mass - LANKY.mass
check("饮食改变体重：增肌期体重上涨，减重期下降", mass_after("增肌") > 0.3 and mass_after("减重") < -0.5,
      f"10 天：增肌 {mass_after('增肌'):+.2f} 公斤，减重 {mass_after('减重'):+.2f} 公斤")

import bodies as BD
f = fresh()
f.mass = BD.WEIGHT_CLASS[2] + 1.0
w = C.weigh_in(f)
f2 = fresh()
check("称重超重：临时脱水掉秤，比赛状态下降；超太多就称重失败",
      w["over"] == 1.0 and not w["failed"] and f.form_mean() < f2.form_mean() - 0.25 and f.mass == BD.WEIGHT_CLASS[2],
      f"超重 1 公斤，状态 {f2.form_mean():+.2f} → {f.form_mean():+.2f}")

nerv = lambda poise: np.mean([C.night(fresh(dataclasses.replace(LANKY, temperament={"沉着": poise}), seed=s), days_left=1, sleep=8)["sleep"] for s in range(6)])
check("赛前紧张：沉着低的人赛前睡不好", nerv(0.2) < nerv(0.9) - 1.0, f"赛前一晚：沉着 0.9 睡 {nerv(0.9):.1f} 小时，沉着 0.2 睡 {nerv(0.2):.1f} 小时")

def spar_change(sleep_q):
    f = fresh()
    f.last_sleep_q = sleep_q
    p0 = [p.copy() for p in f.pi]
    C.spar(f, (BD.SPARRING, __import__("boxing_bots").BoxingSparring(), np.full(B.N_TECH, 0.5), "教练"), iters=2, seed=3)
    return float(sum(np.abs(a - b).sum() for a, b in zip(f.pi, p0)))
check("没睡好，对练学不进去：前一晚没睡，出招网络的更新幅度小得多", spar_change(0.0) < 0.6 * spar_change(1.0),
      f"睡好 {spar_change(1.0):.3f}，没睡 {spar_change(0.0):.3f}")


if __name__ == "__main__":
    ok = sum(r[1] for r in RESULTS)
    for name, passed, detail in RESULTS:
        print(("✓ " if passed else "✗ ") + name + (f"  —— {detail}" if detail else ""))
    print(f"\n{ok}/{len(RESULTS)} 通过")
    raise SystemExit(0 if ok == len(RESULTS) else 1)
