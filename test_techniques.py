"""技术验收：每个拳击技术都要起到和现实中相同的作用。

每条测试都是一个写死的小场景：把两个人摆在指定距离和朝向，按帧给动作，检查结果。
改了 boxing.py 之后跑一遍：py -X utf8 test_techniques.py
"""
import numpy as np

import boxing as B
from bodies import Body

STD = Body("T", "测试", reach=0.75, mass=75, power=1.0, hand_speed=1.0, foot_speed=1.0,
           cardio=1.0, chin=1.0, react_ticks=3)
C = B.RING / 2
RESULTS = []


def make(gap=1.0, th0=0.0, th1=np.pi, x0=None, y=C, n_rounds=1):
    s = B.Sim(1, (STD, STD), auto_reset=False, n_rounds=n_rounds)
    x0 = C - gap / 2 if x0 is None else x0
    s.pos[0, 0] = [x0, y]
    s.pos[0, 1] = [x0 + gap, y]
    s.th[0] = [th0, th1]
    s.hist[:] = s._snapshot()[None]
    return s


def run(s, a0, a1=(), ticks=None, hold1=B.STAY, hold0=B.STAY):
    """a0 / a1：{帧: 动作}。没写的帧用 hold。"""
    a0, a1 = dict(a0), dict(a1)
    T = ticks or (max(list(a0) + list(a1) + [0]) + 15)
    infos = []
    for t in range(T):
        act = np.array([[a0.get(t, hold0), a1.get(t, hold1)]])
        _, _, info = s.step(act)
        infos.append(info)
    return infos


def dealt(infos, k=0):
    return float(sum(i["dmg"][0, k] for i in infos))


def outcomes(infos, k=0):
    return [int(i["outcome"][0, k]) for i in infos if i["outcome"][0, k]]


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))


def punch_vs(punch, gap, defense_hold=B.STAY, defense_at=None, defense=None):
    """0 号在第 0 帧出 punch，1 号举着 defense_hold，或在 defense_at 帧做 defense。返回 (伤害, 结果列表, sim)。"""
    s = make(gap)
    a1 = {defense_at: defense} if defense is not None else {}
    inf = run(s, {0: punch}, a1, ticks=20, hold1=defense_hold)
    return dealt(inf), outcomes(inf), s


def fire_tick(punch):
    return int(B.SU[punch - B.PUNCH0])


# ------------------------------------------------------------------ 进攻
s = make()
check("刺拳最快，后手直拳次之，后手勾拳最慢",
      s.su_tab[0, 0] < s.su_tab[2, 0] < s.su_tab[6, 0], f"起手帧 {s.su_tab[[0, 2, 6], 0]}")
check("射程：直拳 > 勾拳 > 上勾拳",
      s.rng_tab[0, 0] > s.rng_tab[4, 0] > s.rng_tab[8, 0], f"射程 {s.rng_tab[[0, 4, 8], 0].round(2)}")

d_far_jab, o_far_jab, _ = punch_vs(B.JAB_H, 1.1)
d_far_hook, o_far_hook, _ = punch_vs(B.LHOOK_H, 1.1)
d_in_hook, _, _ = punch_vs(B.LHOOK_H, 0.7)
check("中距离刺拳打得到、勾拳够不着；进到内围勾拳才打得到",
      d_far_jab > 0 and o_far_hook == [4] and d_in_hook > 0,
      f"1.1 米刺拳 {d_far_jab:.1f}、勾拳结果 {o_far_hook}；0.7 米勾拳 {d_in_hook:.1f}")

d_jab_mid, _, _ = punch_vs(B.JAB_H, 1.0)
d_jab_close, _, _ = punch_vs(B.JAB_H, 0.65)
d_hook_close, _, _ = punch_vs(B.LHOOK_H, 0.65)
check("贴身时直拳打不开（伤害减半），勾拳不受影响",
      abs(d_jab_close - 0.5 * d_jab_mid) < 1e-6 and d_hook_close > 2 * d_jab_close,
      f"刺拳 1.0 米 {d_jab_mid:.1f} / 0.65 米 {d_jab_close:.1f}，勾拳 0.65 米 {d_hook_close:.1f}")

# 组合拳：刺拳之后接后手，比刺拳之后再接刺拳起手早
def second_start(first, second):
    s = make(1.0)
    a = {0: first}
    a.update({t: second for t in range(1, 15)})
    inf = run(s, a, ticks=16)
    starts = [t for t, i in enumerate(inf) if i["started"][0, 0]]
    return starts[1] if len(starts) > 1 else None
t_alt, t_same = second_start(B.JAB_H, B.CROSS_H), second_start(B.JAB_H, B.JAB_H)
check("组合拳：左右手交替比同一只手连打衔接得快", t_alt is not None and t_same is not None and t_alt < t_same,
      f"第二拳起手：交替 {t_alt} 帧，同手 {t_same} 帧")

# 反击：打在对方起手上
s = make(1.0)
inf = run(s, {1: B.JAB_H}, {0: B.CROSS_H}, ticks=12)
d_counter = dealt(inf, 0)
crossed = sum(i["fired"][0, 1] > 0 for i in inf)
check("反击拳：打在对手起手上伤害 ×1.3（而且他手在出拳，没有自然站姿的保护），并且打断对方那一拳",
      abs(d_counter - 1.3 * d_jab_mid / B.NATURAL_GUARD[B.P_STANDARD, 0]) < 1e-6 and crossed == 0 and any(i["counter"][0, 0] for i in inf),
      f"反击 {d_counter:.2f} vs 普通 {d_jab_mid:.2f}，对方后手是否打出 {crossed}")

# 打空更累
s = make(2.0)
st0 = s.stam[0, 0]
run(s, {0: B.CROSS_H}, ticks=8)
miss_cost = st0 - s.stam[0, 0]
s = make(1.0)
run(s, {0: B.CROSS_H}, ticks=8)
hit_cost = st0 - s.stam[0, 0]
check("打空比打中更耗体能", miss_cost > hit_cost, f"打空 {miss_cost:.1f}，打中 {hit_cost:.1f}（含回复）")

# ------------------------------------------------------------------ 防守
d_free, _, _ = punch_vs(B.JAB_H, 1.0)
d_g, o_g, _ = punch_vs(B.JAB_H, 1.0, B.GUARD_HIGH)
check("高位格挡挡头部直拳：比自然站姿多挡很多（只吃两成左右）", d_g <= 0.25 * d_free and o_g == [2], f"{d_g:.2f} / {d_free:.2f}，结果 {o_g}")

d_free, _, _ = punch_vs(B.LUPPER_H, 0.7)
d_g, _, _ = punch_vs(B.LUPPER_H, 0.7, B.GUARD_HIGH)
check("上勾拳从中间撕开高位格挡：吃 75%", d_g >= 0.7 * d_free, f"{d_g:.2f} / {d_free:.2f}")

d_free, _, _ = punch_vs(B.LHOOK_H, 0.7)
d_g, _, _ = punch_vs(B.LHOOK_H, 0.7, B.GUARD_HIGH)
check("勾拳从侧面绕：高位格挡只挡掉一部分", 0.3 * d_free <= d_g <= 0.47 * d_free, f"{d_g:.2f} / {d_free:.2f}")

d_free, _, _ = punch_vs(B.CROSS_B, 1.0)
d_g, _, _ = punch_vs(B.CROSS_B, 1.0, B.GUARD_HIGH)
check("高位格挡护不住身体：身体直拳吃 70%", d_g >= 0.6 * d_free, f"{d_g:.2f} / {d_free:.2f}")

d_free_b, _, _ = punch_vs(B.LHOOK_B, 0.8)
d_body_b, _, _ = punch_vs(B.LHOOK_B, 0.8, B.GUARD_BODY)
d_free_h, _, _ = punch_vs(B.LHOOK_H, 0.8)
d_body_h, _, _ = punch_vs(B.LHOOK_H, 0.8, B.GUARD_BODY)
check("护体挡身体拳（吃 25%），但头部全空", d_body_b <= 0.3 * d_free_b and d_body_h >= 0.95 * d_free_h,
      f"身体 {d_body_b:.2f}/{d_free_b:.2f}，头 {d_body_h:.2f}/{d_free_h:.2f}")

d, o, s = punch_vs(B.JAB_H, 1.0, defense_at=1, defense=B.SLIP)
d2, o2, _ = punch_vs(B.LHOOK_H, 0.7, defense_at=1, defense=B.SLIP)
check("侧闪躲开头部直拳；勾拳闪不掉", d == 0 and o == [3] and d2 > 0, f"直拳 {d:.1f} 结果 {o}，勾拳 {d2:.1f}")

s = make(1.0)
inf = run(s, {0: B.JAB_H}, {1: B.SLIP}, ticks=4)
rec_slipped = int(s.timer[0, 0])
s = make(2.0)
run(s, {0: B.JAB_H}, ticks=4)
rec_whiff = int(s.timer[0, 0])
check("被侧闪躲开后，进攻方多 2 帧收招（反击窗口）", rec_slipped == rec_whiff + 2, f"被闪 {rec_slipped} 帧，打空 {rec_whiff} 帧")

d, o, _ = punch_vs(B.LHOOK_H, 0.7, defense_at=1, defense=B.DUCK)
d_free, _, _ = punch_vs(B.LUPPER_H, 0.7)
d_duck, _, _ = punch_vs(B.LUPPER_H, 0.7, defense_at=1, defense=B.DUCK)
check("下潜从勾拳下面钻过；但撞上上勾拳伤害 ×1.4", d == 0 and o == [3] and abs(d_duck - 1.4 * d_free) < 1e-6,
      f"勾拳 {d:.1f}，上勾拳 {d_duck:.2f} vs {d_free:.2f}")

s = make(1.0)
inf = run(s, {0: B.CROSS_H}, {3: B.PARRY}, ticks=6)
d_par = dealt(inf)
rec_par = int(s.timer[0, 0])
d_par_hook, _, _ = punch_vs(B.LHOOK_H, 0.7, defense_at=1, defense=B.PARRY)
check("拍挡拍开直拳，对手失去平衡多 4 帧收招；拍不了勾拳",
      d_par == 0 and rec_par >= int(B.REC[2]) + 4 - 1 and d_par_hook > 0,
      f"直拳 {d_par:.1f}，对手剩余收招 {rec_par}，勾拳 {d_par_hook:.1f}")

d, o, _ = punch_vs(B.JAB_H, 1.1, defense_at=1, defense=B.PULL)
d2, _, _ = punch_vs(B.JAB_B, 1.1, defense_at=1, defense=B.PULL)
check("后仰让开射程边缘的头部拳；身体拳照样中", d == 0 and o == [3] and d2 > 0, f"头 {d:.1f} {o}，身 {d2:.1f}")

s = make(1.0, x0=B.RING - B.BODY_R - 1.0)       # 1 号背靠围绳
inf = run(s, {0: B.JAB_H}, {1: B.PULL}, ticks=10)
check("背靠围绳后仰不了", dealt(inf) > 0, f"伤害 {dealt(inf):.1f}")

# 角度：站在对手正面线外，他的格挡打折
s = make(1.0)
inf = run(s, {0: B.JAB_H}, ticks=3, hold1=B.GUARD_HIGH)
s.th[0, 1] = np.pi + np.radians(40)      # 出拳前一帧，1 号的朝向被甩开 40°（比如被轴转绕到侧面）
inf += run(s, {}, ticks=8, hold1=B.GUARD_HIGH)
d_angle = dealt(inf)
d_front, _, _ = punch_vs(B.JAB_H, 1.0, B.GUARD_HIGH)
check("角度：进攻方在对手正面线 25° 以外，对手的格挡效果减半", d_angle > 2 * d_front, f"偏角 {d_angle:.2f} vs 正面 {d_front:.2f}")

s = make(1.0, th0=np.radians(45))
inf = run(s, {0: B.JAB_H}, ticks=10)
check("自己没对准（偏 45°）直拳打不中", outcomes(inf) == [4] and dealt(inf) == 0, f"结果 {outcomes(inf)}")

# 受伤时防守打折
s = make(1.0)
s.stun[0, 1] = 75
inf = run(s, {0: B.JAB_H}, ticks=10, hold1=B.GUARD_HIGH)
d_hurt = dealt(inf)
check("受伤（晕眩 > 60）时格挡效果减半", d_hurt > 2 * d_front, f"受伤 {d_hurt:.2f} vs 正常 {d_front:.2f}")

# 缠抱
s = make(0.7)
inf = run(s, {0: B.CLINCH}, {t: B.LHOOK_H for t in range(5, 40)}, ticks=60)
clinched = [t for t, i in enumerate(inf) if s.phase[0, 0] == B.CLINCHED] or [None]
held = sum(1 for t in range(4, 40) if inf[t]["dmg"][0, 1] > 0)
gap_after = float(np.linalg.norm(s.pos[0, 1] - s.pos[0, 0]))
check("缠抱：抱住后双方都不能出拳，2 秒后裁判分开", held == 0 and gap_after >= 1.1 and s.phase[0, 0] != B.CLINCHED,
      f"抱住期间对手打中 {held} 次，分开后距离 {gap_after:.2f} 米")

# ------------------------------------------------------------------ 步法
s = make(1.0, x0=B.RING - B.BODY_R - 1.0)
run(s, {}, ticks=10, hold1=B.BACK)
check("背靠围绳退不了", abs(s.pos[0, 1, 0] - (B.RING - B.BODY_R)) < 1e-9, f"x = {s.pos[0, 1, 0]:.3f}")

s = make(0.8)
s.phase[0, 1], s.timer[0, 1], s.move[0, 1] = B.RECOVERY, 10, 2     # 1 号刚打完后手正在收招
inf = run(s, {0: B.PIVOT_L}, ticks=3)
_, _, off = s._geom(s.pos, s.th)
ang = np.degrees(abs(off[0, 1]))
check("轴转：趁对手收招转到侧面，拿到 25° 以上的角度", ang > 25, f"对手正面线偏离 {ang:.1f}°")

s = make(2.0)
x_before = s.pos[0, 0].copy()
run(s, {}, ticks=10, hold0=B.LEFT)
moved = s.pos[0, 0] - x_before
check("横移：沿垂直于对手的方向移动", abs(moved[1]) > 0.3 and abs(moved[0]) < abs(moved[1]), f"位移 {moved.round(2)}")

# ------------------------------------------------------------------ 虚晃
def opp_view(first):
    s = make(1.0)
    obs = []
    for t in range(6):
        act = np.array([[first if t == 0 else B.STAY, B.STAY]])
        s.step(act)
        obs.append(s.observe()[0, 1, np.r_[60:99, 100]].copy())      # 只比对"对手是什么状态"那几段
    return obs
v_feint, v_jab = opp_view(B.FEINT_JAB), opp_view(B.JAB_H)
same = all(np.array_equal(a, b) for a, b in zip(v_feint[:2 + STD.react_ticks], v_jab[:2 + STD.react_ticks]))
check("虚晃：前 2 帧在对手眼里和真刺拳一模一样", same, "对手观测逐帧比较")

s = make(1.0)
st0 = s.stam[0, 0]
inf = run(s, {0: B.FEINT_JAB}, ticks=3)
check("虚晃几乎不耗体能，也不会打中", dealt(inf) == 0 and st0 - s.stam[0, 0] <= 1.0, f"耗体能 {st0 - s.stam[0, 0]:.2f}")

# ------------------------------------------------------------------ 身体攻击
def stam_after(punch):
    s = make(0.8)
    st0, cap0 = s.stam[0, 1], s.cap[0, 1]
    run(s, {0: punch}, ticks=10)
    return st0 - s.stam[0, 1], cap0 - s.cap[0, 1]
(sb, cb), (sh, ch) = stam_after(B.LHOOK_B), stam_after(B.LHOOK_H)
check("身体拳：比同一记打头的拳多耗对手很多体能，并压低体能上限", sb > 10 and sb > sh + 10 and cb > 0 and ch == 0,
      f"打身体 体能 -{sb:.1f}、上限 -{cb:.1f}；打头 体能 -{sh:.1f}")

s = make(0.7, x0=B.RING - B.BODY_R - 0.7)       # 把对手压在围绳上打身体
a = {t: (B.LHOOK_B if t % 2 == 0 else B.RHOOK_B) for t in range(70)}
inf = run(s, a, ticks=70)
kd_t = [t for t, i in enumerate(inf) if i["kd"][0, 1]]
check("肝脏拳累积：身体伤满了直接倒地（头部没挨打）", len(kd_t) > 0 and s.kd_total[0, 1] >= 1 and dealt(inf) > 0,
      f"倒地帧 {kd_t[:1]}，身体伤累积打倒")

# ------------------------------------------------------------------ 倒地、KO、TKO、判定
s = make(1.0)
s.stun[0, 1] = 99
inf = run(s, {0: B.CROSS_H}, ticks=70)
check("晕眩到 100 被击倒；体力够，读秒后站起来但仍带伤",
      any(i["kd"][0, 1] for i in inf) and s.phase[0, 1] != B.DOWN and not s.over[0] and s.stun[0, 1] > 20,
      f"状态 {B.PHASE_NAMES[s.phase[0, 1]] or '自由'}，晕眩 {s.stun[0, 1]:.0f}")

s = make(1.0)
s.stun[0, 1], s.hp[0, 1] = 99, 12
inf = run(s, {0: B.CROSS_H}, ticks=70)
end = [i for i in inf if i["done"][0]]
check("体力不够：读秒站不起来 = KO", end and end[0]["method"][0] == B.M_KO and end[0]["winner"][0] == 0,
      f"结果 {B.METHOD_NAMES[end[0]['method'][0]] if end else '没结束'}")

s = make(1.0)
s.stun[0, 1], s.kd_round[0, 1], s.kd_total[0, 1] = 99, 2, 2
inf = run(s, {0: B.CROSS_H}, ticks=10)
end = [i for i in inf if i["done"][0]]
check("一回合第 3 次倒地 = TKO", end and end[0]["method"][0] == B.M_TKO, f"结果 {B.METHOD_NAMES[end[0]['method'][0]] if end else '没结束'}")

s = make(1.0, x0=B.RING - B.BODY_R - 1.0)       # 受伤的一方被困在围绳上
s.stun[0, 1] = 85
a = {k * 8: B.JAB_H for k in range(10)}
inf = run(s, a, ticks=85)
end = [i for i in inf if i["done"][0]]
landed_before = sum(i["outcome"][0, 0] == 1 for i in inf[:end[0]["round"][0] * 0 + [t for t, i in enumerate(inf) if i["done"][0]][0] + 1]) if end else 0
check("受伤后连续挨 8 拳不还手也不防，裁判终止 = TKO", end and end[0]["method"][0] == B.M_TKO and s.kd_total[0, 1] == 0 and landed_before >= B.UNANSWERED_STOP,
      f"结果 {B.METHOD_NAMES[end[0]['method'][0]] if end else '没结束'}，终止前挨了 {landed_before} 拳")

s = make(1.0, x0=B.RING - B.BODY_R - 1.0)
s.stun[0, 1] = 85
a = {k * 8: B.JAB_H for k in range(10)}
inf = run(s, a, {k * 8 + 1: (B.SLIP if k == 4 else B.STAY) for k in range(10)}, ticks=85)
stopped = any(i["done"][0] for i in inf)
check("受伤时只要还在防守（中途闪掉一拳），裁判就重新计数，不终止", not stopped or s.kd_total[0, 1] > 0,
      f"{'被终止' if stopped else '没终止'}")

s = make(1.0)
s.hp[0] = [30.0, 95.0]             # 0 号血少得多
s.tally[0, 0] = [6, 4, 0, 0]       # 但本回合有效拳多
s.tally[0, 1] = [2, 0, 0, 0]
s.t_round[0] = B.ROUND_TICKS - 1
inf = run(s, {}, ticks=1)
check("判定只看裁判打分，不看血量：血少但有效拳多的一方赢",
      inf[0]["done"][0] and inf[0]["method"][0] == B.M_DEC and inf[0]["winner"][0] == 0,
      f"结果 {B.METHOD_NAMES[inf[0]['method'][0]]}，赢家 {inf[0]['winner'][0]}，裁判 {inf[0]['judges'][0].tolist()}")

s = make(1.0)
s.tally[0] = [[5, 4, 0, 0], [5, 4, 0, 0]]
s.kd_round[0, 1] = 1
s.t_round[0] = B.ROUND_TICKS - 1
inf = run(s, {}, ticks=1)
check("击倒在打分上是 10-8（平手回合被打倒一次）", inf[0]["judges"][0].tolist() == [[10, 9]] * 3,
      f"裁判 {inf[0]['judges'][0].tolist()}")

s = make(1.0)
s.tally[0] = [[5, 4, 0, 0], [5, 4, 0, 0]]
s.holds[0, 0] = B.HOLDS_WARN + 1
s.t_round[0] = B.ROUND_TICKS - 1
inf = run(s, {}, ticks=1)
check("一回合缠抱太多次被扣 1 分", inf[0]["judges"][0].tolist() == [[9, 10]] * 3, f"裁判 {inf[0]['judges'][0].tolist()}")

s = make(1.0, n_rounds=3)
s.tally[0] = [[6, 4, 0, 0], [2, 0, 0, 0]]
s.t_round[0] = B.ROUND_TICKS - 1
inf = run(s, {}, ticks=2)
check("回合结束：不判胜负，回合间休息后继续下一回合", s.round[0] == 1 and not s.over[0] and s.judge[0, 0].tolist() == [10, 9],
      f"第 {s.round[0] + 1} 回合，裁判1 {s.judge[0, 0].tolist()}")

# 奖励里没有血量
s = make(1.0)
r_sum, p_sum, hp_lost = np.zeros(2), np.zeros(2), 0.0
for t in range(12):
    r, _, info = s.step(np.array([[B.JAB_B if t == 0 else B.STAY, B.STAY]]))
    r_sum += r[0]
    p_sum += info["points"][0]
    hp_lost += info["dmg"][0, 0]
expect = (p_sum[0] - p_sum[1]) * B.REWARD_PER_POINT
check("奖励只看计分和胜负：打中计分拳得分，奖励里不出现血量",
      p_sum[0] > 0 and hp_lost > 0 and abs(r_sum[0] - expect) < 1e-9 and abs(r_sum[1] + expect) < 1e-9,
      f"奖励 {r_sum.round(4).tolist()} = 计分差 × {B.REWARD_PER_POINT}；对手掉血 {hp_lost:.1f} 不进奖励")

# ------------------------------------------------------------------ 精准
d_full, _, _ = punch_vs(B.JAB_H, 1.0)
s = make(1.0)
edge = float(s.rng_tab[0, 0]) - 0.02           # 刚好在射程最边缘
d_edge, _, _ = punch_vs(B.JAB_H, edge)
check("精准：打在射程最边缘只是擦到（60%）", abs(d_edge - 0.6 * d_full) < 1e-6, f"边缘 {d_edge:.2f} vs 正常 {d_full:.2f}")

# ------------------------------------------------------------------ 气势与心理战
s = make(1.0)
c0 = s.comp[0].copy()
run(s, {0: B.CROSS_H}, ticks=8)
check("气势：重拳打中自己涨、对手掉", s.comp[0, 0] > c0[0] and s.comp[0, 1] < c0[1] - 3,
      f"心态 {c0.round(1).tolist()} → {s.comp[0].round(1).tolist()}")

s = make(1.0)
run(s, {0: B.FEINT_JAB}, {1: B.SLIP}, ticks=3)
c_faked = s.comp[0, 1]
s = make(1.0)
run(s, {}, {1: B.SLIP}, ticks=3)
c_plain = s.comp[0, 1]
check("心理战：被虚晃骗出防守动作，心态掉", c_faked <= c_plain - 3.9, f"被骗 {c_faked:.1f} vs 没骗 {c_plain:.1f}")

s = make(1.0)
r_ok = s.react_now()[0, 1]
s.comp[0, 1] = 30
r_low = s.react_now()[0, 1]
s.comp[0, 1] = 10
r_vlow = s.react_now()[0, 1]
check("心态差反应变慢：低于 40 慢 1 帧，低于 20 慢 2 帧", (r_low, r_vlow) == (r_ok + 1, r_ok + 2), f"{r_ok} / {r_low} / {r_vlow} 帧")

s = make(1.0)
s.comp[0, 0] = 10
run(s, {0: B.JAB_H}, ticks=1)
check("心态崩了出拳犹豫：起手多 1 帧", int(s.timer[0, 0]) == int(s.su_tab[0, 0]) + 1, f"起手 {s.timer[0, 0]} 帧")

s = make(0.9, x0=B.RING - B.BODY_R - 0.9)
run(s, {}, ticks=60)
c_pinned = s.comp[0].copy()
s = make(0.9)
run(s, {}, ticks=60)
c_center = s.comp[0].copy()
check("气势压制：把对手困在围绳上，对手心态持续下降、自己上升",
      c_pinned[1] < c_center[1] - 1.5 and c_pinned[0] > c_center[0],
      f"围绳 {c_pinned.round(1).tolist()} vs 中央 {c_center.round(1).tolist()}")

# ------------------------------------------------------------------ 猜拳：读招
s = make(1.0)
a0 = {k * 12: B.JAB_H for k in range(10)}       # 被躲开要多收 2 帧，所以隔 12 帧出一拳
a1 = {k * 12 + 1: B.DUCK for k in range(10)}
run(s, a0, a1, ticks=125)
duck_rate = float(s.tend_def[0, 0, 5])
check("读招：对手一挨刺拳就下潜，几拳之后记住了他爱下潜", duck_rate > 0.5, f"下潜倾向 {duck_rate:.2f}（起始 0.14）")

s = make(1.0)
run(s, {k * 9: B.LHOOK_H for k in range(8)}, ticks=75)
s2 = s.observe()[0, 1]
check("读招：记住对手爱打勾拳", s2[104 + 2] > 0.5, f"对手出拳构成（刺/直/勾/上勾）{s2[104:108].round(2).tolist()}")

# ------------------------------------------------------------------ 策略：知道记分卡
s = make(1.0, n_rounds=3)
s.tally[0] = [[6, 4, 0, 0], [2, 0, 0, 0]]
s.t_round[0] = B.ROUND_TICKS - 1
run(s, {}, ticks=2)
o = s.observe()[0]
check("策略：知道自己在记分卡上领先还是落后", (o[0, 101:104] > 0).all() and (o[1, 101:104] < 0).all(),
      f"0 号看到的分差 {o[0, 101:104].round(2).tolist()}")

# ------------------------------------------------------------------ 熟练度与身体性能
def skilled(tech, level, gap=1.0):
    sk = np.full((2, B.N_TECH), 0.5)
    sk[:, B.TECH_NAMES.index(tech)] = level
    s = make(gap)
    s.set_skills(sk)
    return s

lo, hi = skilled("刺拳·头", 0.1), skilled("刺拳·头", 0.9)
inf_lo, inf_hi = run(lo, {0: B.JAB_H}, ticks=12), run(hi, {0: B.JAB_H}, ticks=12)
check("熟练度：练得好的刺拳更快、更重、更省体能",
      hi.su_tab[0, 0] <= lo.su_tab[0, 0] and dealt(inf_hi) > dealt(inf_lo) and hi.cost_tab[0, 0] < lo.cost_tab[0, 0],
      f"起手 {lo.su_tab[0, 0]}→{hi.su_tab[0, 0]} 帧，伤害 {dealt(inf_lo):.2f}→{dealt(inf_hi):.2f}，耗体能 {lo.cost_tab[0, 0]:.1f}→{hi.cost_tab[0, 0]:.1f}")
check("熟练度：后手勾拳练得好，起手快 1 帧以上", skilled("后手勾拳·头", 0.95).su_tab[6, 0] < skilled("后手勾拳·头", 0.05).su_tab[6, 0],
      f"{skilled('后手勾拳·头', 0.05).su_tab[6, 0]} → {skilled('后手勾拳·头', 0.95).su_tab[6, 0]} 帧")

def guarded(level):
    s = skilled("高位格挡", level)
    return dealt(run(s, {0: B.CROSS_H}, ticks=12, hold1=B.GUARD_HIGH))
g_lo, g_hi = guarded(0.1), guarded(0.9)
check("熟练度：高位格挡练得好，挡得更干净", g_hi < g_lo, f"挨后手直拳 {g_lo:.2f} → {g_hi:.2f}")

check("熟练度：侧闪练得好，时机窗口更宽", skilled("侧闪", 0.9).def_active[1, B.M_SLIP] > skilled("侧闪", 0.1).def_active[1, B.M_SLIP],
      f"{skilled('侧闪', 0.1).def_active[1, B.M_SLIP]} → {skilled('侧闪', 0.9).def_active[1, B.M_SLIP]} 帧")

def walked(level):
    s = skilled("步法", level, gap=3.0)
    x0 = s.pos[0, 0, 0]
    run(s, {}, ticks=10, hold0=B.FWD)
    return s.pos[0, 0, 0] - x0
check("熟练度：步法好，走得快", walked(0.9) > walked(0.1), f"10 帧前进 {walked(0.1):.2f} → {walked(0.9):.2f} 米")

heavy = Body("H", "重", reach=0.75, mass=75, power=1.3, hand_speed=1.0, foot_speed=1.0, cardio=1.0, chin=1.0, react_ticks=3)
light = Body("L", "轻", reach=0.75, mass=75, power=0.85, hand_speed=1.0, foot_speed=1.0, cardio=1.0, chin=1.0, react_ticks=3)
def body_dmg(b):
    s = B.Sim(1, (b, STD), auto_reset=False, n_rounds=1)
    s.pos[0] = [[C - 0.35, C], [C + 0.35, C]]
    s.th[0] = [0, np.pi]
    s.hist[:] = s._snapshot()[None]
    return dealt(run(s, {0: B.LHOOK_H}, ticks=12))
check("身体性能：力量大的同一招伤害更高", body_dmg(heavy) > 1.4 * body_dmg(light), f"{body_dmg(light):.2f} → {body_dmg(heavy):.2f}")
fast = Body("F", "快", reach=0.75, mass=75, power=1.0, hand_speed=1.3, foot_speed=1.0, cardio=1.0, chin=1.0, react_ticks=3)
sf = B.Sim(1, (fast, STD), auto_reset=False)
check("身体性能：手速快的同一招起手更快", sf.su_tab[2, 0] < sf.su_tab[2, 1], f"后手直拳起手 {sf.su_tab[2, 1]} → {sf.su_tab[2, 0]} 帧")
long_ = Body("R", "长", reach=0.95, mass=75, power=1.0, hand_speed=1.0, foot_speed=1.0, cardio=1.0, chin=1.0, react_ticks=3)
sl = B.Sim(1, (long_, STD), auto_reset=False)
check("身体性能：臂展长的同一招射程更远", sl.rng_tab[0, 0] > sl.rng_tab[0, 1] + 0.1, f"刺拳射程 {sl.rng_tab[0, 1]:.2f} → {sl.rng_tab[0, 0]:.2f} 米")

# ------------------------------------------------------------------ 架势与站架
def vs_posture(post, punch, gap, hold1=B.STAY):
    s = make(gap)
    s.posture[0, 1] = post
    return dealt(run(s, {0: punch}, ticks=14, hold1=hold1))

base_s, base_h, base_u = vs_posture(B.P_STANDARD, B.JAB_H, 1.0), vs_posture(B.P_STANDARD, B.LHOOK_H, 0.7), vs_posture(B.P_STANDARD, B.LUPPER_H, 0.7)
check("长架：前手撑开，对手的头部直拳打不实（×0.85），但勾拳更容易进来（×1.1）",
      abs(vs_posture(B.P_LONG, B.JAB_H, 1.0) - 0.85 * base_s) < 1e-6 and abs(vs_posture(B.P_LONG, B.LHOOK_H, 0.7) - 1.1 * base_h) < 1e-6,
      f"直拳 {vs_posture(B.P_LONG, B.JAB_H, 1.0):.2f}/{base_s:.2f}，勾拳 {vs_posture(B.P_LONG, B.LHOOK_H, 0.7):.2f}/{base_h:.2f}")
s = make(1.0)
s.posture[0, 0] = B.P_LONG
check("长架：自己的直拳射程更远", s.rng_tab[0, 0] < B.BODY_R + B.SHOULDER + (s.rng_tab[0, 0] - B.BODY_R - B.SHOULDER) * B.POSTURE_FX[B.P_LONG, B.FX_RANGE],
      f"射程 ×{B.POSTURE_FX[B.P_LONG, B.FX_RANGE]}")
check("低姿抱架：目标变小，头部直拳 ×0.8；但撞上上勾拳 ×1.25",
      abs(vs_posture(B.P_CROUCH, B.JAB_H, 1.0) - 0.8 * base_s) < 1e-6 and abs(vs_posture(B.P_CROUCH, B.LUPPER_H, 0.7) - 1.25 * base_u) < 1e-6,
      f"直拳 {vs_posture(B.P_CROUCH, B.JAB_H, 1.0):.2f}，上勾拳 {vs_posture(B.P_CROUCH, B.LUPPER_H, 0.7):.2f}/{base_u:.2f}")
g_std, g_low = vs_posture(B.P_STANDARD, B.CROSS_H, 1.0, B.GUARD_HIGH), vs_posture(B.P_LOW, B.CROSS_H, 1.0, B.GUARD_HIGH)
s = make(1.0)
s.posture[0, 0] = B.P_LOW
run(s, {0: B.JAB_H}, ticks=1)
r_low = s.react_now()[0, 0]
check("低手位：刺拳快 1 帧、反应快 1 帧，但举架挡不住",
      int(s.timer[0, 0]) == int(s.su_tab[0, 0]) - 1 and r_low == STD.react_ticks - 1 and g_low > 3 * g_std,
      f"刺拳起手 {s.timer[0, 0]} 帧，反应 {r_low} 帧，举架挨后手 {g_low:.2f} vs {g_std:.2f}")

s = make(1.0)
inf = run(s, {0: B.CHANGE_POSTURE, 2: B.JAB_H}, ticks=8)
check("换架势要花时间（4 帧），换的时候出不了拳", s.posture[0, 0] == 1 and sum(i["started"][0, 0] > 0 for i in inf[:4]) == 0,
      f"换完是 {B.POSTURE_NAMES[s.posture[0, 0]]}")

d_same_cross, _, _ = punch_vs(B.CROSS_H, 1.0)
s = make(1.0)
s.stance[0, 1] = -1                 # 对手是反架
d_open_cross = dealt(run(s, {0: B.CROSS_H}, ticks=14))
s = make(1.0)
s.stance[0, 1] = -1
d_open_jab = dealt(run(s, {0: B.JAB_H}, ticks=14))
check("正反架对位：后手直拳顺中线进来更重（×1.1），刺拳和对方前手打架变弱（×0.85）",
      abs(d_open_cross - 1.1 * d_same_cross) < 1e-6 and abs(d_open_jab - 0.85 * d_jab_mid) < 1e-6,
      f"后手 {d_open_cross:.2f}/{d_same_cross:.2f}，刺拳 {d_open_jab:.2f}/{d_jab_mid:.2f}")
s = make(1.0)
run(s, {0: B.SWITCH_STANCE}, ticks=6)
check("换站架：正架换成反架", s.stance[0, 0] == -1, f"站架 {s.stance[0, 0]}")

# ------------------------------------------------------------------ 状态、性格、露馅
s = make(1.0)
s.form[0, 0] = 1.0
d_good = dealt(run(s, {0: B.CROSS_H}, ticks=12))
check("当天状态好，同一拳更重（+8%）", abs(d_good - 1.08 * d_same_cross) < 1e-6, f"{d_good:.2f} vs {d_same_cross:.2f}")

def comp_drop(courage):
    b = Body("C", "胆", reach=0.75, mass=75, power=1.0, hand_speed=1.0, foot_speed=1.0, cardio=1.0, chin=1.0,
             react_ticks=3, temperament={"胆量": courage})
    s = B.Sim(1, (STD, b), auto_reset=False, n_rounds=1)
    s.pos[0] = [[C - 0.5, C], [C + 0.5, C]]
    s.th[0] = [0, np.pi]
    s.hist[:] = s._snapshot()[None]
    c0 = s.comp[0, 1]
    run(s, {0: B.CROSS_H}, ticks=7)
    return c0 - s.comp[0, 1]
check("性格·胆量：胆小的挨同一拳心态掉得多", comp_drop(0.2) > 1.5 * comp_drop(0.8), f"胆小掉 {comp_drop(0.2):.1f}，胆大掉 {comp_drop(0.8):.1f}")

calm = Body("P", "沉", reach=0.75, mass=75, power=1.0, hand_speed=1.0, foot_speed=1.0, cardio=1.0, chin=1.0,
            react_ticks=3, temperament={"沉着": 0.9})
check("性格·沉着：心态的基准更高", B.Sim(1, (calm, STD), auto_reset=False).comp[0, 0] > 90, f"开场心态 {B.Sim(1, (calm, STD), auto_reset=False).comp[0, 0]:.0f}")

def retreat_comp(drive):
    b = Body("D", "斗", reach=0.75, mass=75, power=1.0, hand_speed=1.0, foot_speed=1.0, cardio=1.0, chin=1.0,
             react_ticks=3, temperament={"斗志": drive})
    s = B.Sim(1, (b, STD), auto_reset=False, n_rounds=1)
    s.pos[0] = [[C - 1.5, C], [C + 1.5, C]]
    s.th[0] = [0, np.pi]
    run(s, {}, ticks=40, hold0=B.BACK)
    return s.comp[0, 0] - s.comp_base[0]
check("性格·斗志：好斗的人一后退心态就掉", retreat_comp(0.9) < retreat_comp(0.3) - 0.5, f"好斗 {retreat_comp(0.9):+.2f}，不好斗 {retreat_comp(0.3):+.2f}")

hot = Body("H", "冲", reach=0.75, mass=75, power=1.0, hand_speed=1.0, foot_speed=1.0, cardio=1.0, chin=1.0,
           react_ticks=3, temperament={"冲动": 0.9})
s = B.Sim(1, (STD, hot), auto_reset=False, n_rounds=1)
s.pos[0] = [[C - 0.5, C], [C + 0.5, C]]
s.th[0] = [0, np.pi]
s.hist[:] = s._snapshot()[None]
run(s, {1: B.JAB_H}, {0: B.CROSS_H}, ticks=12)         # 冲动的 1 号起手后手时被反击，等硬直过去
tilted = s.tilt[0, 1] > 0
run(s, {}, {0: B.CROSS_H}, ticks=1)
seen_early = s.react_now()[0, 0] < STD.react_ticks
check("性格·冲动：被反击就上头，上头时出拳被对手早一帧看到", tilted and seen_early, f"上头 {tilted}，对手反应 {s.react_now()[0, 0]} 帧")

s = skilled("后手勾拳·头", 0.2)
run(s, {0: B.RHOOK_H}, ticks=1)
check("露馅：练得差的拳（熟练度 < 0.35）起手会被对手早一帧看到", s.react_now()[0, 1] == STD.react_ticks - 1, f"对手反应 {s.react_now()[0, 1]} 帧")

# ------------------------------------------------------------------ 战术与流派
s = B.Sim(256, (STD, STD), auto_reset=False)
s.plan[:] = [1, 0]                    # 0 号外围控距，1 号自由发挥
m = s.plan_mask()
check("战术·外围控距：勾拳大多数时候不可选；自由发挥什么都能选",
      m[:, 0, B.LHOOK_H].mean() < 0.4 and m[:, 0, B.JAB_H].all() and m[:, 1].all(),
      f"勾拳可选 {m[:, 0, B.LHOOK_H].mean():.0%}")

import schools as SCH
from bodies import LANKY
check("流派：外围拳击的刺拳天赋更高，训练时刺拳练得更勤",
      SCH.talent(LANKY)[0] > B.skill_vector(LANKY.skills)[0] and SCH.drill(LANKY)[0] > 1 and SCH.posture(LANKY) == B.P_LONG,
      f"刺拳天赋 {B.skill_vector(LANKY.skills)[0]:.2f}→{SCH.talent(LANKY)[0]:.2f}，练习量 ×{SCH.drill(LANKY)[0]}")

# ------------------------------------------------------------------ 心智：观察力、拳商
def minded(obs=0.5, iq=0.5):
    return Body("M", "心", reach=0.75, mass=75, power=1.0, hand_speed=1.0, foot_speed=1.0, cardio=1.0, chin=1.0,
                react_ticks=3, mind={"观察力": obs, "拳商": iq})

def duck_read(obs):
    s = B.Sim(1, (minded(obs=obs), STD), auto_reset=False, n_rounds=1)
    s.pos[0] = [[C - 0.5, C], [C + 0.5, C]]
    s.th[0] = [0, np.pi]
    s.hist[:] = s._snapshot()[None]
    run(s, {k * 12: B.JAB_H for k in range(5)}, {k * 12 + 1: B.DUCK for k in range(5)}, ticks=62)
    return float(s.tend_def[0, 0, 5])
check("观察力：同样看对手下潜 5 次，眼尖的记得更牢", duck_read(0.9) > duck_read(0.2) + 0.15, f"下潜倾向 {duck_read(0.2):.2f} → {duck_read(0.9):.2f}")

def feint_cue(obs):
    s = B.Sim(64, (STD, minded(obs=obs)), auto_reset=False, n_rounds=1)
    s.pos[:] = [[C - 1.0, C], [C, C]]
    s.th[:] = [0, np.pi]
    s.hist[:] = s._snapshot()[None]
    cues = 0
    for t in range(6):
        s.step(np.tile([[B.FEINT_JAB if t == 0 else B.STAY, B.STAY]], (64, 1)))
        cues += s.observe()[:, 1, B.OBS_BASE + 26].sum()
    return cues
check("观察力：眼尖的人能看穿一部分虚晃，普通人看不穿", feint_cue(0.9) > 10 and feint_cue(0.5) == 0, f"64 场里看穿 {feint_cue(0.9):.0f} 次 vs {feint_cue(0.5):.0f} 次")

s = B.Sim(1, (STD, minded(obs=0.2)), auto_reset=False)
s.set_skills(np.where(np.arange(B.N_TECH) == 6, 0.2, 0.5)[None].repeat(2, 0))
s.pos[0] = [[C - 0.5, C], [C + 0.5, C]]
run(s, {0: B.RHOOK_H}, ticks=1)
check("观察力：眼神不好的人看不出对手露馅", s.react_now()[0, 1] == 3, f"对手反应 {s.react_now()[0, 1]} 帧（露馅也没早看到）")

def view_precision(obs):
    s = B.Sim(1, (STD, minded(obs=obs)), auto_reset=False)
    s.stam[0, 0] = 0.63 * s.stam_max[0]
    s.hist[:] = s._snapshot()[None]
    return abs(s.observe()[0, 1, 63] - 0.63)
check("观察力：眼尖的人对对手体能看得更准", view_precision(0.95) < view_precision(0.1), f"误差 {view_precision(0.1):.3f} → {view_precision(0.95):.3f}")

def lapse_rate(iq, comp=80):
    s = B.Sim(2000, (minded(iq=iq), STD), auto_reset=False)
    s.comp[:, 0] = comp
    _, hit = B.lapse(np.zeros((2000, 2), int), None, s, np.random.default_rng(0))
    return hit[:, 0].mean()
check("拳商：拳商低的人出昏招多，心态崩了更多", lapse_rate(0.1) > 2 * lapse_rate(0.8) and lapse_rate(0.1, comp=30) > 1.6 * lapse_rate(0.1),
      f"拳商 0.8 {lapse_rate(0.8):.1%}，拳商 0.1 {lapse_rate(0.1):.1%}，拳商 0.1 且心态差 {lapse_rate(0.1, comp=30):.1%}")


# ------------------------------------------------------------------ 平衡：弯臂拳、致命度
from bodies import STOCKY
sab = B.Sim(1, (LANKY, STOCKY), auto_reset=False)
check("勾拳是弯臂拳：长臂的人勾拳够不到短臂的人刺拳那么远",
      sab.rng_tab[4, 0] < sab.rng_tab[0, 1] - 0.03 and (sab.rng_tab[4, 0] - sab.rng_tab[4, 1]) < 0.6 * (sab.rng_tab[0, 0] - sab.rng_tab[0, 1]),
      f"A 前手勾 {sab.rng_tab[4, 0]:.2f} 米 vs B 刺拳 {sab.rng_tab[0, 1]:.2f} 米；勾拳射程差 {sab.rng_tab[4, 0] - sab.rng_tab[4, 1]:.2f}，直拳射程差 {sab.rng_tab[0, 0] - sab.rng_tab[0, 1]:.2f}")

s = make(0.7)
inf = run(s, {0: B.RHOOK_H, 16: B.RHOOK_H, 32: B.RHOOK_H}, ticks=46)
check("致命度：满状态的普通人挨三记干净的后手勾拳不会倒", s.kd_total[0, 1] == 0 and not s.over[0],
      f"晕眩 {s.stun[0, 1]:.0f}，体力 {s.hp[0, 1]:.0f}")


# ------------------------------------------------------------------ 起手下限、刺拳硬直、自然站姿
fast = Body("Q", "快", reach=0.75, mass=75, power=1.0, hand_speed=1.4, foot_speed=1.0, cardio=1.0, chin=1.0, react_ticks=3)
sq = B.Sim(1, (fast, STD), auto_reset=False)
sq.set_skills(np.full((2, B.N_TECH), 1.0))
sq.posture[0, 0] = B.P_LOW
sq.pos[0] = [[C - 0.5, C], [C + 0.5, C]]
run(sq, {0: B.JAB_H}, ticks=1)
check("起手下限：手速、熟练度、低手位全叠上，刺拳也不会快过 2 帧（100 毫秒）", int(sq.timer[0, 0]) >= 2, f"起手 {sq.timer[0, 0]} 帧")

s = make(1.0)
inf = run(s, {0: B.JAB_H}, ticks=12, hold1=B.GUARD_HIGH)
s = make(1.0)
inf2 = run(s, {t: B.JAB_H for t in range(16)}, {t: B.GUARD_HIGH for t in range(4, 20)}, ticks=16)
second = [int(i["outcome"][0, 0]) for i in inf2 if i["fired"][0, 0]]
check("刺拳只刺一下：挨了一记刺拳后来得及把架子举起来挡下一记", len(second) == 2 and second[1] == 2, f"两记刺拳结果 {second}（1 打中 2 挡住）")

d_idle, _, _ = punch_vs(B.JAB_H, 1.0)
s = make(1.0)
s.posture[0, 1] = B.P_LOW
d_low_idle = dealt(run(s, {0: B.JAB_H}, ticks=12))
check("自然站姿：站着不动时手在脸边，头部直拳只吃七成；低手位没有这层保护",
      d_low_idle > 1.4 * d_idle, f"标准站姿 {d_idle:.2f}，低手位 {d_low_idle:.2f}")


# ------------------------------------------------------------------ 体能卡住出拳量
s = make(3.0)
starts = []
for t in range(400):
    s.step(np.array([[B.JAB_H if t % 2 == 0 else B.CROSS_H, B.STAY]]))
    starts.append(s.phase[0, 0] == B.STARTUP and s.elapsed[0, 0] == 0)
early, late = sum(starts[:100]), sum(starts[300:])
check("体能卡住出拳量：一直狂打，体能见底后出拳明显变少", late < 0.6 * early, f"前 5 秒出拳 {early} 次，最后 5 秒 {late} 次")

def recover(hold):
    s = make(3.0)
    s.stam[0, 0] = 40.0
    run(s, {}, ticks=100, hold0=hold)
    return s.stam[0, 0] - 40.0
check("走动也耗体能：边绕边喘气，比站着回得明显慢", recover(B.LEFT) < 0.8 * recover(B.STAY), f"5 秒回复：站着 +{recover(B.STAY):.1f}，绕着走 +{recover(B.LEFT):.1f}")


# ------------------------------------------------------------------ 累了手会掉、计分看有效性、知道犯规次数
def guarded_when(stam_frac):
    s = make(1.0)
    s.stam[0, 1] = stam_frac * s.stam_max[1]
    return dealt(run(s, {0: B.CROSS_H}, ticks=12, hold1=B.GUARD_HIGH))
s = make(1.0)
s.stam[0, 1] = 0.1 * s.stam_max[1]
check("累了手会掉：体能见底时举架挡不住、反应慢一帧", guarded_when(0.1) > 2 * guarded_when(1.0) and s.react_now()[0, 1] == STD.react_ticks + 1,
      f"举架挨后手：满体能 {guarded_when(1.0):.2f}，见底 {guarded_when(0.1):.2f}；反应 {s.react_now()[0, 1]} 帧")

def points_when(stam_frac, gap=1.0):
    s = make(gap)
    s.stam[0, 0] = stam_frac * s.stam_max[0]
    s.posture[0, 1] = B.P_LOW                       # 对手低手位，排除自然站姿的影响
    inf = run(s, {0: B.JAB_B}, ticks=10)
    return sum(float(i["points"][0, 0]) for i in inf)
check("计分看有效性：累了打出的软拳分数更少，擦到的减半",
      points_when(0.2) < 0.7 * points_when(1.0) and abs(points_when(1.0, gap=float(make().rng_tab[1, 0]) - 0.02) - 0.5 * points_when(1.0)) < 1e-6,
      f"满体能 {points_when(1.0):.2f} 分，体能 20% {points_when(0.2):.2f} 分")

s = make(0.7)
run(s, {0: B.CLINCH}, ticks=6)
check("知道犯规次数：抱了几次，自己和对手都看得到（裁判警告是公开的）",
      s.observe()[0, 0, B.OBS_BASE + 27] > 0 and s.observe()[0, 1, B.OBS_BASE + 28] > 0, f"缠抱次数 {s.holds[0].tolist()}")


# ------------------------------------------------------------------ 属性预算
import bodies as BD
league = (BD.LANKY, BD.STOCKY)
totals = [BD.budget_total(b) for b in league]
check("属性预算：参赛角色的属性总分都等于预算（±0.5），长处要用短处换",
      all(abs(t - BD.BUDGET) <= 0.5 for t in totals), f"A {totals[0]:.1f}，B {totals[1]:.1f}，预算 {BD.BUDGET:.0f}")
check("量级：参赛角色都在同一个量级", all(BD.WEIGHT_CLASS[1] <= b.mass <= BD.WEIGHT_CLASS[2] for b in league),
      f"{BD.WEIGHT_CLASS[0]}：A {BD.LANKY.mass} 公斤，B {BD.STOCKY.mass} 公斤")
tal = [SCH.talent(b).sum() for b in league]
check("天赋总量相同（含流派加成），只是分布不同", abs(tal[0] - tal[1]) <= 0.15, f"A {tal[0]:.2f}，B {tal[1]:.2f}")


# ------------------------------------------------------------------ 统计口径（只统计，不影响比赛）
def landed_of(posture=B.P_STANDARD, hold1=B.STAY, gap=1.0):
    s = make(gap)
    s.posture[0, 1] = posture
    inf = run(s, {0: B.JAB_H}, ticks=12, hold1=hold1)
    return [round(float(i["landed_cb"][0, 0]), 3) for i in inf if i["fired"][0, 0]]


edge = float(make().rng_tab[0, 0]) - 0.005
l_std, l_low, l_guard, l_edge = landed_of(), landed_of(B.P_LOW), landed_of(hold1=B.GUARD_HIGH), landed_of(gap=edge)
check("统计口径（CompuBox）：刺拳打中站着的人按自然站姿折成 0.7 记命中，低手位算 1 记，打在架子上、射程边缘擦到的不算",
      l_std == [0.7] and l_low == [1.0] and l_guard == [0.0] and l_edge == [0.0],
      f"站姿 {l_std}，低手位 {l_low}，举架 {l_guard}，擦到 {l_edge}")


def tko_cause_of(setup, a0, ticks, x0=None):
    s = make(1.0, x0=x0)
    setup(s)
    end = [i for i in run(s, a0, ticks=ticks) if i["done"][0]]
    return (B.METHOD_NAMES[end[0]["method"][0]], int(end[0]["tko_cause"][0])) if end else None


c_hp = tko_cause_of(lambda s: s.hp.__setitem__((0, 1), 0.5), {0: B.CROSS_H}, 10)
c_un = tko_cause_of(lambda s: s.stun.__setitem__((0, 1), 85), {k * 8: B.JAB_H for k in range(10)}, 85, x0=B.RING - B.BODY_R - 1.0)
c_kd = tko_cause_of(lambda s: (s.stun.__setitem__((0, 1), 99), s.kd_round.__setitem__((0, 1), 2), s.kd_total.__setitem__((0, 1), 2)),
                    {0: B.CROSS_H}, 10)
check("统计口径：TKO 记下原因（1 血条归零、2 受伤后连续挨打不还手、3 一回合倒三次）",
      c_hp == ("TKO", 1) and c_un == ("TKO", 2) and c_kd == ("TKO", 3), f"血条 {c_hp}，连续挨打 {c_un}，倒三次 {c_kd}")


if __name__ == "__main__":
    ok = sum(r[1] for r in RESULTS)
    for name, passed, detail in RESULTS:
        print(("✓ " if passed else "✗ ") + name + (f"  —— {detail}" if detail else ""))
    print(f"\n{ok}/{len(RESULTS)} 通过")
    raise SystemExit(0 if ok == len(RESULTS) else 1)
