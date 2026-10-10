"""拳击版的比赛、指标、技术使用表、回放。训练时的阶段评估和赛后分析共用。"""
import numpy as np

import boxing as B
from league import save_json, summarize  # noqa: F401  (给调用方复用)

PUNCH_NAMES = B.ACTION_NAMES[B.PUNCH0:B.PUNCH0 + 12]
DEF_ROWS = [("拍挡", B.PARRY, B.M_PARRY), ("侧闪", B.SLIP, B.M_SLIP), ("下潜", B.DUCK, B.M_DUCK), ("后仰", B.PULL, B.M_PULL)]


class Bout:
    def __init__(self, bodies, names, n, t_max):
        self.bodies, self.names, self.n = bodies, names, n
        self.fields = {}
        self.t_cap = t_max
        self.end = np.full(n, t_max)
        self.winner = np.full(n, -1)
        self.method = np.zeros(n, np.int64)
        self.judges = np.zeros((n, 3, 2))
        self.tko_cause = np.zeros(n, np.int64)
        self.end_round = np.zeros(n, np.int64)
        self.init = None
        self.seed = None
        self.n_rounds = B.N_ROUNDS

    def rec(self, t, **arrays):
        for k, v in arrays.items():
            if k not in self.fields:
                self.fields[k] = np.zeros((self.t_cap,) + v.shape, v.dtype)
            self.fields[k][t] = v

    def trim(self, t):
        for k in self.fields:
            self.fields[k] = self.fields[k][:t]


def run_matches(bodies, ctrls, names, n, seed, n_rounds=B.N_ROUNDS, skills=None, plans=(0, 0),
                form_sd=B.FORM_SD, plan_rank=None, form_mean=None, tend_prior=None):
    """skills：(2, N_TECH) 当前熟练度；不给就用天赋。plans：双方开场的战术计划。
    plan_rank：给了就由角落在回合间调整——这一回合多数裁判判输了，就换成排名里的下一个计划。"""
    sim = B.Sim(n, bodies, seed=seed, auto_reset=False, n_rounds=n_rounds, skills=skills, form_sd=0.0)
    sim.plan_fixed = None if plans is None else np.array(plans)       # None：每场随机抽战术计划
    if form_sd:
        sim.form_sd = form_sd
    if form_mean is not None:
        sim.form_mean = np.array(form_mean, dtype=float)       # 称重时的状态期望（训练营决定）
    if tend_prior is not None:
        sim.tend_prior = list(tend_prior)
    sim.reset(np.ones(n, bool))
    rng = np.random.default_rng(seed + 7919)
    bt = Bout(bodies, names, n, sim.t_max)
    bt.init, bt.seed, bt.n_rounds, bt.skills = sim.init.copy(), seed, n_rounds, sim.skills.copy()
    alive = np.ones(n, bool)
    t = 0
    rank_pos = np.zeros((n, 2), int)
    judge_prev = sim.judge.copy()
    while alive.any() and t < sim.t_max:
        obs = sim.observe()
        msk = sim.plan_mask()
        a = np.stack([ctrls[0].act(obs[:, 0], rng, msk[:, 0]), ctrls[1].act(obs[:, 1], rng, msk[:, 1])], axis=1)
        a, _ = B.lapse(a, msk, sim, rng)                                  # 拳商：昏招
        rnd_before = sim.round.copy()
        _, done, info = sim.step(a)
        if plan_rank is not None:
            new_round = sim.round > rnd_before
            if new_round.any():
                rs = sim.judge - judge_prev                              # 这一回合三位裁判给的分
                lost = (np.sign(rs[:, :, 0] - rs[:, :, 1])[..., None] * np.array([-1, 1]) > 0).sum(1) >= 2
                switch = new_round[:, None] & lost
                for k in (0, 1):
                    if plan_rank[k] is not None:
                        rank_pos[switch[:, k], k] = (rank_pos[switch[:, k], k] + 1) % len(plan_rank[k])
                        sim.plan[switch[:, k], k] = np.array(plan_rank[k])[rank_pos[switch[:, k], k]]
                judge_prev = np.where(new_round[:, None, None], sim.judge, judge_prev)
        bt.rec(t, action=a.astype(np.int8), pos=sim.pos.astype(np.float32), th=sim.th.astype(np.float32),
               phase=sim.phase.astype(np.int8), guard=sim.guard.astype(np.int8), comp=sim.comp.astype(np.float32),
               acted=info["acted"], started=info["started"].astype(np.int8), fired=info["fired"].astype(np.int8),
               outcome=info["outcome"].astype(np.int8), evaded_by=info["evaded_by"].astype(np.int8),
               counter=info["counter"], kd=info["kd"], faked=info["faked"], clinch=info["clinch"],
               pivot=info["pivot"], points=info["points"].astype(np.float32),
               posture=sim.posture.astype(np.int8), stance=sim.stance.astype(np.int8), tilt=sim.tilt > 0,
               plan=sim.plan.astype(np.int8), landed_cb=info["landed_cb"].astype(np.float32),
               stam=(sim.stam / sim.stam_max[None, :]).astype(np.float32))
        newly = done & alive
        bt.end[newly] = t + 1
        bt.winner[newly] = info["winner"][newly]
        bt.method[newly] = info["method"][newly]
        bt.judges[newly] = info["judges"][newly]
        bt.tko_cause[newly] = info["tko_cause"][newly]
        bt.end_round[newly] = info["round"][newly]
        alive &= ~done
        t += 1
    bt.trim(t)
    return bt


def _valid(bt):
    T = len(bt.fields["action"])
    return np.arange(T)[:, None] < bt.end[None, :]


def _geom(bt):
    pos, th = bt.fields["pos"], bt.fields["th"]
    d = pos[:, :, 1] - pos[:, :, 0]
    dist = np.linalg.norm(d, axis=-1)
    b01 = np.arctan2(d[..., 1], d[..., 0])
    bear = np.stack([b01, b01 + np.pi], -1)
    off = B.wrap(bear - th)               # off[..., k]：k 的朝向偏离对手方向多少
    return dist, off


def fight_metrics(bt, k):
    """每场一个值。k 是哪一边。"""
    f = bt.fields
    v = _valid(bt)
    j = 1 - k
    secs = bt.end * B.DT
    dist, off = _geom(bt)
    act, acted = f["action"][:, :, k], f["acted"][:, :, k] & v
    na = np.maximum(acted.sum(0), 1)
    fired = f["fired"][:, :, k] * v
    oc = f["outcome"][:, :, k]
    thrown = (fired > 0).sum(0)
    clean = ((fired > 0) & (oc == 1)).sum(0)
    started = (f["started"][:, :, k] > 0) & v
    pid = np.clip(f["started"][:, :, k].astype(int) - 1, 0, 11)
    pos = f["pos"][:, :, k]
    back = B.ray_to_rope(pos, f["th"][:, :, k] + np.pi)
    pinned = (back < 0.4) & (dist < 1.2) & v
    opp_back = B.ray_to_rope(f["pos"][:, :, j], f["th"][:, :, j] + np.pi)
    pinning = (opp_back < 0.4) & (dist < 1.2) & v
    center = np.linalg.norm(pos - B.RING / 2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = {
            "mean_dist": (dist * v).sum(0) / v.sum(0),
            "punch_rate": started.sum(0) / secs * 10,
            "accuracy": np.where(thrown > 0, clean / thrown, np.nan),
            "power_share": np.where(started.sum(0) > 0, (started & (B.IS_POWER[pid] == 1)).sum(0) / started.sum(0), np.nan),
            "body_share": np.where(started.sum(0) > 0, (started & (B.TGT[pid] == 1)).sum(0) / started.sum(0), np.nan),
            "hook_upper_share": np.where(started.sum(0) > 0, (started & (B.TYPE[pid] >= 1)).sum(0) / started.sum(0), np.nan),
            "forward": (np.isin(act, (B.FWD, B.GUARD_FWD)) & acted).sum(0) / na,
            "backward": ((act == B.BACK) & acted).sum(0) / na,
            "lateral": (np.isin(act, (B.LEFT, B.RIGHT, B.PIVOT_L, B.PIVOT_R)) & acted).sum(0) / na,
            "guard": (np.isin(act, (B.GUARD_HIGH, B.GUARD_FWD, B.GUARD_BODY)) & acted).sum(0) / na,
            "head_move": (np.isin(act, (B.PARRY, B.SLIP, B.DUCK, B.PULL)) & acted).sum(0) / na,
            "feint": (np.isin(act, (B.FEINT_JAB, B.FEINT_CROSS)) & acted).sum(0) / na,
            "pinned": pinned.sum(0) / v.sum(0),
            "pinning": pinning.sum(0) / v.sum(0),
            "center": (center * v).sum(0) / v.sum(0),
            "counters": (f["counter"][:, :, k] & v).sum(0),
            "knockdowns": (f["kd"][:, :, j] & v).sum(0),
            "comp_mean": (f["comp"][:, :, k] * v).sum(0) / v.sum(0),
            "comp_min": np.where(v, f["comp"][:, :, k], 999).min(0),
        }
    m["win"] = (bt.winner == k).astype(float)
    m["loss"] = (bt.winner == j).astype(float)
    m["draw"] = (bt.winner == -1).astype(float)
    m["ko_win"] = ((bt.winner == k) & (bt.method == B.M_KO)).astype(float)
    m["tko_win"] = ((bt.winner == k) & (bt.method == B.M_TKO)).astype(float)
    m["dec_win"] = ((bt.winner == k) & (bt.method == B.M_DEC)).astype(float)
    return m


STYLE_METRICS = {
    "mean_dist": "平均距离 (米)", "punch_rate": "每 10 秒出拳", "accuracy": "命中率",
    "power_share": "重拳占比", "body_share": "打身体占比", "hook_upper_share": "勾拳+上勾拳占比",
    "forward": "压上的比例", "backward": "后退的比例", "lateral": "横移+轴转的比例",
    "guard": "举架的比例", "head_move": "头部闪躲的比例", "feint": "虚晃的比例",
    "pinned": "被压在围绳上的时间", "pinning": "把对手压在围绳上的时间", "center": "离擂台中心的距离 (米)",
    "counters": "每场反击拳", "knockdowns": "每场击倒", "comp_mean": "平均心态", "comp_min": "最低心态",
}


def technique_table(bt, k):
    """技术使用表：每个技术用了几次、成功几次。成功的定义跟着技术的作用走。"""
    f = bt.fields
    v = _valid(bt)
    j = 1 - k
    act, acted = f["action"][:, :, k], f["acted"][:, :, k] & v
    fired, oc = f["fired"][:, :, k] * v, f["outcome"][:, :, k]
    rows = []
    for q, name in enumerate(PUNCH_NAMES):
        sel = fired == q + 1
        rows.append(("进攻", name, int(sel.sum()), int((sel & (oc == 1)).sum()), "打中（没被挡）"))
    combo = (f["started"][:, :, k] > 0) & v & (np.roll(f["phase"][:, :, k], 1, axis=0) == B.RECOVERY)
    rows.append(("进攻", "组合拳衔接", int(combo.sum()), None, "在上一拳收招里起手"))
    rows.append(("进攻", "反击拳", None, int((f["counter"][:, :, k] & v).sum()), "打在对手起手或打空收招上"))
    gprev = np.roll(f["guard"][:, :, k], 1, axis=0)
    opp_oc = f["outcome"][:, :, j] * v
    arrived = np.isin(opp_oc, (1, 2))
    for g, name in ((1, "高位格挡"), (2, "护体")):
        under = arrived & (gprev == g)
        rows.append(("防守", name, int(under.sum()), int((under & (opp_oc == 2)).sum()), "对手的拳落在架子上被挡住"))
    for name, a_code, m_code in DEF_ROWS:
        used = ((act == a_code) & acted & (f["phase"][:, :, k] == B.DEFEND)).sum()
        rows.append(("防守", name, int(used), int(((f["evaded_by"][:, :, k] == m_code) & v).sum()), "躲开 / 拍开对手的拳"))
    rows.append(("防守", "缠抱", int(((act == B.CLINCH) & acted).sum()), int((f["clinch"][:, :, k] & v).sum()), "抱住"))
    dist, off = _geom(bt)
    piv = f["pivot"][:, :, k] & v
    t_idx, n_idx = np.nonzero(piv)
    t_end = np.minimum(t_idx + B.PIVOT_TICKS, len(dist) - 1)
    gained = np.abs(off[t_end, n_idx, j]) > B.ANG_GUARD
    rows.append(("步法", "轴转", int(piv.sum()), int(gained.sum()), "转完对手正面线偏离 25° 以上"))
    rows.append(("步法", "横移", int((np.isin(act, (B.LEFT, B.RIGHT)) & acted).sum()), None, "按帧计"))
    rows.append(("步法", "举架压上", int(((act == B.GUARD_FWD) & acted).sum()), None, "按帧计"))
    rows.append(("架势", "换架势", int(((act == B.CHANGE_POSTURE) & acted).sum()), None, "标准 / 长架 / 低姿 / 低手位轮换"))
    rows.append(("架势", "换站架", int(((act == B.SWITCH_STANCE) & acted).sum()), None, "正架和反架互换"))
    feints = (np.isin(act, (B.FEINT_JAB, B.FEINT_CROSS)) & acted).sum()
    rows.append(("心理", "虚晃", int(feints), int((f["faked"][:, :, j] & v).sum()), "骗出对手的防守动作"))
    return rows


def coverage(rows):
    """多少个技术在对局里至少成功过一次（只算有"成功"定义的）。"""
    have = [r for r in rows if r[3] is not None]
    return sum(1 for r in have if r[3] > 0), len(have)


def snapshot_eval(bodies, pols, bot, ver, n, sparring_body, skills=None):
    """skills：{角色 id: 熟练度数组}。"""
    out = {}
    sk = lambda b: skills[b.id] if skills and b.id in skills else B.skill_vector(b.skills)
    for side, body in enumerate(bodies):
        bt = run_matches((body, sparring_body), (pols[side], bot), (body.id, "S"), n, seed=10_000 + ver,
                         skills=np.stack([sk(body), sk(sparring_body)]))
        out[f"{body.id}_vs_S"] = summarize(fight_metrics(bt, 0))
        out[f"{body.id}_vs_S"]["coverage"] = coverage(technique_table(bt, 0))[0]
    bt = run_matches(bodies, pols, (bodies[0].id, bodies[1].id), n, seed=20_000 + ver,
                     skills=np.stack([sk(bodies[0]), sk(bodies[1])]))
    out["AvB_A"] = summarize(fight_metrics(bt, 0))
    out["AvB_B"] = summarize(fight_metrics(bt, 1))
    out["AvB_A"]["coverage"] = coverage(technique_table(bt, 0))[0]
    out["AvB_B"]["coverage"] = coverage(technique_table(bt, 1))[0]
    a, b = out["A_vs_S"], out["B_vs_S"]
    line = (f"  [v{ver:04d}] vs陪练 胜率 A {a['win']:.2f} B {b['win']:.2f} | 提前结束 A {a['ko_win'] + a['tko_win']:.2f} "
            f"B {b['ko_win'] + b['tko_win']:.2f} | 距离 A {a['mean_dist']:.2f} B {b['mean_dist']:.2f} | 出拳/10s A {a['punch_rate']:.1f} "
            f"B {b['punch_rate']:.1f} | 会用技术 A {a['coverage']} B {b['coverage']} | AvB A胜 {out['AvB_A']['win']:.2f} "
            f"B胜 {out['AvB_B']['win']:.2f}")
    return out, line


# ------------------------------------------------------------------ 回放
def replay_record(bt, i, ctrl_info):
    end = int(bt.end[i])
    acts = bt.fields["action"][:end, i]
    return {
        "format": 1, "sim_version": B.SIM_VERSION, "n_rounds": int(bt.n_rounds),
        "bodies": [b.to_dict() for b in bt.bodies], "controllers": ctrl_info,
        "seed": int(bt.seed), "index": int(i), "init": [float(x) for x in bt.init[i]],
        "skills": bt.skills.round(6).tolist(),
        "plans": plan_timeline(bt.fields["plan"][:end, i]),
        "actions": [",".join(map(str, acts[:, 0])), ",".join(map(str, acts[:, 1]))],
        "result": {"winner": int(bt.winner[i]), "method": int(bt.method[i]), "ticks": end,
                   "judges": bt.judges[i].tolist()},
    }


def plan_timeline(plan):
    """逐帧计划 → 只记变化点 [[帧, 0 号计划, 1 号计划], ...]。"""
    out, last = [], None
    for t, (p0, p1) in enumerate(plan.tolist()):
        if (p0, p1) != last:
            out.append([t, int(p0), int(p1)])
            last = (p0, p1)
    return out


def choose_plan(bodies, pols, side, skills, n=16, seed=4242, opp_plan=0):
    """角落的赛前战术会议：每个计划和对手模拟打一回合，按"赢面"排名。返回 [(计划, 赢面), ...]。"""
    scores = []
    for pl in range(B.N_PLANS):
        plans = (pl, opp_plan) if side == 0 else (opp_plan, pl)
        bt = run_matches(bodies, pols, ("x", "y"), n, seed=seed, n_rounds=1, skills=skills, plans=plans)
        pts = bt.judges[:, :, side].sum(1) - bt.judges[:, :, 1 - side].sum(1)
        win = (bt.winner == side).mean() - (bt.winner == 1 - side).mean()
        scores.append((pl, float(win + 0.05 * pts.mean())))
    return sorted(scores, key=lambda x: -x[1])


def resimulate(rec):
    """按回放档重新模拟，返回逐帧画面数据，并核对结果和存档一致。"""
    from bodies import Body
    bodies = tuple(Body(**b) for b in rec["bodies"])
    sim = B.Sim(1, bodies, auto_reset=False, n_rounds=rec["n_rounds"], skills=rec.get("skills"))
    sim.reset(np.ones(1, bool), init=np.array([rec["init"]]))
    a0 = list(map(int, rec["actions"][0].split(",")))
    a1 = list(map(int, rec["actions"][1].split(",")))

    plans = rec.get("plans") or [[0, 0, 0]]

    def plan_at(t):
        cur = plans[0][1:]
        for tk, p0, p1 in plans:
            if tk <= t:
                cur = [p0, p1]
        return cur

    def frame(info=None, acts=(0, 0), t=0):
        rows = []
        for k in (0, 1):
            rows.append([
                round(float(sim.pos[0, k, 0]), 3), round(float(sim.pos[0, k, 1]), 3), round(float(sim.th[0, k]), 3),
                round(float(sim.hp[0, k]), 1), round(float(sim.stun[0, k]), 1),
                round(float(sim.stam[0, k] / sim.stam_max[k]), 3), round(float(sim.comp[0, k]), 1),
                int(sim.phase[0, k]), int(sim.move[0, k]), int(sim.guard[0, k]),
                int(info["fired"][0, k]) if info else 0, int(info["outcome"][0, k]) if info else 0,
                round(float(info["dmg"][0, k]), 1) if info else 0.0, int(acts[k]),
                int(info["kd"][0, k]) if info else 0, int(info["evaded_by"][0, k]) if info else -1,
                int(info["counter"][0, k]) if info else 0, int(info["faked"][0, k]) if info else 0,
                int(sim.posture[0, k]), int(sim.stance[0, k]), int(sim.tilt[0, k] > 0), int(plan_at(t)[k])])
        return {"f": rows, "r": int(sim.round[0]), "t": int(sim.t_round[0])}

    frames = [frame()]
    last = None
    for t in range(len(a0)):
        _, _, info = sim.step(np.array([[a0[t], a1[t]]]))
        frames.append(frame(info, (a0[t], a1[t]), t))
        if info["done"][0]:
            last = info
    res = rec["result"]
    assert last is not None and int(last["winner"][0]) == res["winner"] and int(last["method"][0]) == res["method"], \
        "回放结果和存档对不上"
    return frames
