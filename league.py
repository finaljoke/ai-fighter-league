"""比赛、打法指标、回放。训练时的阶段评估和赛后分析共用这里。"""
import json

import numpy as np

import sim as S

# 打法指标：名字 -> 中文说明（报告里用）
STYLE_METRICS = {
    "mean_gap": "平均站位距离 (米)",
    "start_gap": "出拳时的距离 (米)",
    "attack_rate": "每 10 秒出拳数",
    "jab_share": "刺拳占出拳比例",
    "fwd": "选择前进的比例",
    "back": "选择后退的比例",
    "block": "选择原地格挡的比例",
    "guard_fwd": "举着架子往前压的比例",
    "cornered": "自己背靠围绳的时间比例",
    "opp_cornered": "把对手逼到围绳的时间比例",
    "accuracy": "出拳命中率（不含被挡）",
    "dmg_per10": "每 10 秒造成伤害",
}
GAP_BINS = np.linspace(S.MIN_GAP, 3.0, 13)


class MatchBatch:
    def __init__(self, bodies, names, n, t_max):
        self.bodies, self.names, self.n = bodies, names, n
        self.fields = {}
        self.end = np.full(n, t_max)
        self.winner = np.full(n, -1)
        self.ko = np.zeros(n, bool)
        self.init_gap = None
        self.seed = None

    def rec(self, t, **arrays):
        for k, v in arrays.items():
            if k not in self.fields:
                self.fields[k] = np.zeros((self.end.max(),) + v.shape, v.dtype)
            self.fields[k][t] = v

    def trim(self, t):
        for k in self.fields:
            self.fields[k] = self.fields[k][:t]


def run_matches(bodies, ctrls, names, n, seed, t_max=S.T_MAX):
    """ctrls: 两个有 act(obs, rng) 的对象。返回 MatchBatch。"""
    sim = S.Sim(n, bodies, seed=seed, auto_reset=False, t_max=t_max)
    rng = np.random.default_rng(seed + 7919)
    mb = MatchBatch(bodies, names, n, t_max)
    mb.init_gap = sim.init_gap.copy()
    mb.seed = seed
    alive = np.ones(n, bool)
    t = 0
    while alive.any() and t < t_max:
        obs = sim.observe()
        a = np.stack([ctrls[0].act(obs[:, 0], rng), ctrls[1].act(obs[:, 1], rng)], axis=1)
        _, done, info = sim.step(a)
        mb.rec(t, action=a.astype(np.int8), x=sim.x.copy(), hp=sim.hp.copy(),
               stam=(sim.stam / sim.stam_max).astype(np.float32), phase=sim.phase.astype(np.int8),
               move=sim.move.astype(np.int8), blocking=sim.blocking.copy(), acted=info["acted"],
               started=info["started"].astype(np.int8), fired=info["fired"].astype(np.int8),
               hit=info["hit"], blocked=info["blocked"], dmg=info["dmg"].astype(np.float32))
        newly = done & alive
        mb.end[newly] = t + 1
        mb.winner[newly] = info["winner"][newly]
        mb.ko[newly] = info["ko"][newly]
        alive &= ~done
        t += 1
    mb.trim(t)
    return mb


def fight_metrics(mb, side):
    """每场一个值，返回 {指标: (n,) 数组}。"""
    f = mb.fields
    T = len(f["x"])
    v = (np.arange(T)[:, None] < mb.end[None, :]).astype(float)
    nt = v.sum(0)
    secs = nt * S.DT
    gap = f["x"][:, :, 1] - f["x"][:, :, 0]
    act = f["action"][:, :, side]
    acted = f["acted"][:, :, side] * v
    na = np.maximum(acted.sum(0), 1)
    started = f["started"][:, :, side] * (v > 0)
    n_start = (started > 0).sum(0)
    fired = (f["fired"][:, :, side] > 0) * v
    hits = f["hit"][:, :, side] * v
    bs = np.stack([f["x"][:, :, 0] - S.BODY_R, S.RING - S.BODY_R - f["x"][:, :, 1]], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = {
            "mean_gap": (gap * v).sum(0) / nt,
            "start_gap": np.where(n_start > 0, (gap * (started > 0)).sum(0) / n_start, np.nan),
            "attack_rate": n_start / secs * 10,
            "jab_share": np.where(n_start > 0, (started == 1).sum(0) / n_start, np.nan),
            "fwd": ((act == S.FWD) * acted).sum(0) / na,
            "back": ((act == S.BACK) * acted).sum(0) / na,
            "block": ((act == S.BLOCK) * acted).sum(0) / na,
            "guard_fwd": ((act == S.GUARD_FWD) * acted).sum(0) / na,
            "cornered": ((bs[:, :, side] < 0.6) * v).sum(0) / nt,
            "opp_cornered": ((bs[:, :, 1 - side] < 0.6) * v).sum(0) / nt,
            "accuracy": np.where(fired.sum(0) > 0, hits.sum(0) / fired.sum(0), np.nan),
            "dmg_per10": (f["dmg"][:, :, side] * v).sum(0) / secs * 10,
        }
    m["win"] = (mb.winner == side).astype(float)
    m["loss"] = (mb.winner == 1 - side).astype(float)
    m["ko_win"] = ((mb.winner == side) & mb.ko).astype(float)
    return m


def summarize(m):
    return {k: float(np.nanmean(v)) for k, v in m.items()}


def fingerprint(mb, side):
    """打法指纹：在每个距离段里，选各个动作的比例。返回 (动作数, 距离段数)。"""
    f = mb.fields
    T = len(f["x"])
    v = np.arange(T)[:, None] < mb.end[None, :]
    gap = f["x"][:, :, 1] - f["x"][:, :, 0]
    sel = v & f["acted"][:, :, side]
    g = gap[sel]
    a = f["action"][:, :, side][sel]
    b = np.clip(np.digitize(g, GAP_BINS) - 1, 0, len(GAP_BINS) - 2)
    counts = np.zeros((S.N_ACTIONS, len(GAP_BINS) - 1))
    np.add.at(counts, (a, b), 1)
    tot = counts.sum(0, keepdims=True)
    return counts / np.maximum(tot, 1), tot[0]


def cohens_d(x, y):
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    s = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / (s + 1e-9))


# ------------------------------------------------------------------ 回放
def replay_record(mb, i, ctrl_info):
    """一场比赛的回放档：初始距离 + 双方逐帧动作。画面靠重新模拟算出来。"""
    end = int(mb.end[i])
    acts = mb.fields["action"][:end, i]
    return {
        "format": 1, "sim_version": S.SIM_VERSION,
        "bodies": [b.to_dict() for b in mb.bodies], "controllers": ctrl_info,
        "seed": int(mb.seed), "index": int(i), "init_gap": float(mb.init_gap[i]),
        "actions": ["".join(map(str, acts[:, 0])), "".join(map(str, acts[:, 1]))],
        "result": {"winner": int(mb.winner[i]), "ko": bool(mb.ko[i]), "ticks": end,
                   "hp": [round(float(h), 4) for h in mb.fields["hp"][end - 1, i]]},
    }


def resimulate(rec):
    """按回放档重新模拟，返回逐帧画面数据，并核对结果和存档一致。"""
    from bodies import Body
    bodies = tuple(Body(**b) for b in rec["bodies"])
    sim = S.Sim(1, bodies, auto_reset=False)
    sim.reset(np.ones(1, bool), gap=np.array([rec["init_gap"]]))
    a0, a1 = rec["actions"]
    # 第 0 帧是开打前的站位
    frames = [[[round(float(sim.x[0, k]), 3), S.HP_MAX, 1.0, 0, 0, 0, 0, 0, 0.0, 0] for k in (0, 1)]]
    for t in range(len(a0)):
        _, _, info = sim.step(np.array([[int(a0[t]), int(a1[t])]]))
        frames.append([
            [round(float(sim.x[0, k]), 3), round(float(sim.hp[0, k]), 2),
             round(float(sim.stam[0, k] / sim.stam_max[k]), 3), int(sim.phase[0, k]),
             int(sim.move[0, k]), int(sim.blocking[0, k]), int(info["fired"][0, k]),
             int(info["hit"][0, k]) + 2 * int(info["blocked"][0, k]), round(float(info["dmg"][0, k]), 2),
             int(a0[t] if k == 0 else a1[t])]
            for k in (0, 1)])
    hp = [round(float(h), 4) for h in sim.hp[0]]
    assert hp == rec["result"]["hp"], f"回放结果对不上: {hp} vs {rec['result']['hp']}"
    return frames


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
