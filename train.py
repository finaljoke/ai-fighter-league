"""自我对练训练：A 和 B 同时学，对手一半是对方的当前版本，一半是对方的历史版本。

记录系统（全部在 runs/<run>/ 下，只追加不删除）：
  config.json      身体参数、超参数、随机种子
  ckpt/            每个版本的出招网络，A_v0003.npz 这样命名
  versions.jsonl   每个版本的文件名、训练到第几帧、sha256
  train_log.jsonl  每轮训练的胜负、熵、KL 等
  evals.jsonl      每个版本对标准陪练、A 对 B 的战绩和打法指标

用法：py -X utf8 train.py --run r1 --iters 300 --seed 1             （一维原型）
      py -X utf8 train.py --run b1 --sim boxing --iters 600 --seed 1  （二维拳击）
"""
import argparse
import json
import os
import time

import numpy as np

import boxing
import league as L
import sim as sim1d
from bodies import LANKY, STOCKY, SPARRING
from bots import SparringBot
from boxing_bots import BoxingSparring
from nets import Agent, save_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-env", type=int, default=512)
    ap.add_argument("--t-roll", type=int, default=128)
    ap.add_argument("--sim", choices=["1d", "boxing"], default="1d")
    ap.add_argument("--snap-every", type=int, default=None)
    ap.add_argument("--eval-n", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--ent-end", type=float, default=0.002)
    # 奖励只围绕胜负，性格不写进奖励。拳击版：计分拳 ±、击倒 ±、KO/TKO/判定 ±1；一维原型用的是伤害差。
    # 下面两条是"擂台规则"，用来打破双方都躲着不打的僵局：
    #   平局算失败：没能击败对手，双方都扣分（只看结果，不规定怎么打）——拳击版默认用这条
    #   裁判催战：双方离得比两人射程都远时，每帧都扣一点——一维原型 s1~s3 用的这条
    ap.add_argument("--stall-gap", type=float, default=2.0)
    ap.add_argument("--stall-penalty", type=float, default=None)
    ap.add_argument("--draw-penalty", type=float, default=None)
    # 接着旧档训练（对照实验用）：--init b2 或 b2:50。检查点只有出招网络，价值网络从头练，先热身 --vf-warmup 轮只练价值网络
    ap.add_argument("--init", default=None)
    ap.add_argument("--vf-warmup", type=int, default=0)
    # 奖励构成（对照实验用）：不给就用 boxing.py 里的默认值（每分 0.02、击倒 0.3）
    ap.add_argument("--point-reward", type=float, default=None)
    ap.add_argument("--kd-reward", type=float, default=None)
    # 留力训练（实验 1b）：每场开打前抽签，floor-frac 的比赛给双方各设一条体能线（floor-lo~floor-hi 之间随机），
    # 体能低于线时不准出拳。只在训练时有，AI 看不到，评估和正式比赛都没有
    ap.add_argument("--floor-frac", type=float, default=0.0)
    ap.add_argument("--floor-lo", type=float, default=0.2)
    ap.add_argument("--floor-hi", type=float, default=0.6)
    args = ap.parse_args()
    box = args.sim == "boxing"
    S = boxing if box else sim1d
    defaults = {"stall_penalty": 0.0 if box else 0.002, "draw_penalty": 0.5 if box else 0.0,
                "snap_every": 20 if box else 10, "eval_n": 48 if box else 128}
    if box and args.ent_end == 0.002:
        args.ent_end = 0.006            # 猜拳需要混合策略，熵别降太低
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if box and args.point_reward is not None:
        boxing.REWARD_PER_POINT = args.point_reward
    if box and args.kd_reward is not None:
        boxing.REWARD_KD = args.kd_reward

    run_dir = os.path.join("runs", args.run)
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_dir} 已存在。规矩是不覆盖旧记录，换个 --run 名字。")
    os.makedirs(os.path.join(run_dir, "ckpt"))
    bodies = (LANKY, STOCKY)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "sim_version": S.SIM_VERSION,
                   "reward": {"per_point": boxing.REWARD_PER_POINT, "kd": boxing.REWARD_KD} if box else None,
                   "bodies": [b.to_dict() for b in bodies],
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)

    rng = np.random.default_rng(args.seed)
    N, T = args.n_env, args.t_roll
    sim = S.Sim(N, bodies, seed=args.seed, **({"form_sd": boxing.FORM_SD} if box else {}))   # 拳击版每场有当天状态
    sim.desync(rng)   # 错开各场的开局时间，不然所有场同时开打同时结束，一批数据全是同一阶段
    agents = [Agent(S.OBS_DIM, S.N_ACTIONS, rng), Agent(S.OBS_DIM, S.N_ACTIONS, rng)]
    pool = [[], []]
    vs_pool = [[], []]             # vs_pool[side][k] = [赢, 场]：side 的当前版本打对方第 k 个历史版本
    pool_share = np.array([0.25, 0.25])
    ctrl = np.full((N, 2), -1)     # -1 = 正在学的当前版本，>=0 = 历史版本编号
    total_ticks = 0
    bot = BoxingSparring() if box else SparringBot()
    # 熟练度（只有拳击版）：天赋是起点，常用、用成功的技术越练越熟，不用的慢慢退回天赋
    import schools as SC
    talent = np.stack([SC.talent(b) for b in bodies]) if box else None      # 天赋 = 身体自带 + 流派加成
    drill = np.stack([SC.drill(b) for b in bodies]) if box else None        # 流派：哪些技术练得勤
    iq = np.array([SC.mind(b)[1] for b in bodies]) if box else None          # 拳商高的练得快
    skills = talent.copy() if box else None
    if args.init:
        from analyze import load_run
        name, _, ver = args.init.partition(":")
        src = load_run(name, int(ver) if ver else None)
        for side, body in enumerate(bodies):
            for p, q in zip(agents[side].pi.params, src["pols"][body.id].params):
                p[...] = q                      # 原地拷贝，优化器还指着这些数组
            if box and "skills" in src["pick"][body.id]:
                skills[side] = np.array(src["pick"][body.id]["skills"])
        with open(os.path.join(run_dir, "config.json"), "r+", encoding="utf-8") as f:
            cfg = json.load(f)
            cfg["init"] = {"run": name, "version": src["version"],
                           "sha256": {c: v["sha256"] for c, v in src["pick"].items()}}
            f.seek(0); json.dump(cfg, f, ensure_ascii=False, indent=2); f.truncate()
        if box:
            sim.set_skills(skills)

    def snapshot(it):
        ver = len(pool[0])
        for side, body in enumerate(bodies):
            pol = agents[side].snapshot()
            pool[side].append(pol)
            vs_pool[1 - side].append([0, 0])
            fn = f"{body.id}_v{ver:04d}.npz"
            sha = save_policy(os.path.join(run_dir, "ckpt", fn), pol.params)
            rec = {"char": body.id, "version": ver, "iter": it, "ticks": total_ticks, "file": fn, "sha256": sha}
            if box:
                rec["skills"] = skills[side].round(4).tolist()
            append(run_dir, "versions.jsonl", rec)
        evaluate(run_dir, bodies, [p[-1] for p in pool], bot, ver, it, total_ticks, args.eval_n, box, skills)

    def assign(mask):
        idx = np.flatnonzero(mask)
        if not idx.size:
            return
        ctrl[idx] = -1
        if not pool[0]:
            return
        u = rng.random(idx.size)
        lo = 1.0 - pool_share.sum()
        for side in (0, 1):
            a0 = lo + pool_share[:side].sum()
            pick = idx[(u >= a0) & (u < a0 + pool_share[side])]
            if pick.size:
                # 优先挑"赢面五五开"的历史对手：太强的打不过学不到东西，太弱的没得学
                st = np.array(vs_pool[side], dtype=float)
                pw = (st[:, 0] + 1) / (st[:, 1] + 2)
                w = pw * (1 - pw) + 0.02
                ctrl[pick, 1 - side] = rng.choice(len(w), size=pick.size, p=w / w.sum())

    floor = np.zeros((N, 2))
    punches = np.arange(boxing.PUNCH0, boxing.PUNCH0 + 12) if box else None

    def new_floors(mask):
        idx = np.flatnonzero(mask)
        if not idx.size or args.floor_frac <= 0:
            return
        on = rng.random(idx.size) < args.floor_frac
        floor[idx] = np.where(on[:, None], rng.uniform(args.floor_lo, args.floor_hi, (idx.size, 2)), 0.0)

    snapshot(0)
    assign(np.ones(N, bool))
    new_floors(np.ones(N, bool))
    obs = sim.observe()
    t0 = time.time()
    for it in range(1, args.iters + 1):
        B_obs = np.zeros((T, N, 2, S.OBS_DIM), np.float32)
        B_act = np.zeros((T, N, 2), np.int64)
        B_lp = np.zeros((T, N, 2), np.float32)
        B_val = np.zeros((T, N, 2), np.float32)
        B_rew = np.zeros((T, N, 2), np.float32)
        B_done = np.zeros((T, N), np.float32)
        B_learn = np.zeros((T, N, 2), np.float32)
        B_acted = np.zeros((T, N, 2), np.float32)
        B_mask = np.ones((T, N, 2, S.N_ACTIONS), bool) if box else None
        B_lapse = np.zeros((T, N, 2), bool)
        res = {"A": 0, "B": 0, "draw": 0, "ko": 0, "eps": 0, "gap": [], "starts": np.zeros(2), "main_n": 0}
        if box:
            usage, succ, learn_ticks = np.zeros((2, boxing.N_TECH)), np.zeros((2, boxing.N_TECH)), np.zeros(2)
        for t in range(T):
            acts = np.zeros((N, 2), np.int64)
            amask = sim.plan_mask() if box else None          # 战术计划限制的动作
            if box and args.floor_frac > 0:
                low = sim.stam < floor * sim.stam_max[None, :]
                amask[:, :, punches] &= ~low[:, :, None]
            for side in (0, 1):
                msk = amask[:, side] if box else None
                a, lp, v = agents[side].act(obs[:, side], rng, msk)
                c = ctrl[:, side]
                for k in np.unique(c[c >= 0]):
                    sel = c == k
                    a[sel] = pool[side][k].act(obs[sel, side], rng, msk[sel] if box else None)
                acts[:, side] = a
                B_lp[t, :, side], B_val[t, :, side] = lp, v
            if box:
                acts, lapsed = boxing.lapse(acts, amask, sim, rng)       # 拳商：昏招
                B_lapse[t] = lapsed
            B_obs[t], B_act[t] = obs, acts
            if box:
                B_mask[t] = amask
            B_learn[t] = ctrl < 0
            rew, done, info = sim.step(acts)
            rew = rew - args.stall_penalty * (info["gap"] > args.stall_gap)[:, None]
            rew = rew - args.draw_penalty * (done & (info["winner"] < 0))[:, None]
            total_ticks += N
            B_rew[t], B_done[t], B_acted[t] = rew, done, info["acted"]
            main = (ctrl < 0).all(1)
            res["gap"].append(info["gap"][main].mean() if main.any() else np.nan)
            res["starts"] += (info["started"][main] > 0).sum(0)
            res["main_n"] += int(main.sum())
            for side in (0, 1):                 # 记录对历史版本的战绩
                vsp = done & (ctrl[:, side] < 0) & (ctrl[:, 1 - side] >= 0)
                for i in np.flatnonzero(vsp):
                    rec = vs_pool[side][ctrl[i, 1 - side]]
                    rec[0] += int(info["winner"][i] == side)
                    rec[1] += 1
            fin = done & main
            if fin.any():
                w = info["winner"][fin]
                res["A"] += int((w == 0).sum()); res["B"] += int((w == 1).sum())
                res["draw"] += int((w == -1).sum()); res["ko"] += int(info["ko"][fin].sum())
                res["eps"] += int(fin.sum())
            if box:
                practice_counts(info, acts, ctrl < 0, sim, usage, succ, learn_ticks)
            assign(done)
            new_floors(done)
            obs = sim.observe()

        if res["eps"] >= 10:                    # 当前版本互打一边倒时，弱的一方多打历史版本
            wa, wb = res["A"] / res["eps"], res["B"] / res["eps"]
            pool_share = np.array([0.45 if wa < 0.25 else 0.25, 0.45 if wb < 0.25 else 0.25])
        frac = (it - 1) / max(1, args.iters - 1)
        ent = args.ent + (args.ent_end - args.ent) * frac
        stats = {}
        for side in (0, 1):
            last_v = agents[side].value(obs[:, side])
            adv = gae(B_rew[..., side], B_val[..., side], B_done, last_v, args.gamma, args.lam)
            ret = adv + B_val[..., side]
            flat = lambda x: x.reshape(T * N, *x.shape[2:])
            st = agents[side].update(
                flat(B_obs[:, :, side]), flat(B_act[..., side]), flat(B_lp[..., side]),
                flat(adv), flat(ret), flat(B_learn[..., side] * B_acted[..., side] * ~B_lapse[..., side]),
                flat(B_learn[..., side]), rng, ent_coef=ent,
                amask=flat(B_mask[:, :, side]) if box else None, pi_update=it > args.vf_warmup)
            stats[bodies[side].id] = st

        if box:
            skills = evolve_skills(skills, talent, usage, succ, learn_ticks, drill * (0.6 + 0.8 * iq[:, None]))
            sim.set_skills(skills)
        el = time.time() - t0
        log = {"iter": it, "ticks": total_ticks, "secs": round(el, 1), "ent_coef": ent,
               "A_win": res["A"], "B_win": res["B"], "draw": res["draw"], "ko": res["ko"], "eps": res["eps"],
               "mean_gap": float(np.nanmean(res["gap"])), "attacks_per_10s": (res["starts"] / max(1, res["main_n"]) / S.DT * 10).round(2).tolist(),
               **{f"{k}_{m}": round(v, 5) for k, s in stats.items() for m, v in s.items()}}
        if box:
            log["skills"] = skills.round(4).tolist()
        append(run_dir, "train_log.jsonl", log)
        if it % 5 == 0 or it == 1:
            print(f"it {it:4d} | {total_ticks/1e6:6.1f}M ticks | {el:6.0f}s | A/B/draw {res['A']}/{res['B']}/{res['draw']} "
                  f"ko {res['ko']} | gap {log['mean_gap']:.2f} | entA {stats['A']['ent']:.2f} entB {stats['B']['ent']:.2f} "
                  f"klA {stats['A']['kl']:.4f}", flush=True)
        if it % args.snap_every == 0:
            snapshot(it)


def practice_counts(info, acts, learn, sim, usage, succ, learn_ticks):
    """记下这一帧里，正在学的那一方用了哪些技术、成功了哪些。"""
    B = boxing
    moves = (B.FWD, B.BACK, B.LEFT, B.RIGHT, B.GUARD_FWD)
    for side in (0, 1):
        lm = learn[:, side]
        if not lm.any():
            continue
        st = info["started"][lm, side]
        usage[side, :12] += np.bincount(st[st > 0] - 1, minlength=12)
        fi, oc = info["fired"][lm, side], info["outcome"][lm, side]
        succ[side, :12] += np.bincount(fi[(fi > 0) & (oc == 1)] - 1, minlength=12)
        ds, ev = info["def_start"][lm, side], info["evaded_by"][lm, side]
        for code, t in B.TECH_OF_DEF.items():
            usage[side, t] += (ds == code).sum()
            succ[side, t] += (ev == code).sum()
        bb, g = info["blocked_by"][lm, side], sim.guard[lm, side]
        usage[side, B.T_GUARD_HIGH] += (g == 1).sum() / 20
        usage[side, B.T_GUARD_BODY] += (g == 2).sum() / 20
        succ[side, B.T_GUARD_HIGH] += (bb == 1).sum()
        succ[side, B.T_GUARD_BODY] += (bb == 2).sum()
        aa, acted = acts[lm, side], info["acted"][lm, side]
        usage[side, B.T_CLINCH] += ((aa == B.CLINCH) & acted).sum()
        succ[side, B.T_CLINCH] += info["clinch"][lm, side].sum()
        usage[side, B.T_PIVOT] += info["pivot"][lm, side].sum()
        usage[side, B.T_FEINT] += (np.isin(aa, (B.FEINT_JAB, B.FEINT_CROSS)) & acted).sum()
        succ[side, B.T_FEINT] += info["faked"][lm, 1 - side].sum()
        usage[side, B.T_FOOT] += (np.isin(aa, moves) & acted).sum() / 20
        usage[side, B.T_SWITCH] += (np.isin(aa, (B.CHANGE_POSTURE, B.SWITCH_STANCE)) & acted).sum()
        learn_ticks[side] += lm.sum()


SKILL_UP, SKILL_DECAY = 0.005, 0.001


def evolve_skills(skills, talent, usage, succ, learn_ticks, drill=None):
    """每 1000 帧里用了几次、成功几次 → 练习量 → 熟练度往上走；不练的往天赋水平退。"""
    per1k = 1000.0 / np.maximum(1, learn_ticks)[:, None]
    ru, rs = usage * per1k, succ * per1k
    practice = 0.3 * np.tanh(ru / 5) + 0.7 * np.tanh(rs / 2)
    use_only = [boxing.T_PIVOT, boxing.T_FOOT, boxing.T_SWITCH]     # 这几项没有"成功"的定义，按用量算
    practice[:, use_only] = np.tanh(ru[:, use_only] / 5)
    if drill is not None:
        practice = practice * drill                    # 流派里练得勤的技术涨得快
    new = skills + SKILL_UP * practice * (1 - skills) - SKILL_DECAY * (skills - talent)
    return np.clip(new, 0.0, 1.0)


def gae(rew, val, done, last_v, gamma, lam):
    T = len(rew)
    adv = np.zeros_like(rew)
    last = np.zeros_like(last_v)
    for t in reversed(range(T)):
        nxt = last_v if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv


def evaluate(run_dir, bodies, pols, bot, ver, it, ticks, n, box=False, skills=None):
    out = {"version": ver, "iter": it, "ticks": ticks}
    if box:
        import boxing_league as BL
        res, line = BL.snapshot_eval(bodies, pols, bot, ver, n, SPARRING,
                                     skills={b.id: skills[i] for i, b in enumerate(bodies)})
        append(run_dir, "evals.jsonl", {**out, **res})
        print(line, flush=True)
        return
    for side, body in enumerate(bodies):
        mb = L.run_matches((body, SPARRING), (pols[side], bot), (body.id, "S"), n, seed=10_000 + ver)
        out[f"{body.id}_vs_S"] = L.summarize(L.fight_metrics(mb, 0))
    mb = L.run_matches(bodies, pols, (bodies[0].id, bodies[1].id), n, seed=20_000 + ver)
    out["AvB_A"] = L.summarize(L.fight_metrics(mb, 0))
    out["AvB_B"] = L.summarize(L.fight_metrics(mb, 1))
    append(run_dir, "evals.jsonl", out)
    a, b = out["A_vs_S"], out["B_vs_S"]
    print(f"  [v{ver:04d}] vs陪练 胜率 A {a['win']:.2f} B {b['win']:.2f} | 距离 A {a['mean_gap']:.2f} B {b['mean_gap']:.2f} "
          f"| 出拳/10s A {a['attack_rate']:.1f} B {b['attack_rate']:.1f} | AvB A胜 {out['AvB_A']['win']:.2f} "
          f"B胜 {out['AvB_B']['win']:.2f}", flush=True)


def append(run_dir, name, obj):
    with open(os.path.join(run_dir, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
