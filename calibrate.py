"""真实度校准：拿一个训练档的 A、B 打一批比赛，按 CompuBox 口径统计，和真实比赛的目标比。

目标在动规则之前定下来，之后不改目标去凑结果（来源和口径见 docs/拳击技术规格.md"真实度目标"）。
用法：py -X utf8 calibrate.py --run b2 [--version 50] [--n 128] [--seed 4242]
结果追加到 runs/<run>/calibration.jsonl。
"""
import argparse
import json
import os
import time

import numpy as np

import boxing as B
import boxing_league as BL
import schools as SC
from analyze import load_run
from bodies import LANKY, STOCKY

# kind：real = 真实比赛数据；design = 设计约束（不是统计数据）；record = 只记录，暂不设目标
TARGETS = [
    ("thrown_pm", "每分钟出拳（每人）", 18.0, 22.0, "real", "职业约 19.5（ESPN 2012）；奥运约 20~21（Davis 2015/2018）"),
    ("connect", "总命中率", 0.30, 0.40, "real", "职业平均约 32%（ESPN 2012）；重量级拳王赛 34.6%（Finlay 2022）"),
    ("connect_jab", "刺拳命中率", 0.18, 0.30, "real", "没有公开总平均；由总命中和重拳命中推算约 22~26%"),
    ("connect_power", "重拳命中率", 0.30, 0.45, "real", "CompuBox 平均 36.9%（Bad Left Hook 2019）"),
    ("tired_share", "累垮时间占比（体能 < 25%）", 0.0, 0.20, "design", "常识约束：真实拳手不会整场空着油箱打"),
    ("posture_max", "最常用架势的时间占比（两人都算）", 0.0, 0.60, "design", "没有一种架势被所有角色都用超过六成时间"),
    ("hp_tko_share", "血条归零判 TKO 的比例", 0.0, 0.0, "design", "原则：血条只是资源池，不判胜负"),
    ("stoppage", "提前结束比例（KO+TKO）", None, None, "record", "职业 47%（KO 15%、TKO 32%，BoxRec 2014~2024）；奥运只有几个百分点。待定"),
    ("kd_per_fight", "每场击倒次数", None, None, "record", "没有可靠数据，不设硬指标"),
    ("last_round_stop", "提前结束里发生在最后一回合的比例", None, None, "record", "真实比赛偏早段（Hutchison 2014；职业平均第 2.59 回合）"),
]


def metrics(bt):
    f = bt.fields
    v = BL._valid(bt)                                   # (T, n)
    n = bt.n
    mins = bt.end * B.DT / 60.0
    fired = f["fired"].astype(int)                      # 0 没出拳，>0 是拳种 +1
    thrown = (fired > 0) & v[..., None]
    jab = thrown & (fired <= 2)
    power = thrown & (fired > 2)
    lw = f["landed_cb"] * thrown
    out = {"n": n, "seconds": float(np.mean(bt.end * B.DT))}
    out["thrown_pm"] = float(np.mean(thrown.sum(0) / mins[:, None]))
    out["connect"] = float(lw.sum() / max(1, thrown.sum()))
    out["connect_jab"] = float((lw * jab).sum() / max(1, jab.sum()))
    out["connect_power"] = float((lw * power).sum() / max(1, power.sum()))
    out["tired_share"] = float(((f["stam"] < B.TIRED) & v[..., None]).sum() / (2 * v.sum()))
    post = np.array([[((f["posture"][:, :, k] == p) & v).sum() / v.sum() for p in range(B.N_POSTURES)] for k in (0, 1)])
    out["posture"] = post.round(3).tolist()
    out["posture_max"] = float(post.max(1).min())      # 两个人都用得最多的那个架势里，较小的那个占比
    defend = ((f["guard"] > 0) | (f["phase"] == 5)) & v[..., None]
    out["defend_share"] = (defend.sum(0).sum(0) / v.sum()).round(3).tolist()
    stop = (bt.method == B.M_KO) | (bt.method == B.M_TKO)
    out["stoppage"] = float(stop.mean())
    out["ko"] = float((bt.method == B.M_KO).mean())
    out["tko"] = float((bt.method == B.M_TKO).mean())
    out["decision"] = float((bt.method == B.M_DEC).mean())
    out["draw"] = float((bt.method == B.M_DRAW).mean())
    tko = bt.method == B.M_TKO
    out["tko_cause"] = {k: int(((bt.tko_cause == c) & tko).sum()) for c, k in ((1, "血条归零"), (2, "连续挨打不还手"), (3, "一回合倒三次"))}
    out["hp_tko_share"] = float(((bt.tko_cause == 1) & tko).sum() / max(1, tko.sum()))
    out["kd_per_fight"] = float((f["kd"] & v[..., None]).sum() / n)
    out["last_round_stop"] = float((stop & (bt.end_round >= bt.n_rounds - 1)).sum() / max(1, stop.sum()))
    out["win"] = [int((bt.winner == 0).sum()), int((bt.winner == 1).sum())]
    return out


def verdict(val, lo, hi, kind):
    if kind == "record" or lo is None:
        return "—"
    return "✓" if lo - 1e-9 <= val <= hi + 1e-9 else "✗"


def report(m):
    lines = []
    for key, label, lo, hi, kind, src in TARGETS:
        val = m[key]
        tgt = "待定" if lo is None else (f"{lo:g}" if lo == hi else f"{lo:g}~{hi:g}")
        pct = key.startswith("connect") or key.endswith("share") or key in ("stoppage", "posture_max", "last_round_stop")
        shown = f"{val:.1%}" if pct else f"{val:.2f}"
        if pct and lo is not None:
            tgt = f"{lo:.0%}" if lo == hi else f"{lo:.0%}~{hi:.0%}"
        lines.append(f"{verdict(val, lo, hi, kind)} {label}：{shown}（目标 {tgt}，{'真实数据' if kind == 'real' else '设计约束' if kind == 'design' else '只记录'}）")
    lines.append(f"  结局：KO {m['ko']:.0%}、TKO {m['tko']:.0%}、判定 {m['decision']:.0%}、平局 {m['draw']:.0%}；TKO 原因 {m['tko_cause']}；"
                 f"A 胜 {m['win'][0]}、B 胜 {m['win'][1]}；平均 {m['seconds']:.0f} 秒")
    lines.append(f"  架势占比（标准/长架/抱架/低手） A {m['posture'][0]}  B {m['posture'][1]}；举架或闪躲占时 A {m['defend_share'][0]:.1%} B {m['defend_share'][1]:.1%}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()
    run = load_run(args.run, args.version)
    bodies = (LANKY, STOCKY)
    skills = np.stack([np.array(run["pick"][c].get("skills", SC.talent(b))) for c, b in zip("AB", bodies)])
    t0 = time.time()
    bt = BL.run_matches(bodies, (run["pols"]["A"], run["pols"]["B"]), ("A", "B"), args.n, seed=args.seed, skills=skills)
    m = metrics(bt)
    print(f"{args.run} v{run['version']}，{args.n} 场 A 对 B（{time.time() - t0:.0f} 秒）")
    print(report(m))
    rec = {"version": run["version"], "n": args.n, "seed": args.seed, "sim_version": B.SIM_VERSION,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), **m}
    with open(os.path.join(run["dir"], "calibration.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
