"""拳击版赛后分析：技术使用表（验收"会不会打拳击"）、打法对比、成长曲线、回放。

用法：py -X utf8 analyze_boxing.py --run b1 [--version 12] [--n 64]
"""
import argparse
import json
import os
import time

import numpy as np

import boxing as B
import boxing_league as BL
from analyze import load_run
from bodies import LANKY, STOCKY, SPARRING
from boxing_bots import BoxingSparring
from league import cohens_d, save_json
import schools as SC

EVAL_SEED = 777
# 打法指纹的动作分组
GROUPS = [("压上", (B.FWD, B.GUARD_FWD)), ("后退", (B.BACK,)), ("横移/轴转", (B.LEFT, B.RIGHT, B.PIVOT_L, B.PIVOT_R)),
          ("举架/护体", (B.GUARD_HIGH, B.GUARD_BODY)), ("闪躲/拍挡", (B.PARRY, B.SLIP, B.DUCK, B.PULL)),
          ("刺拳", (B.JAB_H, B.JAB_B)), ("后手直拳", (B.CROSS_H, B.CROSS_B)),
          ("勾拳", (B.LHOOK_H, B.LHOOK_B, B.RHOOK_H, B.RHOOK_B)), ("上勾拳", (B.LUPPER_H, B.LUPPER_B, B.RUPPER_H, B.RUPPER_B)),
          ("虚晃", (B.FEINT_JAB, B.FEINT_CROSS)), ("缠抱", (B.CLINCH,)), ("不动", (B.STAY,))]
DIST_BINS = np.linspace(0.6, 3.0, 13)


def fingerprint(bt, k):
    f = bt.fields
    v = BL._valid(bt) & f["acted"][:, :, k]
    dist, _ = BL._geom(bt)
    a = f["action"][:, :, k][v]
    b = np.clip(np.digitize(dist[v], DIST_BINS) - 1, 0, len(DIST_BINS) - 2)
    grp = np.zeros(B.N_ACTIONS, int)
    for gi, (_, acts) in enumerate(GROUPS):
        grp[list(acts)] = gi
    counts = np.zeros((len(GROUPS), len(DIST_BINS) - 1))
    np.add.at(counts, (grp[a], b), 1)
    tot = counts.sum(0, keepdims=True)
    return (counts / np.maximum(tot, 1)).round(4).tolist(), (tot[0] / max(1, tot.sum())).round(4).tolist()


def card(body, skills_now):
    """角色卡：身体、流派、架势、站架、性格、天赋和当前熟练度，以及属性预算分。"""
    import bodies as BD
    pt = BD.attr_points(body)
    return {"id": body.id, "label": body.label, "school": body.school or "无", "posture": B.POSTURE_NAMES[SC.posture(body)],
            "stance": "反架" if body.stance == "southpaw" else "正架",
            "budget": [round(BD.budget_total(body), 1), BD.BUDGET],
            "body": [["臂展", f"{body.reach:.2f} 米 · {pt['臂展']:.1f}分"], ["体重", f"{body.mass:g} 公斤"], ["力量", f"{body.power:.2f} · {pt['力量']:.1f}分"],
                     ["手速", f"{body.hand_speed:.2f} · {pt['手速']:.1f}分"], ["步速", f"{body.foot_speed:.2f} · {pt['步速']:.1f}分"],
                     ["体能", f"{body.cardio:.2f} · {pt['体能']:.1f}分"], ["抗击打", f"{body.chin:.2f} · {pt['抗击打']:.1f}分"],
                     ["反应", f"{body.react_ticks * 50} 毫秒 · {pt['反应']:.1f}分"]],
            "temper": SC.temperament(body).round(2).tolist() + SC.mind(body).round(2).tolist(), "talent": SC.talent(body).round(3).tolist(),
            "skills": [round(float(x), 3) for x in skills_now], "drill": SC.drill(body).tolist()}


def _cat(a):
    for gi, (_, acts) in enumerate(GROUPS):
        if a in acts:
            return gi
    return len(GROUPS) - 1


CAT = np.array([_cat(a) for a in range(B.N_ACTIONS)])


def habits(bt, k):
    """从比赛里统计出来的习惯（癖好）。"""
    f = bt.fields
    v = BL._valid(bt)
    j = 1 - k
    T = len(v)
    acted, act = f["acted"][:, :, k] & v, f["action"][:, :, k]
    out = []

    def follow(events, window, title):
        cnt = np.zeros(len(GROUPS))
        for t, i in zip(*events):
            sl = slice(t + 1, min(T, t + 1 + window))
            idx = np.flatnonzero(acted[sl, i] & (act[sl, i] != B.STAY))
            cnt[CAT[act[t + 1 + idx[0], i]] if idx.size else len(GROUPS) - 1] += 1
        tot = cnt.sum()
        if tot >= 10:
            order = np.argsort(-cnt)[:3]
            out.append({"title": title, "n": int(tot),
                        "items": [[GROUPS[g][0], round(float(cnt[g] / tot), 3)] for g in order if cnt[g] > 0]})

    jab = np.nonzero((f["fired"][:, :, k] == 1) & v)
    follow(jab, 10, "刺拳打出去之后，接着")
    opp_start = np.nonzero((f["started"][:, :, j] > 0) & v)
    follow(opp_start, 6, "看到对手起手出拳，第一反应")
    hit = np.nonzero((f["outcome"][:, :, j] == 1) & v)
    follow(hit, 12, "挨了一拳之后，接着")
    post = np.bincount(f["posture"][:, :, k][v].astype(int), minlength=B.N_POSTURES) / max(1, v.sum())
    out.append({"title": "架势用了多久", "n": int(v.sum()), "items": [[B.POSTURE_NAMES[q], round(float(post[q]), 3)] for q in np.argsort(-post) if post[q] > 0.005]})
    return out


def method_counts(bt, k):
    w = bt.winner == k
    return {"KO": int((w & (bt.method == B.M_KO)).sum()), "TKO": int((w & (bt.method == B.M_TKO)).sum()),
            "判定": int((w & (bt.method == B.M_DEC)).sum()), "平局": int((bt.winner == -1).sum()),
            "输": int((bt.winner == 1 - k).sum())}


def pick_replays(run, res, ctrl_info):
    picks = []
    ab = res["AvB"]
    early = np.flatnonzero(np.isin(ab.method, (B.M_KO, B.M_TKO)))
    dec = np.flatnonzero(np.isin(ab.method, (B.M_DEC, B.M_DRAW)))
    if early.size:
        i = int(early[0])
        picks.append(("AvB_%d" % i, f"A 对 B · {B.METHOD_NAMES[ab.method[i]]}", ab, i, ("A", "B")))
    if dec.size:
        i = int(dec[0])
        picks.append(("AvB_%d" % i, f"A 对 B · {B.METHOD_NAMES[ab.method[i]]}", ab, i, ("A", "B")))
    for c in ("A", "B"):
        bt = res[f"{c}vS"]
        picks.append((f"{c}vS_0", f"{c} 对标准陪练 · {B.METHOD_NAMES[bt.method[0]]}", bt, 0, (c, "S")))
    mdir = os.path.join(run["dir"], "matches")
    os.makedirs(mdir, exist_ok=True)
    out = []
    for key, title, bt, i, cs in picks:
        rec = BL.replay_record(bt, i, [ctrl_info(c) for c in cs])
        fn = f"v{run['version']:04d}_{key}.json"
        save_json(os.path.join(mdir, fn), rec)
        frames = BL.resimulate(rec)
        out.append({"title": title, "file": f"{run['name']}/matches/{fn}", "bodies": rec["bodies"],
                    "result": rec["result"], "frames": frames})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = load_run(args.run, args.version)
    bot = BoxingSparring()

    def ctrl_info(c):
        if c == "S":
            return {"name": bot.name}
        v = run["pick"][c]
        return {"name": f"{run['name']}/{v['file']}", "sha256": v["sha256"]}

    t0 = time.time()
    res = {}
    skills = np.stack([np.array(run["pick"][c].get("skills", SC.talent(b))) for c, b in (("A", LANKY), ("B", STOCKY))])
    sk_vs = lambda i: np.stack([skills[i], SC.talent(SPARRING)])
    for c, body, i in (("A", LANKY, 0), ("B", STOCKY, 1)):
        res[f"{c}vS"] = BL.run_matches((body, SPARRING), (run["pols"][c], bot), (c, "S"), args.n, seed=EVAL_SEED, skills=sk_vs(i))
    # 赛前战术会议：每个计划和对手模拟一回合，排名；比赛中角落按排名在回合间调整
    pols = (run["pols"]["A"], run["pols"]["B"])
    rank = {c: BL.choose_plan((LANKY, STOCKY), pols, side, skills, n=16) for side, c in enumerate("AB")}
    plans = (rank["A"][0][0], rank["B"][0][0])
    res["AvB"] = BL.run_matches((LANKY, STOCKY), pols, ("A", "B"), args.n, seed=EVAL_SEED + 1, skills=skills,
                                plans=plans, plan_rank=([p for p, _ in rank["A"]], [p for p, _ in rank["B"]]))
    print(f"评估用了 {time.time() - t0:.0f} 秒；赛前战术 A {B.PLAN_NAMES[plans[0]]}，B {B.PLAN_NAMES[plans[1]]}")

    tech = {}
    for key, bt, k in (("A_vs_S", res["AvS"], 0), ("B_vs_S", res["BvS"], 0), ("AvB_A", res["AvB"], 0), ("AvB_B", res["AvB"], 1)):
        rows = BL.technique_table(bt, k)
        tech[key] = {"rows": rows, "coverage": BL.coverage(rows)}
    mA, mB = BL.fight_metrics(res["AvS"], 0), BL.fight_metrics(res["BvS"], 0)
    style = [{"key": k, "label": lab, "A": float(np.nanmean(mA[k])), "B": float(np.nanmean(mB[k])),
              "d": cohens_d(mA[k].astype(float), mB[k].astype(float))} for k, lab in BL.STYLE_METRICS.items()]
    results = {
        "AvS": method_counts(res["AvS"], 0), "BvS": method_counts(res["BvS"], 0),
        "AvB_A": method_counts(res["AvB"], 0), "AvB_B": method_counts(res["AvB"], 1),
    }
    fps = {}
    for key, bt, k in (("A", res["AvS"], 0), ("B", res["BvS"], 0)):
        fp, occ = fingerprint(bt, k)
        fps[key] = {"fp": fp, "occ": occ}
    for key, t in tech.items():
        print(f"{key}: 成功用过 {t['coverage'][0]}/{t['coverage'][1]} 种技术")
    print("战绩", json.dumps(results, ensure_ascii=False))

    growth = [{"v": e["version"], "ticks": e["ticks"] / 1e6,
               "A_win": e["A_vs_S"]["win"], "B_win": e["B_vs_S"]["win"],
               "A_cov": e["A_vs_S"].get("coverage", 0), "B_cov": e["B_vs_S"].get("coverage", 0),
               "A_fin": e["A_vs_S"]["ko_win"] + e["A_vs_S"]["tko_win"], "B_fin": e["B_vs_S"]["ko_win"] + e["B_vs_S"]["tko_win"],
               "AvB_A": e["AvB_A"]["win"], "AvB_B": e["AvB_B"]["win"],
               "A_comp": e["A_vs_S"]["comp_mean"], "B_comp": e["B_vs_S"]["comp_mean"]} for e in run["evals"]]
    replays = pick_replays(run, res, ctrl_info)
    vers = [json.loads(l) for l in open(os.path.join(run["dir"], "versions.jsonl"), encoding="utf-8")]
    skill_hist = {c: [[v["version"], v["ticks"] / 1e6, v["skills"]] for v in vers if v["char"] == c and "skills" in v] for c in "AB"}
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "n": args.n,
        "run": {"name": run["name"], "version": run["version"], "ticks": run["pick"]["A"]["ticks"],
                "sha": {c: run["pick"][c]["sha256"] for c in ("A", "B")}},
        "bodies": {b.id: b.to_dict() for b in (LANKY, STOCKY, SPARRING)},
        "results": results, "tech": tech, "style": style, "fingerprints": fps,
        "groups": [g[0] for g in GROUPS], "dist_bins": DIST_BINS.round(2).tolist(),
        "growth": growth, "replays": replays,
        "cards": {"A": card(LANKY, skills[0]), "B": card(STOCKY, skills[1])},
        "plans": {"names": B.PLAN_NAMES, "A": rank["A"], "B": rank["B"], "chosen": list(plans)},
        "habits": {"A": habits(res["AvB"], 0), "B": habits(res["AvB"], 1)},
        "skill_hist": skill_hist, "tech_names": B.TECH_NAMES, "temper_names": SC.TEMPER_NAMES + SC.MIND_NAMES,
        "postures": B.POSTURE_NAMES,
        "sim": {"RING": B.RING, "BODY_R": B.BODY_R, "SHOULDER": B.SHOULDER, "DT": B.DT, "ROUND_TICKS": B.ROUND_TICKS,
                "RNG": B.RNG.tolist(), "TYPE": B.TYPE.tolist(), "TGT": B.TGT.tolist(), "HAND": B.HAND.tolist(),
                "version": B.SIM_VERSION, "actions": B.ACTION_NAMES, "phases": B.PHASE_NAMES, "methods": B.METHOD_NAMES},
    }
    out = args.out or os.path.join("runs", f"boxing_report_{run['name']}_v{run['version']:04d}.html")
    tpl = open("boxing_report_template.html", encoding="utf-8").read()
    with open(out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    print("报告:", os.path.abspath(out))


if __name__ == "__main__":
    main()
