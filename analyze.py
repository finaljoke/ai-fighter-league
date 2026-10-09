"""赛后分析：打法分化测试 + 报告页。

做三件事：
  1. 每个 run 的最终版本分别打标准陪练、互打，算打法指标和打法指纹
  2. 拿多个随机种子当对照：A、B 之间的差异要明显大于同一个身体换种子的差异，才算"打法长出来了"
  3. 存几场回放（只存初始距离和动作），重新模拟出画面，生成一个能直接打开的 HTML 报告

用法：py -X utf8 analyze.py --runs s1 s2 s3
"""
import argparse
import json
import os
import time

import numpy as np

import league as L
import sim as S
from bodies import LANKY, STOCKY, SPARRING
from bots import SparringBot
from nets import load_policy

CORE = ["mean_gap", "start_gap", "attack_rate", "jab_share", "fwd", "back", "block", "guard_fwd", "cornered"]
EVAL_SEED = 777


def load_run(name, version=None):
    d = os.path.join("runs", name)
    vers = [json.loads(l) for l in open(os.path.join(d, "versions.jsonl"), encoding="utf-8")]
    if version is None:
        version = max(v["version"] for v in vers)
    pick = {v["char"]: v for v in vers if v["version"] == version}
    pols = {c: load_policy(os.path.join(d, "ckpt", v["file"])) for c, v in pick.items()}
    for c, v in pick.items():
        assert pols[c].sha256() == v["sha256"], f"{name} {c} v{version} 哈希对不上，存档被改过"
    evals = [json.loads(l) for l in open(os.path.join(d, "evals.jsonl"), encoding="utf-8")]
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    return {"name": name, "dir": d, "version": version, "pick": pick, "pols": pols, "evals": evals, "cfg": cfg}


def ctrl_info(run, c):
    if c == "S":
        return {"name": SparringBot.name}
    v = run["pick"][c]
    return {"name": f"{run['name']}/{v['file']}", "sha256": v["sha256"]}


def evaluate_run(run, n):
    bot = SparringBot()
    out = {}
    for c, body in (("A", LANKY), ("B", STOCKY)):
        mb = L.run_matches((body, SPARRING), (run["pols"][c], bot), (c, "S"), n, seed=EVAL_SEED)
        out[f"{c}vS"] = mb
    out["AvB"] = L.run_matches((LANKY, STOCKY), (run["pols"]["A"], run["pols"]["B"]), ("A", "B"), n, seed=EVAL_SEED + 1)
    return out


def verdict(per_run):
    """per_run: [{"A": metrics, "B": metrics}]，都是对标准陪练的每场指标。"""
    rows = []
    prim = per_run[0]
    for k in L.STYLE_METRICS:
        a_means = np.array([np.nanmean(r["A"][k]) for r in per_run])
        b_means = np.array([np.nanmean(r["B"][k]) for r in per_run])
        d = L.cohens_d(prim["A"][k], prim["B"][k])
        signs = np.sign(a_means - b_means)
        consistent = bool(np.all(signs == signs[0]) and signs[0] != 0)
        if len(per_run) >= 2:
            noise = np.sqrt((a_means.var(ddof=1) + b_means.var(ddof=1)) / 2)
            ratio = float(abs(a_means.mean() - b_means.mean()) / (noise + 1e-9))
        else:
            ratio = float("nan")
        sep = consistent and abs(d) >= 0.8 and (len(per_run) < 2 or ratio >= 3)
        rows.append({"key": k, "label": L.STYLE_METRICS[k], "A": a_means.tolist(), "B": b_means.tolist(),
                     "d": d, "ratio": ratio, "consistent": consistent, "separated": bool(sep), "core": k in CORE})
    n_sep = sum(r["separated"] for r in rows if r["core"])
    if n_sep >= 3:
        level, text = "pass", f"分开了：{len(CORE)} 项核心打法指标里有 {n_sep} 项，A 和 B 差得明显，而且换随机种子方向不变。"
    elif n_sep >= 1:
        level, text = "partial", f"有一点分化：只有 {n_sep} 项核心指标分得开，还不够撑起\"不同流派\"。"
    else:
        level, text = "fail", "没分开：A 和 B 的差异不比换随机种子的差异大，身体参数没有长成不同打法。"
    return {"level": level, "text": text, "n_sep": n_sep, "rows": rows}


def pick_replays(run, res):
    picks = []
    ab = res["AvB"]
    for w, label in ((0, "A 赢的一场"), (1, "B 赢的一场")):
        idx = np.flatnonzero(ab.winner == w)
        if idx.size:
            ko = idx[ab.ko[idx]]
            i = int(ko[0] if ko.size else idx[0])
            picks.append((f"AvB_{i}", f"A 对 B · {label}" + ("（KO）" if ab.ko[i] else "（判定）"), ab, i, ("A", "B")))
    if not any(p[2] is ab for p in picks):
        picks.append(("AvB_0", "A 对 B · 平局", ab, 0, ("A", "B")))
    for c in ("A", "B"):
        mb = res[f"{c}vS"]
        picks.append((f"{c}vS_0", f"{c} 对标准陪练", mb, 0, (c, "S")))
    mdir = os.path.join(run["dir"], "matches")
    os.makedirs(mdir, exist_ok=True)
    out = []
    for key, title, mb, i, cs in picks:
        rec = L.replay_record(mb, i, [ctrl_info(run, c) for c in cs])
        fn = f"v{run['version']:04d}_{key}.json"
        L.save_json(os.path.join(mdir, fn), rec)
        frames = L.resimulate(rec)       # 从动作档重新算画面，顺便核对结果
        out.append({"title": title, "file": f"{run['name']}/matches/{fn}", "bodies": rec["bodies"],
                    "result": rec["result"], "frames": frames})
    return out


def growth(runs):
    out = []
    for r in runs:
        pts = [{"v": e["version"], "ticks": e["ticks"] / 1e6,
                "A_win": e["A_vs_S"]["win"], "B_win": e["B_vs_S"]["win"],
                "A_gap": e["A_vs_S"]["mean_gap"], "B_gap": e["B_vs_S"]["mean_gap"],
                "AvB_A": e["AvB_A"]["win"], "AvB_B": e["AvB_B"]["win"]} for e in r["evals"]]
        out.append({"run": r["name"], "points": pts})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="第一个是主 run，其余当种子对照")
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [load_run(r, args.version) for r in args.runs]
    per_run, fps, tables = [], [], []
    for r in runs:
        res = evaluate_run(r, args.n)
        r["res"] = res
        mA = L.fight_metrics(res["AvS"], 0)
        mB = L.fight_metrics(res["BvS"], 0)
        per_run.append({"A": mA, "B": mB})
        fA, occA = L.fingerprint(res["AvS"], 0)
        fB, occB = L.fingerprint(res["BvS"], 0)
        fps.append({"run": r["name"], "A": fA.round(4).tolist(), "B": fB.round(4).tolist(),
                    "occA": (occA / occA.sum()).round(4).tolist(), "occB": (occB / occB.sum()).round(4).tolist()})
        tables.append({"run": r["name"], "AvS": L.summarize(mA), "BvS": L.summarize(mB),
                       "AvB_A": L.summarize(L.fight_metrics(res["AvB"], 0)),
                       "AvB_B": L.summarize(L.fight_metrics(res["AvB"], 1))})
        print(f"{r['name']} v{r['version']}: A胜陪练 {tables[-1]['AvS']['win']:.2f} B胜陪练 {tables[-1]['BvS']['win']:.2f} "
              f"A对B A胜 {tables[-1]['AvB_A']['win']:.2f} B胜 {tables[-1]['AvB_B']['win']:.2f}")

    v = verdict(per_run)
    print(v["text"])
    for row in v["rows"]:
        print(f"  {row['label']:<16} A {np.mean(row['A']):7.3f} B {np.mean(row['B']):7.3f} d {row['d']:6.2f} "
              f"种子比 {row['ratio']:6.1f} {'方向一致' if row['consistent'] else '方向不一'} {'✓' if row['separated'] else ''}")

    replays = pick_replays(runs[0], runs[0]["res"])
    sim0 = S.Sim(1, (LANKY, STOCKY))
    simS = S.Sim(1, (LANKY, SPARRING))
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "runs": [{"name": r["name"], "version": r["version"], "ticks": r["pick"]["A"]["ticks"],
                  "sha": {c: r["pick"][c]["sha256"] for c in ("A", "B")}} for r in runs],
        "n": args.n, "verdict": v, "fingerprints": fps, "tables": tables,
        "metric_labels": L.STYLE_METRICS, "growth": growth(runs), "replays": replays,
        "gap_bins": L.GAP_BINS.round(3).tolist(),
        "ranges": {"A_jab": float(sim0.range[1, 0]), "B_jab": float(sim0.range[1, 1]),
                   "S_jab": float(simS.range[1, 1])},
        "bodies": {b.id: b.to_dict() for b in (LANKY, STOCKY, SPARRING)},
        "sim": {"RING": S.RING, "BODY_R": S.BODY_R, "SHOULDER": S.SHOULDER, "DT": S.DT,
                "MOVE_RANGE": S.MOVE_RANGE.tolist(), "version": S.SIM_VERSION},
    }
    out = args.out or os.path.join("runs", f"report_{'-'.join(args.runs)}.html")
    tpl = open("report_template.html", encoding="utf-8").read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("报告:", os.path.abspath(out))


if __name__ == "__main__":
    main()
