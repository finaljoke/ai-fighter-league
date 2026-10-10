"""试播集：把正史训练档的成长档案、首战、角色卡和立绘拼成一集可以播放、可以导出视频的页面。

用法：py -X utf8 make_episode.py --run canon_s1 [--out runs/episode_pilot.html]
首战按"结果不重来"：第一次生成时打一场并存进 runs/<run>/first_fight.json，以后只读存档。
立绘和人设放在本地 characters/（合伙人的素材，不进仓库）。
"""
import argparse
import base64
import json
import os

import numpy as np

import bodies as BD
import boxing as B
import boxing_league as BL
from analyze import load_run

# 角色卡（人设摘要、台词、立绘位置）放在本地 characters/roster.json：合伙人的素材，不进仓库
CHARS = json.load(open(os.path.join("characters", "roster.json"), encoding="utf-8"))
ORDER = ["LJ", "GY"]          # 红角在前


def portrait_uri(cid):
    """从三视图里裁出正面立绘（ffmpeg），转成 data URI 嵌进页面。"""
    out = os.path.join("runs", "episode_assets", f"{cid}.jpg")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        os.system(f'ffmpeg -hide_banner -loglevel error -y -i "{CHARS[cid]["img"]}" '
                  f'-vf "crop=580:1024:{CHARS[cid]["crop"]}:0,scale=435:768" -q:v 3 "{out}"')
    return "data:image/jpeg;base64," + base64.b64encode(open(out, "rb").read()).decode()


def body_of(cid):
    return BD.ALL[cid]


def char_card(cid, skills):
    b = body_of(cid)
    pts = BD.attr_points(b)
    H = 1.72 + (b.reach - 0.75) * 0.75
    top = B.TECH_NAMES[int(np.argmax(skills[:12]))]
    return {**{k: v for k, v in CHARS[cid].items() if k not in ("img", "crop")},
            "name": b.label.split("·")[0], "school": b.school,
            "attrs": [[k, round(float(v), 1)] for k, v in pts.items()],
            "budget": int(round(BD.budget_total(b))),
            "body": f"身高 {H:.2f} 米 · 臂展 {b.reach:.2f} 米 · {b.mass:g} 公斤", "signature": top}


def first_fight(run, skills_of):
    """首战：只打一次，存档；以后重新生成页面只读存档。"""
    path = os.path.join(run["dir"], "first_fight.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    bodies = tuple(body_of(c) for c in ORDER)
    pols = tuple(run["pols"][c] for c in ORDER)
    skills = np.stack([skills_of[c] for c in ORDER])
    # 赛前战术会议：两边的角落各自模拟每个战术、排名，比赛中回合间按排名调整
    rank = [BL.choose_plan(bodies, pols, side, skills, n=16) for side in (0, 1)]
    plans = (rank[0][0][0], rank[1][0][0])
    seed = 20261010
    bt = BL.run_matches(bodies, pols, tuple(ORDER), 1, seed=seed, skills=skills, plans=plans,
                        plan_rank=([p for p, _ in rank[0]], [p for p, _ in rank[1]]))
    rec = BL.replay_record(bt, 0, [{"name": body_of(c).label} for c in ORDER])
    w = int(bt.winner[0])
    card = {"title": "首战", "fighters": ORDER, "bodies": rec["bodies"], "replay": rec, "plans": [B.PLAN_NAMES[p] for p in plans],
            "result": {"winner": ORDER[w] if w >= 0 else None, "method": B.METHOD_NAMES[int(bt.method[0])],
                       "judges": bt.judges[0].tolist(), "seconds": round(float(bt.end[0]) * B.DT, 1)},
            "source": {"run": run["name"], "version": run["version"], "seed": seed}}
    json.dump(card, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return card


def best_window(frames, secs=9.0):
    """挑这场里最热闹的一段（打中、倒地最多），返回起始帧。"""
    n, win = len(frames), int(secs * 20)
    score = np.zeros(n)
    for t, fr in enumerate(frames):
        for k in (0, 1):
            f = fr["f"][k]
            score[t] += (f[11] == 1) * (1 + 2 * (f[10] - 1 >= 2)) + 8 * f[14]
    if n <= win:
        return 0
    cs = np.concatenate([[0], np.cumsum(score)])
    sums = cs[win:] - cs[:-win]
    return int(np.argmax(sums))


def growth_clips(runs, milestones, coach):
    """成长片段：正史可能分几段训练档接着练，把各段的成长档案按轮数接起来挑。"""
    vs = lambda it: "（对陪练教练）" if it <= coach else ""
    items = []                                                   # (轮数, 训练档, 版本)
    for run in runs:
        gdir = os.path.join(run["dir"], "growth")
        for f in sorted(os.listdir(gdir)):
            if f.endswith(".json"):
                g = json.load(open(os.path.join(gdir, f), encoding="utf-8"))
                if not items or g["iter"] > items[-1][0]:
                    items.append((g["iter"], run, int(f[1:5])))
    if not items:
        return [], 0
    n = len(items)
    picks = sorted(set([0, min(n - 1, 1), n // 4, n // 2, (3 * n) // 4, n - 1]))
    clips, prev_iter = [], -1
    total_fights = 0
    for pi in picks:
        _, run, v = items[pi]
        evals = run["evals"]
        g = json.load(open(os.path.join(run["dir"], "growth", f"v{v:04d}.json"), encoding="utf-8"))
        rec = g["replays"][0]
        frames = BL.resimulate(rec)
        start = best_window(frames)
        new = [m for m in milestones if prev_iter < m["iter"] <= g["iter"]]
        big = [m for m in new if any(x in m["event"] for x in ("击倒", "KO", "TKO"))]
        ev = next((e for e in evals if e["version"] == v), None)
        if g["iter"] == 0:
            cap = {"text": "刚出生：不会控制距离，也不会出拳", "sub": "两具赛博肉体第一次站上擂台"}
            voice = "刚出生的他们，还不知道拳该往哪里打。"
        elif new:
            m = (big or new)[-1]
            cap = {"text": f"{m['name'].split('·')[0]} {m['event']}{vs(m['iter'])}", "sub": f"{'这段之前 · ' if m['iter'] < g['iter'] else ''}训练第 {m['iter']} 轮 · 第 {m['fights']} 场左右", "big": bool(big)}
            voice = f"训练第{m['iter']}轮，{m['name'].split('·')[0]}{m['event']}{'，对手是陪练教练' if m['iter'] <= coach else ''}。"
        else:
            cap = None
            voice = f"训练第{g['iter']}轮。"
        sub = f"已对打 {g['fights']} 场"
        if ev:
            ids = [b["id"] for b in rec["bodies"]]
            rate = [ev.get(f"{i}_vs_S", {}).get("punch_rate", 0) for i in ids]
            sub += f" · 每 10 秒出拳 {rate[0]:.0f} / {rate[1]:.0f}"
        clips.append({"version": v, "iter": g["iter"], "fights": g["fights"], "frames": frames, "bodies": rec["bodies"],
                      "from": start, "dur": 9.0, "speed": 1, "when": f"训练第 {g['iter']} 轮" if g["iter"] else "第 0 轮 · 出生",
                      "when_sub": sub + ("　· 启蒙期：跟陪练教练学" if 0 < g["iter"] <= coach else ""), "caption": cap, "voice": voice})
        prev_iter = g["iter"]
        total_fights = g["fights"]
    return clips, total_fights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="正史训练档，分几段接着练的用逗号隔开，按先后顺序")
    ap.add_argument("--out", default=os.path.join("runs", "episode_pilot.html"))
    args = ap.parse_args()
    runs = [load_run(r) for r in args.run.split(",")]
    run = runs[-1]                                              # 首战用最后一段练出来的最新版本
    skills_of = {c: np.array(run["pick"][c]["skills"]) for c in ORDER}
    milestones = []
    for r in runs:
        mp = os.path.join(r["dir"], "milestones.jsonl")
        if os.path.exists(mp):
            milestones += [json.loads(l) for l in open(mp, encoding="utf-8")]
    coach = runs[0]["cfg"]["args"].get("coach_iters", 0)
    clips, total_fights = growth_clips(runs, milestones, coach)
    card = first_fight(run, skills_of)
    card["frames"] = BL.resimulate(card["replay"])
    chars = {c: char_card(c, skills_of[c]) for c in ORDER}
    # 首战数据（片尾用）
    fr = card["frames"]
    st = {"thrown": [0, 0], "landed": [0, 0], "power": [0, 0], "kd": [0, 0]}
    for f in fr:
        for k in (0, 1):
            x = f["f"][k]
            if x[10]:
                st["thrown"][k] += 1
                if x[11] == 1:
                    st["landed"][k] += 1
                    st["power"][k] += int(x[10] - 1 >= 2)
            st["kd"][k] += int(f["f"][1 - k][14])
    r = card["result"]
    nm = {c: chars[c]["name"] for c in ORDER}
    if r["winner"]:
        loser = [c for c in ORDER if c != r["winner"]][0]
        rnd = fr[-1]["r"] + 1
        outro = (f"{nm[r['winner']]}在第 {rnd} 回合以{r['method']}拿下首战。这是两人从零学拳、"
                 f"自己和自己对打 {total_fights} 场之后，第一次正式交手。{nm[loser]}会怎么回应？")
        voice = f"首战结束，{nm[r['winner']]}以{r['method']}获胜。"
    else:
        outro = f"三位裁判打平。这是两人从零学拳、自己和自己对打 {total_fights} 场之后的第一次正式交手。"
        voice = "首战结束，平局。"
    tale = [["身高", *[c["body"].split(" · ")[0].replace("身高 ", "") for c in chars.values()]],
            ["臂展", *[f"{body_of(c).reach:.2f} 米" for c in ORDER]], ["体重", *[f"{body_of(c).mass:g} 公斤" for c in ORDER]],
            ["流派", *[chars[c]["school"] for c in ORDER]], ["属性总分", *[chars[c]["budget"] for c in ORDER]],
            ["招牌技", *[chars[c]["signature"] for c in ORDER]], ["赛前战术", *card.get("plans", ["", ""])]]
    data = {
        "sim": {"RING": B.RING, "BODY_R": B.BODY_R, "SHOULDER": B.SHOULDER, "DT": B.DT, "ROUND_TICKS": B.ROUND_TICKS,
                "RNG": B.RNG.tolist(), "TYPE": B.TYPE.tolist(), "TGT": B.TGT.tolist(), "HAND": B.HAND.tolist(),
                "actions": B.ACTION_NAMES, "phases": B.PHASE_NAMES, "methods": B.METHOD_NAMES,
                "postures": B.POSTURE_NAMES, "plans": B.PLAN_NAMES, "version": B.SIM_VERSION},
        "order": ORDER, "chars": chars, "portraits": {c: portrait_uri(c) for c in ORDER},
        "growth": clips, "growth_sub": f"两具赛博肉体从不会出拳开始，自己和自己打了 {total_fights} 场",
        "fight": {k: v for k, v in card.items() if k != "replay"},
        "fight_sub": f"红角 {nm['LJ']} 对 蓝角 {nm['GY']} · 3 回合", "tale": tale, "stats": st,
        "outro_text": outro, "outro_voice": voice,
        "next_text": "下集：训练营——输的人，会怎么变？" if r["winner"] else "下集：训练营——谁先变强？",
    }
    tpl = open("episode_template.html", encoding="utf-8").read().replace("/*__VIZ__*/", open("viz.js", encoding="utf-8").read())
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"),
                                                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    open(args.out, "w", encoding="utf-8").write(html)
    print("试播集页面：", os.path.abspath(args.out), f"（首战 {r['winner'] or '平局'} {r['method']}，成长片段 {len(clips)} 段）")


if __name__ == "__main__":
    main()
