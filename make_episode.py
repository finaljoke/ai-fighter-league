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
from boxing_bots import BoxingSparring
from nets import load_policy
from train import MILESTONES

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


CLIP = 8.0                    # 每段成长片段的秒数
FRAMES = os.path.join("runs", "replay_frames")   # 回放算出来的画面帧：规则改版以后，旧回放读这里，不重算
LEAD = 5.0                    # 重演片段里，里程碑那一下出现在第几秒
# 进成长片段的里程碑：只要击倒和提前结束比赛这类大事，数字越大越重要
BIG = {"第一次 KO 获胜": 5, "第一次 TKO 获胜": 4, "第一次肝区击倒": 3, "第一次一拳打中下颌把人打倒": 2, "第一次打倒对手": 1}
TESTS = dict(MILESTONES)


def growth_items(runs):
    """各段训练档的成长档案按轮数接起来：[(轮数, 训练档, 版本), ...]（接着练的那段，开头那个版本和上一段最后一个是同一个，跳过）。"""
    items = []
    for run in runs:
        gdir = os.path.join(run["dir"], "growth")
        for f in sorted(os.listdir(gdir)):
            if f.endswith(".json"):
                g = json.load(open(os.path.join(gdir, f), encoding="utf-8"))
                if not items or g["iter"] > items[-1][0]:
                    items.append((g["iter"], run, int(f[1:5])))
    return items


def pick_milestones(milestones, n=5, gap=30):
    """同一个人 gap 轮之内接连发生的几件大事归成一组（第一次击倒、第一次 KO 往往挨着），一组只放一段，最多 n 组。
    组里按重要程度排好：最大的那件重演不出来（比如一拳下颌击倒太少见），就退一步重演次一级的。"""
    groups = []
    for m in sorted((m for m in milestones if m["event"] in BIG), key=lambda m: m["iter"]):
        g = next((g for g in reversed(groups) if g[0]["char"] == m["char"]), None)
        if g and m["iter"] - g[-1]["iter"] <= gap:
            g.append(m)
        else:
            groups.append([m])
    return [sorted(g, key=lambda m: -BIG[m["event"]]) for g in groups[:n]]


def archive_clip(item, caption, voice, coach):
    """存档录像：存档节点另打的那几场，挑最热闹的一段。"""
    it, run, v = item
    g = json.load(open(os.path.join(run["dir"], "growth", f"v{v:04d}.json"), encoding="utf-8"))
    rec = g["replays"][0]
    frames = BL.frames_of(rec, FRAMES)
    sub = f"已对打 {g['fights']} 场"
    ev = next((e for e in run["evals"] if e["version"] == v), None)
    if ev:
        rate = [ev.get(f"{b['id']}_vs_S", {}).get("punch_rate", 0) for b in rec["bodies"]]
        sub += f" · 每 10 秒出拳 {rate[0]:.0f} / {rate[1]:.0f}"
    return {"kind": "archive", "iter": it, "fights": g["fights"], "frames": frames, "bodies": rec["bodies"],
            "from": best_window(frames, CLIP), "dur": CLIP, "speed": 1, "capAt": 0.3,
            "when": f"训练第 {it} 轮" if it else "第 0 轮 · 出生",
            "when_sub": sub + ("　· 启蒙期：跟陪练教练学" if 0 < it <= coach else ""), "caption": caption, "voice": voice}


def reenact(m, runs, items, coach, cache):
    """里程碑重演：训练里那一场没有录像，用当时的出招网络（新训练档里程碑时另存了一份；
    旧档用里程碑之后最近的存档版本）按当时的对手重新打，直到同样的事发生，挑第一场存档。以后重新生成只读存档。"""
    key = f"{m['char']}|{m['event']}|{m['iter']}"
    if key in cache:
        return cache[key]
    run = next(r for r in runs if r["name"] == m["_run"])
    ids = run["cfg"]["args"]["bodies"].split(",")                  # 训练时的左右（第 0 方、第 1 方）
    bodies = tuple(body_of(c) for c in ids)
    k = ids.index(m["char"])
    if "snap" in m:                                                 # 新训练档：里程碑那一刻的出招网络
        pols = {}
        for c in ids:
            f = m["snap"]["files"][c]
            pols[c] = load_policy(os.path.join(run["dir"], "ckpt", f["file"]))
            assert pols[c].sha256() == f["sha256"], f"{f['file']} 哈希对不上，存档被改过"
        skills, src_iter, src = np.array(m["snap"]["skills"]), m["iter"], "里程碑当时的出招网络"
        vs_list = [m["vs"]]
    else:                                                           # 旧训练档：之后最近的存档版本
        src_iter, vrun, v = next((x for x in items if x[0] >= m["iter"]), items[-1])
        R = load_run(vrun["name"], v)
        pols, skills, src = R["pols"], np.stack([R["pick"][c]["skills"] for c in ids]), f"第 {src_iter} 轮存档"
        run = vrun
        vs_list = ["coach"] if m["iter"] <= coach else ["self", "coach"]  # 旧档没记对手：启蒙期之后先试对方同期版本
    test = TESTS[m["event"]]
    for vs in vs_list:
        ctrls = [pols[c] for c in ids]
        if vs == "coach":
            ctrls[1 - k] = BoxingSparring()
        elif vs.startswith("v"):
            ctrls[1 - k] = load_run(run["name"], int(vs[1:]))["pols"][ids[1 - k]]
        for batch in range(6):
            n, seed = 24, 70_000 + 97 * m["iter"] + 1000 * batch
            hit = np.full(n, -1)

            def on_step(t, info, alive):
                h = np.asarray(test(info, k)) & alive & (hit < 0)
                hit[h] = t
            bt = BL.run_matches(bodies, ctrls, tuple(ids), n, seed=seed, skills=skills, plans=None, on_step=on_step)
            if (hit < 0).all():
                continue
            i = int(np.flatnonzero(hit >= 0)[0])
            names = [{"name": BoxingSparring.name} if (vs == "coach" and j != k) else {"name": body_of(c).label} for j, c in enumerate(ids)]
            rec = BL.replay_record(bt, i, names)
            # 镜头对准倒下的那一下：肝区击倒要过 1~2 秒才倒；KO、TKO 对准最后一次倒地（离结束不远的话）
            kd = bt.fields["kd"][:, i, 1 - k]
            focus = int(hit[i])
            after = np.flatnonzero(kd[focus:])
            if m["event"] == "第一次肝区击倒" and after.size:
                focus += int(after[0])
            elif m["event"] in ("第一次 KO 获胜", "第一次 TKO 获胜"):
                before = np.flatnonzero(kd[:focus + 1])
                if before.size and focus - before[-1] <= 6 * 20:
                    focus = int(before[-1])
            out = {"rec": rec, "focus": focus, "vs": vs, "src": src, "src_iter": int(src_iter), "seed": seed, "index": i}
            cache[key] = out
            return out
    print(f"  没能重演：{m['name']} {m['event']}（训练第 {m['iter']} 轮）")
    cache[key] = None                                               # 重演不出来也记下，下次不再白打
    return None


def reenact_clip(m, r, nick):
    frames = BL.frames_of(r["rec"], FRAMES)
    ids = [b["id"] for b in r["rec"]["bodies"]]
    k = ids.index(m["char"])
    coach = r["vs"] == "coach"
    nm = m["name"].split("·")[0]
    start = max(0, r["focus"] - int(LEAD * 20))
    rs = next(t for t in range(r["focus"], -1, -1) if t == 0 or frames[t - 1]["r"] != frames[r["focus"]]["r"])
    if start < rs <= r["focus"] - 40:                               # 不跨回合剪：从这一回合开头开始，免得两人突然跳回角落
        start = rs
    show =["S" if coach and j != k else c for j, c in enumerate(ids)]
    why = ""
    if m["event"] == "第一次 TKO 获胜":                              # TKO 没人倒地时，观众得知道裁判为什么叫停
        last, rnd = frames[-1]["f"][1 - k], frames[-1]["r"]
        kds = sum(f["f"][1 - k][14] for f in frames if f["r"] == rnd)
        why = " · 裁判叫停：" + ("一回合倒地三次" if kds >= 3 else "体力打空" if last[3] <= 0 else "受伤后连续挨打不还手")
    return {"kind": "reenact", "iter": m["iter"], "fights": m["fights"], "frames": frames, "bodies": r["rec"]["bodies"],
            "ids": show, "labels": [nick[c] for c in show], "names": [nick[c] if c == "S" else body_of(c).label.split("·")[0] for c in show],
            "from": start, "dur": CLIP, "speed": 1, "capAt": (r["focus"] - start) / 20 + 0.15,
            "when": f"训练第 {m['iter']} 轮", "when_sub": f"第 {m['fights']} 场左右 · 用{r['src']}重演" + (" · 对陪练教练" if coach else ""),
            "caption": {"text": f"{nm} {m['event']}", "sub": f"训练第 {m['iter']} 轮 · 版本重演" + ("（对陪练教练）" if coach else "") + why, "big": True},
            "voice": f"训练第{m['iter']}轮，{nm}{m['event']}{'，对手是陪练教练' if coach else ''}。"}


def growth_clips(runs, milestones, coach):
    """成长片段：出生（存档录像）→ 几件大事的版本重演 → 首战前最后一次存档（存档录像）。"""
    items = growth_items(runs)
    if not items:
        return [], 0
    cpath = os.path.join(runs[-1]["dir"], "milestone_reenact.json")
    cache = json.load(open(cpath, encoding="utf-8")) if os.path.exists(cpath) else {}
    clips = [archive_clip(items[0], {"text": "刚出生：不会控制距离，也不会出拳", "sub": "两具赛博肉体第一次站上擂台"},
                          "刚出生的他们，还不知道拳该往哪里打。", coach)]
    nick = {c: CHARS[c]["nick"] for c in CHARS} | {"S": "陪练教练"}
    for group in pick_milestones(milestones):
        for m in group:
            r = reenact(m, runs, items, coach, cache)
            if r:
                clips.append(reenact_clip(m, r, nick))
                break
    json.dump(cache, open(cpath, "w", encoding="utf-8"), ensure_ascii=False)
    last = items[-1]
    clips.append(archive_clip(last, {"text": "首战前的最后一次存档", "sub": f"训练第 {last[0]} 轮"},
                              f"训练第{last[0]}轮，首战前的最后一次存档。", coach))
    return clips, clips[-1]["fights"]


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
            milestones += [{**json.loads(l), "_run": r["name"]} for l in open(mp, encoding="utf-8")]
    coach = runs[0]["cfg"]["args"].get("coach_iters", 0)
    clips, total_fights = growth_clips(runs, milestones, coach)
    card = first_fight(run, skills_of)
    card["frames"] = BL.frames_of(card["replay"], FRAMES)
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
        "order": ORDER, "chars": chars, "portraits": {c: portrait_uri(c) for c in ORDER}, "apron": "第 1 季 · 试播集",
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
