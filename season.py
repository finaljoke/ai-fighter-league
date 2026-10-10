"""第 0 季：A 和 B 的三番战。开季直接打一场，然后每个训练营之后再打一场。

所有东西都存档：每天两人做了什么、数值怎么变、对练片段（只存动作）、每场比赛的完整记录和回放档、
每场比赛时双方出招网络的哈希。最后生成可以直接打开的赛季观看页。

用法：py -X utf8 season.py --run b1 --name s0
      py -X utf8 season.py --run b1 --name test --quick      （小规模对练，用来检查流程和画面）
"""
import argparse
import json
import os
import time

import numpy as np

import boxing as B
import boxing_league as BL
import career as C
import schools as SC
from analyze import load_run
from analyze_boxing import card as char_card
from bodies import LANKY, STOCKY
from nets import params_sha256, save_policy

CLIP_TICKS = 300                # 对练片段录 15 秒


def spar_clip(f, partner, seed):
    """对练日录一段片段：f 和对练对象打一回合，取前 15 秒。只存动作，画面靠重新模拟。"""
    p_body, p_ctrl, p_skills, p_name = partner
    bt = BL.run_matches((f.body(), p_body), (f.policy(), p_ctrl), (f.id, p_name), 1, seed=seed, n_rounds=1,
                        skills=np.stack([f.skills, p_skills]))
    rec = BL.replay_record(bt, 0, [{"name": f.id}, {"name": p_name}])
    frames = BL.resimulate(rec)[:CLIP_TICKS]
    return {"partner": p_name, "bodies": rec["bodies"], "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="从哪个训练 run 接手角色（取最新版本）")
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fights", type=int, default=3)
    ap.add_argument("--camp-days", type=int, default=35)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        C.SPAR_ENVS, C.SPAR_T, C.SPAR_ITERS = 16, 64, 2

    out_dir = os.path.join("runs", f"season_{args.name}")
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} 已存在。规矩是不覆盖旧记录，换个 --name。")
    os.makedirs(os.path.join(out_dir, "ckpt"))
    run = load_run(args.run, args.version)
    fighters = []
    for body, c in ((LANKY, "A"), (STOCKY, "B")):
        sk = np.array(run["pick"][c].get("skills", SC.talent(body)))
        fighters.append(C.Fighter(body, run["pols"][c].params, sk, seed=args.seed + ord(c)))
    fa, fb = fighters
    # 邀请来的对练：一个"打法像对手"的陪练（用对手赛季开始时的版本）
    partners = {"A": (fb.body(), fb.policy(), fb.skills.copy(), f"像{fb.id}的陪练"),
                "B": (fa.body(), fa.policy(), fa.skills.copy(), f"像{fa.id}的陪练")}
    cards0 = {f.id: char_card(f.base, f.skills) for f in fighters}

    timeline, fights, clips = [], [], {}
    day = 0
    t0 = time.time()

    def checkpoint(tag):
        out = {}
        for f in fighters:
            fn = f"{f.id}_{tag}.npz"
            out[f.id] = {"file": fn, "sha256": save_policy(os.path.join(out_dir, "ckpt", fn), f.pi)}
        return out

    for i in range(args.fights):
        if i > 0:
            print(f"训练营 {i}：{args.camp_days} 天", flush=True)
            for d in range(args.camp_days):
                row = {"day": day + 1}
                for f, opp in ((fa, fb), (fb, fa)):
                    act, kw = C.coach_choose(f, opp, args.camp_days - d, partner_available=True)
                    if act == "对练·邀请拳手":
                        kw["partner"], kw["target"] = partners[f.id], opp
                    entry = C.do_day(f, act, seed=args.seed * 1000 + day * 10 + (f is fb), days_left=args.camp_days - d, **kw)
                    row[f.id] = entry
                    if act.startswith("对练") and entry["activity"].startswith("对练"):
                        partner = kw.get("partner") or (C.BD.SPARRING, C.BoxingSparring(), SC.talent(C.BD.SPARRING), "陪练教练")
                        clips[f"{day + 1}_{f.id}"] = spar_clip(f, partner, seed=day * 7 + (f is fb))
                timeline.append(row)
                day += 1
                if (d + 1) % 7 == 0:
                    print(f"  第 {day} 天  A {row['A']['activity']}  B {row['B']['activity']}  "
                          f"疲劳 A {fa.fatigue:.0f} B {fb.fatigue:.0f}  {time.time() - t0:.0f}s", flush=True)
        title = f"第 {i + 1} 战"
        print(f"{title}（第 {day} 天）", flush=True)
        card = C.fight_night(fa, fb, seed=args.seed * 100 + i, title=title)
        card["frames"] = BL.resimulate(card["replay"])
        card["bodies"] = card["replay"]["bodies"]
        card["ckpt"] = checkpoint(f"fight{i + 1}")
        card["day"] = day
        res = card["result"]
        print(f"  结果：{res['winner'] or '平局'} {res['method']}（{res['seconds']} 秒）  体检 "
              + "，".join(f"{m['id']} 伤势 {m['severity']} 恢复 {m['recovery_days']} 天" for m in card["medical"]),
              flush=True)
        fights.append(card)

    data = {
        "name": args.name, "generated": time.strftime("%Y-%m-%d %H:%M"), "source_run": f"{run['name']} v{run['version']}",
        "camp_days": args.camp_days, "cards": cards0,
        "final": {f.id: {"pts": f.pts, "pts0": f.pts0, "skills": f.skills.round(4).tolist(), "record": f.record} for f in fighters},
        "timeline": timeline, "fights": fights, "clips": clips,
        "tech_names": B.TECH_NAMES, "activities": {k: {kk: vv for kk, vv in v.items()} for k, v in C.ACTIVITIES.items()},
        "sim": {"RING": B.RING, "BODY_R": B.BODY_R, "SHOULDER": B.SHOULDER, "DT": B.DT, "ROUND_TICKS": B.ROUND_TICKS,
                "RNG": B.RNG.tolist(), "TYPE": B.TYPE.tolist(), "TGT": B.TGT.tolist(), "HAND": B.HAND.tolist(),
                "actions": B.ACTION_NAMES, "phases": B.PHASE_NAMES, "methods": B.METHOD_NAMES,
                "postures": B.POSTURE_NAMES, "plans": B.PLAN_NAMES, "version": B.SIM_VERSION},
    }
    with open(os.path.join(out_dir, "season.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    tpl = open("season_template.html", encoding="utf-8").read().replace("/*__VIZ__*/", open("viz.js", encoding="utf-8").read())
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"),
                                                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    out_html = os.path.join("runs", f"season_{args.name}.html")
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("赛季观看页:", os.path.abspath(out_html), f"（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
