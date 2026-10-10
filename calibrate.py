"""真实度诊断：拿一个训练档的 A、B 打一批比赛，按 CompuBox 口径统计，和现实参照比。

这些都是诊断，不是 AI 必须达到的目标（用户定：属性按现实，选择归 AI）。偏离现实时先查是不是某个
技术的属性不真实（没代价、缺风险）；属性真实的话，就是 AI 自己的打法。参照的来源见 docs/技术真实度审计.md。
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

# kind：real = 职业比赛数据做参照；design = 拳击常识做参照（没有统计）；record = 只记录
TARGETS = [
    ("thrown_pm", "每分钟出拳（每人）", 15.0, 22.0, "real", "世界冠军赛约 16.9（Puchol 2023）；次中量级约 19.5（ESPN 2012）；重量级冠军赛约 12.5（Finlay 2022）"),
    ("connect", "总命中率", 0.25, 0.40, "real", "世界冠军赛 26.3%（Puchol 2023）；重量级冠军赛 34.6%（Finlay 2022）"),
    ("connect_jab", "刺拳命中率", 0.15, 0.30, "real", "16%（Puchol 2023）~28%（Finlay 2022）"),
    ("connect_power", "重拳命中率", 0.30, 0.45, "real", "34.6%（Puchol 2023）~42.4%（Finlay 2022）"),
    ("jab_share", "刺拳占出拳", 0.40, 0.55, "real", "44%（Puchol 2023）、54%（Finlay 2022）"),
    ("body_share", "打身体占命中", 0.15, 0.35, "real", "CompuBox 平均约 23%（7%~41%）"),
    ("defended_share", "没打中的拳里被挡或被躲的比例", 0.30, 1.0, "design", "常识：挡和躲是主要的防守手段，没有统计"),
    ("evade_success", "闪躲、拍挡的成功率", 0.20, 1.0, "design", "常识，没有统计"),
    ("feint_success", "虚晃骗到的比例", 0.10, 1.0, "design", "常识，没有统计"),
    ("clinch_pm", "缠抱（每人每分钟，含没抱住的）", 0.0, 2.0, "design", "常识：一回合几次"),
    ("stance_pm", "换站架（每人每分钟）", 0.0, 0.5, "design", "常识：只有少数人会换，换得也不多"),
    ("posture_pm", "换架势（每人每分钟）", 0.0, 2.0, "design", "常识，没有统计"),
    # 情境：技术是不是在现实里该用的情境下用（期望按调研里的教练经验定，见 docs/技术真实度审计.md）
    ("sit_clinch_hurt", "被打晃时发起缠抱的频率 ÷ 平时", 3.0, 1e9, "design", "被打晃、累了时抱住喘口气"),
    ("sit_ropes_move", "被逼到绳边时轴转、横移的频率 ÷ 在场中央", 2.0, 1e9, "design", "被逼到绳边时轴转、变向脱离"),
    ("sit_duck_hook", "下潜时对手正在出勾拳的比例", 0.5, 1.0, "design", "下潜从勾拳下钻过"),
    ("sit_body_vs_guard", "对手举高位架时打身体的比例 ÷ 对手护体时", 1.0, 1e9, "design", "对手只顾护头时打身体"),
    ("sit_body_tired", "对手累垮时打身体的比例 ÷ 对手不累时", 1.0, 1e9, "design", "对手累了，打身体收益最大"),
    ("sit_power_behind", "裁判分落后时重拳占出拳的比例 ÷ 领先时", 1.0, 1e9, "design", "落后时加压找 KO，领先时刺拳控距离"),
    ("tired_share", "累垮时间占比（体能 < 25%）", 0.0, 0.20, "design", "常识约束：真实拳手不会整场空着油箱打"),
    ("posture_max", "最常用架势的时间占比（两人都算）", 0.0, 0.60, "design", "没有一种架势被所有角色都用超过六成时间"),
    ("hp_tko_share", "TKO 里体力归零的比例", None, None, "record", "v10 起体力归零判 TKO 是规则的一部分，只记录"),
    ("stoppage", "提前结束比例（KO+TKO）", None, None, "record", "用户定：偏高是好事，不设上限（参照：职业 47%，KO 15%、TKO 32%）"),
    ("kd_per_fight", "每场击倒次数", None, None, "record", "没有可靠数据，不设硬指标"),
    ("last_round_stop", "提前结束里发生在最后一回合的比例", None, None, "record", "真实比赛偏早段（Hutchison 2014；职业平均第 2.59 回合）"),
    ("underdog_win", "爆冷：弱势一方的胜率", None, None, "record", "这批比赛里赢得少的一方赢了多少；用户定：要有爆冷"),
    ("underdog_stop", "爆冷：弱势一方的胜场里靠 KO/TKO 的比例", None, None, "record", "一拳翻盘"),
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
    jab_n, pow_n = int(jab.sum()), int(power.sum())
    out["jab_share"] = float(jab_n / max(1, jab_n + pow_n))
    body = thrown & (B.TGT[np.clip(fired - 1, 0, 11)] == 1)[..., :]
    out["body_share"] = float((lw * body).sum() / max(1e-9, lw.sum()))
    oc = f["outcome"]
    missed = thrown & (oc != 1)
    out["defended_share"] = float((missed & np.isin(oc, (2, 3))).sum() / max(1, missed.sum()))
    # 技术使用：两个人合起来，按技术使用表的口径
    tech = {}
    for k in (0, 1):
        for _, name, used, succ, _ in BL.technique_table(bt, k):
            u, sc = tech.get(name, (0, 0))
            tech[name] = (u + (used or 0), sc + (succ or 0))
    ev = [tech[n] for n in ("拍挡", "侧闪", "下潜", "后仰")]
    out["evade_success"] = float(sum(sc for _, sc in ev) / max(1, sum(u for u, _ in ev)))
    out["feint_success"] = float(tech["虚晃"][1] / max(1, tech["虚晃"][0]))
    fighter_min = 2 * v.sum() * B.DT / 60.0
    out["clinch_pm"] = float(tech["缠抱"][0] / fighter_min)
    out["stance_pm"] = float(tech["换站架"][0] / fighter_min)
    out["posture_pm"] = float(tech["换架势"][0] / fighter_min)
    out.update(situational(bt, v, f))
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
    dog = int(np.argmin(out["win"]))                    # 弱势一方：这批比赛里赢得少的那个
    dog_w = bt.winner == dog
    out["underdog_win"] = float(dog_w.mean())
    out["underdog_stop"] = float((dog_w & stop).sum() / max(1, dog_w.sum()))
    # 风格：某一方开场用计划 p 时，这场提前结束的比例（每场随机抽计划时才有意义）
    plan0 = f["plan"][0]
    out["stop_by_plan"] = {B.PLAN_NAMES[q]: [round(float(stop[(plan0 == q).any(1)].mean()), 3), int((plan0 == q).any(1).sum())]
                           for q in range(B.N_PLANS) if (plan0 == q).any()}
    return out


def _ratio(num_a, den_a, num_b, den_b):
    """(num_a/den_a) ÷ (num_b/den_b)，样本不够时给 nan。"""
    if den_a < 20 or den_b < 20 or num_b == 0:
        return float("nan")
    return float((num_a / den_a) / (num_b / den_b))


def situational(bt, v, f):
    """情境指标：两个人合起来算。"""
    T = v.shape[0]
    vv = v[..., None] & np.ones((1, 1, 2), bool)
    acted = f["acted"] & vv
    act = f["action"].astype(int)
    opp = lambda x: x[:, :, ::-1]
    hurt = f["stun"] > B.STUN_HURT
    cl = f["clinch_start"] & vv
    out = {"sit_clinch_hurt": _ratio(cl[hurt].sum(), (hurt & vv).sum(), cl[~hurt].sum(), (~hurt & vv).sum())}
    # 绳边：背后离围绳不到 0.4 米、对手在 1.2 米内（和模拟里"被压在绳边"同一口径）；场中央：背后离围绳 1.5 米以上
    dist, _ = BL._geom(bt)
    back = np.stack([B.ray_to_rope(f["pos"][:, :, k], f["th"][:, :, k] + np.pi) for k in (0, 1)], -1)
    pinned = (back < 0.4) & (dist[..., None] < 1.2) & vv
    center = (back > 1.5) & vv
    mv = acted & np.isin(act, (B.PIVOT_L, B.PIVOT_R, B.LEFT, B.RIGHT))
    out["sit_ropes_move"] = _ratio(mv[pinned].sum(), pinned.sum(), mv[center].sum(), center.sum())
    # 下潜：下潜那一帧对手正在起手的拳是不是勾拳
    duck = acted & (act == B.DUCK) & (f["phase"] == B.DEFEND)
    om, oph = opp(f["move"]).astype(int), opp(f["phase"])
    opp_punch = (oph == B.STARTUP) & (om < 12)
    d_with = duck & opp_punch
    out["sit_duck_hook"] = float((d_with & (B.TYPE[np.clip(om, 0, 11)] == 1)).sum() / d_with.sum()) if d_with.sum() >= 20 else float("nan")
    # 打身体：出拳那一帧对手的架子、对手累不累
    fired = f["fired"].astype(int)
    thrown = (fired > 0) & vv
    body = thrown & (B.TGT[np.clip(fired - 1, 0, 11)] == 1)
    og = opp(f["guard"])
    out["sit_body_vs_guard"] = _ratio(body[og == 1].sum(), thrown[og == 1].sum(), body[og == 2].sum(), thrown[og == 2].sum())
    otired = opp(f["stam"]) < B.TIRED
    out["sit_body_tired"] = _ratio(body[otired].sum(), thrown[otired].sum(), body[~otired].sum(), thrown[~otired].sum())
    # 裁判分：至少打完一回合以后，自己落后 / 领先时重拳占出拳的比例
    lead = np.stack([f["card"], -f["card"]], -1)
    power = thrown & (fired > 2)
    behind, ahead = thrown & (lead < 0), thrown & (lead > 0)
    out["sit_power_behind"] = _ratio(power[behind].sum(), behind.sum(), power[ahead].sum(), ahead.sum())
    return out


def verdict(val, lo, hi, kind):
    """≈ 接近现实参照，≠ 偏离（诊断用，不是及格线），? 样本不够，— 只记录。"""
    if kind == "record" or lo is None:
        return "—"
    if val != val:                              # nan：样本不够
        return "?"
    return "≈" if lo - 1e-9 <= val <= hi + 1e-9 else "≠"


def report(m):
    lines = []
    for key, label, lo, hi, kind, src in TARGETS:
        val = m[key]
        tgt = "不设" if lo is None else (f"{lo:g}" if lo == hi else f"{lo:g}~{hi:g}")
        pct = key.startswith("connect") or key.endswith("share") or key.endswith("success") or key in ("stoppage", "posture_max", "last_round_stop", "sit_duck_hook")
        shown = f"{val:.1%}" if pct else f"{val:.2f}"
        if pct and lo is not None:
            tgt = f"{lo:.0%}" if lo == hi else (f"{lo:.0%} 以上" if hi >= 1.0 else f"{lo:.0%}~{hi:.0%}")
        elif lo is not None and lo == 0 and hi > 0:
            tgt = f"{hi:g} 以下"
        elif lo is not None and hi >= 1e8:
            tgt = f"{lo:g} 以上" if lo > 1 else "大于 1"
        lines.append(f"{verdict(val, lo, hi, kind)} {label}：{shown}（{'现实参照' if kind == 'real' else '常识参照' if kind == 'design' else '只记录，参照'} {tgt}）")
    lines.append(f"  结局：KO {m['ko']:.0%}、TKO {m['tko']:.0%}、判定 {m['decision']:.0%}、平局 {m['draw']:.0%}；TKO 原因 {m['tko_cause']}；"
                 f"A 胜 {m['win'][0]}、B 胜 {m['win'][1]}；平均 {m['seconds']:.0f} 秒")
    lines.append(f"  架势占比（标准/长架/抱架/低手） A {m['posture'][0]}  B {m['posture'][1]}；举架或闪躲占时 A {m['defend_share'][0]:.1%} B {m['defend_share'][1]:.1%}")
    if len(m["stop_by_plan"]) > 1:
        lines.append("  按开场战术计划的提前结束比例：" + "，".join(f"{k} {v[0]:.0%}（{v[1]} 场）" for k, v in m["stop_by_plan"].items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--plans", choices=["free", "random"], default="free", help="free：双方都自由发挥；random：每场随机抽战术计划（看风格差异）")
    args = ap.parse_args()
    run = load_run(args.run, args.version)
    bodies = (LANKY, STOCKY)
    skills = np.stack([np.array(run["pick"][c].get("skills", SC.talent(b))) for c, b in zip("AB", bodies)])
    t0 = time.time()
    bt = BL.run_matches(bodies, (run["pols"]["A"], run["pols"]["B"]), ("A", "B"), args.n, seed=args.seed, skills=skills,
                        plans=None if args.plans == "random" else (0, 0))
    m = metrics(bt)
    print(f"{args.run} v{run['version']}，{args.n} 场 A 对 B（{time.time() - t0:.0f} 秒）")
    print(report(m))
    rec = {"version": run["version"], "n": args.n, "seed": args.seed, "plans": args.plans, "sim_version": B.SIM_VERSION,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), **m}
    with open(os.path.join(run["dir"], "calibration.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
