"""画面和模拟对不对得上：生成几场编排好的小场景（闪躲、拍挡、格挡、倒地、肝区击倒、缠抱、步法、换站架），
外加正史最新存档打的一场真比赛，逐拳核对画面里的拳套和人是不是在模拟说的地方。

    py -X utf8 tools/viz_check.py            # 生成 runs/viz_check/*.json，再跑 node tools/check_viz.mjs
    py -X utf8 tools/viz_check.py --page     # 另外拼一个观看页 runs/viz_check.html，能逐个场景看画面
    py -X utf8 tools/viz_check.py --page --skin   # 按三维皮肤的骨长核对，核对页里的人换成三维皮肤（characters/<id>.skin.json），出 runs/viz_check_skin.html

场景用标准回放档重算（boxing_league.resimulate），和观看页走的是同一条路。
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import boxing as B                      # noqa: E402
import boxing_league as BL              # noqa: E402
import schools as SC                    # noqa: E402
import skins as SK                      # noqa: E402
from make_episode import body_of        # noqa: E402

OUT = os.path.join("runs", "viz_check")
GY, LJ = body_of("GY"), body_of("LJ")
S = B  # 动作名短一点


def play(acts0, acts1, gap=1.1, seed=1, bodies=(GY, LJ)):
    """按帧给双方动作（{帧: 动作}，或者看着场上情况出手的函数 (帧, sim) → 动作），打到这一回合结束，返回回放档。"""
    T = B.ROUND_TICKS + 200
    a = np.zeros((T, 2), np.int64)
    for k, acts in ((0, acts0), (1, acts1)):
        for t, x in ({} if callable(acts) else acts).items():
            a[t, k] = x
    skills = np.stack([SC.talent(b) for b in bodies])
    init = [gap, 0.0, 0.0, 0.0, 0.0, float(seed)]
    sim = B.Sim(1, bodies, auto_reset=False, n_rounds=1, skills=skills)
    sim.reset(np.ones(1, bool), init=np.array([init]))
    end, res = None, None
    for t in range(T):
        for k, acts in ((0, acts0), (1, acts1)):
            if callable(acts):
                a[t, k] = acts(t, sim)
        _, done, info = sim.step(a[t:t + 1])
        if done[0]:
            end, res = t + 1, {"winner": int(info["winner"][0]), "method": int(info["method"][0])}
            break
    assert end is not None, "场景没打完一回合"
    a = a[:end]
    return {"format": 1, "sim_version": B.SIM_VERSION, "n_rounds": 1, "bodies": [b.to_dict() for b in bodies],
            "init": init, "skills": skills.tolist(), "plans": [[0, 0, 0]],
            "actions": [",".join(map(str, a[:, 0])), ",".join(map(str, a[:, 1]))], "result": res}


def first_kd(acts0, gap, liver, n=256, bodies=(GY, LJ)):
    """同一套动作在 n 个运气种子下同时打，返回第一个打出（肝区）击倒的种子。"""
    skills = np.stack([SC.talent(b) for b in bodies])
    sim = B.Sim(n, bodies, auto_reset=False, n_rounds=1, skills=skills)
    seeds = np.arange(1, n + 1, dtype=np.float64)
    sim.reset(np.ones(n, bool), init=np.stack([np.full(n, gap), *np.zeros((4, n)), seeds], 1))
    hit = np.zeros(n, bool)
    for t in range(max(acts0) + 80):
        a = np.zeros((n, 2), np.int64)
        a[:, 0] = acts0.get(t, B.STAY)
        _, _, info = sim.step(a)
        hit |= info["liver_kd"][:, 0] if liver else (info["flash_kd"][:, 0] & ~(sim.liver_t[:, 1] > 0))
    return float(seeds[np.argmax(hit)]) if hit.any() else None


def attacker(seq, rest=14):
    """依次打 seq 里的 (拳, 距离)：先走到这个中心距以内，站定再打，打完歇 rest 帧。"""
    st = {"i": 0, "wait": 5}

    def act(t, sim):
        if st["i"] >= len(seq) or sim.phase[0, 0] != B.FREE:
            return B.STAY
        if st["wait"] > 0:
            st["wait"] -= 1
            return B.STAY
        p, near = seq[st["i"]]
        if np.linalg.norm(sim.pos[0, 1] - sim.pos[0, 0]) > near:
            return B.FWD
        st["i"] += 1
        st["wait"] = rest
        return p
    return act


def defender(moves, hold=None):
    """对手每出一拳，在拳到前 3 帧依次做 moves 里的防守；其余时候 hold(帧) 给的动作（举架之类）。"""
    st = {"i": 0}

    def act(t, sim):
        if sim.phase[0, 0] == B.STARTUP and sim.move[0, 0] < 12 and sim.timer[0, 0] == 3 and st["i"] < len(moves):
            st["i"] += 1
            return moves[st["i"] - 1]
        return hold(t) if hold else B.STAY
    return act


def every(start, period, count, act):
    return {start + i * period: act for i in range(count)}


def scenarios():
    out = {}
    # 闪躲：拳打过来，对手在拳到前 3 帧侧闪 / 后仰 / 下潜 / 拍挡。后仰只在射程边上有用（头往后 25 厘米就够不着了）
    plan = [(S.JAB_H, 0.95, S.SLIP), (S.CROSS_H, 0.95, S.SLIP), (S.JAB_H, 1.12, S.PULL), (S.LHOOK_H, 0.8, S.DUCK), (S.RHOOK_H, 0.8, S.DUCK),
            (S.JAB_H, 0.95, S.PARRY), (S.CROSS_H, 0.95, S.PARRY), (S.JAB_H, 0.95, S.DUCK), (S.LHOOK_H, 0.8, S.SLIP)]
    out["闪躲与拍挡"] = play(attacker([(p, d) for p, d, _ in plan]), defender([m for _, _, m in plan]), gap=1.4)
    # 格挡：举高位挡头部拳，护体挡身体拳
    seq = [(S.JAB_H, 0.95)] * 3 + [(S.CROSS_H, 0.95)] * 3 + [(S.LHOOK_H, 0.8)] * 3 + [(S.CROSS_B, 0.9)] * 3 + [(S.LHOOK_B, 0.8)] * 3 + [(S.RUPPER_B, 0.75)] * 2
    out["格挡"] = play(attacker(seq), defender([], hold=lambda t: S.GUARD_HIGH if t < 230 else S.GUARD_BODY), gap=1.2)
    # 射程：从远处一步步走近，每走几步打一记刺拳，前几拳够不着
    a0 = {}
    for i in range(10):
        a0[20 + i * 30] = S.JAB_H
        for q in range(4):
            a0[20 + i * 30 + 8 + q] = S.FWD
    out["射程"] = play(a0, {}, gap=2.2)
    # 步法：前进、后退、左右横移、轴转、换站架、换架势
    a0 = {}
    seq = [S.FWD] * 20 + [S.BACK] * 20 + [S.LEFT] * 20 + [S.RIGHT] * 20 + [S.STAY] * 5 + [S.PIVOT_L, S.STAY, S.STAY, S.STAY] * 5 \
        + [S.PIVOT_R, S.STAY, S.STAY, S.STAY] * 5 + [S.SWITCH_STANCE] + [S.STAY] * 10 + [S.FWD] * 10 + [S.CHANGE_POSTURE] + [S.STAY] * 10 \
        + [S.CHANGE_POSTURE] + [S.STAY] * 10 + [S.CHANGE_POSTURE] + [S.STAY] * 10 + [S.CHANGE_POSTURE]
    for i, x in enumerate(seq):
        a0[5 + i] = x
    out["步法"] = play(a0, {}, gap=1.6)
    # 缠抱：贴近了抱住，抱一会儿往后退开；够不着去扑空
    a0 = {5: S.FWD, 6: S.FWD, 7: S.FWD, 10: S.CLINCH, 40: S.BACK, 41: S.BACK, 80: S.CLINCH}
    out["缠抱"] = play(a0, {}, gap=0.9)
    # 一拳打倒：重拳连打，一次试 256 个运气种子，挑第一个打出头部击倒的；肝区：前手勾拳打身体，挑第一个肝区击倒的
    for name, punches in (("头部击倒", [S.CROSS_H, S.LHOOK_H, S.RHOOK_H, S.RUPPER_H]), ("肝区击倒", [S.LHOOK_B])):
        a0 = {}
        for i in range(70):                    # 每打一拳往前跟一步，抵掉击退
            a0[10 + 14 * i] = punches[i % len(punches)]
            a0[10 + 14 * i + 11] = S.FWD
        seed = first_kd(a0, gap=0.85, liver=name == "肝区击倒")
        if seed is not None:
            out[name] = play(a0, {}, gap=0.85, seed=seed)
    return out


def real_match():
    """正史最新存档的两人打一场（有存档才打）。"""
    try:
        from analyze import load_run
        R = load_run("canon_s1c")
    except (FileNotFoundError, OSError):
        return None
    ids = R["cfg"]["args"]["bodies"].split(",")
    bodies = tuple(body_of(c) for c in ids)
    skills = np.stack([R["pick"][c]["skills"] for c in ids])
    bt = BL.run_matches(bodies, tuple(R["pols"][c] for c in ids), tuple(ids), 1, seed=777, skills=skills, plans=(0, 0))
    return BL.replay_record(bt, 0, [{"name": c} for c in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", action="store_true")
    ap.add_argument("--skin", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    recs = scenarios()
    m = real_match()
    if m:
        recs["正史比赛"] = m
    paths, cards = [], []
    for name, rec in recs.items():
        frames = BL.resimulate(rec)
        card = {"title": name, "fighters": [b["id"] for b in rec["bodies"]], "bodies": rec["bodies"], "frames": frames,
                "result": {"winner": None, "method": "", "seconds": 0, "judges": [[0, 0]] * 3}}
        p = os.path.join(OUT, f"{name}.json")
        json.dump({"sim": B.viz_sim(), "card": card}, open(p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
        paths.append(p)
        cards.append(card)
        print(f"{name}：{len(frames)} 帧")
    if args.page:
        page(cards, args.skin)
    sys.exit(subprocess.call(["node", os.path.join("tools", "check_viz.mjs"), *(["--skin"] if args.skin else []), *paths]))


def page(cards, skin=False):
    """把场景拼成一个能逐个看的观看页（用 season 观看页的模板会牵扯训练营数据，这里只要比赛画面）。"""
    html = """<!doctype html><meta charset="utf-8"><title>画面核对</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui}#bar{padding:8px}canvas{display:block;width:100%;max-width:1280px}</style>
<div id="bar"><select id="sel"></select> <input id="t" type="range" min="0" max="1" step="0.001" style="width:60%"> <span id="lab"></span>
<button id="play">播放</button> <select id="spd"><option value="1">1×</option><option value="0.25">0.25×</option><option value="0.1">0.1×</option></select></div>
<canvas id="cv" width="1280" height="720"></canvas>
<script>/*__VIZ__*/</script><script>
const D = /*__DATA__*/null;
VIZ.setSim(D.sim);
if (D.skins) VIZ.setSkins(D.skins);
const cv = document.getElementById('cv'), c = cv.getContext('2d'), sel = document.getElementById('sel'), tr = document.getElementById('t');
D.cards.forEach((cd, i) => { const o = document.createElement('option'); o.value = i; o.textContent = cd.title; sel.appendChild(o); });
let tf = 0, playing = false, last = 0;
function draw() {
  const cd = D.cards[+sel.value]; VIZ.prep(cd); const n = cd.frames.length;
  c.clearRect(0, 0, 1280, 720); VIZ.drawBoutFrame(c, cd, tf, [0, 0, 1280, 720], { ids: cd.fighters, labels: cd.fighters });
  VIZ.hud(c, cd, Math.floor(tf), [0, 0, 1280, 720], cd.fighters, false);
  document.getElementById('lab').textContent = `第 ${tf.toFixed(2)} 帧 / ${n - 1}`; tr.value = tf / (n - 1);
}
tr.oninput = () => { tf = tr.value * (D.cards[+sel.value].frames.length - 1); draw(); };
sel.onchange = () => { tf = 0; draw(); };
document.getElementById('play').onclick = () => { playing = !playing; last = performance.now(); requestAnimationFrame(loop); };
function loop(now) { if (!playing) return; tf = Math.min(D.cards[+sel.value].frames.length - 1, tf + (now - last) / 50 * +document.getElementById('spd').value); last = now; draw(); requestAnimationFrame(loop); }
window.EXPORT = { items: () => D.cards.map((cd, i) => ({ i, title: cd.title, dur: cd.frames.length / 20, type: 'fight' })),
  frame: (i, t, w, h, z = 1) => {      // z > 1：放大画，再把两个人挪到画面中间（镜头不拍出擂台，人可能在画面边上）
    sel.value = i; tf = t * 20; cv.width = w; cv.height = h; const cd = D.cards[i]; VIZ.prep(cd);
    if (z === 1) { c.clearRect(0, 0, w, h); VIZ.drawBoutFrame(c, cd, tf, [0, 0, w, h], { ids: cd.fighters, labels: cd.fighters }); VIZ.hud(c, cd, Math.floor(tf), [0, 0, w, h], cd.fighters, false); return cv.toDataURL('image/jpeg', 0.9); }
    const off = document.createElement('canvas'); off.width = w * z; off.height = h * z;
    const { cam } = VIZ.drawBoutFrame(off.getContext('2d'), cd, tf, [0, 0, w * z, h * z], { ids: cd.fighters, labels: cd.fighters, hud: false });
    const f = cd.frames[Math.floor(tf)].f, p = cam.P((f[0][0] + f[1][0]) / 2, (f[0][1] + f[1][1]) / 2, 1.0);
    c.fillStyle = '#000'; c.fillRect(0, 0, w, h); c.drawImage(off, w / 2 - p[0], h * 0.55 - p[1]); return cv.toDataURL('image/jpeg', 0.9); },
  audio: () => [], ready: () => VIZ.skinsReady() };
(function first() { if (VIZ.skinsReady()) draw(); else setTimeout(first, 50); })();
</script>"""
    viz = open("viz.js", encoding="utf-8").read()
    skins = SK.load([fid for cd in cards for fid in cd["fighters"]], skin)
    data = json.dumps({"sim": B.viz_sim(), "cards": cards, "skins": skins}, ensure_ascii=False, separators=(",", ":"))
    out = os.path.join("runs", "viz_check_skin.html" if skin else "viz_check.html")
    open(out, "w", encoding="utf-8").write(html.replace("/*__VIZ__*/", viz).replace("/*__DATA__*/null", data))
    print("核对页：", os.path.abspath(out))


if __name__ == "__main__":
    main()
