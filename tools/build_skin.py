"""从三视图立绘生成角色的三维皮肤（画面验证用）。

输入 characters/<id>.rig.json：立绘、三个视图在图上的位置、关节点的像素坐标（正面 + 侧面）。
输出 characters/<id>.skin.json：每段身体（躯干、头颈、两臂、两腿、两脚）沿骨骼一圈圈的截面、
每个顶点在三视图上的贴图坐标和混合权重、贴图本身。viz.js 每帧按骨骼把截面摆上去（见 viz.js 的 SKIN）。

截面怎么量：正面量左右宽（往角色右边、左边各量到轮廓），侧面量前后厚；胳膊在侧面挡在身子前面量不出厚，按左右宽算。
贴图怎么贴：顶点朝前贴正面、朝后贴背面、朝两侧贴侧面（右侧当成左侧的镜像），交界处按法线方向渐变。
侧面里胳膊挡住的躯干、大腿不贴侧面，免得把胳膊贴到肋上。

用法：py -X utf8 tools/build_skin.py LJ [--debug 目录]（--debug 把截面描回立绘上，核对量得准不准）
"""
import argparse
import base64
import io
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

NT = 32                 # 一圈的顶点数
STEP = 0.012            # 截面间距（米）
BLEND = 0.25            # 关节处切线渐变的范围（按段长的比例），viz.js 的 SKIN 用同一个数
FOOT_ZC = 0.045         # 脚的中轴离地高度（米）
SHRINK = 0.985          # 截面略往里收，贴图不采到轮廓外面
EXP = {"torso": 2.6, "head": 2.2, "arm": 2.1, "leg": 2.2, "foot": 2.6}
NECK_RX, NECK_RY = 0.068, 0.075     # 脖子最粗多少（正面、侧面在脖子那几行会量到斜方肌、头发）


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def perp(v, t):
    return norm(v - t * np.dot(v, t))


def minrot(a, b, x):
    """把 a 转到 b 的最小旋转，作用在 x 上（viz.js 的 minrot 同一个算法）。"""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.9999:
        ax = perp(np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0]), a)
        return 2 * ax * np.dot(ax, x) - x
    return x * c + np.cross(v, x) + v * np.dot(v, x) / (1 + c)


def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


class Chain:
    """一串关节点。截面的位置用 (k, v) 表示：k 在 0..n-1 时是第 k 段上的比例 v；k = -1 / n 时是首尾往外延的米数。
    段内按比例走（骨头长短跟着模拟），往外延的按米数走（头顶、拳套尖、三角肌顶不跟着骨头拉长）。"""

    def __init__(self, pts):
        self.P = [np.asarray(p, float) for p in pts]
        self.d = [norm(self.P[i + 1] - self.P[i]) for i in range(len(self.P) - 1)]
        self.len = [np.linalg.norm(self.P[i + 1] - self.P[i]) for i in range(len(self.P) - 1)]
        self.cum = np.concatenate([[0], np.cumsum(self.len)])

    def kv(self, u):
        n = len(self.len)
        if u < 0:
            return -1, u
        if u > self.cum[-1]:
            return n, u - self.cum[-1]
        k = min(n - 1, int(np.searchsorted(self.cum, u, side="right")) - 1)
        return k, (u - self.cum[k]) / self.len[k]

    def at(self, k, v):
        n = len(self.len)
        if k < 0:
            return self.P[0] + self.d[0] * v, self.d[0]
        if k >= n:
            return self.P[-1] + self.d[-1] * v, self.d[-1]
        c = self.P[k] + (self.P[k + 1] - self.P[k]) * v
        t = self.d[k]
        if v < BLEND and k > 0:
            w = smoothstep((v + BLEND) / (2 * BLEND))
            t = norm(self.d[k - 1] * (1 - w) + self.d[k] * w)
        elif v > 1 - BLEND and k + 1 < n:
            w = smoothstep((v - 1 + BLEND) / (2 * BLEND))
            t = norm(self.d[k] * (1 - w) + self.d[k + 1] * w)
        return c, t


class Rig:
    def __init__(self, path):
        R = json.load(open(path, encoding="utf-8"))
        self.R = R
        self.S = R["scale"]
        self.G = R["ground"]
        self.vf, self.vs, self.vb = R["views"]["front"], R["views"]["side"], R["views"]["back"]
        self.img = Image.open(R["image"]).convert("RGB")
        self.W, self.H = self.img.size
        self.mask = silhouette(np.asarray(self.img).astype(float))
        self.J = {k: self.j3(v) for k, v in R["joints"].items()}
        self.arm_side = capsules(R["sideArm"], self.W, self.H)

    def j3(self, v):
        f, s = v.get("f"), v.get("s")
        X = (self.vf[2] - f[0]) / self.S if f else 0.0
        Y = (self.vs[2] - s[0]) / self.S if s else 0.0
        Z = (self.G - (f[1] if f else s[1])) / self.S
        return np.array([X, Y, Z])

    def z_of_row(self, y):
        return (self.G - y) / self.S

    # 三维点 → 三个视图上的像素
    def pf(self, P):
        return np.array([self.vf[2] - P[0] * self.S, self.G - P[2] * self.S])

    def ps(self, P):
        return np.array([self.vs[2] - P[1] * self.S, self.G - P[2] * self.S])

    def pb(self, P):
        return np.array([self.vb[2] + P[0] * self.S, self.G - P[2] * self.S])

    def split_x(self, side, y):
        """正面里胳膊和躯干的分界线在第 y 行的 x（分界线以下胳膊和躯干之间本来就隔着空）。"""
        (x0, y0), (x1, y1) = self.R["split"][side]
        if y > y0 or y < y1 - 40:
            return None
        return x0 + (x1 - x0) * (y - y0) / (y1 - y0)


def silhouette(im):
    """灰底上抠出人：拟合背景的渐变，离背景远或颜色鲜艳的算人，再去掉小块、补上小洞。"""
    H, W, _ = im.shape
    ys, xs = np.mgrid[0:H:8, 0:W:8]
    samp = im[::8, ::8]
    sat, br = samp.max(-1) - samp.min(-1), samp.mean(-1)
    cand = (sat < 12) & (br > 178) & (br < 228)
    A = np.stack([np.ones(cand.sum()), ys[cand] / H, xs[cand] / W, (xs[cand] / W) ** 2, (ys[cand] / H) ** 2], 1)
    coef = [np.linalg.lstsq(A, samp[cand][:, c], rcond=None)[0] for c in range(3)]
    Y, X = np.mgrid[0:H, 0:W]
    bg = np.stack([coef[c][0] + coef[c][1] * Y / H + coef[c][2] * X / W + coef[c][3] * (X / W) ** 2 + coef[c][4] * (Y / H) ** 2
                   for c in range(3)], -1)
    d = np.sqrt(((im - bg) ** 2).sum(-1))
    fg = ndi.binary_opening((d > 22) | (im.max(-1) - im.min(-1) > 30), iterations=1)
    lab, n = ndi.label(fg)
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sizes > 20000)[0])
    hl, hn = ndi.label(~keep)
    hs = ndi.sum(~keep, hl, range(1, hn + 1))
    return ndi.binary_closing(keep | np.isin(hl, 1 + np.where(hs < 1500)[0]), iterations=2)


def capsules(caps, W, H):
    """侧面里胳膊占的地方：沿着关节点连起来的一串胶囊。"""
    Y, X = np.mgrid[0:H, 0:W]
    m = np.zeros((H, W), bool)
    for (x0, y0, r0), (x1, y1, r1) in zip(caps[:-1], caps[1:]):
        ax, ay = x1 - x0, y1 - y0
        t = np.clip(((X - x0) * ax + (Y - y0) * ay) / (ax * ax + ay * ay), 0, 1)
        r = r0 + (r1 - r0) * t
        m |= (X - x0 - t * ax) ** 2 + (Y - y0 - t * ay) ** 2 <= r * r
    return m


def march(ok, p, d, maxd=400):
    """从 p 沿 d 一直走到出轮廓，返回走了多远（像素）和出去前最后一格是不是"可疑"（被胳膊挡着）。"""
    last, sus = 0.0, False
    t = 0.0
    while t < maxd:
        x, y = p + d * t
        st = ok(int(round(x)), int(round(y)))
        if st == 0:
            break
        if st == 1:
            last, sus = t, False
        else:
            sus = True                # 走在胳膊上：还在人里面，但量不出这一段是不是躯干
        t += 0.5
    return last, sus


def build(rid, debug=None):
    rg = Rig(os.path.join("characters", f"{rid}.rig.json"))
    J, S, M = rg.J, rg.S, rg.mask
    rows = rg.R["rows"]
    H, W = M.shape
    inside = lambda x, y: 0 <= x < W and 0 <= y < H and M[y, x]

    # 躯干：骨盆 → 两肩中点
    torso = Chain([J["pel"], J["chest"]])
    T0 = torso.d[0]
    r_hip, r_sh = perp(J["hipR"] - J["hipL"], T0), perp(J["shR"] - J["shL"], T0)
    top = (r_sh, np.cross(T0, r_sh), T0)              # 躯干顶的标架（右、前、上），胳膊挂在这上面
    bot = (r_hip, np.cross(T0, r_hip), T0)            # 躯干底的标架，腿挂在这上面
    dQ = J["head"][2] - J["chin"][2]
    Q = J["head"] + norm(J["chest"] - J["head"]) * dQ
    head = Chain([J["chest"], Q, J["head"]])
    dWr = {s: float(np.linalg.norm(J["gc" + s] - J["wr" + s])) for s in "RL"}
    dTip = {s: float(np.linalg.norm(J["gt" + s] - J["gc" + s])) for s in "RL"}
    foot = {"heel": float(J["heel"][1]), "toe": float(J["toe"][1])}

    parts = []

    def part(name, kind, ch, u0, u1, frames, measure, parent=None):
        n = max(2, int(round((u1 - u0) / STEP))) + 1
        us = np.linspace(u0, u1, n)
        rings, C, Fx, Fy, Tt = [], [], [], [], []
        X = Y = prevT = root = None
        for i, u in enumerate(us):
            k, v = ch.kv(u)
            c, t = ch.at(k, v)
            tw = 0.0                                   # 躯干：从胯的朝向扭到肩的朝向；头颈：从躯干的朝向转到脸的朝向
            if frames == "transport":
                if i == 0:
                    X = perp(parent[0], t)
                    Y = np.cross(X, t)
                    root = {"X": [float(np.dot(X, a)) for a in parent], "Y": [float(np.dot(Y, a)) for a in parent],
                            "T": [float(np.dot(t, a)) for a in parent]}
                else:
                    X, Y = minrot(prevT, t, X), minrot(prevT, t, Y)
                prevT = t
            else:
                X, Y, tw = frames(k, v, u, t)
            rings.append([k, v, tw])
            C.append(c)
            Fx.append(X)
            Fy.append(Y)
            Tt.append(t)
        ext = np.array([measure(C[i], Fx[i], Fy[i], us[i]) for i in range(n)])   # (n, 4)：+X、-X、+Y、-Y，nan = 量不出
        for j in range(4):
            col = ext[:, j]
            bad = np.isnan(col)
            if bad.all():
                raise SystemExit(f"{name}：第 {j} 个方向一个也没量出来")
            idx = np.arange(n)
            col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
            col = ndi.median_filter(col, size=5, mode="nearest")
            ext[:, j] = ndi.gaussian_filter1d(col, 1.2, mode="nearest")
        ext *= SHRINK
        e = EXP[kind]
        th = 2 * np.pi * np.arange(NT) / NT
        cx = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2 / e)
        sy = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2 / e)
        V = np.zeros((n, NT, 3))
        geo = []
        for i in range(n):
            ox, rx = (ext[i, 0] - ext[i, 1]) / 2, (ext[i, 0] + ext[i, 1]) / 2
            oy, ry = (ext[i, 2] - ext[i, 3]) / 2, (ext[i, 2] + ext[i, 3]) / 2
            geo.append([int(rings[i][0]), round(float(rings[i][1]), 5), round(ox, 4), round(oy, 4), round(rx, 4), round(ry, 4),
                        round(float(rings[i][2]), 4)])
            V[i] = C[i] + np.outer(ox + rx * cx, Fx[i]) + np.outer(oy + ry * sy, Fy[i])
        caps = [C[0] + Fx[0] * geo[0][2] + Fy[0] * geo[0][3], C[-1] + Fx[-1] * geo[-1][2] + Fy[-1] * geo[-1][3]]
        Nn = normals(V, C)
        P = np.concatenate([V.reshape(-1, 3), np.array(caps)])
        Nv = np.concatenate([Nn.reshape(-1, 3), np.array([-Tt[0], Tt[-1]])])
        uv, w = texcoords(rg, P, Nv, kind)
        out = {"name": name, "kind": kind, "e": e, "rings": geo, "uv": uv, "w": w}
        if frames == "transport":
            out["root"] = {k2: [round(x, 6) for x in v2] for k2, v2 in root.items()}
        parts.append(out)
        return V

    # ---- 量截面的办法
    def ok_front(cond=None):
        def ok(x, y):
            if not inside(x, y):
                return 0
            return 1 if cond is None or cond(x, y) else 0
        return ok

    def ok_side_masked(arm_ok):
        def ok(x, y):
            if not inside(x, y):
                return 0
            return 2 if (rg.arm_side[y, x] and not arm_ok) else 1
        return ok

    def meas(view, P, V, ok):
        p = rg.pf(P) if view == "f" else rg.ps(P)
        dv = np.array([-V[0], -V[2]]) if view == "f" else np.array([-V[1], -V[2]])
        if np.linalg.norm(dv) < 0.3:
            return np.nan
        t, sus = march(ok, p, dv / np.linalg.norm(dv))
        return np.nan if (sus or t == 0) else t / S / np.linalg.norm(dv)

    mid = rg.vf[2]

    def torso_ok(x, y):
        a, b = rg.split_x("R", y), rg.split_x("L", y)
        return (a is None or x >= a) and (b is None or x <= b)

    def m_torso(c, X, Y, u):
        okf = ok_front(cond=torso_ok)
        oks = ok_side_masked(False)
        return [meas("f", c, X, okf), meas("f", c, -X, okf), meas("s", c, Y, oks), meas("s", c, -Y, oks)]

    def m_head(c, X, Y, u):
        okf, oks = ok_front(), ok_side_masked(True)
        r = [meas("f", c, X, okf), meas("f", c, -X, okf), meas("s", c, Y, oks), meas("s", c, -Y, oks)]
        if u < head.cum[1]:            # 脖子：正面、侧面在这几行会量到斜方肌和后脑的头发
            r = [min(r[0], NECK_RX), min(r[1], NECK_RX), min(r[2], NECK_RY), min(r[3], NECK_RY * 0.9)]
        return r

    def m_arm(side):
        def cond(x, y):
            a = rg.split_x(side, y)
            return a is None or (x < a if side == "R" else x > a)
        okf = ok_front(cond=cond)

        def m(c, X, Y, u):
            a, b = meas("f", c, X, okf), meas("f", c, -X, okf)
            r = (a + b) / 2 if not (np.isnan(a) or np.isnan(b)) else np.nan
            return [a, b, r, r]
        return m

    def m_leg(side):
        okf = ok_front(cond=(lambda x, y: x <= mid) if side == "R" else (lambda x, y: x >= mid))
        oks = ok_side_masked(False)
        return lambda c, X, Y, u: [meas("f", c, X, okf), meas("f", c, -X, okf), meas("s", c, Y, oks), meas("s", c, -Y, oks)]

    def m_foot(side, half):
        oks = ok_side_masked(True)

        def m(c, X, Y, u):
            v = u / (foot["toe"] - foot["heel"])
            w = half * (0.78 + 0.22 * np.sin(np.pi * min(1, max(0, v))))
            up = meas("s", c, Y, oks)
            return [w, w, min(up, 0.065) if not np.isnan(up) else np.nan, FOOT_ZC]
        return m

    # ---- 各段
    def torso_frames(k, v, u, t):
        w = smoothstep(u / torso.cum[-1])
        X = perp(r_hip + (r_sh - r_hip) * w, t)
        return X, np.cross(t, X), w
    part("torso", "torso", torso, rg.z_of_row(rows["crotch"]) - J["pel"][2], torso.cum[-1] + rg.z_of_row(rows["neckBase"]) - J["chest"][2],
         torso_frames, m_torso)
    # 头颈：脖子跟着躯干朝向，头跟着脸的朝向（立绘里两者一样，画面里脸会转向对手）
    fw0 = top[1]

    def head_frames(k, v, u, t):
        w = smoothstep((u - (head.cum[1] - 0.05)) / 0.07)
        Y = perp(fw0 * (1 - w) + np.array([0, 1.0, 0]) * w, t)
        return np.cross(Y, t), Y, w
    part("head", "head", head, 0.01, head.cum[-1] + rg.z_of_row(rows["hairTop"]) - J["head"][2], head_frames, m_head)
    for s in "RL":
        arm = Chain([J["sh" + s], J["el" + s], J["wr" + s], J["gc" + s], J["gt" + s]])
        part("arm" + s, "arm", arm, -0.07, arm.cum[-1] + 0.006, "transport", m_arm(s), parent=top)
        leg = Chain([J["hip" + s], J["kn" + s], J["an" + s]])
        part("leg" + s, "leg", leg, -0.05, leg.cum[-1], "transport", m_leg(s), parent=bot)
        fy, row = rg.pf(J["an" + s]), int(rg.G - 30)          # 脚宽：离地 30 像素那一行（再低就量到台面上的影子）
        run = [x for x in range(int(fy[0]) - 80, int(fy[0]) + 80) if inside(x, row)]
        half = (max(run) - min(run)) / 2 / S
        ft = Chain([[J["an" + s][0], foot["heel"], FOOT_ZC], [J["an" + s][0], foot["toe"], FOOT_ZC]])
        part("foot" + s, "foot", ft, 0.0, ft.cum[-1], lambda k, v, u, t: (lambda U: (np.cross(t, U), U, 0.0))(perp(np.array([0, 0, 1.0]), t)),
             m_foot(s, half))

    tex = texture(rg)
    out = {"id": rid, "v": 1, "nt": NT, "blend": BLEND, "tex": tex, "size": [W, H], "parts": parts,
           "bind": {"dQ": round(float(dQ), 4), "dWr": dWr, "dTip": dTip, "heel": foot["heel"] - float(J["anR"][1]),
                    "toe": foot["toe"] - float(J["anR"][1]), "zc": FOOT_ZC}}
    path = os.path.join("characters", f"{rid}.skin.json")
    json.dump(out, open(path, "w", encoding="utf-8"), separators=(",", ":"))
    nv = sum(len(p["w"]) // 3 for p in parts)
    print(f"{path}：{len(parts)} 段、{nv} 个顶点，{os.path.getsize(path) / 1e6:.2f} MB")
    if debug:
        overlay(rg, parts, debug)


def normals(V, C):
    n, m, _ = V.shape
    dth = np.roll(V, -1, 1) - np.roll(V, 1, 1)
    ds = np.concatenate([V[1:2] - V[0:1], V[2:] - V[:-2], V[-1:] - V[-2:-1]], 0)
    N = np.cross(ds, dth)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    out = V - np.asarray(C)[:, None, :]
    N[(N * out).sum(-1) < 0] *= -1
    return N


def texcoords(rg, P, N, kind):
    """每个顶点在正面、侧面、背面上的像素坐标（整数）和三张的权重（0~255）。"""
    pf = np.array([rg.pf(p) for p in P])
    ps = np.array([rg.ps(p) for p in P])
    pb = np.array([rg.pb(p) for p in P])
    h = N[:, :2] / (np.linalg.norm(N[:, :2], axis=1, keepdims=True) + 1e-9)
    h[np.linalg.norm(N[:, :2], axis=1) < 0.05] = [0, 1]
    wf, wb, ws = np.maximum(h[:, 1], 0) ** 3, np.maximum(-h[:, 1], 0) ** 3, np.abs(h[:, 0]) ** 3
    if kind != "arm":                  # 侧面里被胳膊挡住的地方不贴侧面
        H, W = rg.arm_side.shape
        xs, ys = np.clip(ps[:, 0].round().astype(int), 0, W - 1), np.clip(ps[:, 1].round().astype(int), 0, H - 1)
        ws[rg.arm_side[ys, xs]] = 0
    tot = wf + wb + ws
    zero = tot < 1e-6
    wf[zero] = (N[zero, 1] >= 0).astype(float)
    wb[zero] = (N[zero, 1] < 0).astype(float)
    tot = wf + wb + ws
    w = np.stack([wf, ws, wb], 1) / tot[:, None]
    uv = np.concatenate([pf, ps, pb], 1).round().astype(int)
    return uv.reshape(-1).tolist(), (w * 255).round().astype(int).reshape(-1).tolist()


def texture(rg):
    """贴图：轮廓外的灰底换成离它最近的人身上的颜色（采到轮廓边上不会带出一圈灰边）。"""
    im = np.asarray(rg.img)
    core = ndi.binary_erosion(rg.mask, iterations=2)
    _, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    out = im[iy, ix]
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay(rg, parts, outdir):
    """把每一圈截面描回三视图：轮廓线贴着人走才算量对了。"""
    os.makedirs(outdir, exist_ok=True)
    im = rg.img.copy()
    dr = ImageDraw.Draw(im)
    for p in parts:
        uv = np.array(p["uv"]).reshape(-1, 6)
        nr = len(p["rings"])
        for i in range(0, nr, 2):
            ring = uv[i * NT:(i + 1) * NT]
            for a, col in ((0, (0, 255, 255)), (2, (255, 255, 0)), (4, (255, 0, 255))):
                pts = [tuple(q) for q in ring[:, a:a + 2]]
                dr.line(pts + [pts[0]], fill=col, width=1)
    for k, v in rg.J.items():
        for f in (rg.pf, rg.ps):
            x, y = f(v)
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 0, 0), width=2)
    im.save(os.path.join(outdir, "overlay.png"))
    print("核对图：", os.path.join(outdir, "overlay.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--debug")
    a = ap.parse_args()
    build(a.id, a.debug)
