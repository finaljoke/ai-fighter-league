"""从三视图立绘生成角色的三维皮肤。规范（立绘要求、标注、比例约定、核对）见 docs/角色皮肤规范.md。

输入 characters/<id>.rig.json：立绘、三个视图在图上的位置、关节点的像素坐标（正面 + 侧面）。
     胳膊和躯干的分界、侧面胳膊挡住的区域、裆和脖子根的位置都从轮廓里推，推得不对再在标注里写 split / sideArm / rows 覆盖。
     角色的身体参数要先在 bodies.py 里定义：身高按它算（同 viz.js 的 build），立绘只管形状和粗细。
输出 characters/<id>.skin.json：
  parts：每段身体（躯干、头颈、两臂、两腿、两脚）沿骨骼一圈圈的截面，每个顶点在三视图上的贴图坐标和混合权重
  rig：  从立绘量出来的骨长和身体尺寸，画面里替换按身体参数估的那一套（胳膊总长除外，它跟着模拟的臂展）
  ref：  做皮肤时的身体参数，体重、力量变了以后画面按比例加减粗细
  tex：  贴图（三视图原图 + 侧面补色图拼在一起）
  标注里有 model（AI 生成的全身模型，tools/reduce_model.mjs 减过面）时：截面只当骨头，不带贴图坐标；
  模型绑在截面上写进 model，tex 换成模型的贴图（tools/skin_model.py）

截面怎么量：正面量左右宽，侧面量前后厚；胳膊在侧面挡在身子前面量不出厚，按左右宽算。
贴图怎么贴：顶点朝前贴正面、朝后贴背面、朝两侧贴侧面（右侧当成左侧的镜像），交界处按法线方向渐变。
侧面里躯干、大腿被胳膊挡住的地方，先按同一行两边露出来的身体颜色补上（补色图），躯干、腿的侧面贴补色图。

用法：py -X utf8 tools/build_skin.py LJ GY [--debug 目录]（--debug 把截面描回立绘上，核对量得准不准）
"""
import argparse
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bodies as BD                    # noqa: E402
import skin_head as SH                 # noqa: E402
import skin_model as SMD               # noqa: E402

NT = 32                 # 一圈的顶点数
STEP = 0.012            # 截面间距（米）
BLEND = 0.25            # 关节前后圆过去的范围（按段长的比例），viz.js 的 SKIN 用同一个数
FOOT_ZC = 0.045         # 脚的中轴离地高度（米）
SHRINK = 0.985          # 截面略往里收，贴图不采到轮廓外面
EXP = {"torso": 2.3, "head": 2.2, "arm": 2.1, "leg": 2.2, "foot": 2.6}
# 管子两头收成圆顶，不留平盖：out = 往外多接一个半球（胳膊根、大腿根），in = 最后一段收圆（拳面、脚尖脚跟、躯干顶，长度不变）
# out 的半球球心在关节上、半径取第一圈截面的平均半径：胳膊、腿绕关节怎么转，这个球都在原地，不会往外戳出一截管子头
DOME = {"arm": ("out", "in"), "leg": ("out", None), "foot": ("in", "in"), "torso": (None, "in"), "head": (None, "in")}
DOME_LEN = {"arm": (0, 0.03), "foot": (0.03, 0.04), "torso": (0, 0.03), "head": (0, 0.025)}  # in 的长度（米）；躯干顶收到比脖子略粗，再收几圈圆进脖子里，只收一圈就是脖子根一圈平台
NT_HEAD, STEP_HEAD = 48, 0.006     # 头的截面密一些：鼻梁三四厘米宽、鼻尖到嘴不到两厘米，32 个点、1.2 厘米一圈摆不出来
DOME_T = (1.0, 0.95, 0.85, 0.7, 0.5, 0.25)                                       # out 半球的几圈：离第一圈的距离占半径的比例
YOKE = (-0.06, 0.0)     # 肩关节往下 6 厘米到肩关节这一段，躯干的宽度从"只算躯干"渐变到"连三角肌顶"（相对肩关节高度，米）
NECK_RX, NECK_RY = 0.068, 0.075     # 脖子最粗多少（正面、侧面在脖子那几行会量到斜方肌、头发）
TRAP_NECK = 1.25        # 躯干顶停在正面轮廓收到脖子最粗（NECK_RX）这么多倍的那一行：斜方肌在这儿并进脖子
HEAD_FRAC = 0.465 + 0.3 + 0.045 + 0.064 * 0.95   # viz.js rig 里头心离地占身高 H 的比例：立绘的头心高按这个对上身高


def body_H(b):
    return 1.72 + (b.reach - 0.75) * 0.75                     # 同 viz.js build 的 H


def body_arm(b):
    return 0.36 * body_H(b) * (1 + 0.8 * (b.reach - 0.75))    # 同 viz.js rig 的胳膊长（肩到拳套中心），跟着臂展


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


def runs_of(row):
    """一行里连续为真的几段 [(起, 止)]。"""
    d = np.diff(np.concatenate([[0], row.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


class Chain:
    """一串关节点。截面的位置用 (k, v) 表示：k 在 0..n-1 时是第 k 段上的比例 v；k = -1 / n 时是首尾往外延的米数。
    段内按比例走（骨头长短跟着画面的骨骼），往外延的按米数走（头顶、拳套尖、三角肌顶不跟着骨头拉长）。"""

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
        """关节前后各 BLEND 段长的范围里，中心走一段二次贝塞尔（两端在两根骨头上、控制点是关节），同 viz.js 的 chainAt。"""
        n = len(self.len)
        if k < 0:
            return self.P[0] + self.d[0] * v, self.d[0]
        if k >= n:
            return self.P[-1] + self.d[-1] * v, self.d[-1]
        J, s = (k, 0.5 + v / (2 * BLEND)) if v < 0.5 else (k + 1, (v - 1 + BLEND) / (2 * BLEND))
        if 0 < J < n and 0 <= s <= 1:
            A = self.P[J] - (self.P[J] - self.P[J - 1]) * BLEND
            B = self.P[J] + (self.P[J + 1] - self.P[J]) * BLEND
            return (1 - s) ** 2 * A + 2 * s * (1 - s) * self.P[J] + s * s * B, norm((self.P[J] - A) * (1 - s) + (B - self.P[J]) * s)
        return self.P[k] + (self.P[k + 1] - self.P[k]) * v, self.d[k]


class Rig:
    """标注 + 立绘：关节点换成米（身高按 bodies.py 的身体参数），没标的辅助位置从轮廓里推。"""

    def __init__(self, rid):
        R = json.load(open(os.path.join("characters", f"{rid}.rig.json"), encoding="utf-8"))
        if rid not in BD.ALL:
            raise SystemExit(f"bodies.py 里没有 {rid}：先定义身体参数，再做皮肤")
        self.body = BD.ALL[rid]
        self.R = R
        self.G = R["ground"]
        self.vf, self.vs, self.vb = R["views"]["front"], R["views"]["side"], R["views"]["back"]
        self.img = Image.open(R["image"]).convert("RGB")
        self.W, self.H = self.img.size
        self.mask = silhouette(np.asarray(self.img).astype(float))
        # 补画的头部三视图：贴回立绘的头上得到头的轮廓（head_mask），量头那一段用。有全身模型时不用（头就是模型的头）
        self.ref = SH.HeadRef(self) if "headRef" in R and "model" not in R else None
        self.head_mask = self.ref.paste() if self.ref else self.mask
        jf = R["joints"]
        self.S = (self.G - jf["head"]["f"][1]) / (HEAD_FRAC * body_H(self.body))     # 像素 / 米
        self.J = {k: self.j3(v) for k, v in jf.items()}
        # 全身模型（标注里的 model）：截面只当骨头用，画的是绑在截面上的模型（skin_model.py）；胳膊的关节按模型重定
        self.model = SMD.Model(self, R["model"]["glb"]) if "model" in R else None
        if self.model:
            self.model.fit_arms(self.J)
        self.rows = {**self.derive_rows(), **R.get("rows", {})}
        if self.ref and "hairTop" not in R.get("rows", {}):           # 头顶按补画的头发
            self.rows["hairTop"] = int(np.where(self.head_mask[:, self.vf[0]:self.vf[1]].any(1))[0].min()) + 2
        self.split = R.get("split") or {s: self.derive_split(s) for s in "RL"}
        self.side_caps = R.get("sideArm") or self.derive_side_arm()
        self.arm_side = capsules(self.side_caps, self.W, self.H)

    def j3(self, v):
        f, s = v.get("f"), v.get("s")
        X = (self.vf[2] - f[0]) / self.S if f else 0.0
        Y = (self.vs[2] - s[0]) / self.S if s else 0.0
        Z = (self.G - (f[1] if f else s[1])) / self.S
        return np.array([X, Y, Z])

    def px(self, name, view="f"):
        return self.R["joints"][name][view]

    def z_of_row(self, y):
        return (self.G - y) / self.S

    # 三维点 → 三个视图上的像素
    def pf(self, P):
        return np.array([self.vf[2] - P[0] * self.S, self.G - P[2] * self.S])

    def ps(self, P):
        return np.array([self.vs[2] - P[1] * self.S, self.G - P[2] * self.S])

    def pb(self, P):
        return np.array([self.vb[2] + P[0] * self.S, self.G - P[2] * self.S])

    def front_run(self, y, x):
        """正面第 y 行里包含 x 的那一段。"""
        for a, b in runs_of(self.mask[y, self.vf[0]:self.vf[1]]):
            if a + self.vf[0] <= x <= b + self.vf[0]:
                return a + self.vf[0], b + self.vf[0]
        return None

    def derive_rows(self):
        cx = self.vf[2]
        hair = int(np.where(self.mask[:, self.vf[0]:self.vf[1]].any(1))[0].min())
        chest, chin, pel = self.px("chest")[1], self.px("chin")[1], self.px("pel")[1]
        crotch = next(y for y in range(pel, self.G) if not self.mask[y, cx - 2:cx + 3].any())    # 两腿之间第一次露出背景
        # 短裤的腰头上沿、裤脚：正面中线往下，颜色离开胸口皮肤的第一行；沿右腿往下，颜色回到皮肤的第一行（侧面补色不跨这两行）
        im = np.asarray(self.img).astype(float)
        skin = im[chest + 60:chest + 90, cx - 20:cx + 20].reshape(-1, 3).mean(0)
        far = lambda y, x: np.linalg.norm(im[y, x - 12:x + 13].mean(0) - skin) > 60
        waist = next(y for y in range(chest + 80, pel + 60) if all(far(y + d, cx) for d in range(4)))
        (xh, yh), (xk, yk) = self.px("hipR"), self.px("knR")
        leg_x = lambda y: int(round(xh + (xk - xh) * (y - yh) / (yk - yh)))
        hem = next(y for y in range(crotch, yk) if not any(far(y + d, leg_x(y + d)) for d in range(4)))
        return {"hairTop": hair + 2, "trapTop": self.derive_trap_top(chest, chin), "neckBase": int(round(chest - 0.35 * (chest - chin))),
                "crotch": crotch, "waist": waist, "hem": hem}

    def derive_trap_top(self, chest, chin):
        """躯干顶：两肩连线往上，正面轮廓顺着斜方肌一行比一行窄，收到比脖子略粗的那一行停。
        按固定比例取会停在三角肌顶上：上面那段坡全归了脖子，脖子成了插在平板肩膀上的一根管子。最多找到下巴往上三成，再往上是脸。"""
        if "trapTop" in self.R.get("rows", {}):
            return self.R["rows"]["trapTop"]
        lim = TRAP_NECK * NECK_RX * self.S
        for y in range(chest, int(chin - 0.3 * (chest - chin)), -1):
            run = self.front_run(y, self.vf[2])
            if run and (run[1] - run[0]) / 2 <= lim:
                return y
        raise SystemExit("推不出躯干顶（斜方肌的轮廓一直没收到脖子粗细），在标注里写 rows.trapTop")

    def derive_split(self, side):
        """正面里胳膊和躯干的分界线：从胳肢窝（胳膊和躯干第一次分开的那一行）往上斜到脖子根旁边。"""
        (xs, ys), (xe, ye) = self.px("sh" + side), self.px("el" + side)
        sg = 1 if side == "R" else -1                          # 角色右臂在图左：往中线是 +x
        for y in range(ys + 10, ye):
            xl = xs + (xe - xs) * (y - ys) / (ye - ys)
            run = self.front_run(y, self.vf[2])
            if run and (run[0] > xl + 5 if side == "R" else run[1] < xl - 5):
                xa = run[0] if side == "R" else run[1]
                yt = self.rows["neckBase"] - 7
                return [[xa, y], [round(xa + sg * 0.11 * (y - yt)), yt]]
        raise SystemExit(f"推不出 {side} 臂和躯干的分界（胳膊一直贴着身子？），在标注里写 split")

    def derive_side_arm(self):
        """侧面里胳膊占的地方：沿侧面的肩、肘、腕、拳套中心、拳套尖连一串胶囊，粗细按正面同一处的宽度。"""
        def half(name):
            x, y = self.px(name)
            run = self.front_run(y, x)
            return (run[1] - run[0]) / 2 if run else 30
        (xw, yw), (xt, yt) = self.px("wrL", "s"), self.px("gtL", "s")
        r_gc = max(half("gcL") + 8, 0.48 * np.hypot(xt - xw, yt - yw) + 4)   # 拳套在侧面比正面看着大：按腕到拳套尖的长度
        r_el, r_wr = half("elL") + 6, max(half("wrL") + 10, 0.8 * r_gc)      # 腕上的白绷带在侧面是斜着的，盖宽一点
        rs = [r_el * 1.3, r_el, r_wr, r_gc, r_gc * 0.8]
        return [[*self.px(n, "s"), round(r)] for n, r in zip(["shL", "elL", "wrL", "gcL", "gtL"], rs)]

    def split_x(self, side, y):
        """分界线在第 y 行的 x（分界线以下胳膊和躯干之间本来就隔着空）。"""
        (x0, y0), (x1, y1) = self.split[side]
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
    """一串胶囊盖住的地方（侧面里的胳膊）。"""
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
    rg = Rig(rid)
    J, S, M, rows = rg.J, rg.S, rg.mask, rg.rows
    H, W = M.shape
    inside = lambda x, y: 0 <= x < W and 0 <= y < H and M[y, x]
    inside_head = lambda x, y: 0 <= x < W and 0 <= y < H and rg.head_mask[y, x]

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
    model = rg.model                            # 有全身模型就不按三视图贴图
    side_tex = None if model else inpaint_side(rg)
    atlas = None if model else atlas_of(rg, side_tex)
    heads = []                                  # 头的展开图（skin_head.bake），最后拼到贴图右边

    parts, meas, bones = [], {}, []

    def part(name, kind, ch, u0, u1, frames, measure, parent=None, step=STEP, nt=NT, face=None):
        n = max(2, int(round((u1 - u0) / step))) + 1
        us = np.linspace(u0, u1, n)
        d0, d1 = DOME[kind]
        L0, L1 = DOME_LEN.get(kind, (0, 0))
        pre = [u0 - 0.04 * t for t in DOME_T] if d0 == "out" else []      # 往外接的半球：先占位，量完第一圈按它的半径重新摆
        us = np.concatenate([pre, us])
        n = len(us)
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
        raw = np.zeros_like(ext)                       # 没滤过的：头要用侧面轮廓上的鼻尖、嘴、下巴
        for j in range(4):
            col = ext[:, j]
            bad = np.isnan(col)
            if bad.all():
                raise SystemExit(f"{name}：第 {j} 个方向一个也没量出来")
            idx = np.arange(n)
            col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
            raw[:, j] = col * SHRINK
            col = ndi.median_filter(col, size=5, mode="nearest")
            ext[:, j] = ndi.gaussian_filter1d(col, 1.2, mode="nearest")
        ext *= SHRINK
        if pre:                                        # 往外的半球：照第一圈真截面的样子，按四分之一圆往里收
            R0 = float(ext[len(pre)].mean())
            for i, t in enumerate(DOME_T):
                us[i] = u0 - R0 * t
                k, v = ch.kv(us[i])
                rings[i][:2] = [k, v]
                C[i] = ch.at(k, v)[0]                  # 贴图坐标按挪过去的位置算
                ext[i] = ext[len(pre)] * np.sqrt(max(0.0, 1 - t ** 2)) + 1e-4
        for d, L, a, b in ((d0, L0, us[0], 1), (d1, L1, us[-1], -1)):
            if d == "in":                              # 往里收的圆顶：最后 L 米按四分之一圆收，长度不变
                t = np.clip(1 - b * (us - a) / L, 0, 1) if b == 1 else np.clip(1 - (a - us) / L, 0, 1)
                ext *= np.sqrt(np.clip(1 - t ** 2, 0.04, 1))[:, None]
        meas[name] = (us, ext)
        e = EXP[kind]
        th = 2 * np.pi * np.arange(nt) / nt
        cx = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2 / e)
        sy = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2 / e)
        loc = SH.shape(face, C, Fy, ext, raw, nt, step) if face else None       # 头：截面换成模板脸（见 skin_head.py）
        V = np.zeros((n, nt, 3))
        geo = []
        for i in range(n):
            ox, rx = (ext[i, 0] - ext[i, 1]) / 2, (ext[i, 0] + ext[i, 1]) / 2
            oy, ry = (ext[i, 2] - ext[i, 3]) / 2, (ext[i, 2] + ext[i, 3]) / 2
            geo.append([int(rings[i][0]), round(float(rings[i][1]), 5), round(ox, 4), round(oy, 4), round(rx, 4), round(ry, 4),
                        round(float(rings[i][2]), 4)])
            if face:
                V[i] = C[i] + np.outer(loc[i, :, 0], Fx[i]) + np.outer(loc[i, :, 1], Fy[i])
            else:
                V[i] = C[i] + np.outer(ox + rx * cx, Fx[i]) + np.outer(oy + ry * sy, Fy[i])
        caps = [C[0] + Fx[0] * geo[0][2] + Fy[0] * geo[0][3], C[-1] + Fx[-1] * geo[-1][2] + Fy[-1] * geo[-1][3]]
        Nn = normals(V, C)
        if model:
            # 只留截面（画面里按它摆骨头）和立绘站姿下每圈的中心、标架（模型的顶点按它算相对位置）
            bones.append({"name": name, "kind": kind, "C": np.array(C), "X": np.array(Fx), "Y": np.array(Fy), "T": np.array(Tt),
                           "ext": ext.copy(), "kv": [(r[0], r[1]) for r in rings]})
            rest = np.concatenate([np.array(C), np.array(Fx), np.array(Fy), np.array(Tt)], 1)
            out = {"name": name, "kind": kind, "e": e, "rings": geo, "rest": [round(float(x), 5) for x in rest.reshape(-1)]}
        elif face:
            # 头贴烘焙好的展开图：三组贴图坐标都指向展开图上同一个点，权重全给第一组。每圈多一列接缝（和第一列同位置、贴图在展开图最右边）
            img, uvs = SH.bake(face, V, Nn, atlas)
            heads.append(img)
            uvs[..., 0] += atlas.shape[1]
            cap_uv = [[uvs[0, :, 0].mean(), uvs[0, 0, 1]], [uvs[-1, :, 0].mean(), uvs[-1, 0, 1]]]
            q = np.concatenate([uvs.reshape(-1, 2), np.array(cap_uv)]).round().astype(int)
            uv = np.concatenate([q, q, q], 1).reshape(-1).tolist()
            w = [255, 0, 0] * len(q)
            loc = np.concatenate([loc, loc[:, :1]], 1)
            out = {"name": name, "kind": kind, "e": e, "rings": geo, "uv": uv, "w": w, "nt": nt, "seam": 1,
                   "loc": [round(float(x), 4) for x in loc.reshape(-1)]}
        else:
            P = np.concatenate([V.reshape(-1, 3), np.array(caps)])
            Nv = np.concatenate([Nn.reshape(-1, 3), np.array([-Tt[0], Tt[-1]])])
            uv, w = texcoords(rg, P, Nv, kind)
            out = {"name": name, "kind": kind, "e": e, "rings": geo, "uv": uv, "w": w}
        if frames == "transport":
            out["root"] = {k2: [round(x, 6) for x in v2] for k2, v2 in root.items()}
        parts.append(out)

    # ---- 量截面的办法
    def ok_front(cond=None, ins=inside):
        def ok(x, y):
            if not ins(x, y):
                return 0
            return 1 if cond is None or cond(x, y) else 0
        return ok

    def ok_side_masked(arm_ok, ins=inside):
        def ok(x, y):
            if not ins(x, y):
                return 0
            return 2 if (rg.arm_side[y, x] and not arm_ok) else 1
        return ok

    def meas_dir(view, P, V, ok):
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

    sh_z = (J["shR"][2] + J["shL"][2]) / 2

    # 躯干：正面按分界线只量躯干；肩关节附近渐变到连三角肌顶一起量（YOKE）。胳膊往前、往里、往上抬时，
    # 圆圆的肩头留在躯干上，胳膊从它底下出来；不然胳膊顶端那截管子会从躯干侧面伸出来，像翅膀、柱子
    def m_torso(c, X, Y, u):
        okf, oks = ok_front(cond=torso_ok), ok_side_masked(False)
        r = [meas_dir("f", c, X, okf), meas_dir("f", c, -X, okf), meas_dir("s", c, Y, oks), meas_dir("s", c, -Y, oks)]
        w = smoothstep((c[2] - sh_z - YOKE[0]) / (YOKE[1] - YOKE[0]))
        if w > 0:
            for j, D in ((0, X), (1, -X)):
                a = meas_dir("f", c, D, ok_front())
                if not (np.isnan(a) or np.isnan(r[j])):
                    r[j] += (a - r[j]) * w
        return r

    def m_head(c, X, Y, u):
        okf, oks = ok_front(ins=inside_head), ok_side_masked(True, ins=inside_head)
        r = [meas_dir("f", c, X, okf), meas_dir("f", c, -X, okf), meas_dir("s", c, Y, oks), meas_dir("s", c, -Y, oks)]
        if u < head.cum[1]:            # 脖子：正面、侧面在这几行会量到斜方肌和后脑的头发
            r = [min(r[0], NECK_RX), min(r[1], NECK_RX), min(r[2], NECK_RY), min(r[3], NECK_RY * 0.9)]
        return r

    def m_arm(side):
        def cond(x, y):
            a = rg.split_x(side, y)
            return a is None or (x < a if side == "R" else x > a)
        okf = ok_front(cond=cond)

        def m(c, X, Y, u):
            a, b = meas_dir("f", c, X, okf), meas_dir("f", c, -X, okf)
            r = (a + b) / 2 if not (np.isnan(a) or np.isnan(b)) else np.nan
            return [a, b, r, r]
        return m

    def m_leg(side):
        okf = ok_front(cond=(lambda x, y: x <= mid) if side == "R" else (lambda x, y: x >= mid))
        oks = ok_side_masked(False)
        return lambda c, X, Y, u: [meas_dir("f", c, X, okf), meas_dir("f", c, -X, okf), meas_dir("s", c, Y, oks), meas_dir("s", c, -Y, oks)]

    def m_foot(half):
        oks = ok_side_masked(True)

        def m(c, X, Y, u):
            v = u / (foot["toe"] - foot["heel"])
            w = half * (0.78 + 0.22 * np.sin(np.pi * min(1, max(0, v))))
            up = meas_dir("s", c, Y, oks)
            return [w, w, min(up, 0.065) if not np.isnan(up) else np.nan, FOOT_ZC]
        return m

    # ---- 各段
    def torso_frames(k, v, u, t):
        w = smoothstep(u / torso.cum[-1])
        X = perp(r_hip + (r_sh - r_hip) * w, t)
        return X, np.cross(t, X), w
    # 躯干顶接到脖子根（trapTop）：上面几圈按正面轮廓一圈比一圈窄，就是斜方肌从脖子斜到肩头的坡
    part("torso", "torso", torso, rg.z_of_row(rows["crotch"]) - J["pel"][2], torso.cum[-1] + rg.z_of_row(rows["trapTop"]) - J["chest"][2],
         torso_frames, m_torso)
    # 头颈：脖子跟着躯干朝向，头跟着脸的朝向（立绘里两者一样，画面里脸会转向对手）
    fw0 = top[1]

    def head_frames(k, v, u, t):
        w = smoothstep((u - (head.cum[1] - 0.05)) / 0.07)
        Y = perp(fw0 * (1 - w) + np.array([0, 1.0, 0]) * w, t)
        return np.cross(Y, t), Y, w
    hq = {"step": STEP_HEAD, "nt": NT_HEAD, "face": SH.Face(rg)} if rg.ref and not model else {}     # 没补头部三视图的角色照旧用超椭圆
    part("head", "head", head, 0.01, head.cum[-1] + rg.z_of_row(rows["hairTop"]) - J["head"][2], head_frames, m_head, **hq)
    glove_w = []
    for s in "RL":
        arm = Chain([J["sh" + s], J["el" + s], J["wr" + s], J["gc" + s], J["gt" + s]])
        # 从肩关节起，往外只接一个半球（藏在躯干的肩头里）；从肩关节往外先接一截整粗的管子，胳膊一转那截管子头就戳出来
        part("arm" + s, "arm", arm, 0.0, arm.cum[-1] + 0.006, "transport", m_arm(s), parent=top)
        us, ext = meas["arm" + s]
        glove_w.append(float(((ext[:, 0] + ext[:, 1]) / 2)[us >= arm.cum[2]].max()))   # 拳套最宽处的一半
        leg = Chain([J["hip" + s], J["kn" + s], J["an" + s]])
        part("leg" + s, "leg", leg, 0.0, leg.cum[-1], "transport", m_leg(s), parent=bot)      # 同胳膊：从胯关节起接半球
        fy, row = rg.pf(J["an" + s]), int(rg.G - 30)          # 脚宽：离地 30 像素那一行（再低就量到台面上的影子）
        run = [x for x in range(int(fy[0]) - 80, int(fy[0]) + 80) if inside(x, row)]
        half = (max(run) - min(run)) / 2 / S
        ft = Chain([[J["an" + s][0], foot["heel"], FOOT_ZC], [J["an" + s][0], foot["toe"], FOOT_ZC]])
        part("foot" + s, "foot", ft, 0.0, ft.cum[-1], lambda k, v, u, t: (lambda U: (np.cross(t, U), U, 0.0))(perp(np.array([0, 0, 1.0]), t)),
             m_foot(half))

    # ---- 骨长和身体尺寸（米）：画面里替换按身体参数估的那一套
    L = lambda a, b: float(np.linalg.norm(J[a] - J[b]))
    mean2 = lambda f: (f("R") + f("L")) / 2
    ua, fa = mean2(lambda s: L("sh" + s, "el" + s)), mean2(lambda s: L("el" + s, "gc" + s))

    def ring_at(name, u):
        us, ext = meas[name]
        return ext[int(np.argmin(np.abs(us - u)))]
    hc = ring_at("head", head.cum[-1])
    head_r = float((hc[0] + hc[1] + hc[2] + hc[3]) / 4)          # 头心那一圈（带头发）的平均半径
    sp = torso.cum[-1]
    rig = {"H": round(body_H(rg.body), 4), "uaFrac": round(ua / (ua + fa), 4), "spine": round(sp, 4),
           "thigh": round(mean2(lambda s: L("hip" + s, "kn" + s)), 4), "shin": round(mean2(lambda s: L("kn" + s, "an" + s)), 4),
           "ankle": round(float((J["anR"][2] + J["anL"][2]) / 2), 4), "headR": round(head_r, 4),
           # 脖子只算顺着脊柱的那段：头往前探的那部分画面里另有前探量，算进脖子长会把头顶高
           "neckL": round(float(np.dot(J["head"] - J["chest"], T0)) - 0.95 * head_r, 4), "shHalf": round(L("shL", "shR") / 2, 4), "hipHalf": round(L("hipL", "hipR") / 2, 4),
           "chestD": round(float(ring_at("torso", 0.62 * sp)[2]), 4), "waistD": round(float(ring_at("torso", 0.45 * sp)[2]), 4),
           "glove": round(sum(glove_w) / 2, 4), "gloveTip": round((dTip["R"] + dTip["L"]) / 2, 4)}
    b = rg.body
    stretch = body_arm(b) / (ua + fa)
    if model:
        r_sh = perp(J["shR"] - J["shL"], T0)
        half = [max(0.05, float(np.dot(J["shR"] - J["chest"], r_sh))), max(0.05, -float(np.dot(J["shL"] - J["chest"], r_sh)))]
        model.bind(bones, {"spine": sp, "half": half})
        mdl, tex, size = model.export()
    else:
        full = np.concatenate([atlas] + [np.pad(h, ((0, H - h.shape[0]), (0, 0), (0, 0))) for h in heads], 1)   # 头的展开图拼在最右边
        tex, size, mdl = encode(full), [full.shape[1], H], None
    out = {"id": rid, "v": 2, "nt": NT, "blend": BLEND, "tex": tex, "size": size, "parts": parts,
           "bind": {"dQ": round(float(dQ), 4), "dWr": dWr, "dTip": dTip, "heel": foot["heel"] - float(J["anR"][1]),
                    "toe": foot["toe"] - float(J["anR"][1]), "zc": FOOT_ZC},
           "rig": rig, "ref": {k: getattr(b, k) for k in ("reach", "mass", "power", "chin", "foot_speed")}}
    if mdl:
        out["model"] = mdl
    path = os.path.join("characters", f"{rid}.skin.json")
    json.dump(out, open(path, "w", encoding="utf-8"), separators=(",", ":"))
    if model:
        cnt = np.bincount(model.label, minlength=len(parts))
        print(f"{path}：全身模型 {mdl['n']} 个顶点、{len(model.tri)} 个三角面，挂在 {sum(len(p['rings']) for p in parts)} 圈截面上，"
              f"{os.path.getsize(path) / 1e6:.2f} MB；模型缩放 {model.scale:.3f}；各段顶点 "
              + "、".join(f"{p['name']} {c}" for p, c in zip(parts, cnt)) + f"；短裤底下看不见的：皮 {model.shrunk} 个顶点往里收，里子删了 {model.dropped} 个")
    else:
        nv = sum(len(p["w"]) // 3 for p in parts)
        print(f"{path}：{len(parts)} 段、{nv} 个顶点，{os.path.getsize(path) / 1e6:.2f} MB；身高 {body_H(b):.3f} 米，立绘 {S:.0f} 像素/米")
    print(f"  骨长：上臂 {ua:.3f}、前臂到拳套 {fa:.3f}，按臂展 {b.reach} 胳膊要 {body_arm(b):.3f} 米，整条胳膊均匀拉长 {100 * (stretch - 1):+.0f}%"
          + ("（超过 15%：立绘和臂展设定差得多，跟画师或数值确认）" if abs(stretch - 1) > 0.15 else ""))
    print(f"  大腿 {rig['thigh']:.3f}、小腿 {rig['shin']:.3f}、躯干 {sp:.3f}、头半径 {head_r:.3f}、肩宽 {2 * rig['shHalf']:.3f}、"
          f"拳套半宽 {rig['glove']:.3f}、拳套中心到拳面 {rig['gloveTip']:.3f}")
    # 关节离地占身高（头顶不算头发：头心往上和到下巴一样远），和真人比一比，标偏了一眼能看出来
    top = 2 * rg.px("head")[1] - rg.px("chin")[1]
    frac = lambda n: (rg.G - rg.px(n)[1]) / (rg.G - top)
    print(f"  关节离地占身高：肩 {frac('shR'):.3f}、胯 {frac('hipR'):.3f}、膝 {frac('knR'):.3f}（真人约 0.82、0.52、0.28，差 0.02 以上回去看标注）")
    print(f"  推出来的：头顶 {rows['hairTop']} 行、躯干顶 {rows['trapTop']} 行、脖子根 {rows['neckBase']} 行、腰头 {rows['waist']} 行、裆 {rows['crotch']} 行、裤脚 {rows['hem']} 行；"
          f"胳膊分界 {json.dumps(rg.split, default=int)}")
    if debug and model:
        os.makedirs(os.path.join(debug, rid), exist_ok=True)
        model.overlay(os.path.join(debug, rid, "model.png"))
        print("  核对图：", os.path.abspath(os.path.join(debug, rid, "model.png")))
    elif debug:
        overlay(rg, parts, side_tex, os.path.join(debug, rid), heads[0] if heads else None)


def normals(V, C):
    dth = np.roll(V, -1, 1) - np.roll(V, 1, 1)
    ds = np.concatenate([V[1:2] - V[0:1], V[2:] - V[:-2], V[-1:] - V[-2:-1]], 0)
    N = np.cross(ds, dth)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    out = V - np.asarray(C)[:, None, :]
    N[(N * out).sum(-1) < 0] *= -1
    return N


def texcoords(rg, P, N, kind):
    """每个顶点在正面、侧面、背面上的贴图坐标（像素，整数）和三张的权重（0~255）。
    躯干、腿、头脚的侧面贴补色图（拼在原图右边），胳膊贴原来的侧面（胳膊本来就在侧面里露着）。"""
    pf = np.array([rg.pf(p) for p in P])
    ps = np.array([rg.ps(p) for p in P])
    pb = np.array([rg.pb(p) for p in P])
    if kind != "arm":
        ps[:, 0] += rg.W - rg.vs[0]
    h = N[:, :2] / (np.linalg.norm(N[:, :2], axis=1, keepdims=True) + 1e-9)
    h[np.linalg.norm(N[:, :2], axis=1) < 0.05] = [0, 1]
    wf, wb, ws = np.maximum(h[:, 1], 0) ** 3, np.maximum(-h[:, 1], 0) ** 3, np.abs(h[:, 0]) ** 3
    w = np.stack([wf, ws, wb], 1) / (wf + wb + ws)[:, None]
    uv = np.concatenate([pf, ps, pb], 1).round().astype(int)
    return uv.reshape(-1).tolist(), (w * 255).round().astype(int).reshape(-1).tolist()


def inpaint_side(rg):
    """侧面补色图：胳膊挡住躯干、大腿的地方，先按同一行里胳膊两边露出来的身体颜色连起来，
    再反复取上下左右四邻的平均抹匀（露出来的身体颜色不动）：胳膊竖着挡的地方横着接上，拳套挡住的短裤上下也接得上，不留一道道横纹。"""
    x0, x1 = rg.vs[0], rg.vs[1]
    im = np.asarray(rg.img).astype(float)[:, x0:x1].copy()
    m = rg.mask[:, x0:x1]
    arm = rg.arm_side[:, x0:x1] & m
    body = m & ~arm
    fill = np.zeros_like(arm)
    w = x1 - x0

    def edge(y, x, step):        # 胳膊边上往外 2~4 像素的身体颜色（贴着轮廓的那一两像素混着灰底）
        xs = [x + step * d for d in range(2, 5) if 0 <= x + step * d < w and body[y, x + step * d]]
        return im[y, xs].mean(0) if xs else None
    for y in np.where(arm.any(1))[0]:
        for s, e in runs_of(arm[y]):
            cl = edge(y, s, -1) if s > 0 and body[y, s - 1] else None
            cr = edge(y, e, 1) if e + 1 < w and body[y, e + 1] else None
            if cl is None and cr is None:
                continue
            cl, cr = (cr, cr) if cl is None else (cl, cl) if cr is None else (cl, cr)
            t = ((np.arange(s, e + 1) - s + 1) / (e - s + 2))[:, None]
            im[y, s:e + 1] = cl * (1 - t) + cr * t
            fill[y, s:e + 1] = True
    # 短裤布面（腰头往下一截到裤脚）：侧面的黑条、金边、褶子都是竖着的，横着接会把红布和黑条搅成一团黑。
    # 按同一列往下（下面没露出来就往上）最近露出来的布料颜色接上，这些点不再参与下面的抹匀
    keep = np.zeros_like(fill)
    fa, fb = rg.rows["waist"] + int(round(0.065 * rg.S)), rg.rows["hem"]
    src = im.copy()
    for x in range(w):
        col = body[fa:fb, x]
        if not col.any():
            continue
        vis = np.where(col)[0] + fa
        for y in np.where(fill[fa:fb, x])[0] + fa:
            below, above = vis[vis > y], vis[vis < y]
            near = below[:24] if len(below) else above[-24:]       # 最近的二十来行布料取平均：一行的噪点、绗缝的菱格不拉成竖线
            im[y, x] = src[near[2:] if len(near) > 4 else near, x].mean(0)
            keep[y, x] = True
    if keep.any():                                             # 横着略微抹一下，竖纹不那么像条形码
        soft = ndi.gaussian_filter1d(im, 2.5, axis=1)
        im[keep] = soft[keep]
    ys, xs = np.where(fill)
    y0, y1, xa, xb = max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3
    sub, f = im[y0:y1, xa:xb], (fill & ~keep)[y0:y1, xa:xb]
    ok = (body | fill)[y0:y1, xa:xb].astype(float)[..., None]
    # 腰头上沿、裤脚两行是分界：皮肤的颜色不往短裤里抹，短裤的颜色也不往皮肤上抹
    zone = np.digitize(np.arange(y0, y1), [rg.rows["waist"], rg.rows["hem"]])[:, None, None] * np.ones_like(ok)
    for _ in range(500):
        acc, cnt = np.zeros_like(sub), np.zeros_like(ok)
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            same = np.roll(ok, sh, ax) * (np.roll(zone, sh, ax) == zone)
            acc += np.roll(sub, sh, ax) * same
            cnt += same
        sub[f] = (acc / np.maximum(cnt, 1))[f]
    return {"img": im.clip(0, 255).astype(np.uint8), "mask": m, "fill": fill}


def atlas_of(rg, side):
    """贴图：原图 + 侧面补色图左右拼起来；轮廓外的灰底换成离它最近的人身上的颜色（采到轮廓边上不带出灰边）。头的展开图最后再拼上。"""
    def dilate(im, mask):
        core = ndi.binary_erosion(mask, iterations=2)
        _, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
        return im[iy, ix]
    return np.concatenate([dilate(np.asarray(rg.img), rg.mask), dilate(side["img"], side["mask"])], 1)


def encode(atlas):
    buf = io.BytesIO()
    Image.fromarray(atlas).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay(rg, parts, side, outdir, head_img):
    """核对图：每隔一圈的截面描回三视图（青 = 正面、黄 = 侧面、紫 = 背面），红点是关节，绿线是推出来的胳膊分界，
    另存侧面补色图（胳膊挡住的地方补成了什么颜色）、头的展开图。"""
    os.makedirs(outdir, exist_ok=True)
    im = rg.img.copy()
    dr = ImageDraw.Draw(im)
    for p in parts:
        if p.get("seam"):              # 头贴的是展开图，描不回立绘
            continue
        uv = np.array(p["uv"]).reshape(-1, 6)
        off = rg.W - rg.vs[0] if p["kind"] != "arm" else 0
        for i in range(0, len(p["rings"]), 2):
            ring = uv[i * NT:(i + 1) * NT].copy()
            ring[:, 2] -= off
            for a, col in ((0, (0, 255, 255)), (2, (255, 255, 0)), (4, (255, 0, 255))):
                pts = [tuple(q) for q in ring[:, a:a + 2]]
                dr.line(pts + [pts[0]], fill=col, width=1)
    for v in rg.J.values():
        for f in (rg.pf, rg.ps):
            x, y = f(v)
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 0, 0), width=2)
    for s in "RL":
        dr.line([tuple(q) for q in rg.split[s]], fill=(0, 255, 0), width=2)
    for x, y, r in rg.side_caps:
        dr.ellipse([x - r, y - r, x + r, y + r], outline=(0, 255, 0), width=1)
    im.save(os.path.join(outdir, "overlay.png"))
    if head_img is not None:
        Image.fromarray(head_img).save(os.path.join(outdir, "head_bake.png"))
    Image.fromarray(side["img"]).save(os.path.join(outdir, "side_inpainted.png"))
    print("  核对图：", os.path.abspath(outdir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--debug")
    a = ap.parse_args()
    for rid in a.ids:
        build(rid, a.debug)
