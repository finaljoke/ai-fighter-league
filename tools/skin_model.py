"""全身模型（AI 图生三维，tools/reduce_model.mjs 减过面）绑到截面骨骼上：build_skin.py 在标注里有 model 时用。规范见 docs/角色皮肤规范.md。

模型是拿同一张三视图立绘生成的，站姿和立绘一样：按轮廓对齐到立绘的坐标（米），每个顶点挂到最近那段身体的两圈截面上
（截面就是骨头：画面里每圈截面怎么摆，挂在上面的顶点就怎么跟着动）。肩、脖子、脚踝这几处两段身体交界，
沿着网格表面往两边各渐变几厘米，弯的时候不裂、不折。胯按位置渐变（见 HIP），短裤底下看不见的几层收进去或删掉（见 HIDE_KEEP）。
"""
import base64
import io
import json
import struct

import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.sparse import csgraph

TEX = 2048              # 贴图边长：原图 4096，脸只占一小块，2048 在画面里看不出差别
AX = 0.02               # 找最近那圈截面：顺着骨头差 2 厘米，和截面上差一个半径一样远
# 两段身体交界处沿表面往两边各渐变多远（米）；没列的两段不混（拳套和大腿）
BAND = {("arm", "torso"): 0.05, ("head", "torso"): 0.04, ("foot", "leg"): 0.03}
# 胯（躯干和两条大腿）按位置渐变，不按表面：短裤里面还有一层大腿的皮（离短裤一厘米），按表面量两层的权重不一样，
# 腿一抬大腿就从短裤里戳出来。顺着大腿从胯关节往上 9 厘米到往下 9 厘米，躯干渐变到腿；中线左右 3 厘米，两条腿互相渐变
HIP = (0.09, 0.03)
# 短裤底下看不见的那几层：短裤里子（布的反面，离布面一两毫米，贴图是皮色）、大腿和裆里的皮（再往里一两厘米）。
# 大腿一抬胯前挤成一团、两腿一分裆部拉开，蒙皮混出来的位置一差就互相穿过去，皮色从侧缝、胯前、裆部露出来。
# 腰头到裤脚之间，从四周 HIDE_VIEWS 这些方向都看不见的点：里子（法线朝着身体中心）删掉，折缝里透出来的是布的反面；
# 皮不删，往中心收 HIDE_SHRINK 米——倒地时镜头从脚那头往裤腿里看得进去，删了大腿就在裤腿里断了。
# 离看得见的地方沿表面 HIDE_KEEP 米以内的里子不删；皮离看得见的地方越远收得越多，HIDE_FADE 米收满
# （渐变短了，模型里看不见的地方三角形大，相邻两个点一个收满一个没收，往裤腿里看大腿一截一截的）
HIDE_VIEWS = [(el, az) for el in (-10, 15, 40) for az in range(0, 360, 30)]        # （仰角、方位角）度
HIDE_KEEP = 0.03
HIDE_FADE = 0.10
HIDE_SHRINK = 0.04
EASE = lambda t: np.where(t < 0, 0, np.where(t > 1, 1, t * t * (3 - 2 * t)))      # 同 viz.js 的 ease


def read_glb(path):
    """一个网格的 GLB → 顶点、法线、贴图坐标（左上角为原点）、三角形、底色贴图。"""
    data = open(path, "rb").read()
    off, js, binary = 12, None, None
    while off < len(data):
        ln, typ = struct.unpack_from("<II", data, off)
        chunk = data[off + 8:off + 8 + ln]
        if typ == 0x4E4F534A:
            js = json.loads(chunk)
        elif typ == 0x004E4942:
            binary = chunk
        off += 8 + ln
    if js is None or binary is None:
        raise SystemExit(f"{path} 不是 GLB")

    def view(i):
        v = js["bufferViews"][i]
        return binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]

    def acc(i, dt, comps):
        a = js["accessors"][i]
        b = view(a["bufferView"])[a.get("byteOffset", 0):]
        return np.frombuffer(b, dt, a["count"] * comps).reshape(a["count"], comps) if comps > 1 else np.frombuffer(b, dt, a["count"])
    pr = js["meshes"][0]["primitives"][0]
    ia = js["accessors"][pr["indices"]]
    idx = acc(pr["indices"], np.uint32 if ia["componentType"] == 5125 else np.uint16, 1).astype(np.int64).reshape(-1, 3)
    tex = js["materials"][pr["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    img = Image.open(io.BytesIO(view(js["images"][js["textures"][tex]["source"]]["bufferView"]))).convert("RGB")
    at = pr["attributes"]
    return acc(at["POSITION"], np.float32, 3).astype(float), acc(at["NORMAL"], np.float32, 3).astype(float), \
        acc(at["TEXCOORD_0"], np.float32, 2).astype(float), idx, img


def seen_from(P, N, tri, test, px=0.003, eps=0.003):
    """test 里的点从 HIDE_VIEWS 各个方向正交着看，有没有哪个方向看得见：朝着镜头，前面没被 tri 挡住（逐像素深度，px 米一个像素）。"""
    seen = np.zeros(len(test), bool)
    used = np.unique(tri)
    for el, az in HIDE_VIEWS:
        e, a = np.radians(el), np.radians(az)
        d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])          # 视线方向
        u = np.array([-np.sin(a), np.cos(a), 0.0])
        x, y, z = P @ u / px, P @ np.cross(u, d) / px, P @ d
        x, y = x - x[used].min() + 2, y - y[used].min() + 2
        W, H = int(x[used].max()) + 3, int(y[used].max()) + 3
        zb = np.full(W * H, np.inf)
        tx, ty, tz = x[tri], y[tri], z[tri]
        x0, y0 = np.floor(tx.min(1)).astype(int), np.floor(ty.min(1)).astype(int)
        size = np.maximum(np.ceil(tx.max(1)) - x0, np.ceil(ty.max(1)) - y0).astype(int) + 1
        lo = 0
        for k in (3, 5, 8, 12, 20, 34, 60, 120, 400):         # 按三角形在画面上占多大分批，一批一起算盖住哪些像素
            i = np.where((size > lo) & (size <= k))[0]
            lo = k
            if not len(i):
                continue
            gx, gy = np.meshgrid(np.arange(k), np.arange(k))
            qx, qy = x0[i, None] + gx.ravel() + 0.5, y0[i, None] + gy.ravel() + 0.5
            (ax, bx, cx), (ay, by, cy) = tx[i].T[..., None], ty[i].T[..., None]
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            den = np.where(np.abs(den) < 1e-12, 1e-12, den)
            w0 = ((by - cy) * (qx - cx) + (cx - bx) * (qy - cy)) / den
            w1 = ((cy - ay) * (qx - cx) + (ax - cx) * (qy - cy)) / den
            ok = (w0 >= -1e-6) & (w1 >= -1e-6) & (w0 + w1 <= 1 + 1e-6) & (qx < W) & (qy < H)
            zz = w0 * tz[i, 0, None] + w1 * tz[i, 1, None] + (1 - w0 - w1) * tz[i, 2, None]
            np.minimum.at(zb, (qy.astype(int) * W + qx.astype(int))[ok], zz[ok])
        zb = np.pad(zb.reshape(H, W), 1, constant_values=np.inf)
        xi, yi = np.clip(x[test].astype(int), 0, W - 1), np.clip(y[test].astype(int), 0, H - 1)
        back = np.max([zb[yi + dy, xi + dx] for dy in range(3) for dx in range(3)], 0)       # 周围 3×3 取最远的：贴着轮廓边不算被挡
        seen |= (z[test] <= back + eps) & (N[test] @ d < -0.05)
    return seen


class Model:
    def __init__(self, rg, path):
        self.rg = rg
        P, N, self.uv, self.tri, self.img = read_glb(path)
        # 模型 y 朝上、脸朝 +z，角色的右手在 -x；立绘的坐标是 [右, 前, 上]（米）
        q = np.stack([-P[:, 0], P[:, 2], P[:, 1]], 1)
        self.N = np.stack([-N[:, 0], N[:, 2], N[:, 1]], 1)
        self.N /= np.linalg.norm(self.N, axis=1, keepdims=True) + 1e-12
        self.P = self.align(q)

    def align(self, q):
        """按轮廓的外框对齐：头顶到脚底的高度定缩放（三个方向一样），正面外框的左右中点、侧面外框的前后中点定平移。"""
        rg, M = self.rg, self.rg.mask
        fr, sd = M[:, rg.vf[0]:rg.vf[1]], M[:, rg.vs[0]:rg.vs[1]]
        top = int(np.where(fr.any(1))[0].min())
        cf = np.where(fr.any(0))[0][[0, -1]].mean() + rg.vf[0]
        cs = np.where(sd.any(0))[0][[0, -1]].mean() + rg.vs[0]
        lo, hi = q.min(0), q.max(0)
        s = (rg.G - top) / rg.S / (hi[2] - lo[2])
        fit = self.rg.R["model"].get("fit")             # 对不上时在标注里写 [缩放, 右移, 前移, 上移]（米）微调
        P = (q - (lo + hi) / 2 * [1, 1, 0] - [0, 0, lo[2]]) * s + [(rg.vf[2] - cf) / rg.S, (rg.vs[2] - cs) / rg.S, 0]
        if fit:
            P = P * fit[0] + fit[1:]
        self.scale = s
        return P

    def fit_arms(self, J):
        """胳膊的关节按模型重定（就地改 J）：AI 生成时胳膊垂得比立绘的侧面直，拳套往后差了 7~12 厘米，照立绘的关节绑，
        拳套网格会偏离画面里拳套该在的位置（打中时拳面不贴落点）。肘、腕取同一高度胳膊截面的中心，拳套中心取腕以下那一坨的外框中心，
        拳面取它的最低点。正面左右（x）本来就对得上，只改前后；拳套中心、拳面连高度一起改（模型的拳套比立绘的小一圈）。"""
        P, out = self.P, []
        for s in "RL":
            for n in ("el", "wr"):
                j = J[n + s]
                sel = (np.abs(P[:, 2] - j[2]) < 0.012) & (np.abs(P[:, 0] - j[0]) < 0.05)
                y = (P[sel, 1].min() + P[sel, 1].max()) / 2
                out.append(f"{n}{s} 前后 {100 * (y - j[1]):+.1f}")
                j[1] = y
            g = (P[:, 2] < J["wr" + s][2]) & (P[:, 2] > J["gt" + s][2] - 0.06) & (np.abs(P[:, 0] - J["gc" + s][0]) < 0.09)   # 再往下是靴子
            c = (P[g].min(0) + P[g].max(0)) / 2
            out.append(f"gc{s} 前后 {100 * (c[1] - J['gc' + s][1]):+.1f} 上下 {100 * (c[2] - J['gc' + s][2]):+.1f}")
            J["gc" + s] = np.array([J["gc" + s][0], c[1], c[2]])
            z0 = P[g, 2].min()
            bot = g & (P[:, 2] < z0 + 0.015)
            t = np.array([J["gt" + s][0], P[bot, 1].mean(), z0])
            out.append(f"gt{s} 前后 {100 * (t[1] - J['gt' + s][1]):+.1f} 上下 {100 * (t[2] - J['gt' + s][2]):+.1f}")
            J["gt" + s] = t
        print("  胳膊关节按模型重定（厘米）：" + "、".join(out))

    # ---- 绑骨骼
    def bind(self, frames, lift):
        """frames：每段身体 {name, kind, C, X, Y, T, ext, kv}（立绘站姿下每圈截面的中心、标架、四个方向的半径、(k, v)），按皮肤里的顺序。
        每个顶点挂两段身体（主、混），每段挂相邻两圈：一共 4 个骨头和权重。"""
        P, n = self.P, len(self.P)
        names = [f["name"] for f in frames]
        base = np.cumsum([0] + [len(f["C"]) for f in frames])
        best = np.full((len(frames), n), np.inf)
        ring, nb = np.zeros((len(frames), n), int), np.zeros((len(frames), n), int)     # 最近那圈、顺着骨头往顶点那边的下一圈
        alpha = np.zeros((len(frames), n))                                                # 下一圈占多少
        for p, f in enumerate(frames):
            C, X, Y, T, e = f["C"], f["X"], f["Y"], f["T"], f["ext"]
            ox, rx = (e[:, 0] - e[:, 1]) / 2, (e[:, 0] + e[:, 1]) / 2
            oy, ry = (e[:, 2] - e[:, 3]) / 2, (e[:, 2] + e[:, 3]) / 2
            for a0 in range(0, n, 8000):
                sl = slice(a0, a0 + 8000)
                D = P[sl, None, :] - C[None]
                a, b, s = (D * X[None]).sum(-1), (D * Y[None]).sum(-1), (D * T[None]).sum(-1)
                q = ((a - ox) / rx) ** 2 + ((b - oy) / ry) ** 2 + (s / AX) ** 2
                i = q.argmin(1)
                rows = np.arange(len(i))
                si = s[rows, i]
                j = np.clip(i + np.where(si > 0, 1, -1), 0, len(C) - 1)      # 贴着首尾那圈往外就是自己
                sj = s[rows, j]
                best[p, sl], ring[p, sl], nb[p, sl] = q[rows, i], i, j
                alpha[p, sl] = np.where((j != i) & (si * sj < 0), si / (si - sj + 1e-12), 0.0)
        label = best.argmin(0)
        kinds = [f["kind"] for f in frames]

        # 交界处沿表面渐变：位置相同的顶点（贴图接缝两边）先并成一个，按边长建图，量每段身体沿表面到这个点多远
        _, weld = np.unique(np.round(P / 1e-6).astype(np.int64), axis=0, return_inverse=True)
        weld = weld.reshape(-1)
        nw = weld.max() + 1
        lab_w = np.zeros(nw, int)
        lab_w[weld] = label
        e = np.concatenate([self.tri[:, [0, 1]], self.tri[:, [1, 2]], self.tri[:, [2, 0]]])
        L = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1) + 1e-9
        ew = weld[e]
        G = sparse.coo_matrix((L, (ew[:, 0], ew[:, 1])), shape=(nw, nw)).tocsr()
        G = G.maximum(G.T)
        far = max(BAND.values())
        dist = np.full((len(frames), nw), np.inf)
        for p in range(len(frames)):
            src = np.where(lab_w == p)[0]
            if len(src):
                dist[p] = csgraph.dijkstra(G, directed=False, indices=src, min_only=True, limit=far)
        w2 = np.zeros(nw)
        sec = np.full(nw, -1)
        for p in range(len(frames)):
            for qk in set(kinds):
                b = BAND.get(tuple(sorted((kinds[p], qk))), 0)
                if not b:
                    continue
                m = (np.array(kinds)[lab_w] == qk) & (lab_w != p)
                w = 0.5 * (1 - EASE(dist[p] / b))
                upd = m & (w > w2)
                w2[upd], sec[upd] = w[upd], p
        w2, sec = w2[weld], sec[weld]
        rows = np.arange(n)
        Wp = np.zeros((n, len(frames)))                    # 每个点挂在各段身体上的权重
        Wp[rows, label] = 1 - w2
        Wp[rows[sec >= 0], sec[sec >= 0]] += w2[sec >= 0]

        # 胯：躯干、两条腿上的点按位置重新分（HIP）。顺着各自大腿量离胯关节多远，中线两边按左右分
        t, lR, lL = names.index("torso"), names.index("legR"), names.index("legL")
        J, (a, b) = self.rg.J, HIP
        hipz = np.isin(label, [t, lR, lL])
        down = lambda s: EASE(((P - J["hip" + s]) @ (J["kn" + s] - J["hip" + s]) / np.linalg.norm(J["kn" + s] - J["hip" + s]) + a) / (2 * a))
        side = EASE((P[:, 0] + b) / (2 * b))              # 角色的右边是 +x
        wR, wL = side * down("R"), (1 - side) * down("L")
        Wp[hipz] = 0
        Wp[hipz, lR], Wp[hipz, lL], Wp[hipz, t] = wR[hipz], wL[hipz], 1 - wR[hipz] - wL[hipz]

        # 每段身体挂相邻两圈，按权重取最大的 4 个
        cand_b = np.concatenate([base[None, :-1] + ring.T, base[None, :-1] + nb.T], 1)      # (n, 2 × 段数)
        cand_w = np.concatenate([Wp * (1 - alpha.T), Wp * alpha.T], 1)
        top = np.argsort(-cand_w, 1)[:, :4]
        bone, wt = np.take_along_axis(cand_b, top, 1), np.take_along_axis(cand_w, top, 1)
        wt /= wt.sum(1, keepdims=True)
        self.label, self.bone, self.wt = label, bone, wt
        self.names, self.base = names, base

        # 躯干上部三成跟着两肩的高低斜过去（同 viz.js posePart 里的 lift）：每个点记 系数 × 哪一侧，画面里乘这一帧肩抬了多高
        f = frames[t]
        hf = np.array([0.0 if k < 0 else v if k == 0 else 1 + v / lift["spine"] for k, v in f["kv"]])
        wl = EASE((hf - 0.7) / 0.3)
        wt_torso = Wp[:, t]
        i, j, al = ring[t], nb[t], alpha[t]
        a = ((P - f["C"][i]) * f["X"][i]).sum(1)
        half = np.where(a >= 0, lift["half"][0], lift["half"][1])
        self.lift = np.sign(a) * wt_torso * ((1 - al) * wl[i] + al * wl[j]) * np.minimum(1.3, np.abs(a) / half)
        self.shrunk, self.dropped = self.hide_inner(hipz & np.isin(label, [t, lR, lL]), label, (t, lR, lL), G, weld)

    def hide_inner(self, zone, label, parts, G, weld):
        """短裤底下看不见的那几层（HIDE_SHRINK）：里子删掉，皮往中心收，返回收了几个、删了几个顶点。中心线：躯干上的点是骨盆到两肩中点，
        腿上的点是胯关节到膝盖。躯干和两条腿一起挡着看：立绘站姿里大腿内侧被另一条腿挡住的地方，离看得见的地方（两腿之间露出的那一截）
        沿表面很近，不动。"""
        P, N, rg, J = self.P, self.N, self.rg, self.rg.J
        zlo, zhi = rg.z_of_row(rg.rows["hem"]) - 0.04, rg.z_of_row(rg.rows["waist"]) + 0.04
        occ = zone & (P[:, 2] > zlo - 0.25) & (P[:, 2] < zhi + 0.25)
        test = np.where(zone & (P[:, 2] > zlo) & (P[:, 2] < zhi))[0]
        ok = seen_from(P, N, self.tri[occ[self.tri].all(1)], test)
        hid = test[~ok]
        dist = csgraph.dijkstra(G, directed=False, indices=np.unique(weld[test[ok]]), min_only=True, limit=HIDE_FADE)[weld[hid]]
        fade = EASE(np.minimum(dist, HIDE_FADE) / HIDE_FADE)[:, None]          # 超出 limit 的距离是 inf
        r = np.zeros((len(hid), 3))                        # 从中心线指到这个点
        for p, (a, b) in zip(parts, ((J["pel"], J["chest"]), (J["hipR"], J["knR"]), (J["hipL"], J["knL"]))):
            m = label[hid] == p
            u = (b - a) / np.linalg.norm(b - a)
            r[m] = (P[hid[m]] - a) - ((P[hid[m]] - a) @ u)[:, None] * u
        rn = np.linalg.norm(r, axis=1, keepdims=True) + 1e-9
        lining = (N[hid] * r).sum(1) < 0                   # 法线朝着中心：布的反面
        skin = ~lining
        P[hid[skin]] -= (r / rn * np.minimum(HIDE_SHRINK * fade, rn / 2))[skin]
        hide = np.zeros(len(P), bool)
        hide[hid[lining & (dist >= HIDE_KEEP)]] = True
        tri = self.tri[~hide[self.tri].all(1)]
        keep = np.zeros(len(P), bool)
        keep[tri] = True
        self.tri = (np.cumsum(keep) - 1)[tri]
        for a in ("P", "N", "uv", "label", "bone", "wt", "lift"):
            setattr(self, a, getattr(self, a)[keep])
        return int(skin.sum()), int((~keep).sum())

    # ---- 写进皮肤
    def export(self):
        n = len(self.P)
        head = self.names.index("head")
        tri_head = (self.label[self.tri] == head).sum(1) >= 2          # 头的三角形放前面：挨重拳闪白只白头
        tri = np.concatenate([self.tri[tri_head], self.tri[~tri_head]])
        lo, hi = self.P.min(0), self.P.max(0)
        pos = np.round((self.P - lo) / (hi - lo) * 65535).astype("<u2")
        nor = np.round(self.N * 127).astype("i1")
        uv = np.round(np.clip(self.uv, 0, 1) * 65535).astype("<u2")
        w = np.round(self.wt * 255).astype(int)
        w[np.arange(n), w.argmax(1)] += 255 - w.sum(1)                  # 四个权重加起来正好 255
        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
        img = self.img.resize((TEX, TEX), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
        tex = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        model = {"n": n, "lo": [round(float(x), 5) for x in lo], "hi": [round(float(x), 5) for x in hi],
                 "pos": b64(pos), "nor": b64(nor), "uv": b64(uv), "bone": b64(self.bone.astype("<u2")), "wt": b64(w.astype("u1")),
                 "lift": b64(np.round(np.clip(self.lift / 1.3, -1, 1) * 127).astype("i1")), "idx": b64(tri.astype("<u4")),
                 "head": int(tri_head.sum() * 3)}
        return model, tex, [TEX, TEX]

    # ---- 核对图
    def overlay(self, path):
        """模型按各段身体上色，投到立绘的正面、侧面上（半透明）：轮廓对不对得上、每段身体分得对不对。"""
        rg = self.rg
        cols = {"torso": (230, 80, 60), "head": (240, 200, 40), "armR": (60, 160, 240), "armL": (40, 220, 200),
                "legR": (170, 90, 230), "legL": (240, 110, 200), "footR": (120, 200, 60), "footL": (60, 120, 40)}
        im = rg.img.copy()
        lay = Image.new("RGB", im.size, (0, 0, 0))
        mk = Image.new("L", im.size, 0)
        dr, dm = ImageDraw.Draw(lay), ImageDraw.Draw(mk)
        lab = self.label
        tri = self.tri
        c3 = self.P[tri].mean(1)
        for proj, depth in ((rg.pf, c3[:, 1]), (rg.ps, -c3[:, 0])):     # 正面从前往后看、侧面从角色左边看
            pts = np.array([proj(p) for p in self.P])
            for t in np.argsort(depth):                     # 远的先画，近的盖上去
                part = self.names[np.bincount(lab[tri[t]]).argmax()]
                poly = [tuple(pts[v]) for v in tri[t]]
                dr.polygon(poly, fill=cols[part])
                dm.polygon(poly, fill=150)
        out = Image.composite(lay, im, mk)
        do = ImageDraw.Draw(out)
        for v in rg.J.values():                            # 关节（胳膊是按模型重定过的）
            for proj in (rg.pf, rg.ps):
                x, y = proj(v)
                do.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(0, 0, 0), fill=(255, 255, 255), width=2)
        do.text((10, 10), "model parts: torso red, head yellow, arms blue/cyan, legs purple/pink, feet green; dots = joints", fill=(0, 0, 0))
        out.save(path)
