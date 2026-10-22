"""头：补画的头部三视图 + 模板脸 + 烘焙成一张展开的贴图（build_skin.py 用）。规范见 docs/角色皮肤规范.md 的"头"。

身体别的段是一圈圈超椭圆，头不行：超椭圆是圆的，脸前面没有平面、鼻子立不起来，正面立绘贴上去一转到 3/4 就被拉宽。
老游戏拿照片做脸的办法（通用头模按特征点对齐、正侧两张照片烘焙进一张贴图）照搬过来：

素材：全身立绘里头只有一百来像素，另补头部的正面、左侧面、右侧面三张（rig.json 的 headRef；脸左右的疤不对称，两边侧面各用各的）。
      正面按两眼间距缩放到立绘的大小，侧面按眼睛到下巴的高度对齐正面，都是下巴对齐立绘的下巴（脖子长短不变）。补画的脸比立绘长，多出来的往头顶长。
      先缩小贴回全身立绘的头上（量轮廓用），烘贴图时直接从补画原图取色（清楚）。
形状：每圈截面的外轮廓按轮廓量（正面量左右宽、侧面量前后），前半边往前收窄成蛋形，前面换成模板脸——
      眉眼那一带前面压平（嘴、下巴往前收成 U 形），正中按侧面轮廓（鼻梁、鼻尖、嘴、下巴）往前凸，
      凸出去的宽度按模板（鼻子窄、别处宽）；眼窝按侧面图上眼睛的深度往里凹（真人的眼睛在鼻梁后面三四厘米，脸是有进深的）。
贴图：每个顶点都在一张展开图上有固定位置（绕头一圈是横向，从脖子到头顶是纵向，接缝在后脑正中），
      展开图离线烘焙：脸前面取正面，侧面取侧面，两张只在一窄条里渐变；补画的颜色按脸上一块皮肤调到立绘的肤色，
      脖子往下渐变回全身立绘（接上躯干不出色差）。
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

BAKE_W, BAKE_H = 1024, 512     # 展开图大小（像素）：头一圈约 0.6 米，补画的头约 1100 像素/米，烘到 1700 像素/米不亏细节
PAD = 8                        # 展开图左右各多烘几列（绕过接缝接着烘），缩小采样时不混进贴图里旁边的侧面图
SEAM = (0.4, 0.25)             # 正面、侧面贴图的交界：法线朝前的分量在 0.4~0.65 之间渐变（约 49°~66°）
EYE_SOCKET = (0.013, 0.010)    # 眼窝往外落下去的左右、上下宽度（米）；深多少按侧面图上眼睛的位置定
TAPER = 0.5                    # 头从上往下看是蛋形：前半边越往前越窄，到脸正前方左右收掉这么多（比例）。不收就是方头，眼睛深度那儿还是宽宽的侧面，侧面图上的眼睛就落到那上面


def smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def sample(img, x, y):
    """img (H, W, 3) 上按浮点像素坐标双线性取色。"""
    return np.stack([ndi.map_coordinates(img[..., k], [y, x], order=1, mode="nearest") for k in range(3)], -1)


def matte(im):
    """补画的灰底抠人：底色比全身立绘暗，立绘那套按亮度挑背景样本挑不出来。这里拿图的上边、左右两边（上六成）当背景样本，
    拟合渐变，离背景远或颜色鲜艳的算人，再去掉小块、补上小洞。"""
    H, W, _ = im.shape
    Y, X = np.mgrid[0:H, 0:W]
    edge = (Y < 30) | (((X < 30) | (X >= W - 30)) & (Y < 0.6 * H))
    edge &= (im.max(-1) - im.min(-1)) < 20
    A = lambda y, x: np.stack([np.ones_like(y, dtype=float), y / H, x / W, (x / W) ** 2, (y / H) ** 2], -1)
    coef = [np.linalg.lstsq(A(Y[edge], X[edge]), im[edge][:, c], rcond=None)[0] for c in range(3)]
    bg = np.stack([A(Y, X) @ coef[c] for c in range(3)], -1)
    fg = ndi.binary_opening((np.sqrt(((im - bg) ** 2).sum(-1)) > 18) | (im.max(-1) - im.min(-1) > 30), iterations=1)
    lab, n = ndi.label(fg)
    keep = np.isin(lab, 1 + np.where(ndi.sum(fg, lab, range(1, n + 1)) > 20000)[0])
    hl, hn = ndi.label(~keep)
    return ndi.binary_closing(keep | np.isin(hl, 1 + np.where(ndi.sum(~keep, hl, range(1, hn + 1)) < 1500)[0]), iterations=2)


class HeadRef:
    """补画的头部三视图（rig.json 的 headRef）：f 正面，l 左侧面（脸朝左，看到的是他的左脸，同立绘的侧面），
    r 右侧面（脸朝右，看到右脸；flip 了以后也是脸朝左，坐标按翻过来的图标）。三张可以分开画，大小不必一样：
    正面按两眼间距缩放到立绘，侧面按"眼睛到下巴的高度"对齐正面，都是下巴对齐立绘的下巴（脖子长短不变）。"""

    def __init__(self, rg):
        H = rg.R["headRef"]
        self.rg, self.H = rg, H
        self.img, self.mask, self.k, self.anchor, self.gain = {}, {}, {}, {}, {}
        for v in "flr":
            L = H[v]
            im = np.asarray(Image.open(L["image"]).convert("RGB")).astype(float)
            if L.get("flip"):
                im = im[:, ::-1]
            m = matte(im)
            core = ndi.binary_erosion(m, iterations=3)
            _, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
            self.img[v], self.mask[v] = im[iy, ix], m     # 轮廓外抹成离它最近的人身上的颜色，采到边上不带灰边
            self.gain[v] = np.ones(3)
        F, f = rg.R["face"], H["f"]
        self.k["f"] = abs(F["eyeR"]["f"][0] - F["eyeL"]["f"][0]) / abs(f["eyeR"][0] - f["eyeL"][0])
        eye_chin = f["chin"][1] - (f["eyeR"][1] + f["eyeL"][1]) / 2
        for v in "lr":
            self.k[v] = self.k["f"] * eye_chin / (H[v]["chin"][1] - H[v]["eye"][1])
        self.anchor["f"] = (np.array(rg.px("chin", "f"), float), np.array(f["chin"], float))
        for v in "lr":
            self.anchor[v] = (np.array(rg.px("chin", "s"), float), np.array(H[v]["chin"], float))

    def to_ref(self, v, x, y):
        a, b = self.anchor[v]
        return b[0] + (x - a[0]) / self.k[v], b[1] + (y - a[1]) / self.k[v]

    def to_rig(self, v, p):
        a, b = self.anchor[v]
        return a + (np.asarray(p, float) - b) * self.k[v]

    def look(self, v, x, y):
        """立绘坐标上的点（f 用正面视图的坐标，l、r 用侧面视图的坐标），去补画上取色（已调到立绘的肤色）。"""
        return sample(self.img[v], *self.to_ref(v, x, y)) * self.gain[v]

    def paste(self):
        """正面、左侧面缩小贴回全身立绘的头上，返回贴好的轮廓：量头那一段截面、推头顶那一行按它（躯干等别的还按原立绘）。
        贴到下巴往下 12 像素，再往下 15 像素渐变回原图。先各比一块脸颊上的皮肤，算出每张的调色。贴好的图存在 self.pasted（核对用）。"""
        rg = self.rg
        arr = np.asarray(rg.img).astype(float)
        mask = rg.mask.copy()
        cy = rg.px("chin", "f")[1]
        # 调色：比脖子（头要接的是脖子和躯干）。正面取下巴底下的喉咙，侧面取下颌后面的颈侧；左右侧面都对立绘的侧面（立绘只画了左脸）
        (fx, _), (sx, sy) = rg.px("chin", "f"), rg.px("chin", "s")
        patches = {"f": np.mgrid[cy + 6:cy + 18, fx - 8:fx + 9], "l": np.mgrid[sy + 6:sy + 24, sx + 38:sx + 52]}
        patches["r"] = patches["l"]
        for v, (ys, xs) in patches.items():
            ys, xs = ys.ravel().astype(float), xs.ravel().astype(float)
            ref = sample(self.img[v], *self.to_ref(v, xs, ys)).mean(0)
            self.gain[v] = np.clip(arr[ys.astype(int), xs.astype(int)].mean(0) / ref, 0.7, 1.4)
        bg = arr[2:12, 2:12].reshape(-1, 3).mean(0)          # 立绘的灰底：补画上人以外的地方贴这个
        for v, (x0, x1) in (("f", rg.vf[:2]), ("l", rg.vs[:2])):
            Y, X = np.mgrid[0:cy + 27, x0:x1].astype(float)
            w = (1 - smooth01((Y - cy - 12) / 15))[..., None]
            u, t = self.to_ref(v, X, Y)
            m = ndi.map_coordinates(self.mask[v].astype(float), [t, u], order=1, mode="constant") > 0.5
            col = np.where(m[..., None], self.look(v, X, Y), bg)
            arr[0:cy + 27, x0:x1] = w * col + (1 - w) * arr[0:cy + 27, x0:x1]
            mask[0:cy + 27, x0:x1] = np.where(w[..., 0] > 0.5, m, mask[0:cy + 27, x0:x1])
        self.pasted = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return mask


class Face:
    """脸部特征点（从补画上标的换到立绘坐标，再换成米）。"""

    def __init__(self, rg):
        self.rg, ref = rg, rg.ref
        f, s = rg.R["headRef"]["f"], rg.R["headRef"]["l"]
        eR, eL = ref.to_rig("f", f["eyeR"]), ref.to_rig("f", f["eyeL"])
        self.eye_z = (rg.z_of_row(eR[1]) + rg.z_of_row(eL[1])) / 2
        self.eye_x = abs(eR[0] - eL[0]) / 2 / rg.S
        self.nose_z = rg.z_of_row(ref.to_rig("f", f["nose"])[1])
        self.eye_y = (rg.vs[2] - ref.to_rig("l", s["eye"])[0]) / rg.S          # 眼睛在侧面上往前多少（世界坐标 Y）
        self.chin_z = rg.J["chin"][2]
        self.brow_z = self.eye_z + 0.012

    def zone(self, z):
        """脸的范围：下巴底到眉骨，往外 1.2 厘米渐变。再往下是脖子，再往上侧面轮廓量到的是刘海。"""
        return smooth01((z - (self.chin_z - 0.012)) / 0.012) * smooth01((self.brow_z + 0.012 - z) / 0.012)

    def ridge(self, z):
        """正中凸起（鼻、嘴、下巴）往两边落下去的宽度（米）：鼻子那几圈窄，别处宽。"""
        wn = smooth01((z - (self.nose_z - 0.008)) / 0.006) * smooth01((self.eye_z - 0.004 - z) / 0.006)
        return 0.018 - 0.007 * wn

    def head(self, z):
        """下巴往上才是头（前半边收成蛋形），往下是脖子。"""
        return smooth01((z - (self.chin_z - 0.015)) / 0.015)

    def flat(self, z):
        """脸前面有多平：眉眼那一带宽而平，嘴、下巴往前收窄（从上往下看是 U 形，压平了就是一张方面具）。"""
        return 0.4 * np.exp(-0.5 * ((z - self.eye_z) / 0.015) ** 2)


def shape(face, C, Fy, ext, raw, nt, step):
    """每圈截面上每个点在截面标架里的位置 (n, nt, 2)：a 顺着 X（人的右边），b 顺着 Y（脸朝向）。
    ext：滤过的四向半径（+X、-X、+Y、-Y）；raw：没滤过的，raw[:, 2] 就是侧面轮廓（鼻梁、鼻尖、嘴、下巴）。
    脸平面取侧面轮廓的下包络（2.5 厘米里最靠后的再抹平）：正中一条照轮廓走，鼻、嘴、下巴凸出包络的部分按模板宽度往两边落下去。"""
    n = len(C)
    th = 1.5 * np.pi + 2 * np.pi * np.arange(nt) / nt          # 第一列在后脑正中（展开图的接缝藏在头发里），往人的右边转到脸
    c, s = np.cos(th), np.sin(th)
    prof = ndi.gaussian_filter1d(raw[:, 2], 0.7, mode="nearest")
    plane = ndi.gaussian_filter1d(ndi.minimum_filter1d(prof, max(3, int(round(0.025 / step))), mode="nearest"), 0.012 / step, mode="nearest")
    plane = np.minimum(plane, prof)
    ax, az = EYE_SOCKET
    loc = np.zeros((n, nt, 2))
    for i in range(n):
        z = C[i][2]
        w = face.zone(z)
        front = w * plane[i] + (1 - w) * ext[i, 2]
        e = np.where(s > 0, 2.2 + face.flat(z) * w, 2.2)
        sx = np.sign(c) * np.abs(c) ** (2 / e)
        sy = np.sign(s) * np.abs(s) ** (2 / e)
        a = sx * np.where(c >= 0, ext[i, 0], ext[i, 1]) * (1 - TAPER * face.head(z) * np.maximum(sy, 0) ** 2)
        b = sy * np.where(s > 0, front, ext[i, 3])
        if w > 0:
            b = b + w * np.where(s > 0, (prof[i] - plane[i]) * np.exp(-0.5 * (a / face.ridge(z)) ** 2) * s, 0)
            # 眼窝：眼睛那一点按侧面图该在的深度，现在的面比它靠前多少就往回凹多少，往四周落下去
            gz = np.exp(-0.5 * ((z - face.eye_z) / az) ** 2)
            if gz > 0.01:
                want = (face.eye_y - C[i][1]) * Fy[i][1]
                for side in (1, -1):
                    m = (s > 0) & (np.sign(a) == side)
                    order = np.argsort(np.abs(a[m]))
                    now = np.interp(face.eye_x, np.abs(a[m])[order], b[m][order])
                    dig = max(0.0, now - want) * gz * w
                    b = b - np.where(m, dig * np.exp(-0.5 * ((np.abs(a) - face.eye_x) / ax) ** 2), 0)
        loc[i, :, 0], loc[i, :, 1] = a, b
    return loc


def bake(face, V, N, atlas):
    """烘焙展开图。V、N：(n, nt) 的顶点和法线（立绘姿势、世界坐标）。atlas：全身立绘拼好的贴图（背面、脖子往下用它）。
    返回 (展开图 (BAKE_H, BAKE_W + 2 * PAD, 3), 每个顶点（含接缝多出来的那一列）在展开图上的像素坐标 (n, nt + 1, 2))。"""
    rg, ref = face.rg, face.rg.ref
    n, nt = V.shape[:2]
    Vc, Nc = np.concatenate([V, V[:, :1]], 1), np.concatenate([N, N[:, :1]], 1)
    gy, gx = np.mgrid[0:BAKE_H, 0:BAKE_W + 2 * PAD].astype(float)
    fi, fj = gy / (BAKE_H - 1) * (n - 1), np.mod((gx - PAD) / (BAKE_W - 1) * nt, nt)   # 展开图上每个像素对应第几圈、第几列（浮点）
    P = np.stack([ndi.map_coordinates(Vc[..., k], [fi, fj], order=1) for k in range(3)], -1)
    Nn = np.stack([ndi.map_coordinates(Nc[..., k], [fi, fj], order=1) for k in range(3)], -1)
    h = Nn[..., :2] / (np.linalg.norm(Nn[..., :2], axis=-1, keepdims=True) + 1e-9)

    S, G = rg.S, rg.G
    fx, fy = rg.vf[2] - P[..., 0] * S, G - P[..., 2] * S
    bx = rg.vb[2] + P[..., 0] * S
    sx = rg.vs[2] - P[..., 1] * S
    # 脖子下半截取全身立绘（接躯干），下巴往下 3~5 厘米渐变到补画
    wr = smooth01((P[..., 2] - (face.chin_z - 0.05)) / 0.02)[..., None]
    Fc = wr * ref.look("f", fx, fy) + (1 - wr) * sample(atlas, fx, fy)
    # 侧面：人的右半边（法线朝 +X）取右侧面，左半边取左侧面，正中前后一窄条渐变（那里本来就是正面、背面图在管）
    wsr = smooth01(h[..., 0] / 0.3 + 0.5)[..., None]
    Sc = wr * (wsr * ref.look("r", sx, fy) + (1 - wsr) * ref.look("l", sx, fy)) + (1 - wr) * sample(atlas, sx + rg.W - rg.vs[0], fy)
    # 后脑也取侧面图（侧面图上整个后脑的头发都画了）。全身立绘的背面不用：补画的头发比立绘高，头顶那截在立绘背面上是灰底
    # 脖子下半截还是全身立绘的背面
    Bc = sample(atlas, bx, fy)
    a, b = SEAM
    wf = smooth01((h[..., 1] - a) / b)
    wb = smooth01((-h[..., 1] - a) / b) * (1 - wr[..., 0])
    ws = np.clip(1 - wf - wb, 0, 1)
    out = wf[..., None] * Fc + ws[..., None] * Sc + wb[..., None] * Bc
    out = np.clip(out, 0, 255).astype(np.uint8)

    uvs = np.zeros((n, nt + 1, 2))
    uvs[..., 0] = PAD + np.arange(nt + 1)[None, :] / nt * (BAKE_W - 1)
    uvs[..., 1] = np.arange(n)[:, None] / (n - 1) * (BAKE_H - 1)
    return out, uvs
