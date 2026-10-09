"""纯 numpy 的小网络和 PPO。网络很小（两层 128），CPU 足够，不依赖 torch。"""
import hashlib
import io

import numpy as np


def _ortho(rng, fan_in, fan_out, gain):
    a = rng.standard_normal((max(fan_in, fan_out), min(fan_in, fan_out)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if fan_in < fan_out:
        q = q.T
    return (gain * q[:fan_in, :fan_out]).astype(np.float32)


def _tmul(a, b):
    """a.T @ b。OpenBLAS 在结果行数很少（如 6×128）时走慢路径，慢几十倍，所以总让大的维度当行。"""
    if a.shape[1] >= b.shape[1]:
        return a.T @ b
    return np.ascontiguousarray((b.T @ a).T)


class MLP:
    def __init__(self, sizes, rng, out_gain):
        self.params = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_gain if i == len(sizes) - 2 else np.sqrt(2)
            self.params += [_ortho(rng, a, b, gain), np.zeros(b, np.float32)]

    def forward(self, x):
        acts = [x]
        h = x
        n = len(self.params) // 2
        for l in range(n):
            h = h @ self.params[2 * l] + self.params[2 * l + 1]
            if l < n - 1:
                h = np.tanh(h)
            acts.append(h)
        return h, acts

    def backward(self, acts, dout):
        grads = [None] * len(self.params)
        d = dout
        for l in reversed(range(len(self.params) // 2)):
            grads[2 * l] = _tmul(acts[l], d)
            grads[2 * l + 1] = d.sum(0)
            if l > 0:
                d = (d @ self.params[2 * l].T) * (1 - acts[l] ** 2)
        return grads


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-5):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, max_norm=0.5):
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        scale = min(1.0, max_norm / (norm + 1e-8))
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
        return norm


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def sample(logits, rng, mask=None):
    """按 logits 抽动作，返回 (动作, log 概率)。mask 为 False 的动作不可选（战术计划限制）。"""
    if mask is not None:
        logits = np.where(mask, logits, -1e9)
    lp = log_softmax(logits)
    g = -np.log(-np.log(rng.random(lp.shape) + 1e-12) + 1e-12)
    a = np.argmax(lp + g, axis=1)
    return a, lp[np.arange(len(a)), a]


class Policy:
    """只用来出招的那部分（快照、比赛都只要它）。"""

    def __init__(self, params):
        self.params = params
        self._mlp = MLP.__new__(MLP)
        self._mlp.params = params

    def logits(self, obs):
        return self._mlp.forward(obs)[0]

    def act(self, obs, rng, mask=None):
        return sample(self.logits(obs), rng, mask)[0]

    def sha256(self):
        return params_sha256(self.params)


def params_sha256(params):
    h = hashlib.sha256()
    for p in params:
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()


class Agent:
    def __init__(self, obs_dim, n_act, rng, hidden=128, lr=3e-4, vf_lr=1e-3):
        self.pi = MLP([obs_dim, hidden, hidden, n_act], rng, 0.01)
        self.vf = MLP([obs_dim, hidden, hidden, 1], rng, 1.0)
        self.opt_pi = Adam(self.pi.params, lr)
        self.opt_vf = Adam(self.vf.params, vf_lr)

    def act(self, obs, rng, mask=None):
        logits, _ = self.pi.forward(obs)
        a, lp = sample(logits, rng, mask)
        v = self.vf.forward(obs)[0][:, 0]
        return a, lp, v

    def value(self, obs):
        return self.vf.forward(obs)[0][:, 0]

    def snapshot(self):
        return Policy([p.copy() for p in self.pi.params])

    def update(self, obs, act, old_lp, adv, ret, pmask, vmask, rng,
               epochs=4, mb=4096, clip=0.2, ent_coef=0.01, amask=None, pi_update=True):
        """pi_update=False：只练价值网络（接着旧档训练时先热身，检查点里没存价值网络）。"""
        n = len(obs)
        sel = pmask > 0
        if sel.sum() > 1:
            adv = (adv - adv[sel].mean()) / (adv[sel].std() + 1e-8)
        stats = {"kl": [], "clipfrac": [], "ent": [], "vloss": []}
        ar = np.arange(mb)
        for _ in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, mb):
                idx = perm[s:s + mb]
                o, a, A = obs[idx], act[idx], adv[idx]
                m, vm = pmask[idx], vmask[idx]
                M, Mv = max(m.sum(), 1.0), max(vm.sum(), 1.0)
                k = len(idx)

                logits, acts = self.pi.forward(o)
                if amask is not None:
                    logits = np.where(amask[idx], logits, -1e9)
                lp_all = log_softmax(logits)
                p = np.exp(lp_all)
                lp = lp_all[ar[:k], a]
                ratio = np.exp(lp - old_lp[idx])
                flow = ((A >= 0) & (ratio < 1 + clip)) | ((A < 0) & (ratio > 1 - clip))
                g_lp = -(A * ratio * flow) * m / M
                onehot = np.zeros_like(p)
                onehot[ar[:k], a] = 1
                H = -(p * lp_all).sum(1)
                d = g_lp[:, None] * (onehot - p) + ent_coef * (m / M)[:, None] * p * (lp_all + H[:, None])
                if pi_update:
                    self.opt_pi.step(self.pi.backward(acts, d.astype(np.float32)))

                v, vacts = self.vf.forward(o)
                err = v[:, 0] - ret[idx]
                self.opt_vf.step(self.vf.backward(vacts, (err * vm / Mv)[:, None].astype(np.float32)))

                stats["kl"].append(float((((ratio - 1) - np.log(ratio)) * m).sum() / M))
                stats["clipfrac"].append(float(((np.abs(ratio - 1) > clip) * m).sum() / M))
                stats["ent"].append(float((H * m).sum() / M))
                stats["vloss"].append(float((err ** 2 * vm).sum() / Mv))
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def state(self):
        return {f"pi{i}": p for i, p in enumerate(self.pi.params)} | {f"vf{i}": p for i, p in enumerate(self.vf.params)}


def save_policy(path, params):
    buf = io.BytesIO()
    np.savez(buf, *params)
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return params_sha256(params)


def load_policy(path):
    z = np.load(path)
    return Policy([z[f"arr_{i}"] for i in range(len(z.files))])
