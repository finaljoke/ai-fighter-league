"""标准陪练：写死的打法，永远不变，用来当量尺。

和 AI 一样只看观测（对手状态也有反应延迟），不偷看真实状态。
打法：走到自己刺拳距离，在距离内随机刺拳/重拳/举架子，看到对方起手有一半概率格挡。
"""
import numpy as np

from sim import STAY, FWD, BACK, JAB, POWER, BLOCK


class SparringBot:
    name = "sparring-v1"

    def act(self, obs, rng, mask=None):
        n = len(obs)
        u = rng.random(n)
        u2 = rng.random(n)
        to_jab = obs[:, 13]            # 距离减自己的刺拳射程，>0 表示打不到
        opp_startup = obs[:, 20] > 0.5
        a = np.full(n, STAY)
        a = np.where(u < 0.22, JAB, a)
        a = np.where((u >= 0.22) & (u < 0.30), POWER, a)
        a = np.where((u >= 0.30) & (u < 0.40), BLOCK, a)
        a = np.where(to_jab > 0.0, FWD, a)
        a = np.where(to_jab < -0.35, BACK, a)
        a = np.where(opp_startup & (u2 < 0.5), BLOCK, a)
        return a
