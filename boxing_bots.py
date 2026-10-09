"""拳击标准陪练：打法写死，永远不变，当量尺。

只看观测（对手状态也有反应延迟）。打法：远了就压上，刺拳距离里刺拳、1-2、举架、绕步、偶尔预判性地侧闪，
进了内围打勾拳上勾拳、偶尔缠抱；看到慢的后手勾拳起手会下潜（快拳反应不过来，只能预判），背靠围绳会轴转出去。
"""
import numpy as np

import boxing as B


class BoxingSparring:
    name = "box-sparring-v1"

    def act(self, o, rng, mask=None):
        n = len(o)
        u, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
        to_jab, to_hook = o[:, 52], o[:, 53]
        opp_su = o[:, 65] > 0.5
        opp_mv = np.argmax(o[:, 73:93], axis=1)
        own_mv = np.argmax(o[:, 14:34], axis=1)
        combo = o[:, 39] > 0.5
        on_ropes = o[:, 41] * B.RING < 0.4

        mid = np.select([u < 0.22, u < 0.28, u < 0.32, u < 0.35, u < 0.55, u < 0.62, u < 0.69],
                        [B.JAB_H, B.CROSS_H, B.JAB_B, B.FEINT_JAB, B.GUARD_HIGH, B.LEFT, B.RIGHT], B.STAY)
        close = np.select([u < 0.12, u < 0.18, u < 0.24, u < 0.30, u < 0.36, u < 0.56, u < 0.58, u < 0.68],
                          [B.LHOOK_H, B.RHOOK_H, B.LUPPER_H, B.LHOOK_B, B.CROSS_H, B.GUARD_HIGH, B.CLINCH, B.BACK], B.STAY)
        far = np.where(u < 0.7, B.FWD, B.GUARD_FWD)
        a = np.where(to_hook <= 0, close, np.where(to_jab <= -0.03, mid, far))
        a = np.where(combo & (own_mv == 0) & (u3 < 0.5), B.CROSS_H, a)
        a = np.where(on_ropes & (u3 < 0.15), np.where(u2 < 0.5, B.PIVOT_L, B.PIVOT_R), a)
        opp_in_range = o[:, 55] <= 0                       # 我在对手刺拳射程里
        a = np.where(opp_in_range & (u2 < 0.05), B.SLIP, a)  # 预判性侧闪
        slow_hook = opp_su & (opp_mv == 6)                   # 后手勾拳起手 6 帧，来得及反应
        a = np.where(slow_hook & (u2 < 0.5), B.DUCK, a)
        return a
