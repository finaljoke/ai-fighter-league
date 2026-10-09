"""一维擂台对打模拟器（向量化，N 场同时跑）。

确定性：给定初始距离和双方每帧动作，结果完全确定。所以回放只需要存
"初始距离 + 动作序列"，画面随时可以重新算出来。

坐标：擂台是 [0, RING] 的一条线，0 号选手永远在左边面朝右，1 号在右边面朝左
（两人碰撞不能穿过，所以不会换边）。
"""
import numpy as np

SIM_VERSION = "1d-v2"

DT = 0.05            # 一帧 0.05 秒
RING = 6.0           # 擂台宽，米
BODY_R = 0.30        # 身体半径
SHOULDER = 0.15      # 肩膀到身体中心的偏移
MIN_GAP = 2 * BODY_R
WALK = 1.6           # 基准步速 m/s
BACK_MULT = 0.85     # 后退比前进慢
HP_MAX = 100.0
STAM_BASE = 100.0
STAM_REGEN = 6.0     # 每秒
T_MAX = 600          # 一回合 30 秒

# 动作
STAY, FWD, BACK, JAB, POWER, BLOCK, GUARD_FWD = range(7)
N_ACTIONS = 7
ACTION_NAMES = ["stay", "fwd", "back", "jab", "power", "block", "guard_fwd"]
GUARD_WALK = 0.5     # 举着架子往前压，速度打对折

# 状态
FREE, STARTUP, RECOVERY, HITSTUN, BLOCKSTUN = range(5)
N_PHASES = 5

# 招式表，下标 0 = 无，1 = 刺拳，2 = 重拳
MOVE_STARTUP = np.array([0, 3, 6])
MOVE_RECOVERY = np.array([0, 5, 9])
MOVE_DMG = np.array([0.0, 4.0, 13.0])
MOVE_COST = np.array([0.0, 4.0, 10.0])
MOVE_HITSTUN = np.array([0, 5, 11])
MOVE_BLOCKSTUN = np.array([0, 3, 5])
MOVE_KB = np.array([0.0, 0.12, 0.40])
MOVE_RANGE = np.array([0.0, 1.0, 0.9])     # 乘臂展
BLOCK_MULT = 0.15
BLOCK_STAM_DRAIN = 0.6                      # 格挡时按招式基础伤害扣体能
BLOCK_KB_MULT = 0.25                        # 挡住了也会被推一点
COUNTER_MULT = 1.3                          # 打断对方前摇的加成
CRAMP_FRAC = 0.55                           # 贴得比"肩 + 0.55 倍臂展"还近，长臂打不开
CRAMP_MULT = 0.6

FACING = np.array([1.0, -1.0])
OBS_DIM = 31
HIST = 8


def _ticks(base, speed):
    return np.maximum(1, np.floor(base / speed + 0.5)).astype(np.int32)


class Sim:
    def __init__(self, n, bodies, seed=0, auto_reset=True, t_max=T_MAX):
        self.n = n
        self.bodies = bodies
        self.auto_reset = auto_reset
        self.t_max = t_max
        self.rng = np.random.default_rng(seed)
        b0, b1 = bodies
        pick = lambda k: np.array([getattr(b0, k), getattr(b1, k)], dtype=np.float64)
        self.reach, self.mass, self.power = pick("reach"), pick("mass"), pick("power")
        self.chin, self.cardio = pick("chin"), pick("cardio")
        hand = pick("hand_speed")
        self.react = np.array([b0.react_ticks, b1.react_ticks])
        self.stam_max = STAM_BASE * self.cardio
        self.regen = STAM_REGEN * self.cardio * DT
        self.walk = WALK * pick("foot_speed") * np.sqrt(75.0 / self.mass) * DT
        # [招式, 选手]
        self.startup = np.stack([_ticks(MOVE_STARTUP[m], hand) if m else np.zeros(2, np.int32) for m in range(3)])
        self.recovery = np.stack([_ticks(MOVE_RECOVERY[m], hand) if m else np.zeros(2, np.int32) for m in range(3)])
        self.range = BODY_R + SHOULDER + MOVE_RANGE[:, None] * self.reach[None, :]  # 中心距 <= 这个就打得到
        self.cramp = BODY_R + SHOULDER + CRAMP_FRAC * self.reach

        self.x = np.zeros((n, 2))
        self.hp = np.zeros((n, 2))
        self.stam = np.zeros((n, 2))
        self.phase = np.zeros((n, 2), np.int32)
        self.move = np.zeros((n, 2), np.int32)
        self.timer = np.zeros((n, 2), np.int32)
        self.blocking = np.zeros((n, 2), bool)
        self.t = np.zeros(n, np.int32)
        self.init_gap = np.zeros(n)
        self.hist = np.zeros((HIST, n, 2, 7))
        self.ptr = 0
        self.reset(np.ones(n, bool))

    # ---------------------------------------------------------------- 状态
    def reset(self, mask, gap=None):
        k = int(mask.sum())
        if k == 0:
            return
        if gap is None:
            gap = 2.0 + self.rng.uniform(-0.3, 0.3, size=k)
        self.init_gap[mask] = gap
        self.x[mask, 0] = RING / 2 - gap / 2
        self.x[mask, 1] = RING / 2 + gap / 2
        self.hp[mask] = HP_MAX
        self.stam[mask] = self.stam_max
        self.phase[mask] = FREE
        self.move[mask] = 0
        self.timer[mask] = 0
        self.blocking[mask] = False
        self.t[mask] = 0
        snap = self._snapshot()
        self.hist[:, mask] = snap[mask][None]

    def desync(self, rng):
        """训练开头把各场的进度错开，避免所有场同时开打同时结束。"""
        self.t[:] = rng.integers(0, self.t_max, self.n)

    def _snapshot(self):
        return np.stack([self.x, self.hp, self.stam / self.stam_max, self.phase,
                         self.move, self.timer, self.blocking], axis=-1)

    def _push_hist(self):
        self.ptr = (self.ptr + 1) % HIST
        self.hist[self.ptr] = self._snapshot()

    def back_space(self, x):
        """各自身后到围绳的距离。"""
        return np.stack([x[..., 0] - BODY_R, RING - BODY_R - x[..., 1]], axis=-1)

    # ---------------------------------------------------------------- 一帧
    def step(self, actions):
        """actions: (n, 2) int。返回 (reward, done, info)。"""
        a = actions
        ph, mv, tm = self.phase, self.move, self.timer
        side = np.arange(2)[None, :]

        # 1) 推进计时器
        busy = ph != FREE
        tm = np.where(busy, tm - 1, tm)
        fire = (ph == STARTUP) & (tm <= 0)
        ends = (ph >= RECOVERY) & (tm <= 0)
        ph = np.where(ends, FREE, ph)
        mv = np.where(ends, 0, mv)

        # 2) 出拳判定（双方同时结算，可以互换）
        fired = np.where(fire, mv, 0)
        gap = (self.x[:, 1] - self.x[:, 0])[:, None]
        rng = self.range[mv, side]
        hit = fire & (gap <= rng + 1e-9)
        guard = self.blocking | (ph == BLOCKSTUN)
        tgt_guard = guard[:, ::-1]
        tgt_counter = ((ph == STARTUP) & ~fire)[:, ::-1]
        stam_frac = self.stam / self.stam_max
        base = MOVE_DMG[mv] * self.power[None, :] * (0.5 + 0.5 * stam_frac)
        mult = np.where(tgt_guard, BLOCK_MULT, np.where(tgt_counter, COUNTER_MULT, 1.0))
        mult = mult * np.where(gap < self.cramp[None, :], CRAMP_MULT, 1.0)
        dmg = np.where(hit, base * mult / self.chin[::-1][None, :], 0.0)   # dmg[:, i] = i 打出的伤害
        blocked = hit & tgt_guard
        clean = hit & ~tgt_guard

        att_move = mv[:, ::-1]          # 打到我的那一招
        got_clean = clean[:, ::-1]
        got_blocked = blocked[:, ::-1]
        hp = self.hp - dmg[:, ::-1]
        stam = self.stam - np.where(got_blocked, MOVE_DMG[att_move] * BLOCK_STAM_DRAIN, 0.0)

        # 出拳的人进后摇（mv 保留，画面上显示伸出的拳）
        ph = np.where(fire, RECOVERY, ph)
        tm = np.where(fire, self.recovery[mv, side], tm)
        # 挨打的人进硬直，会打断自己的招
        ph = np.where(got_blocked, BLOCKSTUN, ph)
        tm = np.where(got_blocked, MOVE_BLOCKSTUN[att_move], tm)
        ph = np.where(got_clean, HITSTUN, ph)
        tm = np.where(got_clean, MOVE_HITSTUN[att_move], tm)
        mv = np.where(got_clean | got_blocked, 0, mv)

        kb = MOVE_KB[att_move] * self.power[::-1][None, :] * (75.0 / self.mass[None, :])
        kb = np.where(got_clean, kb, np.where(got_blocked, kb * BLOCK_KB_MULT, 0.0))
        x = self.x - kb * FACING[None, :]

        # 3) 自由状态下执行动作
        acted = ph == FREE
        mid = np.where(a == JAB, 1, np.where(a == POWER, 2, 0))
        start = acted & (mid > 0) & (stam >= MOVE_COST[mid])
        ph = np.where(start, STARTUP, ph)
        mv = np.where(start, mid, mv)
        tm = np.where(start, self.startup[mid, side], tm)
        stam = stam - np.where(start, MOVE_COST[mid], 0.0)
        started = np.where(start, mid, 0)

        blocking = acted & ((a == BLOCK) | (a == GUARD_FWD))
        moving = acted & ((a == FWD) | (a == BACK) | (a == GUARD_FWD))
        tired = 0.6 + 0.4 * np.clip(stam / self.stam_max, 0, 1)
        speed = np.select([a == FWD, a == BACK, a == GUARD_FWD], [1.0, -BACK_MULT, GUARD_WALK], 0.0)
        step = self.walk[None, :] * speed * tired
        x = x + np.where(moving, step * FACING[None, :], 0.0)

        # 4) 碰撞（按体重互推）和围绳
        overlap = np.maximum(0.0, MIN_GAP - (x[:, 1] - x[:, 0]))
        m0, m1 = self.mass
        x[:, 0] -= overlap * m1 / (m0 + m1)
        x[:, 1] += overlap * m0 / (m0 + m1)
        x[:, 0] = np.maximum(x[:, 0], BODY_R)
        x[:, 1] = np.minimum(x[:, 1], RING - BODY_R)
        x[:, 1] = np.maximum(x[:, 1], x[:, 0] + MIN_GAP)
        x[:, 0] = np.minimum(x[:, 0], x[:, 1] - MIN_GAP)

        # 5) 体能回复（前摇时不回，举架子回一半）
        regen = np.where(ph == STARTUP, 0.0, np.where(blocking, 0.5, 1.0)) * self.regen[None, :]
        stam = np.clip(stam + regen, 0.0, self.stam_max[None, :])

        self.x, self.hp, self.stam = x, hp, stam
        self.phase, self.move, self.timer, self.blocking = ph, mv, tm, blocking
        self.t = self.t + 1

        # 6) 胜负
        ko = (hp <= 0).any(axis=1)
        timeout = self.t >= self.t_max
        done = ko | timeout
        diff = hp[:, 0] - hp[:, 1]
        winner = np.where(diff > 0.5, 0, np.where(diff < -0.5, 1, -1))
        winner = np.where(done, winner, -1)
        win_sign = np.where(winner == 0, 1.0, np.where(winner == 1, -1.0, 0.0))
        reward = (dmg - dmg[:, ::-1]) / HP_MAX
        reward = reward + np.stack([win_sign, -win_sign], axis=1)

        info = {"acted": acted, "started": started, "fired": fired,
                "hit": clean, "blocked": blocked, "dmg": dmg, "ko": ko & done, "winner": winner,
                "gap": x[:, 1] - x[:, 0], "t": self.t.copy()}
        self._push_hist()
        if self.auto_reset and done.any():
            self.reset(done)
        return reward, done, info

    # ---------------------------------------------------------------- 观测
    def observe(self):
        """返回 (n, 2, OBS_DIM)。自己的状态是当前的，对手的状态延迟 react_ticks 帧。"""
        out = np.zeros((self.n, 2, OBS_DIM), np.float32)
        cur = self._snapshot()
        bs_now = self.back_space(self.x)
        t_left = 1.0 - self.t / self.t_max
        for i in (0, 1):
            j = 1 - i
            old = self.hist[(self.ptr - self.react[i]) % HIST]
            opp = old[:, j]
            own = cur[:, i]
            gap = np.abs(opp[:, 0] - own[:, 0])
            opp_x = old[:, :, 0]
            opp_bs = self.back_space(opp_x)[:, j]
            o = out[:, i]
            o[:, 0] = own[:, 1] / HP_MAX
            o[:, 1] = own[:, 2]
            o[:, 2:7] = np.eye(N_PHASES)[own[:, 3].astype(int)]
            o[:, 7] = own[:, 4] == 1
            o[:, 8] = own[:, 4] == 2
            o[:, 9] = own[:, 5] / 12.0
            o[:, 10] = own[:, 6]
            o[:, 11] = bs_now[:, i] / RING
            o[:, 12] = gap / 3.0
            o[:, 13] = gap - self.range[1, i]
            o[:, 14] = gap - self.range[2, i]
            o[:, 15] = gap - self.range[1, j]
            o[:, 16] = gap - self.range[2, j]
            o[:, 17] = opp[:, 1] / HP_MAX
            o[:, 18] = opp[:, 2]
            o[:, 19:24] = np.eye(N_PHASES)[opp[:, 3].astype(int)]
            o[:, 24] = opp[:, 4] == 1
            o[:, 25] = opp[:, 4] == 2
            o[:, 26] = opp[:, 5] / 12.0
            o[:, 27] = opp[:, 6]
            o[:, 28] = opp_bs / RING
            o[:, 29] = t_left
            o[:, 30] = (own[:, 1] - opp[:, 1]) / HP_MAX
        return out
