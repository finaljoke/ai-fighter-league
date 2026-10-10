"""按声音事件表合成音轨。全部程序合成：拳击音效、观众、回合铃、训练音效、训练节拍；解说用 edge-tts（有缓存，连不上网就跳过）。

用法：py -X utf8 tools/make_audio.py events.json out.wav
events.json：{"duration": 总秒数, "events": [{t, type, ...}, ...]}（由 tools/export_video.mjs 从观看页导出）
"""
import asyncio
import hashlib
import json
import os
import subprocess
import time
import sys
import wave

import numpy as np

SR = 48000
RNG = np.random.default_rng(7)
HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, "voice_cache")
VOICE = "zh-CN-YunxiNeural"          # 解说：年轻男声


# ------------------------------------------------------------------ 基础
def tt(d):
    return np.arange(int(d * SR)) / SR


def noise(d):
    return RNG.standard_normal(int(d * SR))


def band(x, lo=None, hi=None):
    """频域滤波：保留 lo~hi 赫兹（边缘平滑过渡）。"""
    n = len(x)
    if n < 8:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** 4)
    if hi:
        g *= 1 / (1 + (f / hi) ** 4)
    return np.fft.irfft(X * g, n)


def env(d, attack, decay):
    t = tt(d)
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)


def sweep(d, f0, f1, decay):
    """频率从 f0 滑到 f1 的正弦（闷响的核心）。"""
    t = tt(d)
    f = f1 + (f0 - f1) * np.exp(-t / (d * 0.35))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / decay)


def norm(x, peak=1.0):
    m = np.max(np.abs(x)) if len(x) else 0
    return x * (peak / m) if m > 0 else x


def mix(*parts):
    n = max(len(p) for p in parts)
    out = np.zeros(n)
    for p in parts:
        out[:len(p)] += p
    return out


# ------------------------------------------------------------------ 音效库
def snd_hit(dmg, power, body, counter, slow=False, gym=False):
    s = 2.4 if slow else 1.0
    f0 = (95 if body else 145) * (0.85 if power else 1.0) * (0.6 if slow else 1.0)
    d = 0.35 * s
    thump = sweep(d, f0 * 1.6, f0 * 0.55, (0.12 if body else 0.085) * s)
    slap = band(noise(0.06 * s), 400 if body else 800, 2200 if body else 3800) * env(0.06 * s, 0.001, 0.018 * s)
    parts = [thump * 1.0, slap * 0.9]
    if counter:
        parts.append(band(noise(0.03), 3000, 9000) * env(0.03, 0.0005, 0.008) * 0.8)
    if power:
        parts.append(band(noise(0.25 * s), 60, 400) * env(0.25 * s, 0.002, 0.06 * s) * 0.6)
    amp = float(np.clip(0.35 + dmg / 10.0, 0.3, 1.0)) * (1.15 if power else 1.0)
    y = norm(mix(*parts), amp)
    if slow:
        y = band(y, None, 2500)
        y = mix(y, np.concatenate([np.zeros(int(0.09 * SR)), y * 0.35]))
    return y * (0.6 if gym else 1.0)                              # 训练场：护具、大拳套，声音轻


def snd_block(power):
    y = mix(band(noise(0.05), 1000, 4500) * env(0.05, 0.0005, 0.012), sweep(0.12, 260, 180, 0.03) * 0.5)
    return norm(y, 0.45 if power else 0.32)


def snd_whiff(power):
    d = 0.22 if power else 0.14
    t = tt(d)
    x = noise(d)
    y = band(x, 500, 2400) * np.sin(np.pi * t / d) ** 2
    return norm(y, 0.2 if power else 0.12)


def snd_evade():
    d = 0.2
    t = tt(d)
    return norm(band(noise(d), 300, 1500) * np.sin(np.pi * t / d) ** 2, 0.14)


def snd_kd():
    y = mix(sweep(0.6, 110, 45, 0.18), band(noise(0.5), 80, 900) * env(0.5, 0.003, 0.12) * 0.8,
            band(noise(0.9), 600, 2500) * env(0.9, 0.01, 0.4) * 0.15)
    return norm(y, 1.0)


def snd_bell(kind):
    def strike():
        t = tt(2.6)
        y = np.zeros_like(t)
        for ratio, a, dec in ((1.0, 1.0, 1.4), (2.0, 0.5, 0.9), (2.76, 0.45, 0.7), (4.07, 0.3, 0.45), (5.4, 0.2, 0.3)):
            y += a * np.sin(2 * np.pi * 820 * ratio * t) * np.exp(-t / dec)
        return y * np.minimum(1, t / 0.002)
    n = {"start": 1, "end": 3, "final": 5}.get(kind, 1)
    gap = int(0.2 * SR)
    one = strike()
    y = np.zeros(len(one) + gap * (n - 1))
    for i in range(n):
        y[i * gap:i * gap + len(one)] += one * (1 - 0.08 * i)
    return norm(y, 0.55)


def snd_cheer(level):
    d = 2.6
    t = tt(d)
    base = band(noise(d), 350, 3200) * (np.minimum(1, t / 0.25) * np.exp(-np.maximum(0, t - 0.4) / 1.2))
    # 零散的"喔"声：几个滑音
    whoo = np.zeros_like(t)
    for _ in range(int(6 + 10 * level)):
        st = RNG.uniform(0, 1.2)
        dd = RNG.uniform(0.3, 0.8)
        f0 = RNG.uniform(260, 520)
        i0 = int(st * SR)
        tl = tt(dd)
        seg = np.sin(2 * np.pi * np.cumsum(f0 * (1 + 0.25 * tl / dd)) / SR) * np.sin(np.pi * tl / dd) ** 2
        whoo[i0:i0 + len(seg)] += seg[:len(whoo) - i0] * RNG.uniform(0.2, 0.5)
    y = base + band(whoo, 200, 2000) * 0.5
    return norm(y, 0.25 + 0.45 * level)


def snd_step():
    return norm(mix(sweep(0.06, 120, 70, 0.02), band(noise(0.03), 1500, 5000) * env(0.03, 0.0005, 0.006) * 0.4), 0.18)


def snd_rope():
    return norm(mix(band(noise(0.02), 2500, 9000) * env(0.02, 0.0003, 0.004), sweep(0.08, 140, 80, 0.025) * 0.6), 0.22)


def snd_clank():
    t = tt(0.6)
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / d) for f, a, d in ((610, 1.0, 0.18), (1580, 0.6, 0.1), (2930, 0.4, 0.06)))
    return norm(mix(y, sweep(0.2, 120, 60, 0.05) * 0.8), 0.35)


def snd_bag():
    t = tt(0.4)
    chain = sum(np.sin(2 * np.pi * f * t) * np.exp(-t / 0.05) for f in (3200, 4700, 6100)) * 0.08
    return norm(mix(sweep(0.4, 110, 55, 0.11), band(noise(0.06), 500, 2500) * env(0.06, 0.0005, 0.015) * 0.7, chain), 0.6)


def snd_mitt():
    return norm(mix(band(noise(0.04), 1500, 5000) * env(0.04, 0.0003, 0.009), sweep(0.1, 420, 260, 0.02) * 0.6), 0.5)


def snd_ball():
    t = tt(0.12)
    return norm(mix(np.sin(2 * np.pi * 880 * t) * np.exp(-t / 0.03), band(noise(0.01), 2000, 8000) * 0.4), 0.2)


def snd_tap():
    return norm(band(noise(0.012), 1500, 6000) * env(0.012, 0.0003, 0.003), 0.12)


def snd_marker():
    t = tt(0.14)
    return norm(np.sin(2 * np.pi * np.cumsum(2200 + 600 * t / 0.14) / SR) * np.sin(np.pi * t / 0.14) * (0.6 + 0.4 * np.sin(2 * np.pi * 37 * t)), 0.06)


def snd_night():
    t = tt(2.0)
    chime = (np.sin(2 * np.pi * 1046.5 * t) + 0.6 * np.sin(2 * np.pi * 1568 * t)) * np.exp(-t / 0.7) * np.minimum(1, t / 0.005)
    crickets = np.sin(2 * np.pi * 4500 * t) * (np.sin(2 * np.pi * 28 * t) > 0.6) * (np.sin(2 * np.pi * 1.6 * t) > 0) * 0.15
    return norm(mix(chime, crickets), 0.3)


def snd_hurt():
    t = tt(0.3)
    return norm(band(noise(0.3), 250, 900) * np.sin(np.pi * t / 0.3) * (1 + 0.5 * np.sin(2 * np.pi * 110 * t)), 0.25)


def snd_sting():
    t = tt(1.6)
    kick = sweep(0.5, 150, 45, 0.15)
    crash = band(noise(1.6), 3000, 12000) * np.exp(-t / 0.5) * 0.35
    chord = np.zeros_like(t)
    for f in (110, 164.8, 220, 261.6):
        ph = (t * f) % 1.0
        chord += (2 * ph - 1)
    chord = band(chord, None, 1800) * np.minimum(1, t / 0.02) * np.exp(-t / 0.9) * 0.35
    return norm(mix(kick, crash, chord), 0.7)


def snd_swoosh():
    d = 0.6
    t = tt(d)
    return norm(band(noise(d), 400, 4000) * (t / d) ** 2 * np.exp(-np.maximum(0, t - 0.5) / 0.05), 0.3)


SFX = {"hit": snd_hit, "block": snd_block, "whiff": snd_whiff, "evade": snd_evade, "kd": snd_kd, "bell": snd_bell,
       "cheer": snd_cheer, "step": snd_step, "rope": snd_rope, "clank": snd_clank, "thud": lambda **k: snd_step() * 2.5,
       "bag": snd_bag, "mitt": snd_mitt, "ball": snd_ball, "tap": snd_tap, "marker": snd_marker, "night": snd_night,
       "hurt": snd_hurt, "sting": snd_sting, "swoosh": snd_swoosh}


# ------------------------------------------------------------------ 环境声和音乐（整段生成）
CR = 200                                                         # 包络的控制率（赫兹）


def slow_env(d, rate, depth=1.0):
    """随机起伏的包络：起伏速度约 rate 赫兹，depth 越大起伏越明显。"""
    n = int(d * CR) + 2
    x = RNG.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / CR)
    x = np.fft.irfft(np.fft.rfft(x) / (1 + (f / rate) ** 4), n)
    e = np.maximum(0, 1 + depth * x / (np.std(x) + 1e-9))
    return np.interp(tt(d), np.arange(n) / CR, e)


def bed_crowd(d, muffled=False):
    """人声嘈杂：十几簇共振峰频段的噪声，各自按说话的音节节奏（3~6 赫兹）起伏，叠起来就是"嗡嗡"的人群。"""
    y = np.zeros(int(d * SR))
    for _ in range(12):
        fc = RNG.uniform(300, 2000)
        y += band(noise(d), fc * 0.75, fc * 1.3) * slow_env(d, RNG.uniform(3, 6), 0.9)
    y += band(noise(d), 110, 380) * slow_env(d, 1.2, 0.4) * 1.5
    return norm(band(y, 90, 1100 if muffled else 3800), 1.0) * 0.10


def bed_roar(d):
    """观众激动时的吼声：更亮、更密，音量由激动程度控制。"""
    y = band(noise(d), 450, 3200) * slow_env(d, 8, 0.5)
    for _ in range(6):
        fc = RNG.uniform(500, 1400)
        y += band(noise(d), fc * 0.9, fc * 1.15) * slow_env(d, 2, 0.6) * 2
    return norm(band(y, 200, 5000), 1.0) * 0.10


def bed_gym(d):
    t = tt(d)
    hum = 0.04 * np.sin(2 * np.pi * 60 * t) + 0.02 * np.sin(2 * np.pi * 120 * t)
    room = band(noise(d), 60, 600) * 0.05
    return hum + norm(room, 0.05)


def bed_tv(d):
    return band(bed_crowd(d), 300, 1800) * 0.6


def bed_treadmill(d):
    t = tt(d)
    return 0.05 * np.sin(2 * np.pi * 110 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t)) + norm(band(noise(d), 800, 3000), 0.02)


def bed_music(d, start_t=0.0):
    """训练节拍：96 BPM，小调贝斯 + 鼓。按绝对时间对齐，几段连起来不会断拍。"""
    bpm = 96
    beat = 60 / bpm
    t = tt(d) + start_t
    y = np.zeros(len(t))
    kick, snare, hat = sweep(0.25, 150, 48, 0.08), mix(band(noise(0.18), 900, 5000) * env(0.18, 0.001, 0.05), sweep(0.12, 220, 160, 0.04) * 0.5), band(noise(0.04), 6000, 14000) * env(0.04, 0.0005, 0.012)
    bass_notes = [55.0, 55.0, 43.65, 49.0]                     # A1 A1 F1 G1，一小节一个
    first = int(np.floor(start_t / (beat / 2)))
    last = int(np.ceil((start_t + d) / (beat / 2)))
    for k in range(first, last + 1):
        at = int((k * beat / 2 - start_t) * SR)
        if at >= len(y):
            break
        def put(s, g):
            a0 = max(0, at)
            s0 = a0 - at
            n = min(len(s) - s0, len(y) - a0)
            if n > 0:
                y[a0:a0 + n] += s[s0:s0 + n] * g
        put(hat, 0.22)
        if k % 4 == 0:
            put(kick, 0.9)
        if k % 4 == 2:
            put(snare, 0.6)
        if k % 2 == 0:
            f = bass_notes[(k // 8) % 4]
            tl = tt(beat * 0.9)
            ph = (tl * f) % 1.0
            put(band((2 * ph - 1) * np.exp(-tl / 0.35), None, 600), 0.35)
    return y * 0.5


# ------------------------------------------------------------------ 解说（edge-tts，缓存）
def voice_wav(text, rate):
    os.makedirs(VOICE_DIR, exist_ok=True)
    key = hashlib.md5(f"{VOICE}|{rate}|{text}".encode("utf-8")).hexdigest()[:16]
    mp3, wav = os.path.join(VOICE_DIR, key + ".mp3"), os.path.join(VOICE_DIR, key + ".wav")
    if not os.path.exists(wav):
        import edge_tts
        for attempt in range(3):                                 # 语音服务偶尔不返回音频，重试几次
            try:
                asyncio.run(edge_tts.Communicate(text, VOICE, rate=rate).save(mp3))
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", mp3, "-ac", "1", "-ar", str(SR), wav], check=True)
                break
            except Exception as e:                               # 连不上网：只出音效
                if attempt == 2:
                    print(f"  解说生成失败（{text}）：{e}", file=sys.stderr)
                    return None
                time.sleep(1.5)
    with wave.open(wav, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    return data


# ------------------------------------------------------------------ 混音
def main():
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    out = sys.argv[2]
    dur = float(spec["duration"])
    N = int(dur * SR) + SR
    NC = int(dur * CR) + CR
    fx = np.zeros((2, N))                                        # 音效
    vo = np.zeros(N)                                             # 解说（居中）
    bd = np.zeros((2, N))                                        # 环境声和音乐（解说时压低）
    excite = np.zeros(NC)

    def place(buf, sig, t, pan=0.0, gain=1.0):
        i0 = int(t * SR)
        if i0 >= N or len(sig) == 0:
            return
        n = min(len(sig), N - i0)
        if buf.ndim == 1:
            buf[i0:i0 + n] += sig[:n] * gain
            return
        a = (pan + 1) * np.pi / 4
        buf[0, i0:i0 + n] += sig[:n] * gain * np.cos(a)
        buf[1, i0:i0 + n] += sig[:n] * gain * np.sin(a)

    def bump(t, amount):
        excite[min(NC - 1, int(t * CR))] += amount

    cache = {}
    beds = []
    last_cheer = -9.0
    for e in sorted(spec["events"], key=lambda e: e["t"]):
        typ, t, pan = e["type"], float(e["t"]), float(e.get("pan", 0.0))
        if typ == "bed":
            beds.append(e)
            continue
        if typ == "voice":
            v = voice_wav(e["text"], "+30%" if e.get("count") else "+8%")
            if v is not None:
                place(vo, v, t, gain=0.85)
            continue
        if typ == "hit":
            key = ("hit", round(e["dmg"]), e["power"], e["body"], e["counter"], e.get("slow", False), e.get("gym", False), RNG.integers(3))
            if key not in cache:
                cache[key] = snd_hit(e["dmg"], e["power"], e["body"], e["counter"], e.get("slow", False), e.get("gym", False))
            place(fx, cache[key], t, pan, 0.8)
            if not e.get("gym") and not e.get("slow"):
                bump(t, 0.12 + e["dmg"] / 30 + (0.3 if e["counter"] else 0))
            continue
        if typ == "cheer":
            # 欢呼要稀才有分量：离上一次不到 3 秒就只让观众声涨一点，不再单独喊一声（击倒例外）
            if e["level"] >= 1.0 or t - last_cheer >= 3.0:
                place(fx, snd_cheer(e["level"]), t, 0.0, 0.8)
                last_cheer = t
                bump(t, e["level"])
            else:
                bump(t, 0.3)
            continue
        kw = {k: v for k, v in e.items() if k in ("power", "kind", "level")}
        key = (typ, tuple(sorted(kw.items())), RNG.integers(2) if typ in ("whiff", "step", "bag", "mitt") else 0)
        if key not in cache:
            fn = SFX.get(typ)
            if fn is None:
                continue
            cache[key] = fn(**kw) if kw else fn()
        place(fx, cache[key], t, pan, 0.8)

    # 观众的激动程度：打中、反击、欢呼让它上涨（约 0.3 秒涨上去），之后约 2 秒回落
    ex = np.zeros(NC)
    up, down, acc, sm = 1 - np.exp(-1 / (0.3 * CR)), np.exp(-1 / (2.0 * CR)), 0.0, 0.0
    for i in range(NC):
        acc = acc * down + excite[i]
        sm += (min(acc, 2.5) - sm) * up
        ex[i] = sm

    def ctl(curve, t0, n):                                       # 控制率曲线 → 音频采样
        return np.interp(t0 + np.arange(n) / SR, np.arange(len(curve)) / CR, curve)

    for b in beds:
        t0, t1 = float(b["t"]), float(b["to"])
        d = max(0.0, t1 - t0)
        if d <= 0.01:
            continue
        kind, lvl, pan = b["kind"], float(b.get("level", 0.5)), float(b.get("pan", 0.0))
        if kind == "crowd":
            murmur = bed_crowd(d, b.get("muffled", False))
            roar = bed_roar(d)
            e = ctl(ex, t0, len(murmur))
            sig = (murmur * (1 + 0.4 * e) + roar * 0.9 * e) * lvl
            if b.get("muffled"):
                sig = band(sig, None, 1100)
        elif kind == "gym":
            sig = bed_gym(d) * lvl * 0.6
        elif kind == "tv":
            sig = bed_tv(d) * lvl
        elif kind == "treadmill":
            sig = bed_treadmill(d) * lvl
        elif kind == "music":
            sig = bed_music(d, t0) * lvl
        else:
            continue
        fade = np.minimum(1, np.minimum(np.arange(len(sig)), len(sig) - np.arange(len(sig))) / (0.03 * SR))
        place(bd, sig * fade, t0, pan)

    # 解说开口时，环境声和音乐压低一半（0.05 秒压下，0.4 秒放开）
    v_env = np.abs(vo[::SR // CR][:NC])
    duck, g = np.ones(NC), 1.0
    for i in range(NC):
        target = 0.5 if v_env[i] > 0.02 else 1.0
        g += (target - g) * (0.6 if target < g else 0.012)
        duck[i] = g
    bd *= ctl(duck, 0.0, N)

    st = (fx + bd + vo[None, :]).T[: int(dur * SR)] * 2.3          # 固定增益，各条视频响度一致
    st = np.tanh(st)                                              # 软限幅：叠到一起的峰值被圆滑压住
    st *= 0.89                                                    # 约 -1 dBFS
    pcm = (st * 32767).astype(np.int16)
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"音轨：{out}（{dur:.1f} 秒，{len(spec['events'])} 个事件）")


if __name__ == "__main__":
    main()
