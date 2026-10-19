"""旧回放的画面帧补上画面字段（落点、擦到、被什么挡住、肝区击倒前弯腰、心窝喘不上气）。

规则改版以后，旧回放在新规则下重算会走样，所以 runs/replay_frames/ 里存着当初算出来的帧（正史首战"结果不重来"）。
那些帧是 box2d-v15 以前存的，没有画面字段，观看页只能按"这种拳最常见的落点"去猜。这里从 git 里取出当时那一版
boxing.py 重算一遍：原来的 22 项必须一项不差，才把补全的帧写进 <key>.viz.json。原来的帧文件不动（runs/ 只追加）。

    py -X utf8 tools/enrich_frames.py
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import boxing as B                      # noqa: E402
import boxing_league as BL              # noqa: E402

CACHE = os.path.join("runs", "replay_frames")


def all_recs():
    """runs/ 里所有存着回放档的地方：首战、里程碑重演、成长存档录像。"""
    for f in glob.glob(os.path.join("runs", "*", "first_fight.json")):
        yield f, json.load(open(f, encoding="utf-8"))["replay"]
    for f in glob.glob(os.path.join("runs", "*", "milestone_reenact.json")):
        for k, v in json.load(open(f, encoding="utf-8")).items():
            if v:
                yield f"{f}：{k}", v["rec"]
    for f in glob.glob(os.path.join("runs", "*", "growth", "v*.json")):
        for r in json.load(open(f, encoding="utf-8")).get("replays", []):
            yield f, r


def key_of(rec):
    return hashlib.sha1((rec.get("sim_version", "") + "|" + "|".join(rec["actions"]) + "|" + json.dumps(rec["init"])).encode()).hexdigest()[:16]


def version_commits():
    """规则版本 → 最后一个还是这一版的提交。"""
    out = {}
    for c in subprocess.run(["git", "log", "--format=%H", "--", "boxing.py"], capture_output=True, text=True, check=True).stdout.split():
        src = subprocess.run(["git", "show", f"{c}:boxing.py"], capture_output=True, check=True).stdout.decode("utf-8")
        ver = next((line.split('"')[1] for line in src.splitlines() if line.startswith("SIM_VERSION")), None)
        if ver and ver not in out:
            out[ver] = (c, src)
    return out


def load_old(ver, commit, src):
    """把当时那一版 boxing.py 载成一个模块，结算信息里加上画面要的两项（擦到、被什么挡住）。"""
    anchor = '"blocked_by": np.where(blocked, gtype, 0)[:, ::-1],'
    if anchor in src:
        kind = "np.where(nat_block, 3, gtype)" if "nat_block = " in src else "gtype"
        src = src.replace(anchor, anchor + f' "glance": landed & glance, "block_kind": np.where(blocked, {kind}, 0),', 1)
    mod = types.ModuleType("boxing_" + ver.replace("-", "_"))
    mod.__file__ = f"<git {commit[:7]}:boxing.py>"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def main():
    commits, mods, done = None, {}, set()
    for src, rec in all_recs():
        ver, key = rec.get("sim_version"), key_of(rec)
        path = os.path.join(CACHE, f"{key}.json")
        viz = os.path.join(CACHE, f"{key}.viz.json")
        if ver == B.SIM_VERSION or key in done or not os.path.exists(path) or os.path.exists(viz):
            continue
        done.add(key)
        commits = commits or version_commits()
        if ver not in commits:
            print(f"  {key}（{src}）：git 里找不到 {ver} 那一版规则，跳过")
            continue
        if ver not in mods:
            mods[ver] = load_old(ver, *commits[ver])
        old = json.load(open(path, encoding="utf-8"))
        new = BL.resimulate(rec, mods[ver])
        same = len(old) == len(new) and all(o["r"] == q["r"] and o["t"] == q["t"] and all(o["f"][k] == q["f"][k][:len(o["f"][k])] for k in (0, 1))
                                              for o, q in zip(old, new))
        if not same:
            print(f"  ✗ {key}（{src}）：用 {ver}（{commits[ver][0][:7]}）重算和存好的帧对不上，不补")
            continue
        json.dump(new, open(viz, "w", encoding="utf-8"), separators=(",", ":"))
        print(f"  ✓ {key}（{src}）：{ver} 重算 {len(new)} 帧一项不差，补好画面字段 → {viz}")


if __name__ == "__main__":
    main()
