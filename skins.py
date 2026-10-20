"""三维皮肤：页面生成时把出场角色的 characters/<id>.skin.json 读进来（tools/build_skin.py 生成，规范见 docs/角色皮肤规范.md）。

有皮肤的角色画成立绘做的三维网格；页面里只要有一个角色有皮肤，同场没有皮肤的角色就按体型和配色画成素色三维，画法统一。
characters/ 不进仓库，没有皮肤文件时页面照旧用 2D 小人。
"""
import json
import os


def load(ids, enabled=True):
    """{id: 皮肤数据}，没有皮肤文件的角色不在里面；enabled=False 或一个都没有时返回 None（页面不启用三维皮肤）。"""
    if not enabled:
        return None
    out = {}
    for rid in dict.fromkeys(ids):
        p = os.path.join("characters", f"{rid}.skin.json")
        if os.path.exists(p):
            out[rid] = json.load(open(p, encoding="utf-8"))
    if out:
        print("三维皮肤：", "、".join(out))
    return out or None
