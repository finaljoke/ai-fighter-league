"""拿已经存档的赛季数据重新生成观看页（改了画面代码以后用，不用重跑比赛和训练）。

用法：py -X utf8 render_season.py --name s0 [--no-skin]
"""
import argparse
import json
import os

import skins as SK


def build_html(data, out_html, skin=True):
    """赛季数据 → 观看页。出场角色有三维皮肤（characters/<id>.skin.json）就带上，skin=False 画回 2D 小人。"""
    ids = list(data.get("final", {})) + [fid for fg in data.get("fights", []) for fid in fg.get("fighters", [])]
    tpl = open("season_template.html", encoding="utf-8").read().replace("/*__VIZ__*/", open("viz.js", encoding="utf-8").read())
    html = tpl.replace("/*__DATA__*/null", json.dumps({**data, "skins": SK.load(ids, skin)}, ensure_ascii=False, separators=(",", ":"),
                                                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.abspath(out_html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-skin", action="store_true", help="不用三维皮肤，画回 2D 小人")
    args = ap.parse_args()
    data = json.load(open(os.path.join("runs", f"season_{args.name}", "season.json"), encoding="utf-8"))
    print("赛季观看页:", build_html(data, os.path.join("runs", f"season_{args.name}.html"), not args.no_skin))


if __name__ == "__main__":
    main()
