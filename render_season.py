"""拿已经存档的赛季数据重新生成观看页（改了画面代码以后用，不用重跑比赛和训练）。

用法：py -X utf8 render_season.py --name s0
"""
import argparse
import json
import os


def build_html(data, out_html):
    tpl = open("season_template.html", encoding="utf-8").read().replace("/*__VIZ__*/", open("viz.js", encoding="utf-8").read())
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.abspath(out_html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    data = json.load(open(os.path.join("runs", f"season_{args.name}", "season.json"), encoding="utf-8"))
    print("赛季观看页:", build_html(data, os.path.join("runs", f"season_{args.name}.html")))


if __name__ == "__main__":
    main()
