# AI 格斗家联赛 · 原型

`runs/`（训练产出、版本存档、报告、回放）不在仓库里，跑训练和分析脚本会自动生成。

两个阶段：

- **v1 一维原型**：只回答"只给不同的身体条件、给一样的奖励，能不能长出看得出区别的打法"。结论：能（报告 `runs/report_s1-s2-s3.html`）。
- **v2 二维拳击**：先能打拳击，再延伸到别的流派。12 种拳、7 种防守、步法、虚晃，每个技术都要起到和现实中相同的作用。
  胜负只有 KO / TKO / 点数判定；体力、体能、心态都是资源池。规则全文见 `docs/拳击技术规格.md`。

## 二维拳击

```powershell
cd fighter_league
py -X utf8 test_techniques.py                     # 技术验收：每个技术的作用都写成了测试，必须全过
$env:OPENBLAS_NUM_THREADS=1
py -X utf8 train.py --run b1 --sim boxing --seed 1 --iters 1000   # 训练（每 20 轮存一个版本并自测）
py -X utf8 analyze_boxing.py --run b1               # 报告：技术使用表、战绩、打法对比、成长曲线、回放
py -X utf8 calibrate.py --run b1                    # 真实度校准：命中率、出拳量、累垮等对照真实比赛的目标
```

报告在 `runs/boxing_report_b1_v<版本>.html`。训练没跑完也能对任意已存版本出报告（`--version 12`）。

| 文件 | 干什么 |
|---|---|
| `boxing.py` | 二维拳击模拟器：12 种拳、防守、步法、虚晃、缠抱、架势、站架、倒地读秒、三位裁判 10 分制、心态、读招记忆、熟练度、性格、心智、战术计划 |
| `schools.py` | 流派数据、性格和心智的读取 |
| `test_techniques.py` | 技术验收测试 |
| `boxing_bots.py` | 拳击标准陪练（量尺） |
| `boxing_league.py` | 比赛、打法指标、技术使用表、回放 |
| `analyze_boxing.py` + `boxing_report_template.html` | 拳击报告（俯视回放播放器） |
| `calibrate.py` | 真实度校准：按 CompuBox 口径统计，对照规格文档"真实度目标"打勾 |

角色模型（全文见规格文档"角色模型"一节）：
身体属性、天赋熟练度、性格（胆量 / 沉着 / 斗志 / 冲动）、心智（观察力 / 拳商）、流派（`schools.py`）、
熟练度（随训练演化）、习惯、当天状态、战术计划（角落赛前模拟选、回合间调整）、架势（4 种，可换）、站架（正反架，可换），
都会影响这个角色在这一局的发挥。角色定义在 `bodies.py`，流派在 `schools.py`。

## 赛季、训练营与观看页

```powershell
py -X utf8 test_career.py                                   # 训练营和赛事流程的验收测试
py -X utf8 season.py --run b1 --name s0                     # 接手训练好的角色，跑第 0 季：开季一战 + 两个 35 天训练营
py -X utf8 render_season.py --name s0                       # 改了画面代码后，只重新生成观看页（不重跑比赛）
node tools/export_video.mjs runs/season_s0.html runs/第1战.mp4 --items 0                 # 导出一场比赛
node tools/export_video.mjs runs/season_s0.html runs/训练营.mp4 --items 1-35 --day-len 3  # 导出训练营（每天 3 秒）
py -X utf8 make_episode.py --run canon_s1,canon_s1b          # 试播集页面（正史分几段练的按先后列出）
# 正史第三段（首战之后，box2d-v15 规则）：从 canon_s1b 第 550 轮接着两人一起练，只练一次，练出什么就是什么
py -X utf8 train.py --run canon_s1c --sim boxing --bodies GY,LJ --seed 3 --iters 300 --init canon_s1b:12 --record 2 --coach-frac 0.3 --vf-warmup 10 --gamma 0.999 --ent 0.005 --ent-end 0.003 --snap-every 25 --eval-n 32 --point-reward 0 --kd-reward 0
node tools/export_video.mjs runs/episode_pilot.html runs/试播集.mp4 --items all
# 画面核对：编排好的小场景（闪躲、拍挡、格挡、射程、倒地、缠抱、步法）+ 正史存档打一场，逐拳核对画面和模拟对不对得上
py -X utf8 tools/viz_check.py --page                        # 另出一个能逐个场景看的核对页 runs/viz_check.html
node tools/preview_frames.mjs runs/viz_check.html 截图目录 "0@3.45@1.8"   # 条目@秒@放大倍数，放大只有核对页认
py -X utf8 tools/enrich_frames.py                           # 旧回放存好的帧补上画面字段（用 git 里当时那一版规则重算）
node tools/audit_rig.mjs                                    # 小人体检：关节跳变、够不够得着、穿插、遮挡闪烁，报数和例子
# 三维皮肤（验证中）：三视图立绘做成网格，关节点标在 characters/<id>.rig.json；要装 pillow 和 scipy
py -X utf8 tools/build_skin.py LJ --debug 核对目录           # 生成 characters/LJ.skin.json，核对目录里是截面描回立绘的图
py -X utf8 tools/viz_check.py --page --skin                 # 核对页里的人换成三维皮肤：runs/viz_check_skin.html
```

导出视频要先在 `tools/` 下 `npm install`（只装 puppeteer-core，用本机的 Edge），本机要有 ffmpeg，Python 要装 numpy 和 edge-tts。

视频默认带声音：击打、格挡、挥空、回合铃、观众（跟着攻防起落）、训练场的器械和节拍，全部程序合成；中文解说用 edge-tts 生成，缓存在 `tools/voice_cache/`，连不上网就只出音效。`--mute 1` 出无声版，`--keep 1` 保留声音事件表和音轨方便调音。

| 文件 | 干什么 |
|---|---|
| `career.py` | 角色在两场比赛之间的生活：训练项目、疲劳伤病、睡眠饮食、称重、比赛夜、体检 |
| `season.py` / `render_season.py` | 跑一个赛季并存档 / 用存档重新生成观看页 |
| `viz.js` | 转播画面渲染。拳手是三维骨骼：比赛里按模拟的位置、朝向和动作时间摆好再用机位投影，所见即模拟（规则见规格文档"观看体验·所见即模拟"）；训练馆、开场卡里用同一副骨骼按训练项目摆姿势。另有多机位导播（整场预先排好机位，两人不叠在一起）、擂台、开场卡、回合统计、读秒、慢镜回放、判定、训练场景。观看页和导出视频共用 |
| `tools/viz_check.py` + `tools/check_viz.mjs` | 画面核对：每一拳在模拟结算那一帧，拳套是不是正好在落点上、被挡的是不是停在护手上、被躲开的是不是没碰到头、打空的是不是差了模拟算出来的那一截；拳不提前到、胳膊不拉长、脚不交叉 |
| `tools/enrich_frames.py` | `runs/replay_frames/` 里旧回放的帧是 v15 以前存的，没有落点这些画面字段。从 git 取当时那一版 `boxing.py` 重算，原来的项一项不差才另存 `<key>.viz.json`，原来的帧不动 |
| `season_template.html` | 赛季观看页（时间线、比赛、训练场、疲劳曲线） |
| `tools/export_video.mjs` | 把观看页的画面逐帧导出成 mp4，再配上音轨 |
| `make_episode.py` + `episode_template.html` | 试播集：片头、人物介绍、成长片段（出生和首战前用 `train.py --record` 存档节点另打的录像；中间几件大事是里程碑的版本重演：训练里那一场没录像，用当时的出招网络按当时的对手重打到同样的事发生，第一次找到的那场存进 `runs/<run>/milestone_reenact.json`，以后只读存档）、首战（只打一次存档，结果不重来）、片尾。人设和立绘读本地 `characters/`，不进仓库 |
| `tools/preview_frames.mjs` | 抽几帧截图看画面，不用导整片 |
| `tools/make_audio.py` | 按声音事件表合成音轨（音效、观众、训练节拍、解说），事件表由 `viz.js` 从同一条时间线算出 |
| `tools/build_skin.py` | 三维皮肤（验证中）：从三视图量出每段身体沿骨骼一圈圈的截面，算出每个顶点在正面、侧面、背面上的贴图坐标。`viz.js` 的 `SKIN` 每帧按骨骼摆好截面、用 WebGL 画，页面调了 `setSkins` 才启用，目前只有比赛画面用；没有皮肤的角色画成素色三维 |

## 一维原型

## 跑起来

只依赖 numpy（`py -m pip install numpy`），CPU 就够，不用显卡。

```powershell
cd fighter_league
# 训练（每个种子 300 轮约 2000 万帧，单进程约 25~40 分钟；多个种子可以并行跑）
$env:OPENBLAS_NUM_THREADS=1
py -X utf8 train.py --run s1 --seed 1 --iters 300
py -X utf8 train.py --run s2 --seed 2 --iters 300
py -X utf8 train.py --run s3 --seed 3 --iters 300
# 分析 + 报告（第一个是主种子，其余当对照）
py -X utf8 analyze.py --runs s1 s2 s3
```

报告在 `runs/report_s1-s2-s3.html`，浏览器直接打开。

## 文件

| 文件 | 干什么 |
|---|---|
| `bodies.py` | 身体参数：实验体 A（长臂轻快）、B（短臂拳重抗揍）、标准陪练 |
| `sim.py` | 一维擂台模拟器，向量化，完全确定 |
| `nets.py` | 纯 numpy 的小网络和 PPO |
| `bots.py` | 标准陪练，打法写死，当量尺用 |
| `train.py` | 自我对练训练 + 记录系统 |
| `league.py` | 比赛、打法指标、打法指纹、回放存取 |
| `analyze.py` | 打法分化测试，生成报告 |
| `report_template.html` | 报告页模板（指纹热力图、成长曲线、回放播放器） |

## 规则

- **身体**：臂展、体重、力量、手速、步速、体能、抗击打、反应时间。反应时间的实现方式是让它看到的对手状态延迟几帧。
- **动作**：不动、前进、后退、刺拳、重拳、格挡、举着架子往前压。
- **招式**：每一招都有起手、出拳、收招的帧数。挨打会硬直，打在对方起手上伤害加成。贴得太近时长臂打不开，伤害打折。出拳耗体能，体能越低伤害越低、步子越慢。
- **奖励**：A、B 完全一样：打出的伤害减去挨的伤害，赢了 +1，输了 −1。另有一条裁判催战：两人离得比 2 米还远时，双方每帧都扣一点。

## 记录系统（从第一天就做）

每个 run 一个目录 `runs/<run>/`，只追加，不覆盖（同名 run 会拒绝启动）：

- `ckpt/A_v0012.npz`：每个版本都留着，新秀版本也在
- `versions.jsonl`：版本号、训练帧数、sha256。正式比赛前公布哈希，观众可以核对没换人
- `train_log.jsonl`：每轮训练的胜负、熵、KL
- `evals.jsonl`：每个版本对标准陪练、对对手的战绩和打法指标，成长曲线就从这里画
- `matches/*.json`：回放档，只存初始距离和双方逐帧动作。模拟器是确定的，画面随时可以重新算，换机位、加特效都不用重新比赛
- `milestones.jsonl` + `ckpt/<角色>_ms<轮数>.npz`（拳击版 `--record`）：第一次打中、第一次击倒之类的里程碑，记下对手是谁，并把那一刻两边的出招网络和熟练度另存一份，做节目时用它重演
- `runs/replay_frames/`：试播集用到的回放算出来的画面帧。规则改版以后，旧回放在新规则下重算会走样，就读这里存好的，正史首战永远是当初那一场。`<key>.viz.json` 是补上画面字段的同一批帧（`tools/enrich_frames.py`），有就优先读

## 已知局限

- 一维擂台，没有横向移动，被逼到围绳就只能硬打，所以外围型吃亏。要做真正的绕步、切角得上二维擂台（v2）。
- 网络很小、训练量也小，动作会显得笨拙。这正好是"新秀期"。
