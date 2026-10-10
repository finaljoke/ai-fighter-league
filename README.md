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
node tools/export_video.mjs runs/episode_pilot.html runs/试播集.mp4 --items all
```

导出视频要先在 `tools/` 下 `npm install`（只装 puppeteer-core，用本机的 Edge），本机要有 ffmpeg，Python 要装 numpy 和 edge-tts。

视频默认带声音：击打、格挡、挥空、回合铃、观众（跟着攻防起落）、训练场的器械和节拍，全部程序合成；中文解说用 edge-tts 生成，缓存在 `tools/voice_cache/`，连不上网就只出音效。`--mute 1` 出无声版，`--keep 1` 保留声音事件表和音轨方便调音。

| 文件 | 干什么 |
|---|---|
| `career.py` | 角色在两场比赛之间的生活：训练项目、疲劳伤病、睡眠饮食、称重、比赛夜、体检 |
| `season.py` / `render_season.py` | 跑一个赛季并存档 / 用存档重新生成观看页 |
| `viz.js` | 转播画面渲染：拳手骨骼与造型、动作、跟拍镜头、擂台、开场卡、回合统计、读秒、慢镜回放、判定、训练场景。观看页和导出视频共用 |
| `season_template.html` | 赛季观看页（时间线、比赛、训练场、疲劳曲线） |
| `tools/export_video.mjs` | 把观看页的画面逐帧导出成 mp4，再配上音轨 |
| `make_episode.py` + `episode_template.html` | 试播集：片头、人物介绍、成长片段（`train.py --record` 的成长档案和里程碑）、首战（只打一次存档，结果不重来）、片尾。人设和立绘读本地 `characters/`，不进仓库 |
| `tools/preview_frames.mjs` | 抽几帧截图看画面，不用导整片 |
| `tools/make_audio.py` | 按声音事件表合成音轨（音效、观众、训练节拍、解说），事件表由 `viz.js` 从同一条时间线算出 |

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

## 已知局限

- 一维擂台，没有横向移动，被逼到围绳就只能硬打，所以外围型吃亏。要做真正的绕步、切角得上二维擂台（v2）。
- 网络很小、训练量也小，动作会显得笨拙。这正好是"新秀期"。
