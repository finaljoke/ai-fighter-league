// 小人模型体检：node tools/audit_rig.mjs（先跑 py -X utf8 tools/viz_check.py 生成 runs/viz_check/*.json）
// 每 1/20 帧采样所有核对场景和第 0 季的比赛，量：关节跳变、骨长、拳套和脚够不够得着、膝盖手肘反弯、自己的拳套陷进自己身体、
// 两个人互相穿插、头部画法来回跳。只报数和例子，不判对错：有的是模拟本身的行为（AI 只举一帧架、两人贴到 0.6 米），见 docs/赛季与训练场规格.md
import { readFileSync, existsSync } from 'node:fs';
const ROOT = new URL('..', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
const VIZ = new Function(readFileSync(ROOT + 'viz.js', 'utf8') + ';return VIZ;')();
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], len = a => Math.hypot(a[0], a[1], a[2]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const segDist = (p, a, b) => { const ab = sub(b, a), t = Math.max(0, Math.min(1, dot(sub(p, a), ab) / (dot(ab, ab) || 1))); return len(sub(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t])); };
function segSeg(p1, q1, p2, q2) { let best = 1e9; for (let i = 0; i <= 8; i++) { const a = [0, 1, 2].map(j => p1[j] + (q1[j] - p1[j]) * i / 8); best = Math.min(best, segDist(a, p2, q2)); } return best; }

const sets = [];
for (const f of ['闪躲与拍挡', '格挡', '射程', '步法', '缠抱', '头部击倒', '肝区击倒', '正史比赛', '旧回放_正史首战']) {
  const D = JSON.parse(readFileSync(ROOT + `runs/viz_check/${f}.json`, 'utf8'));
  sets.push({ name: f, sim: D.sim, card: D.card });
}
const S0 = ROOT + 'runs/season_s0/season.json';
if (existsSync(S0)) { const D = JSON.parse(readFileSync(S0, 'utf8')); D.fights.forEach((fg, i) => sets.push({ name: `第0季第${i + 1}战`, sim: D.sim, card: fg })); }

const H = 0.05;
const JOINTS = ['pel', 'chest', 'head', 'shL', 'shR', 'hipL', 'hipR'];
// 胳膊按解剖学的左右比（换站架时前后手对调，按前后手比会误报跳变）
const get = (J, n) => { if (!n.includes('.')) return J[n]; const [a, i] = n.split('.'); const idx = (a === 'gloves' || a === 'elbows') ? (((+i === 0) === (J.ls > 0)) ? 0 : 1) : +i; return J[a][idx]; };
const ALL = [...JOINTS, 'knees.0', 'knees.1', 'ankles.0', 'ankles.1', 'elbows.0', 'elbows.1', 'gloves.0', 'gloves.1'];
const R = {};          // 类别 → {n, ex: []}
const add = (cat, ex, val) => { const r = R[cat] || (R[cat] = { n: 0, ex: [], max: -1e9 }); r.n++; if (val > r.max) r.max = val; if (r.ex.length < 6) r.ex.push(ex); else if (val > (r.exv || 0)) { r.ex[5] = ex; r.exv = val; } };
let samples = 0;
const ctxOf = (rep, k, tf) => {
  const t = Math.floor(tf), f = rep.frames[t].f[k], mo = rep._mo[k], c = [];
  const ph = ['自由', '起手', '收招', '被打中', '格挡硬直', '防守', '缠抱', '倒地', '移动'][f[7]];
  c.push(ph + (f[7] === 8 ? '·' + ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '左轴转', '右轴转', '', '换架势', '换站架'][f[8]] : ''));
  for (let h = 0; h < 2; h++) { const E = VIZ.prep && rep._mo[k].byHand[h][rep._mo[k].ptr[h][t]]; if (E && tf >= E.t0 && tf < E.t2) c.push(`${h ? '后' : '前'}手${E.feint ? '虚晃' : ['直', '勾', '上勾'][E.typ]}${tf < E.ta ? '预备' : tf < (E.cut ?? E.t1) ? '出拳' : '收'}${E.cut !== undefined ? '(被打断)' : ''}`); }
  const D = mo.defs[mo.dptr[t]]; if (D && tf >= D.t0 && tf < D.t2) c.push(['拍挡', '侧闪', '下潜', '后仰'][D.m - 13]);
  const Hh = mo.hits[mo.hptr[t]]; if (Hh && tf - Hh.t < 3) c.push(Hh.E.oc === 1 ? '刚挨打' : '刚被挡');
  if (mo.liver[t]) c.push('弯腰');
  const P = rep._prep; if (P.upAt[t][k] > 0 && tf - P.upAt[t][k] < 12) c.push('站起来');
  if (t + 1 < rep.frames.length && rep.frames[t + 1].r !== rep.frames[t].r) c.push('回合交界');
  return c.join('+');
};

for (const set of sets) {
  VIZ.setSim(set.sim);
  const rep = VIZ.prep({ frames: set.card.frames, bodies: set.card.bodies });
  const n = rep.frames.length, Rg = [0, 1].map(k => VIZ.rigOf(rep, k));
  let prev = null, prevOrder = null, prevMode = [null, null], modeLast = [[], []];
  for (let si = 0; si < (n - 1) / H; si++) {
    const tf = si * H;
    samples++;
    const Js = [0, 1].map(k => VIZ.bodyPose(rep, tf, k));
    Js.forEach((J, k) => VIZ.armsOf(rep, tf, k, J, Js[1 - k]));
    const t = Math.floor(tf), down = Js.map(J => J.down !== undefined);
    const tag = (k, extra = '') => `${set.name} ${k}号 第${tf.toFixed(2)}帧 ${ctxOf(rep, k, tf)}${extra}`;
    for (let k = 0; k < 2; k++) {
      const J = Js[k], Rk = Rg[k];
      // A. 跳变：1/20 帧里挪了 5 厘米以上（每秒 20 米）
      if (prev && !(t + 1 < n && rep.frames[Math.floor(tf - H) + 1] && rep.frames[Math.floor(tf - H) + 1].r !== rep.frames[Math.floor(tf - H)].r)) {
        for (const j of ALL) {
          const d = len(sub(get(J, j), get(prev[k], j)));
          if (d > 0.05) add(`A 跳变·${j.replace(/\.\d/, '')}·${ctxOf(rep, k, tf).replace(/\(.*\)/, '')}`, tag(k, ` ${j} ${(d * 100).toFixed(0)}cm`), d);
        }
      }
      if (down[k]) continue;
      // B. 骨长
      for (let h = 0; h < 2; h++) {
        const u = len(sub(J.elbows[h], J.sh[h])), f = len(sub(J.gloves[h], J.elbows[h]));
        if (Math.abs(u - Rk.ua) > 0.01 || Math.abs(f - Rk.fa) > 0.01) add('B 胳膊骨长不对', tag(k), Math.max(Math.abs(u - Rk.ua), Math.abs(f - Rk.fa)));
        // C1. 胳膊够不着目标（拉直了还差）
        const miss = len(sub(J.gloves[h], J.gloveT[h]));
        if (miss > 0.02) add(`C 拳套够不着目标·${ctxOf(rep, k, tf).replace(/\(.*\)/, '')}`, tag(k, ` ${h ? '后' : '前'}手差${(miss * 100).toFixed(0)}cm`), miss);
      }
      for (let i = 0; i < 2; i++) {
        const hip = i ? J.hipR : J.hipL, th = len(sub(J.knees[i], hip)), sh = len(sub(J.ankles[i], J.knees[i]));
        if (Math.abs(th - Rk.thigh) > 0.01 || Math.abs(sh - Rk.shin) > 0.01) add('B 腿骨长不对', tag(k), 1);
        const miss = len(sub(J.ankles[i], J.ankT[i]));
        if (miss > 0.02) add(`C 脚够不着落脚点（腿拉直，脚悬空或被拖走）·${ctxOf(rep, k, tf).replace(/\(.*\)/, '')}`, tag(k, ` ${i ? '右' : '左'}脚差${(miss * 100).toFixed(0)}cm`), miss);
        // F. 膝盖反弯：膝盖相对髋-踝连线往后凸
        const mid = [(hip[0] + J.ankles[i][0]) / 2, (hip[1] + J.ankles[i][1]) / 2, (hip[2] + J.ankles[i][2]) / 2];
        const bend = sub(J.knees[i], mid), fw = [J.c, J.s, 0], toe = [Math.cos(J.footYaw[i]), Math.sin(J.footYaw[i]), 0];
        if (dot(bend, fw) < -0.03) add('F 膝盖往后弯（相对身体朝向）', tag(k, ` ${i ? '右' : '左'}膝 ${(dot(bend, fw) * 100).toFixed(0)}cm`), -dot(bend, fw));
        if (dot(bend, toe) < -0.03) add('F 膝盖和脚尖方向相反', tag(k), -dot(bend, toe));
      }
      // G. 手肘：护手时手肘高过肩
      for (let h = 0; h < 2; h++) {
        const E = J.elbows[h], Sh = J.sh[h];
        const act = rep._mo[k].byHand[h][rep._mo[k].ptr[h][t]], punching = act && tf >= act.t0 && tf < act.t2;
        if (!punching && E[2] > Sh[2] + 0.05 && rep.frames[t].f[k][7] !== 6) add('G 护手时手肘高过肩', tag(k, ` ${h ? '后' : '前'}手`), E[2] - Sh[2]);
        if (prev && !down[k]) {
          const ax = sub(J.gloves[h], Sh), ax0 = sub(prev[k].gloves[h], prev[k].sh[h]);
          const perp = (E2, S2, a) => { const v = sub(E2, S2), l = dot(a, a) || 1; return sub(v, a.map(x => x * dot(v, a) / l)); };
          const p1 = perp(E, Sh, ax), p0 = perp(prev[k].elbows[h], prev[k].sh[h], ax0);
          if (len(p1) > 0.05 && len(p0) > 0.05 && dot(p1, p0) / (len(p1) * len(p0)) < 0.3) add('G 手肘翻面（弯的方向突然变）', tag(k, ` ${h ? '后' : '前'}手`), 1);
        }
      }
      // E. 自己的拳套陷进自己身体
      for (let h = 0; h < 2; h++) {
        const g = J.gloves[h], dh = len(sub(g, J.head)) - Rk.headR - Rk.glove, dt = segDist(g, J.pel, J.chest) - Rk.chestD - Rk.glove;
        if (dh < -0.07) add('E 自己的拳套陷进自己脸里', tag(k, ` ${h ? '后' : '前'}手 ${(dh * 100).toFixed(0)}cm`), -dh);
        if (dt < -0.06) add('E 自己的拳套陷进自己躯干', tag(k, ` ${h ? '后' : '前'}手 ${(dt * 100).toFixed(0)}cm`), -dt);
      }
      // H. 头部画法（侧面 / 正面 / 背面）来回跳
      const cam = rep._cam[t], rot = cam[3], ang = J.faceAng + rot * Math.PI / 2;
      const cs = Math.cos(ang), sn = Math.sin(ang), mode = sn > 0.6 ? '背面' : Math.abs(cs) > 0.5 ? (cs > 0 ? '侧面右' : '侧面左') : '正面';
      if (prevMode[k] && mode !== prevMode[k]) {
        const ml = modeLast[k]; ml.push(tf); while (ml.length && tf - ml[0] > 10) ml.shift();
        add('H 头部画法切换', tag(k, ` ${prevMode[k]}→${mode}`), 1);
        if (ml.length >= 3) add('H 头部画法半秒内来回跳 3 次以上', tag(k, ` ${prevMode[k]}→${mode}`), ml.length);
      }
      prevMode[k] = mode;
    }
    // D. 两个人互相穿插（倒地的不算）
    if (!down[0] && !down[1]) {
      const [a, b] = Js, ra = Rg[0], rb = Rg[1];
      const hh = len(sub(a.head, b.head)) - ra.headR - rb.headR;
      if (hh < -0.04) add('D 两个人的头叠在一起', tag(0, ` ${(hh * 100).toFixed(0)}cm`), -hh);
      const tt = segSeg(a.pel, a.chest, b.pel, b.chest) - ra.chestD - rb.chestD;
      if (tt < -0.06) add('D 两个人的躯干插在一起', tag(0, ` ${(tt * 100).toFixed(0)}cm ${ctxOf(rep, 1, tf)}`), -tt);
      for (const [x, y, rx, ry, kx] of [[a, b, ra, rb, 0], [b, a, rb, ra, 1]]) {
        const th = segDist(x.head, y.pel, y.chest) - rx.headR - ry.chestD;
        if (th < -0.05) add('D 头插进对方躯干', tag(kx, ` ${(th * 100).toFixed(0)}cm`), -th);
        for (let h = 0; h < 2; h++) {
          const E = rep._mo[kx].byHand[h][rep._mo[kx].ptr[h][t]];
          const near = E && E.oc && Math.abs(tf - E.t1) < 3;
          const gh = len(sub(x.gloves[h], y.head)) - ry.headR - rx.glove;
          if (gh < -0.04 && !(near && E.oc === 1 && E.tgt === 0)) add(`D 拳套插进对方头里（这一下不是打中头）·${near ? ['', '打中', '被挡', '被躲开', '打空'][E.oc] : '没在出拳'}`, tag(kx, ` ${h ? '后' : '前'}手 ${(gh * 100).toFixed(0)}cm`), -gh);
          const gt = segDist(x.gloves[h], y.pel, y.chest) - ry.chestD - rx.glove;
          if (gt < -0.05 && !(near && E.tgt === 1 && E.oc <= 2)) add(`D 拳套插进对方躯干（这一下不是打身体）·${near ? ['', '打中', '被挡', '被躲开', '打空'][E.oc] : '没在出拳'}`, tag(kx, ` ${h ? '后' : '前'}手 ${(gt * 100).toFixed(0)}cm`), -gt);
          for (let h2 = 0; h2 < 2; h2++) { const gg = len(sub(x.gloves[h], y.gloves[h2])) - rx.glove - ry.glove; if (gg < -0.05 && kx === 0) add('D 两个人的拳套叠在一起', tag(0, ` ${(gg * 100).toFixed(0)}cm`), -gg); }
        }
      }
    }
    prev = Js;
  }
}
console.log(`采样 ${samples} 次（${sets.length} 场，每 1/20 帧）\n`);
for (const [cat, r] of Object.entries(R).sort((a, b) => a[0].localeCompare(b[0]) || b[1].n - a[1].n)) {
  console.log(`${cat}：${r.n} 次，最大 ${r.max.toFixed(2)}`);
  r.ex.forEach(e => console.log('    ' + e));
}
