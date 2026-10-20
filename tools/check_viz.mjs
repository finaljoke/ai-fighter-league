// 画面和模拟对不对得上：node tools/check_viz.mjs [--skin] <比赛.json> [更多.json]
// --skin：有三维皮肤的角色（characters/<id>.skin.json）按皮肤的骨长和身体尺寸核对，和皮肤画面一致
// json 是 {sim: boxing.viz_sim(), card: {bodies, frames}}，由 tools/viz_check.py 生成。逐拳核对"所见即模拟"：
//   打中：结算那一帧拳套正好贴在对手身上（头部拳贴着头）
//   被挡：拳套停在对手的护手（或小臂、肩）上
//   被躲开：拳套到了头原来的位置，头已经不在那里
//   打空：射程不够的，拳套离对手还差模拟算出来的那一截
//   拳到之前一帧还没到；脚不交叉；胳膊没有为了够到落点被拉长
import { readFileSync, existsSync } from 'node:fs';
const VIZ = new Function(readFileSync(new URL('../viz.js', import.meta.url), 'utf8') + ';return VIZ;')();
// 出场角色的皮肤（只要骨长和身体尺寸，贴图不读）
const skinsOf = bodies => Object.fromEntries(bodies.map(b => [b.id, new URL(`../characters/${b.id}.skin.json`, import.meta.url)])
  .filter(([, u]) => existsSync(u)).map(([id, u]) => { const s = JSON.parse(readFileSync(u, 'utf8')); return [id, { ...s, tex: null, parts: [] }]; }));
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], len = a => Math.hypot(a[0], a[1], a[2]);
const OC = ['', '打中', '被挡', '被躲开', '打空'];
let fails = 0;
const check = (ok, msg) => { if (!ok) { fails++; if (fails <= 40) console.log('  ✗ ' + msg); } return ok; };
const args = process.argv.slice(2), SKIN = args.includes('--skin');
for (const path of args.filter(a => a !== '--skin')) {
  const D = JSON.parse(readFileSync(path, 'utf8'));
  VIZ.setSim(D.sim);
  if (SKIN) VIZ.setSkins(skinsOf(D.card.bodies));
  const rep = VIZ.prep({ frames: D.card.frames, bodies: D.card.bodies });
  const n = rep.frames.length, stat = { 1: [], 2: [], 3: [], 4: [] }, lunge = [];
  let early = 0, stretched = 0, total = 0;
  for (let k = 0; k < 2; k++) {
    const d = 1 - k, Rd = VIZ.rigOf(rep, d), Ra = VIZ.rigOf(rep, k);
    for (const E of rep._mo[k].punches) {
      if (!E.oc) continue;
      total++;
      const Ja = VIZ.armsOf(rep, E.t1, k, VIZ.bodyPose(rep, E.t1, k));
      const Jd = VIZ.bodyPose(rep, E.t1, d); VIZ.armsOf(rep, E.t1, d, Jd, Ja);
      const G = Ja.gloves[E.hand], K = VIZ.targetOf(rep, k, E), off = len(sub(G, K.T));
      const tag = `${k}号 第${E.t1}帧 ${D.sim.actions[7 + E.p]} ${OC[E.oc]}`;
      if (!check(off < 0.012, `${tag}：拳套离该在的位置 ${(off * 100).toFixed(1)} 厘米（胳膊够不着）`)) stretched++;
      lunge.push(E.lunge);
      const head = len(sub(G, Jd.head)) - Rd.headR - (Ra.gloveTip || Ra.glove);      // 拳面到头表面（皮肤的拳套比宽长，按拳面算）
      if (E.oc === 1 && E.tgt === 0) { stat[1].push(head); check(head < 0.04 && head > -0.06, `${tag}：打中了头，拳套和头隔着 ${(head * 100).toFixed(1)} 厘米`); }
      if (E.oc === 2) {
        // 头部拳挡在拳套（或肩卷的前肩）上，身体拳挡在小臂、手肘上：量拳套到对方两条小臂（肘到拳套那一段）的距离
        const seg = (p, a, b) => { const ab = sub(b, a), t = Math.max(0, Math.min(1, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / (len(ab) ** 2 || 1))); return len(sub(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t])); };
        const g = E.tgt === 0 ? Math.min(...Jd.gloves.map(q => len(sub(G, q))), len(sub(G, Jd.sh[0]))) - 2 * Ra.glove
          : Math.min(...[0, 1].map(h => seg(G, Jd.elbows[h], Jd.gloves[h]))) - Ra.glove - 0.04;
        stat[2].push(g); check(g < 0.12 && g > -0.08, `${tag}：被挡，拳套和挡它的护手（小臂）隔着 ${(g * 100).toFixed(1)} 厘米`);
      }
      if (E.oc === 3 && E.tgt === 0) { stat[3].push(head); check(head > 0.02, `${tag}：被躲开，拳套却碰到了头（${(head * 100).toFixed(1)} 厘米）`); }
      if (E.oc === 4 && E.tgt === 0) {
        const a0 = rep.frames[E.t1 - 1].f[k], d0 = rep.frames[E.t1 - 1].f[d];
        const gap = Math.hypot(d0[0] - a0[0], d0[1] - a0[1]) - VIZ.reachLim(rep, k, E);
        if (gap > 0) { stat[4].push(head - gap); check(head > gap * 0.6 - 0.03, `${tag}：射程差 ${(gap * 100).toFixed(0)} 厘米，画面上只差 ${(head * 100).toFixed(0)} 厘米`); }
      }
      // 拳到之前一帧，拳套还在路上（还差这一拳行程的一成半以上）
      if (E.t1 - E.ta >= 2) {
        const Jp = VIZ.armsOf(rep, E.t1 - 1, k, VIZ.bodyPose(rep, E.t1 - 1, k));
        const J0 = VIZ.armsOf(rep, E.ta, k, VIZ.bodyPose(rep, E.ta, k));
        if (len(sub(Jp.gloves[E.hand], K.T)) < 0.15 * len(sub(J0.gloves[E.hand], K.T))) early++;
      }
    }
  }
  // 脚：前脚一直在前、两脚不交叉
  let crossed = 0, frames = 0;
  for (let k = 0; k < 2; k++) for (let t = 0; t < n; t += 2) {
    const f = rep.frames[t].f[k];
    if (f[7] === 7) continue;
    const J = VIZ.bodyPose(rep, t, k), lead = J.ls > 0 ? 0 : 1;
    const fwd = q => (q[0] - J.x) * Math.cos(J.th) + (q[1] - J.y) * Math.sin(J.th);
    frames++;
    if (fwd(J.ankles[lead]) - fwd(J.ankles[1 - lead]) < 0.05) crossed++;
  }
  const q = (a, p) => a.length ? a.slice().sort((x, y) => x - y)[Math.floor((a.length - 1) * p)] : NaN;
  const cm = v => (v * 100).toFixed(1);
  console.log(`${path}\n  ${n} 帧，核对 ${total} 拳（打中 ${stat[1].length} 头部，被挡 ${stat[2].length}，被躲开 ${stat[3].length}，射程不够打空 ${stat[4].length}）`);
  if (stat[1].length) console.log(`  打中头部：拳套表面到头表面 中位 ${cm(q(stat[1], 0.5))} 厘米，最远 ${cm(Math.max(...stat[1]))}`);
  if (stat[2].length) console.log(`  被挡：拳套到对方护手 中位 ${cm(q(stat[2], 0.5))} 厘米`);
  if (stat[3].length) console.log(`  被躲开：拳套离头 最近 ${cm(Math.min(...stat[3]))} 厘米`);
  if (stat[4].length) console.log(`  射程不够：画面差距 − 模拟差距 中位 ${cm(q(stat[4], 0.5))} 厘米`);
  console.log(`  往前送步：中位 ${cm(q(lunge, 0.5))}、九成 ${cm(q(lunge, 0.9))}、最大 ${cm(Math.max(...lunge))} 厘米；胳膊够不着 ${stretched} 拳；拳提前一帧就到了 ${early} 拳`);
  console.log(`  脚：${frames} 帧里前后脚交叉或并拢 ${crossed} 帧`);
  check(early === 0, `${early} 拳提前到了`);
  check(crossed <= frames * 0.01, `脚交叉 ${crossed} 帧`);
}
console.log(fails ? `\n${fails} 项没对上` : '\n全部对上');
process.exit(fails ? 1 : 0);
