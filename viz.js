// AI 格斗家联赛 · 转播画面渲染。观看页和导出视频共用这一份代码，所以网页里看到的就是导出的视频。
// 输入只有回放档重算出来的逐帧数据（只存动作的回放 → 重新模拟），画面全部由代码画出。
'use strict';
const VIZ = (() => {
  // 22 起是画面专用的几项（box2d-v15 起才有）：落点、擦到、被什么挡住、肝区击倒前弯腰、心窝喘不上气、身体预备帧数、起手总帧数
  const F = { x: 0, y: 1, th: 2, hp: 3, stun: 4, stam: 5, comp: 6, ph: 7, mv: 8, g: 9, fired: 10, oc: 11, dmg: 12, act: 13, kd: 14, ev: 15, ctr: 16, fk: 17, post: 18, st: 19, tilt: 20, plan: 21,
    zone: 22, gl: 23, blk: 24, liv: 25, wind: 26, prep: 27, su: 28 };
  const PH = { FREE: 0, STARTUP: 1, RECOVERY: 2, HITSTUN: 3, BLOCKSTUN: 4, DEFEND: 5, CLINCH: 6, DOWN: 7, MOVING: 8 };
  const M = { CLINCH: 12, PARRY: 13, SLIP: 14, DUCK: 15, PULL: 16, PIVL: 17, PIVR: 18, POSTURE: 20, STANCE: 21 };
  const ACT = { GUARD_FWD: 20, FEINT_JAB: 27, FEINT_CROSS: 28 };
  let SIM = null;                       // 由页面设置：boxing.viz_sim() 给的规则常数（RING、DT、TYPE、TGT、HAND、RNG、射程系数、击退……）
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = t => t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t);
  const easeOut = t => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
  const hash = n => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

  // ---------------------------------------------------------------- 角色造型
  const LOOK = {
    A: { skin: '#e9c29c', hair: 'spiky', hairCol: '#16161c', trunk: '#1f57b8', trunk2: '#f4f4f4', glove: '#2f7de6', cuff: '#f4f4f4', shoe: '#f2f2f2', sock: '#1f57b8', brow: 0.15, nick: '长臂' },
    B: { skin: '#c98d62', hair: 'buzz', hairCol: '#2a1c14', trunk: '#b8351c', trunk2: '#f2c14e', glove: '#d9461f', cuff: '#f2c14e', shoe: '#1d1d22', sock: '#f2f2f2', brow: -0.25, nick: '重拳' },
    // 第一季正史角色：照合伙人给的立绘配色（characters/）
    GY: { skin: '#e8c4a2', hair: 'slick', hairCol: '#121216', trunk: '#17161b', trunk2: '#6b45b8', stripe: true, glove: '#1d1c22', cuff: '#efeae0', shoe: '#141417', sock: '#141417', brow: 0.35, smirk: true, accent: '#9a6be0', nick: '黑狐' },
    LJ: { skin: '#dcb18b', hair: 'shaggy', hairCol: '#0f0c0d', trunk: '#8f1a20', trunk2: '#141214', stripe: true, trim: '#c9a14a', glove: '#a11d24', cuff: '#efeae0', shoe: '#1a1416', sock: '#8f1a20', brow: -0.4, scar: true, accent: '#e0383e', nick: '赤獠' },
    S: { skin: '#d9b08c', hair: 'cap', hairCol: '#555', trunk: '#555b66', trunk2: '#999', glove: '#8a8f99', cuff: '#ddd', shoe: '#333', sock: '#ccc', brow: 0, nick: '陪练' },
  };
  const look = id => LOOK[id] || LOOK.S;
  const accent = id => { const L = look(id); return L.accent || L.glove; };
  // 体型：按身体参数算。身高、臂长跟臂展走（出拳够不够得着是按这个画的，不能动）；
  // 体格比真实差别夸大一些，远景里也一眼分得开：力量大的肩宽、胸厚、胳膊粗，抗击打好的脖子粗、下巴宽，步速快的腿细
  function build(body) {
    const reach = body ? body.reach : 0.75, mass = body ? body.mass : 72;
    const power = body ? body.power : 1, chin = body ? body.chin : 1, foot = body ? body.foot_speed : 1;
    const H = 1.72 + (reach - 0.75) * 0.75;
    const mus = clamp(1 + 1.6 * (power - 1) + 0.01 * (mass - 72), 0.65, 1.45);
    const leg = clamp(1 + 0.6 * (power - 1) + 0.012 * (mass - 72) - 0.9 * (foot - 1), 0.75, 1.3);
    return {
      H, hipH: 0.5 * H, shH: 0.81 * H, headR: 0.064 * H, neck: 0.03 * H,
      ua: 0.165 * H * (0.85 + 0.6 * (reach - 0.75)), fa: 0.15 * H * (0.85 + 0.6 * (reach - 0.75)),
      thigh: 0.25 * H, shin: 0.25 * H, mus,
      chest: 0.15 * (0.25 + 0.75 * mus) + 0.003 * (mass - 72), waist: 0.125 + 0.003 * (mass - 72),
      arm: 0.085 * mus, legW: 0.13 * leg, neckW: 0.1 * clamp(1 + 1.5 * (chin - 1), 0.7, 1.4), jaw: clamp(1 + 0.4 * (chin - 1), 0.9, 1.12),
      glove: 0.072 + (mass - 70) * 0.0005,
    };
  }

  // 两段骨骼的反向运动学：给定根和末端，求中间关节（肘、膝）。bend = +1/-1 决定往哪边弯
  function ik(ax, ay, bx, by, l1, l2, bend) {
    const dx = bx - ax, dy = by - ay;
    let d = Math.hypot(dx, dy);
    const dmax = (l1 + l2) * 0.999;
    if (d > dmax) { const k = dmax / d; bx = ax + dx * k; by = ay + dy * k; d = dmax; }
    const a = Math.atan2(by - ay, bx - ax);
    const cosA = clamp((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1);
    const ang = a + bend * Math.acos(cosA);
    return { jx: ax + Math.cos(ang) * l1, jy: ay + Math.sin(ang) * l1, ex: bx, ey: by };
  }

  function tube(c, pts, w, fill, ink, lw) {
    c.lineCap = 'round'; c.lineJoin = 'round';
    c.strokeStyle = ink; c.lineWidth = w + lw * 2; c.beginPath(); pts.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.stroke();
    c.strokeStyle = fill; c.lineWidth = w; c.beginPath(); pts.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.stroke();
  }
  // 有粗细变化的肢体（肌肉往关节收细）：pts 是沿肢体的点，ws 是每个点处的粗细（像素）。先整个描一圈墨线，再填色
  function limb(c, pts, ws, fill, ink, lw) {
    const pass = grow => {
      for (let i = 0; i < pts.length - 1; i++) {
        const [ax, ay] = pts[i], [bx, by] = pts[i + 1], L = Math.hypot(bx - ax, by - ay) || 1;
        const nx = -(by - ay) / L, ny = (bx - ax) / L, wa = ws[i] / 2 + grow, wb = ws[i + 1] / 2 + grow;
        c.beginPath(); c.moveTo(ax + nx * wa, ay + ny * wa); c.lineTo(bx + nx * wb, by + ny * wb); c.lineTo(bx - nx * wb, by - ny * wb); c.lineTo(ax - nx * wa, ay - ny * wa); c.fill();
      }
      pts.forEach((p, i) => { c.beginPath(); c.arc(p[0], p[1], ws[i] / 2 + grow, 0, Math.PI * 2); c.fill(); });
    };
    c.fillStyle = ink; pass(lw);
    c.fillStyle = fill; pass(0);
  }
  const along = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];

  // ---------------------------------------------------------------- 画一个拳手
  // 姿势用"身体局部坐标"：x 朝对手为正（米），y 从地面往上（米）。屏幕上 sx = 脚下中心 + dir * x * s。
  // pose: {crouch, lean, twist, head:[dx,dy], lead:[x,y], rear:[x,y], feet:[leadX, rearX, leadLift, rearLift],
  //        down (0..1 倒地进度), face:{hurt, tilt, sweat, look}, leadBig, rearBig, leadBody, rearBody, label}
  function drawBoxer(c, sx, sy, s, dir, id, body, pose, stance = 1) {
    const L = look(id), B = build(body), ink = '#14161c', lw = Math.max(1.2, 0.012 * s);
    const X = x => sx + dir * x * s, Y = y => sy - y * s;
    const down = pose.down || 0;
    if (down > 0.001) return drawDown(c, sx, sy, s, dir, id, body, pose);
    const crouch = pose.crouch || 0, lean = pose.lean || 0, twist = pose.twist || 0;
    const hipH = B.hipH - crouch * 0.16 * B.H;
    const hip = [lean * 0.25, hipH];
    const shC = [hip[0] + Math.sin(lean) * (B.shH - B.hipH), hipH + Math.cos(lean) * (B.shH - B.hipH)];
    // 前后肩：侧面看，转腰时后肩往前送
    const shSpan = 0.11 + B.chest * 0.25;
    const shLead = [shC[0] + shSpan * (0.55 - 0.9 * twist), shC[1] + 0.01];
    const shRear = [shC[0] - shSpan * (0.75 - 1.2 * twist), shC[1] - 0.01];
    const headC = [shC[0] + 0.02 + Math.sin(lean) * 0.12 + (pose.head ? pose.head[0] : 0), shC[1] + B.neck + B.headR * 1.1 + (pose.head ? pose.head[1] : 0)];
    // 近侧和远侧：正架面朝右时，后手在镜头这边；面朝左时，前手在镜头这边
    const nearIsRear = (dir * stance) > 0;
    const feet = pose.feet || [0.24, -0.2, 0, 0];
    const P = p => [X(p[0]), Y(p[1])];
    const leg = (fx, lift, far) => {
      const top = [hip[0] + (far ? -0.03 : 0.03), hipH], ft = [fx, 0.06 + lift];
      const k = ik(top[0], top[1], ft[0], ft[1], B.thigh, B.shin, 1);      // 膝盖往前弯
      const knee = [k.jx, k.jy], w = B.legW * s;
      const col = far ? shade(L.skin, -0.18) : L.skin;
      // 大腿 → 膝盖收细 → 小腿肚 → 脚踝
      limb(c, [top, along(top, knee, 0.4), knee, along(knee, ft, 0.3), ft].map(P), [w, w * 0.95, w * 0.62, w * 0.7, w * 0.42], col, ink, lw);
      // 袜子和鞋
      limb(c, [along(knee, ft, 0.72), ft].map(P), [w * 0.58, w * 0.5], far ? shade(L.sock, -0.2) : L.sock, ink, lw);
      c.fillStyle = far ? shade(L.shoe, -0.25) : L.shoe; c.strokeStyle = ink; c.lineWidth = lw;
      c.beginPath(); c.ellipse(X(fx + 0.04), Y(0.035 + lift), 0.11 * s, 0.045 * s, 0, 0, Math.PI * 2); c.fill(); c.stroke();
    };
    const arm = (sh, g, big, bodyShot, far, glove) => {
      const k = ik(sh[0], sh[1], g[0], g[1], B.ua, B.fa, -1);                                     // 手肘往下
      const col = far ? shade(L.skin, -0.18) : L.skin, el = [k.jx, k.jy], w = B.arm * s;
      // 三角肌 → 二头肌 → 手肘收细 → 前臂 → 手腕（被拳套盖住）
      limb(c, [sh, along(sh, el, 0.4), el, along(el, g, 0.35), g].map(P), [w * 1.15, w, w * 0.68, w * 0.74, w * 0.52], col, ink, lw);
      const r = (big ? 1.15 : 1) * B.glove * s;
      c.fillStyle = far ? shade(glove, -0.22) : glove; c.strokeStyle = ink; c.lineWidth = lw;
      c.beginPath(); c.ellipse(X(g[0]), Y(g[1]), r * 1.1, r, 0, 0, Math.PI * 2); c.fill(); c.stroke();
      c.strokeStyle = far ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.32)'; c.lineWidth = lw * 1.3;     // 皮面高光
      c.beginPath(); c.ellipse(X(g[0]), Y(g[1]), r * 0.72, r * 0.62, 0, -Math.PI * 0.85, -Math.PI * 0.35); c.stroke(); c.strokeStyle = ink;
      c.fillStyle = L.cuff; c.beginPath(); c.ellipse(X(g[0] - 0.06), Y(g[1]), r * 0.35, r * 0.75, 0, 0, Math.PI * 2); c.fill();
      if (bodyShot) { c.strokeStyle = 'rgba(255,255,255,0.6)'; c.setLineDash([3, 3]); c.beginPath(); c.arc(X(g[0]), Y(g[1]), r * 1.35, 0, Math.PI * 2); c.stroke(); c.setLineDash([]); }
    };
    const lead = pose.lead, rear = pose.rear;
    // 远侧的腿和手先画
    leg(nearIsRear ? feet[0] : feet[1], nearIsRear ? feet[2] : feet[3], true);
    if (nearIsRear) arm(shLead, lead, pose.leadBig, pose.leadBody, true, L.glove); else arm(shRear, rear, pose.rearBig, pose.rearBody, true, L.glove);
    // 躯干：胸背轮廓 + 短裤
    const tw = B.chest * (1 - 0.25 * Math.abs(twist - 0.3));
    c.fillStyle = L.skin; c.strokeStyle = ink; c.lineWidth = lw;
    c.beginPath();
    c.moveTo(X(hip[0] + B.waist), Y(hipH + 0.05)); c.quadraticCurveTo(X(shC[0] + tw * 1.35), Y(shC[1] - 0.12), X(shC[0] + tw * 0.8), Y(shC[1] + 0.03));
    c.lineTo(X(shC[0] - tw * 0.9), Y(shC[1] + 0.03)); c.quadraticCurveTo(X(shC[0] - tw * 1.2), Y(shC[1] - 0.2), X(hip[0] - B.waist), Y(hipH + 0.05));
    c.closePath(); c.fill(); c.stroke();
    // 胸肌、腹肌线：越壮越明显
    c.strokeStyle = `rgba(0,0,0,${0.06 + 0.08 * B.mus})`; c.lineWidth = lw;
    c.beginPath(); c.moveTo(X(shC[0] + tw * 1.0), Y(shC[1] - 0.05)); c.quadraticCurveTo(X(shC[0] + tw * 0.95), Y(shC[1] - 0.17), X(shC[0] + tw * 0.15), Y(shC[1] - 0.15)); c.stroke();
    c.strokeStyle = 'rgba(0,0,0,0.12)'; c.beginPath(); c.moveTo(X(shC[0] + tw * 0.55), Y(shC[1] - 0.17)); c.quadraticCurveTo(X(hip[0] + B.waist * 1.1), Y((hipH + shC[1]) / 2), X(hip[0] + B.waist * 0.8), Y(hipH + 0.08)); c.stroke();
    // 近侧的腿：画在短裤前面，大腿根被短裤盖住
    leg(nearIsRear ? feet[1] : feet[0], nearIsRear ? feet[3] : feet[2], false);
    // 短裤（腿粗的短裤也宽一点）
    const sw = Math.max(B.waist, B.legW * 0.95);
    c.fillStyle = L.trunk; c.strokeStyle = ink;
    c.beginPath(); c.moveTo(X(hip[0] - sw * 1.15), Y(hipH + 0.08)); c.lineTo(X(hip[0] + sw * 1.15), Y(hipH + 0.08));
    c.lineTo(X(hip[0] + sw * 1.3), Y(hipH - 0.2)); c.lineTo(X(hip[0] - sw * 1.3), Y(hipH - 0.2)); c.closePath(); c.fill(); c.stroke();
    c.fillStyle = L.stripe ? shade(L.trunk, -0.35) : L.trunk2; c.fillRect(Math.min(X(hip[0] - sw * 1.15), X(hip[0] + sw * 1.15)), Y(hipH + 0.08), Math.abs(X(hip[0] + sw * 1.15) - X(hip[0] - sw * 1.15)), 0.035 * s);
    if (L.stripe) {                 // 短裤外侧的竖条（立绘里的紫条 / 黑金条）
      c.fillStyle = L.trunk2; c.beginPath();
      c.moveTo(X(hip[0] + sw * 0.78), Y(hipH + 0.045)); c.lineTo(X(hip[0] + sw * 1.02), Y(hipH + 0.045));
      c.lineTo(X(hip[0] + sw * 1.18), Y(hipH - 0.2)); c.lineTo(X(hip[0] + sw * 0.9), Y(hipH - 0.2)); c.closePath(); c.fill();
      if (L.trim) { c.strokeStyle = L.trim; c.lineWidth = lw * 0.8; c.beginPath(); c.moveTo(X(hip[0] + sw * 0.78), Y(hipH + 0.045)); c.lineTo(X(hip[0] + sw * 0.9), Y(hipH - 0.2)); c.stroke(); c.strokeStyle = ink; }
    }
    // 脖子：抗击打越好越粗
    limb(c, [[shC[0] + 0.01, shC[1] - 0.03], [headC[0] - 0.015, headC[1] - B.headR * 0.55]].map(P), [B.neckW * 1.2 * s, B.neckW * s], shade(L.skin, -0.06), ink, lw);
    // 头
    drawHead(c, X(headC[0]), Y(headC[1]), B.headR * s, dir, L, { ...(pose.face || {}), jaw: B.jaw }, lw, pose.flash);
    // 近侧的手最后画
    if (nearIsRear) arm(shRear, rear, pose.rearBig, pose.rearBody, false, L.glove); else arm(shLead, lead, pose.leadBig, pose.leadBody, false, L.glove);
    if (pose.label) {
      c.font = `700 ${Math.max(11, 0.15 * s)}px system-ui, "Microsoft YaHei"`; c.textAlign = 'center';
      const ly = Y(headC[1] + B.headR * 2.4), lx = X(headC[0]);
      c.fillStyle = 'rgba(0,0,0,0.45)'; c.fillText(pose.label, lx + 1, ly + 1);
      c.fillStyle = '#ffffff'; c.fillText(pose.label, lx, ly);
    }
    return { hx: X(headC[0]), hy: Y(headC[1]), hr: B.headR * s };
  }

  // 头。psi 是脸朝向在画面里的角度：0 朝右、π 朝左、−π/2 正对镜头、π/2 背对镜头。五官按在头上的方位角投影，转到背面的不画
  function drawHead(c, hx, hy, r, dir, L, face, lw, flash) { drawHead3(c, hx, hy, r, dir > 0 ? 0 : Math.PI, L, face, lw, flash); }
  function drawHead3(c, hx, hy, r, psi, L, face, lw, flash) {
    const ink = '#14161c', cs = Math.cos(psi), sn = Math.sin(psi), dir = cs >= 0 ? 1 : -1;
    const vis = az => Math.sin(psi + az) < 0.2, ax = (az, k) => hx + Math.cos(psi + az) * r * k;
    c.fillStyle = flash ? '#ffffff' : (face.tilt ? mixHex(L.skin, '#e0503a', 0.45) : L.skin);
    c.strokeStyle = ink; c.lineWidth = lw;
    c.beginPath(); c.ellipse(hx, hy, r * 0.95 * (face.jaw || 1), r, 0, 0, Math.PI * 2); c.fill(); c.stroke();
    // 头发：侧面按发型画，正面、背面画成发顶
    if (sn > 0.9) hairFront(c, hx, hy, r, L, true);
    else if (Math.abs(cs) > 0.3) hairProfile(c, hx, hy, r, dir, L);
    else hairFront(c, hx, hy, r, L, false);
    // 耳朵
    c.fillStyle = shade(L.skin, -0.1);
    for (const az of [Math.PI / 2, -Math.PI / 2]) if (vis(az)) { c.beginPath(); c.ellipse(ax(az, 0.82) - cs * r * 0.15, hy + r * 0.05, r * 0.18, r * 0.26, 0, 0, Math.PI * 2); c.fill(); }
    if (sn > 0.9) return;
    // 眼睛、眉毛（眉尾朝鼻子那头按眉形压低或抬高）
    const br = (L.brow || 0) + (face.tilt ? -0.35 : 0) + (face.hurt ? 0.3 : 0), nose = ax(0, 0.95), ey = hy - r * 0.05;
    for (const az of [0.42, -0.42]) {
      if (!vis(az)) continue;
      const ex = ax(az, 0.6), sq = clamp(-Math.sin(psi + az) * 1.3, 0.45, 1), inn = Math.sign(nose - ex) || dir;
      if (face.hurt) {
        c.strokeStyle = ink; c.lineWidth = lw * 1.2; c.beginPath(); c.moveTo(ex - r * 0.15 * sq, ey - r * 0.1); c.lineTo(ex + r * 0.15 * sq, ey + r * 0.1); c.moveTo(ex - r * 0.15 * sq, ey + r * 0.1); c.lineTo(ex + r * 0.15 * sq, ey - r * 0.1); c.stroke();
      } else {
        c.fillStyle = '#ffffff'; c.beginPath(); c.ellipse(ex, ey, r * 0.16 * sq, r * 0.13, 0, 0, Math.PI * 2); c.fill();
        c.fillStyle = ink; c.beginPath(); c.arc(ex + cs * r * 0.06, ey, r * 0.075, 0, Math.PI * 2); c.fill();
      }
      c.strokeStyle = ink; c.lineWidth = lw * 1.6;
      c.beginPath(); c.moveTo(ex - inn * r * 0.2 * sq, ey - r * 0.25 - br * r * 0.15); c.lineTo(ex + inn * r * 0.22 * sq, ey - r * 0.25 + br * r * 0.15); c.stroke();
    }
    if (L.scar) {                   // 脸上的旧伤疤，在朝镜头的那边脸颊上
      const az = Math.sin(psi - 0.75) < Math.sin(psi + 0.75) ? -0.75 : 0.75;
      if (vis(az)) {
        const sx = ax(az, 0.64);
        c.strokeStyle = '#a3262b'; c.lineWidth = lw * 1.1; c.beginPath(); c.moveTo(sx - dir * r * 0.15, hy + r * 0.05); c.lineTo(sx + dir * r * 0.15, hy + r * 0.38); c.stroke();
        c.lineWidth = lw * 0.7; c.beginPath(); c.moveTo(sx - dir * r * 0.07, hy + r * 0.24); c.lineTo(sx + dir * r * 0.05, hy + r * 0.15); c.stroke();
      }
    }
    if (sn > 0.45) return;
    const mx = ax(0, 0.67), wide = 1 + 0.6 * clamp(-sn, 0, 1);
    if (L.smirk && !face.hurt) {    // 带着笑意打量对手
      c.strokeStyle = ink; c.lineWidth = lw; c.beginPath();
      c.moveTo(mx - dir * r * 0.17 * wide, hy + r * 0.5); c.quadraticCurveTo(mx - dir * r * 0.01, hy + r * 0.58, mx + dir * r * 0.17 * wide, hy + r * 0.4); c.stroke();
    }
    c.fillStyle = L.trunk2; c.beginPath(); c.ellipse(ax(0, 0.72), hy + r * 0.45, r * 0.2 * wide, r * 0.1, 0, 0, Math.PI * 2); c.fill();     // 护齿
    if (face.sweat) { c.fillStyle = 'rgba(160,210,255,0.85)'; c.beginPath(); c.ellipse(hx - cs * r * 0.6, hy - r * 0.9, r * 0.1, r * 0.16, 0, 0, Math.PI * 2); c.fill(); }
  }
  function hairProfile(c, hx, hy, r, dir, L) {
    c.fillStyle = L.hairCol;
    if (L.hair === 'spiky') {
      c.beginPath(); c.moveTo(hx - dir * r * 0.95, hy - r * 0.1);
      for (let i = 0; i <= 5; i++) { const a = Math.PI + i / 5 * Math.PI; const rr = i % 2 ? r * 1.25 : r * 0.95; c.lineTo(hx + Math.cos(a) * rr * 0.98 * (dir), hy + Math.sin(a) * rr); }
      c.closePath(); c.fill();
    } else if (L.hair === 'slick') {        // 背头：顶上往后梳、两侧剃短，额前一绺垂下来
      c.beginPath(); c.moveTo(hx + dir * r * 0.75, hy - r * 0.55);
      c.quadraticCurveTo(hx + dir * r * 0.2, hy - r * 1.45, hx - dir * r * 0.95, hy - r * 0.75);
      c.quadraticCurveTo(hx - dir * r * 1.12, hy - r * 0.1, hx - dir * r * 0.8, hy + r * 0.1);
      c.quadraticCurveTo(hx - dir * r * 0.5, hy - r * 0.55, hx + dir * r * 0.75, hy - r * 0.55); c.fill();
      c.beginPath(); c.moveTo(hx + dir * r * 0.55, hy - r * 0.62); c.quadraticCurveTo(hx + dir * r * 0.95, hy - r * 0.35, hx + dir * r * 0.78, hy - r * 0.05);
      c.quadraticCurveTo(hx + dir * r * 0.7, hy - r * 0.35, hx + dir * r * 0.35, hy - r * 0.55); c.fill();
    } else if (L.hair === 'shaggy') {       // 乱发：一圈参差的发梢，刘海压到眉毛
      c.beginPath(); c.moveTo(hx - dir * r * 1.0, hy + r * 0.2);
      const tips = [[-1.15, -0.25], [-1.05, -0.75], [-0.75, -1.25], [-0.25, -1.3], [0.2, -1.38], [0.6, -1.1], [1.05, -0.85], [0.8, -0.55], [1.02, -0.32], [0.62, -0.38], [0.5, -0.2]];
      for (const [tx, ty] of tips) c.lineTo(hx + dir * r * tx, hy + r * ty);
      c.lineTo(hx - dir * r * 0.2, hy - r * 0.45); c.lineTo(hx - dir * r * 0.75, hy + r * 0.05); c.closePath(); c.fill();
    } else if (L.hair === 'buzz') {
      c.beginPath(); c.ellipse(hx - dir * r * 0.08, hy - r * 0.35, r * 0.92, r * 0.65, 0, Math.PI, 0); c.fill();
    } else {
      c.beginPath(); c.ellipse(hx, hy - r * 0.45, r * 1.0, r * 0.55, 0, Math.PI, 0); c.fill();
      c.fillRect(hx, hy - r * 0.5, dir * r * 1.35, r * 0.18);
    }
  }
  function hairFront(c, hx, hy, r, L, back) {
    c.fillStyle = L.hairCol;
    if (back) { c.beginPath(); c.ellipse(hx, hy - r * 0.12, r * 0.93, r * 0.86, 0, 0, Math.PI * 2); c.fill(); return; }
    c.beginPath(); c.ellipse(hx, hy - r * 0.42, r * 0.98, r * 0.62, 0, Math.PI, 0); c.fill();
    if (L.hair === 'shaggy' || L.hair === 'spiky') {   // 刘海
      c.beginPath(); c.moveTo(hx - r * 0.9, hy - r * 0.42);
      for (let i = 0; i <= 6; i++) c.lineTo(hx - r * 0.9 + i * r * 0.3, hy - r * (i % 2 ? 0.18 : 0.42));
      c.closePath(); c.fill();
    }
  }

  function drawDown(c, sx, sy, s, dir, id, body, pose) {
    // 倒地：从站姿到平躺的过渡（down 0→1）
    const L = look(id), B = build(body), ink = '#14161c', lw = Math.max(1.2, 0.012 * s), d = ease(pose.down);
    const ang = d * Math.PI / 2 * 0.98;                     // 身体从竖直倒到水平（往后倒）
    const hipX = -0.25 * d, hipY = lerp(B.hipH, 0.14, d);
    const len = B.shH - B.hipH + B.neck + B.headR * 2;
    const X = x => sx + dir * x * s, Y = y => sy - y * s;
    const shx = hipX - Math.sin(ang) * (B.shH - B.hipH), shy = hipY + Math.cos(ang) * (B.shH - B.hipH);
    const hx = hipX - Math.sin(ang) * len, hy = hipY + Math.cos(ang) * len;
    tube(c, [[X(hipX), Y(hipY)], [X(0.25 - 0.2 * d), Y(lerp(0.3, 0.18, d))], [X(0.45 - 0.05 * d), Y(0.06)]], B.legW * 0.85 * s, L.skin, ink, lw);
    tube(c, [[X(hipX), Y(hipY)], [X(shx), Y(shy)]], B.chest * 1.6 * s, L.skin, ink, lw);
    c.fillStyle = L.trunk; c.beginPath(); c.ellipse(X(hipX), Y(hipY), B.waist * 1.3 * s, 0.1 * s, -dir * ang, 0, Math.PI * 2); c.fill();
    tube(c, [[X(shx), Y(shy)], [X(shx + 0.25), Y(shy + 0.15 - 0.2 * d)]], B.arm * 0.85 * s, L.skin, ink, lw);
    c.fillStyle = L.glove; c.beginPath(); c.arc(X(shx + 0.25), Y(shy + 0.15 - 0.2 * d), B.glove * s, 0, Math.PI * 2); c.fill(); c.stroke();
    drawHead(c, X(hx), Y(hy), B.headR * s, dir, L, { hurt: true }, lw);
    return { hx: X(hx), hy: Y(hy), hr: B.headR * s };
  }

  function shade(hex, k) { return mixHex(hex, k < 0 ? '#000000' : '#ffffff', Math.abs(k)); }
  function mixHex(a, b, t) {
    const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    const r = Math.round(lerp(pa >> 16, pb >> 16, t)), g = Math.round(lerp((pa >> 8) & 255, (pb >> 8) & 255, t)), bl = Math.round(lerp(pa & 255, pb & 255, t));
    return '#' + ((1 << 24) + (r << 16) + (g << 8) + bl).toString(16).slice(1);
  }

  // ---------------------------------------------------------------- 回放预处理
  function prep(rep) {
    if (rep._prep) return rep;
    const n = rep.frames.length, fr = rep.frames;
    const P = { downAt: [], upAt: [], rounds: [] };
    const dn = [-999, -999], up = [-999, -999];
    for (let t = 0; t < n; t++) {
      const d = [], u = [];
      for (let k = 0; k < 2; k++) {
        const f = fr[t].f[k], pf = t ? fr[t - 1].f[k] : f;
        if (f[F.ph] === PH.DOWN && pf[F.ph] !== PH.DOWN) dn[k] = t;
        if (f[F.ph] !== PH.DOWN && pf[F.ph] === PH.DOWN) up[k] = t;
        d.push(dn[k]); u.push(up[k]);
      }
      P.downAt.push(d); P.upAt.push(u);
    }
    // 回合分界
    for (let t = 1; t < n; t++) if (fr[t].r !== fr[t - 1].r) P.rounds.push(t);
    // 事件
    P.events = [];
    const PUN = SIM.actions.slice(7, 19), DEF = { 13: '拍开', 14: '侧闪躲过', 15: '下潜躲过', 16: '后仰躲过' };
    for (let t = 1; t < n; t++) for (let k = 0; k < 2; k++) {
      const f = fr[t].f[k];
      if (f[F.kd]) P.events.push({ t, k: 1 - k, big: 3, text: '击倒！', kdOf: k });
      else if (f[F.fired] && f[F.oc] === 1 && f[F.ctr]) P.events.push({ t, k, big: 2, text: '反击！' + PUN[f[F.fired] - 1] });
      else if (f[F.fired] && f[F.oc] === 1 && f[F.dmg] >= 5) P.events.push({ t, k, big: 1, text: PUN[f[F.fired] - 1] });
      if (f[F.ev] >= 13) P.events.push({ t, k, big: 1, text: DEF[f[F.ev]], def: true });
      if (f[F.fk]) P.events.push({ t, k, big: 1, text: '被晃了', def: true });
    }
    rep._prep = P;
    rep._mo = [motion(rep, 0), motion(rep, 1)];
    link(rep);
    rep._mo.forEach((mo, k) => { mo.feet = planFeet(rep, k); });
    rep._mo.forEach((mo, k) => mo.punches.forEach(E => { E.lunge = lungeFor(rep, k, E); }));
    rep._cam = cameraPath(rep);
    return rep;
  }

  // ---------------------------------------------------------------- 动作时间表
  // 回放是先算完再播的，往后看得到：每一拳从哪一帧开始预备、哪一帧打到、哪一帧收回、结果如何，都先排出来。
  // 时间单位是帧（模拟一帧 50 毫秒）。第 s 帧开始处于起手状态的拳，动作从 s−1 开始（那一步做的决定），
  // 在打出的那一帧（回放里 fired 的那一帧，模拟就在这一帧结算）正好到达落点。
  const KB_DEFAULT = [0.08, 0.05, 0.25, 0.15, 0.12, 0.08, 0.15, 0.10, 0.10, 0.06, 0.14, 0.08];
  const TWIST_DEFAULT = [0, 0, 0, 0, 0.25, 0.10, 0.30, 0.10, 0, 0, 0, 0];
  // 胳膊真正打出去的帧数：发力从脚、胯开始，肩跟上，胳膊最后才出去（刺拳约 100 毫秒，后手、勾拳、上勾拳约 150 毫秒）
  const armFrames = E => E.typ === 0 && E.hand === 0 ? 2 : 3;
  // 每帧指向"开始时间不晚于这一帧的最后一个事件"（事件按开始时间排好）
  function pointers(n, list, key = 't0') {
    const out = new Int32Array(n).fill(-1);
    for (let t = 0, j = -1; t < n; t++) { while (j + 1 < list.length && list[j + 1][key] <= t) j++; out[t] = j; }
    return out;
  }
  function motion(rep, k) {
    const fr = rep.frames, n = fr.length, row = t => fr[clamp(t, 0, n - 1)].f[k], opp = t => fr[clamp(t, 0, n - 1)].f[1 - k];
    const mo = { punches: [], defs: [], clinches: [], switches: [], hits: [], kb: [], liver: new Uint8Array(n), winded: new Uint8Array(n) };
    // 出拳、虚晃、扑抱
    for (let t = 1; t < n;) {
      const f = row(t);
      if (f[F.ph] !== PH.STARTUP) { t++; continue; }
      const mv = f[F.mv], s = t, feint = f[F.act] === ACT.FEINT_JAB || f[F.act] === ACT.FEINT_CROSS;
      let e = s + 1;
      while (e < n && row(e)[F.ph] === PH.STARTUP && row(e)[F.mv] === mv && !(feint && e - s >= 2)) e++;
      const recEnd = from => { let r = from; while (r < n && row(r)[F.ph] === PH.RECOVERY && row(r)[F.mv] === mv) r++; return r; };
      if (mv === M.CLINCH) {
        const ok = e < n && row(e)[F.ph] === PH.CLINCH;      // 抱住了：扑过去的前探在 3 帧里收成抱住的姿势；没抱住：收招时慢慢站回来
        mo.clinches.push({ t0: s - 1, t1: e, t2: ok ? e + 3 : Math.max(e + 1, recEnd(e)), ok });
      } else if (mv < 12) {
        const E = { hand: SIM.HAND[mv], p: mv, typ: SIM.TYPE[mv], tgt: SIM.TGT[mv], s, t0: s - 1, feint, oc: 0,
          left: (SIM.HAND[mv] === 0) === (f[F.st] >= 0) };                  // 是不是左手打的（肝在对手右侧，左勾拳最容易打到）
        if (feint) { E.tb = E.ta = E.t0; E.t1 = e; E.t2 = e + 2; }
        else if (e < n && row(e)[F.fired] === mv + 1) {
          const g = row(e), og = opp(e), pr = f[F.prep] !== undefined ? f[F.prep] : 0;     // 旧回放（v13 以前）没有身体预备
          E.tb = E.t0 + clamp(pr, 0, e - s - 1); E.t1 = e; E.ta = Math.max(E.tb, E.t1 - armFrames(E));
          E.oc = g[F.oc]; E.zone = g[F.zone]; E.glance = !!g[F.gl]; E.ctr = !!g[F.ctr]; E.dmg = g[F.dmg]; E.ev = og[F.ev]; E.kd = !!og[F.kd];
          // 被什么挡住：旧回放没记，按防守方上一帧举没举架推出来（模拟里就是这么判的）
          E.blk = g[F.blk] !== undefined ? g[F.blk] : E.oc === 2 ? (opp(e - 1)[F.g] > 0 ? opp(e - 1)[F.g] : 3) : 0;
          const r = recEnd(e), next = r < n ? row(r)[F.ph] : PH.FREE;
          // 收招被下一拳接上（组合拳）时，这只手还要按原来的节奏收回来
          E.t2 = e + (next === PH.STARTUP ? Math.max(2 * (r - e), r - e + 2) : Math.max(2, r - e + (next === PH.FREE ? 0 : 1)));
        } else {
          // 起手被打断（挨打、被打倒、被抱住、回合结束）：胳膊走到一半收回来
          const su = f[F.su] > 0 ? f[F.su] : e - s + 2, pr = f[F.prep] !== undefined ? f[F.prep] : 0;
          E.tb = E.t0 + clamp(pr, 0, su - 1); E.t1 = s + su; E.ta = Math.max(E.tb, E.t1 - armFrames(E)); E.cut = e; E.t2 = e + 3;
        }
        mo.punches.push(E);
      }
      t = e;
    }
    // 同一只手的下一拳开始时，上一拳一定已经收回
    for (let h = 0; h < 2; h++) {
      const list = mo.punches.filter(E => E.hand === h);
      for (let i = 0; i + 1 < list.length; i++) {
        const E = list[i], nx = list[i + 1].t0;           // 虚晃没有打到那一下，连着晃就在下一次开始时正好收完
        E.t2 = E.feint ? Math.max(E.t0 + 0.5, Math.min(E.t2, nx)) : Math.max(Math.min(E.t2, nx), (E.cut ?? E.t1) + 0.5);
      }
    }
    // 防守：拍挡、侧闪、下潜、后仰。时机窗口（防守中）一开始就有效，所以动作在窗口开始时已经做到位；收招时慢慢回位
    for (let t = 1; t < n;) {
      const f = row(t);
      if (f[F.ph] !== PH.DEFEND) { t++; continue; }
      const m = f[F.mv], s = t;
      let e = s + 1; while (e < n && row(e)[F.ph] === PH.DEFEND && row(e)[F.mv] === m) e++;
      let r = e; while (r < n && row(r)[F.ph] === PH.RECOVERY && row(r)[F.mv] === m) r++;
      mo.defs.push({ m, s, t0: s - 1, ta: s + 0.3, t1: e, t2: Math.max(r, e + 1) });
      t = e;
    }
    // 换架势、换站架：模拟在换完那一帧才改，画面在这几帧里换过去
    for (let t = 1; t < n;) {
      const f = row(t);
      if (!(f[F.ph] === PH.MOVING && (f[F.mv] === M.POSTURE || f[F.mv] === M.STANCE))) { t++; continue; }
      const m = f[F.mv], s = t;
      let e = s + 1; while (e < n && row(e)[F.ph] === PH.MOVING && row(e)[F.mv] === m) e++;
      // 换到一半被打断（挨打、倒地）的，模拟里架势没变，画面也不换
      const done = e < n && (m === M.POSTURE ? row(e)[F.post] !== row(s - 1)[F.post] : row(e)[F.st] !== row(s - 1)[F.st]);
      if (done) mo.switches.push({ kind: m === M.POSTURE ? 'post' : 'stance', t0: s - 1, t1: e, to: row(e)[F.post] });
      t = e;
    }
    // 旧回放（换架势、换站架还不花时间的规则）里是一帧之内换的：补 3 帧过渡
    for (let t = 1; t < n; t++) {
      const f = row(t), pf = row(t - 1);
      if (fr[t].r !== fr[t - 1].r) continue;
      for (const kind of ['post', 'stance']) {
        if (kind === 'post' ? f[F.post] === pf[F.post] : f[F.st] === pf[F.st]) continue;
        if (mo.switches.some(q => q.kind === kind && q.t1 === t)) continue;
        mo.switches.push({ kind, t0: Math.max(0, t - 3), t1: t, to: f[F.post] });
      }
    }
    mo.switches.sort((a, b) => a.t0 - b.t0);
    for (let t = 0; t < n; t++) { mo.liver[t] = row(t)[F.liv] || 0; mo.winded[t] = row(t)[F.wind] || 0; }
    // 缠抱住的那几段（模拟里两个人同时进、同时出）
    mo.holds = [];
    for (let t = 1; t < n;) {
      if (row(t)[F.ph] !== PH.CLINCH) { t++; continue; }
      let e = t + 1; while (e < n && row(e)[F.ph] === PH.CLINCH) e++;
      mo.holds.push({ s: t, e }); t = e;
    }
    mo.hdptr = pointers(n, mo.holds.map(q => ({ t0: q.s - 1 })));
    return mo;
  }

  // 两个人的时间表对上：挨了哪几拳（挨打的反应、击退）、防守是冲着哪一拳去的（往哪边闪、用哪只手拍）
  function link(rep) {
    const fr = rep.frames, n = fr.length, KB = SIM.KB || KB_DEFAULT, TW = SIM.TWIST || TWIST_DEFAULT;
    for (let k = 0; k < 2; k++) {
      const mo = rep._mo[k], om = rep._mo[1 - k], row = t => fr[clamp(t, 0, n - 1)].f[k], opp = t => fr[clamp(t, 0, n - 1)].f[1 - k];
      for (const E of om.punches) {
        if (E.oc !== 1 && E.oc !== 2) continue;
        let h = E.t1; while (h < n && (row(h)[F.ph] === PH.HITSTUN || row(h)[F.ph] === PH.BLOCKSTUN)) h++;
        mo.hits.push({ t: E.t1, E, dur: Math.max(2, h - E.t1), side: E.left ? -1 : 1 });     // 挨打的是自己哪一侧（对手左手打来落在右侧）
        // 击退、勾拳把身体打歪：模拟里和这一拳同一帧生效
        const a0 = opp(E.t1 - 1), d0 = row(E.t1 - 1), d1 = row(E.t1);
        let ux = d0[F.x] - a0[F.x], uy = d0[F.y] - a0[F.y]; const ul = Math.hypot(ux, uy) || 1; ux /= ul; uy /= ul;
        const ab = rep.bodies[1 - k], db = rep.bodies[k];
        let mag = KB[E.p] * (ab.power || 1) * (75 / (db.mass || 72)) * (E.oc === 1 ? 1 : 0.25);
        mag = clamp(Math.min(mag, (d1[F.x] - d0[F.x]) * ux + (d1[F.y] - d0[F.y]) * uy), 0, mag);
        const tw = E.oc === 1 && E.typ === 1 ? TW[E.p] * (E.hand === 0 ? 1 : -1) * (opp(E.t1)[F.st] < 0 ? -1 : 1) : 0;
        mo.kb.push({ t1: E.t1, v: [ux * mag, uy * mag], tw, dur: 1.5 });
      }
      // 被拉开、拉拢（击倒后回中立角、裁判分开缠抱、抱上去时两人贴到一起）：模拟里一帧之内移过去，画面上走过去
      for (let t = 1; t < n; t++) {
        if (fr[t].r !== fr[t - 1].r) continue;            // 回合开始回角落：转播在这里切到回合间
        const f = row(t), pf = row(t - 1), q = mo.kb.find(b => b.t1 === t), hold = f[F.ph] === PH.CLINCH && pf[F.ph] !== PH.CLINCH;
        const vx = f[F.x] - pf[F.x] - (q ? q.v[0] : 0), vy = f[F.y] - pf[F.y] - (q ? q.v[1] : 0);
        if (Math.hypot(vx, vy) < (hold ? 0.03 : 0.25)) continue;
        mo.kb.push({ t1: t, v: [vx, vy], tw: 0, dur: f[F.ph] === PH.DOWN || opp(t)[F.ph] === PH.DOWN ? 14 : hold ? 3 : 5 });
      }
      mo.kb.sort((a, b) => a.t1 - b.t1);
      mo.hits.sort((a, b) => a.t - b.t);
      // 旧回放没记肝区击倒前弯腰：倒地那一帧对手没打中人、之前 1~2 秒有一记干净的身体拳，就是它
      if (row(0)[F.liv] === undefined) for (let t = 1; t < n; t++) {
        if (!(row(t)[F.ph] === PH.DOWN && row(t - 1)[F.ph] !== PH.DOWN) || (opp(t)[F.fired] && opp(t)[F.oc] === 1)) continue;
        const E = om.punches.filter(q => q.oc === 1 && q.tgt === 1 && t - q.t1 >= 19 && t - q.t1 <= 41).pop();
        if (E) { for (let q = E.t1 + 1; q < t; q++) mo.liver[q] = 1; E.zone = 3; }
      }
      // 防守冲着哪一拳：窗口里打过来的那一拳（被它躲开的优先）
      for (const D of mo.defs) {
        const inc = om.punches.filter(E => !E.feint && E.oc && E.t1 >= D.s && E.t1 < D.t2);
        const E = inc.find(q => q.oc === 3 && q.ev === D.m) || inc[0] || null;
        const ls = row(D.s)[F.st] < 0 ? -1 : 1, lo = opp(D.s)[F.st] < 0 ? -1 : 1;
        const sideA = E ? (E.left ? 1 : -1) : lo;            // 来拳是对手的左手还是右手（没有来拳按对手的前手）
        D.inc = E;
        D.side = -sideA;                                     // 往来拳那只胳膊的外侧闪：对手的左手在自己右侧
        D.hand = D.side === ls ? 0 : 1;                      // 拍挡用来拳那一侧的手
      }
      mo.hptr = pointers(n, mo.hits, 't');
      mo.kbptr = pointers(n, mo.kb, 't1');
      mo.dptr = pointers(n, mo.defs);
      mo.cptr = pointers(n, mo.clinches);
      mo.sptr = pointers(n, mo.switches);
      mo.byHand = [0, 1].map(h => mo.punches.filter(E => E.hand === h));
      mo.ptr = mo.byHand.map(list => pointers(n, list));
    }
  }
  function clinchWeight(mo, tf) {
    const sp = mo.holds[mo.hdptr[clamp(Math.floor(tf) + 1, 0, mo.hdptr.length - 1)]];
    if (!sp) return 0;
    return tf < sp.s ? ease(clamp((tf - (sp.s - 0.5)) / 0.5, 0, 1)) : tf < sp.e ? 1 : 1 - ease(clamp((tf - sp.e) / 4, 0, 1));
  }
  // 打到的时刻离 tf 不到 win 帧的拳（不管那一拳这时开始起手了没有：快拳从起手到打到不足 win 帧，按"正在出拳"找会在起手那一刻突然冒出来）
  function punchesNear(mo, tf, win) {
    const out = [], n = mo.ptr[0].length;
    for (let h = 0; h < 2; h++) {
      const list = mo.byHand[h];
      for (let i = mo.ptr[h][clamp(Math.floor(tf + win), 0, n - 1)]; i >= 0; i--) {
        const E = list[i];
        if (E.t1 < tf - win - 8) break;
        if (E.oc && Math.abs(tf - E.t1) <= win) out.push(E);
      }
    }
    return out;
  }
  function activePunch(mo, h, tf) {
    const t = Math.floor(tf), arr = mo.ptr[h];
    if (t < 0 || t >= arr.length || arr[t] < 0) return null;
    const E = mo.byHand[h][arr[t]];
    return tf >= E.t0 && tf < E.t2 ? E : null;
  }

  // ---------------------------------------------------------------- 三维拳手
  // 模拟是俯视的二维：位置 (x, y)、朝向 th。画面里每个拳手是一副三维骨骼（z 是离台面的高度），按模拟的位置、朝向、
  // 动作时间摆好，再用转播机位投影。局部坐标：前（朝向 th）、左、上；左右按解剖学，正架左脚在前、反架右脚在前。
  // 所见即模拟：拳在打出那一帧到达落点；被挡停在对方的护手上，被躲开就打到头原来的位置，射程不够就停在半空差那么一截。
  const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul3 = (a, k) => [a[0] * k, a[1] * k, a[2] * k], dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const len3 = a => Math.hypot(a[0], a[1], a[2]), norm3 = a => mul3(a, 1 / (len3(a) || 1));
  const lerp3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
  const bez3 = (a, c, b, t) => lerp3(lerp3(a, c, t), lerp3(c, b, t), t);
  const wrapPi = a => a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));
  const fatigueOf = sf => sf >= 0.5 ? 0 : sf >= 0.2 ? 0.4 * (0.5 - sf) / 0.3 : 0.4 + 0.6 * (0.2 - Math.max(0, sf)) / 0.2;   // 同 boxing.fatigue

  // 骨骼尺寸。胳膊长按臂展定：满射程的拳胳膊伸直还差一截，要往前送一步才打得到（现实里刺拳也是上步打的）
  function rig(body) {
    const B = build(body), H = B.H, reach = body ? body.reach : 0.75, mass = body ? body.mass : 72;
    const arm = 0.36 * H * (1 + 0.8 * (reach - 0.75));
    return { ...B, armLen: arm, ua: arm * 0.52, fa: arm * 0.48, thigh: 0.245 * H, shin: 0.245 * H, ankle: 0.07,
      hipZ: 0.465 * H, spine: 0.3 * H, neckL: 0.045 * H, shHalf: 0.15 + 0.035 * B.mus, hipHalf: 0.085 + 0.002 * (mass - 72),
      chestD: 0.09 + 0.035 * B.mus, waistD: 0.085 };
  }
  const rigOf = (rep, k) => (rep._rig || (rep._rig = [rig(rep.bodies[0]), rig(rep.bodies[1])]))[k];

  // 绕着肩膀 S 在 P、Q 两个位置之间过渡：方向按球面插值，离肩膀的远近线性插值。护手从腹前抬到额前（AI 常常只举一帧高位格挡）时，
  // 拳套从身前绕个弧上去，不会擦着肩膀走直线把胳膊折死、肘绕着胳膊乱转
  function polar(S, P, Q, w) {
    if (w <= 0) return P;
    if (w >= 1) return Q;
    const a = sub3(P, S), b = sub3(Q, S), ra = len3(a), rb = len3(b);
    if (ra < 1e-6 || rb < 1e-6) return lerp3(P, Q, w);
    const na = mul3(a, 1 / ra), nb = mul3(b, 1 / rb), om = Math.acos(clamp(dot3(na, nb), -1, 1));
    const dir = om < 1e-4 ? lerp3(na, nb, w) : add3(mul3(na, Math.sin((1 - w) * om) / Math.sin(om)), mul3(nb, Math.sin(w * om) / Math.sin(om)));
    return add3(S, mul3(norm3(dir), lerp(ra, rb, w)));
  }
  // 点 p 离线段 a–b 不到 r 就沿垂直方向推到 r 处（a = b 时就是球）
  function outside(p, a, b, r) {
    const ab = sub3(b, a), t = clamp(dot3(sub3(p, a), ab) / (dot3(ab, ab) || 1), 0, 1), q = add3(a, mul3(ab, t)), v = sub3(p, q), d = len3(v);
    return d >= r || d < 1e-6 ? p : add3(q, mul3(v, r / d));
  }
  // 两段骨骼的三维反向运动学：根 a、末端 b，pole 指向关节（肘、膝）往哪边弯。够不着就伸直指向末端。
  // pole 和骨头几乎平行时（比如手垂在腹前、提示又是"肘往下"），往哪边弯定不下来，关节会绕着骨头乱转：按夹角平滑地掺进备用方向 alt
  function ik3(a, b, l1, l2, pole, alt) {
    let d = sub3(b, a), L = len3(d);
    const Lmax = (l1 + l2) * 0.999, Lmin = Math.abs(l1 - l2) + 0.02;
    if (L > Lmax) { b = add3(a, mul3(d, Lmax / L)); d = sub3(b, a); L = Lmax; }
    else if (L < Lmin) { b = add3(a, mul3(norm3(L > 1e-6 ? d : pole), Lmin)); d = sub3(b, a); L = Lmin; }
    const u = mul3(d, 1 / L), x = (l1 * l1 - l2 * l2 + L * L) / (2 * L), hgt = Math.sqrt(Math.max(0, l1 * l1 - x * x));
    const perp = v => { const q = norm3(v); return sub3(q, mul3(u, dot3(q, u))); };
    let p = perp(pole);
    if (alt) p = add3(p, mul3(perp(alt), clamp(1 - len3(p) / 0.5, 0, 1)));
    if (len3(p) < 1e-6) p = Math.abs(u[2]) < 0.9 ? add3([0, 0, -1], mul3(u, u[2])) : [1, 0, 0];
    return { E: add3(add3(a, mul3(u, x)), mul3(norm3(p), hgt)), G: b };
  }

  // 下一帧（插值用）。回合开始两人回到角落：不往那边插值，这一回合的最后一帧就停在原地
  const nextRow = (fr, t, k) => t + 1 < fr.length && fr[t + 1].r === fr[t].r ? fr[t + 1].f[k] : fr[t].f[k];
  // 位置和朝向（连续时间）。击退、勾拳打歪身体、被拉开，在模拟里都是一帧之内的事：画面上推迟到拳到了以后再发生，
  // 挨打的人不会在拳到之前先往后飘；被拉开的两个人用几帧走开，不瞬移
  function placeAt(rep, tf, k) {
    const fr = rep.frames, n = fr.length, t = clamp(Math.floor(tf), 0, n - 1), a = clamp(tf - t, 0, 1);
    const f = fr[t].f[k], g = nextRow(fr, t, k), mo = rep._mo[k];
    let x = lerp(f[F.x], g[F.x], a), y = lerp(f[F.y], g[F.y], a), th = f[F.th] + wrapPi(g[F.th] - f[F.th]) * a;
    for (let i = mo.kbptr[Math.min(n - 1, t + 1)]; i >= 0; i--) {
      const q = mo.kb[i];
      if (q.t1 < tf - 16) break;
      let w = 0;
      if (tf > q.t1 - 1 && tf <= q.t1) w = tf - (q.t1 - 1);
      else if (tf > q.t1 && tf < q.t1 + q.dur) w = 1 - (q.dur < 2 ? easeOut : ease)((tf - q.t1) / q.dur);
      x -= q.v[0] * w; y -= q.v[1] * w; th -= q.tw * w;
    }
    return [x, y, th];
  }

  // 脚步：拳击是滑步，两只脚不交叉。脚落地以后钉在台面上，离该在的位置远了才迈一步：
  // 往前走前脚先动、后脚跟上，往后退后脚先动，横移靠那一侧的脚先动；轴转时前脚钉住当轴
  const STANCE_W = [1, 1.12, 1.15, 1];
  function footHome(f, side) {                 // side：+1 左脚、−1 右脚。返回局部 [前, 左, 脚尖朝向]
    const ls = f[F.st] < 0 ? -1 : 1, w = STANCE_W[f[F.post]] || 1;
    return side === ls ? [0.2 * w, ls * 0.1 * w, -ls * 0.25] : [-0.22 * w, -ls * 0.12 * w, -ls * 0.85];
  }
  function planFeet(rep, k) {
    const fr = rep.frames, n = fr.length, out = new Array(n);
    const home = (t, side) => {
      const [x, y, th] = placeAt(rep, t, k), h = footHome(fr[t].f[k], side), c = Math.cos(th), s = Math.sin(th);
      return [x + c * h[0] - s * h[1], y + s * h[0] + c * h[1], th + h[2]];
    };
    let feet = [home(0, 1), home(0, -1)], step = null, prevFoot = -1;
    const roundEnd = new Int32Array(n);                    // 这一回合的最后一帧：往后看落脚点不看到下一回合（回合开始两人回到角落）
    for (let t = n - 1; t >= 0; t--) roundEnd[t] = t + 1 < n && fr[t + 1].r === fr[t].r ? roundEnd[t + 1] : t;
    for (let t = 0; t < n; t++) {
      const f = fr[t].f[k];
      if (step && t >= step.t1) { const s = step; feet = feet.map((p, i) => i === s.i ? s.b : p); step = null; }
      if (t > 0 && fr[t].r !== fr[t - 1].r) { step = null; feet = [home(t, 1), home(t, -1)]; }
      else if (!step && f[F.ph] !== PH.DOWN) {
        const last = roundEnd[t], la = Math.min(last, t + 2), want = [home(la, 1), home(la, -1)];
        const err = want.map((w, i) => Math.hypot(w[0] - feet[i][0], w[1] - feet[i][1]) + 0.12 * Math.abs(wrapPi(w[2] - feet[i][2])));
        const ls = f[F.st] < 0 ? -1 : 1, lead = ls > 0 ? 0 : 1;
        const [x0, y0] = placeAt(rep, Math.max(0, t - 1), k), [x1, y1, th] = placeAt(rep, Math.min(last, t + 1), k);
        const vf = Math.cos(th) * (x1 - x0) + Math.sin(th) * (y1 - y0), vl = -Math.sin(th) * (x1 - x0) + Math.cos(th) * (y1 - y0);
        const first = Math.abs(vf) >= Math.abs(vl) ? (vf >= 0 ? lead : 1 - lead) : (vl * ls >= 0 ? lead : 1 - lead);
        const pivot = f[F.ph] === PH.MOVING && (f[F.mv] === M.PIVL || f[F.mv] === M.PIVR);
        // 离该在的位置远了的脚才迈。都远了：该先动的那只优先，刚迈过的那只让一让——一直往一边绕、连续轴转时两只脚交替跟上，
        // 不会只迈一只脚，把另一只甩在后面把腿拉直
        const need = [0, 1].map(i => err[i] - (pivot && i === lead ? 0.2 : 0.07));
        const pri = [0, 1].map(i => need[i] + (i === first ? 0.05 : 0) + (i !== prevFoot ? 0.06 : 0));
        const i = [0, 1].filter(q => need[q] > 0).sort((a, b) => pri[b] - pri[a])[0] ?? -1;
        if (i >= 0) {
          const T = err[i] > 0.25 ? 2 : 3;
          step = { i, t0: t, t1: t + T, a: feet[i], b: home(Math.min(last, t + T + 1), i ? -1 : 1), lift: i === first && !pivot ? 0.035 : 0.012 };
          prevFoot = i;
        }
      }
      out[t] = { p: feet, step };
    }
    return out;
  }
  function feetAt(rep, k, tf) {
    const pl = rep._mo[k].feet, st = pl[clamp(Math.floor(tf), 0, pl.length - 1)], s = st.step;
    return [0, 1].map(i => {
      if (s && s.i === i && tf >= s.t0 && tf < s.t1) {
        const u = (tf - s.t0) / (s.t1 - s.t0), e = ease(u);
        return { p: [lerp(s.a[0], s.b[0], e), lerp(s.a[1], s.b[1], e)], yaw: s.a[2] + wrapPi(s.b[2] - s.a[2]) * e, lift: s.lift * Math.sin(Math.PI * u) };
      }
      return { p: [st.p[i][0], st.p[i][1]], yaw: st.p[i][2], lift: 0 };
    });
  }

  // 架势：肩线侧过去多少（前肩在前）、重心降多少、前倾多少；护手位置相对头的中心，局部 [前, 朝前手一侧, 上]
  const POST_ALPHA = [0.6, 0.75, 0.45, 0.9], POST_CROUCH = [0.02, 0.02, 0.12, 0.02], POST_PITCH = [0.12, 0.1, 0.3, 0.06];
  const GUARD_POS = [
    [[0.2, 0.06, -0.06], [0.12, -0.08, -0.08]],        // 标准高位：前手在颧骨前，后手贴下巴
    [[0.42, 0.04, -0.14], [0.12, -0.08, -0.08]],       // 长架：前手伸出去顶住对手
    [[0.14, 0.06, 0.0], [0.12, -0.06, -0.02]],         // 低姿抱架：两手贴着额头和脸颊
    [[0.16, -0.02, -0.46], [0.1, -0.07, -0.08]],       // 低手位（肩卷）：前手横在腹前，后手贴下巴，前肩抬起
  ];
  const HIGH_GUARD = [[0.12, 0.06, 0.03], [0.11, -0.06, 0.01]];       // 高位格挡：两手挡在额前
  const BODY_GUARD = [[0.14, 0.07, -0.15], [0.12, -0.07, -0.16]];     // 护体：手在下巴前，小臂、手肘护住肋部
  function guardLocal(f, ls, post, post2, pu, su) {
    const gt = f[F.g];
    const pick = h => gt === 1 ? HIGH_GUARD[h] : gt === 2 ? BODY_GUARD[h] : lerp3(GUARD_POS[post][h], GUARD_POS[post2][h], pu);
    const anat = (v, s) => [v[0], v[1] * s, v[2]];
    // 换站架时前后手对调：每只手从原来的角色换到新的角色
    const gl = [0, 1].map(h => su > 0 ? lerp3(anat(pick(h), ls), anat(pick(1 - h), -ls), su) : anat(pick(h), ls));
    return { gl, crouch: lerp(POST_CROUCH[post], POST_CROUCH[post2], pu) + (gt === 1 ? 0.03 : gt === 2 ? 0.06 : 0),
      pitch: lerp(POST_PITCH[post], POST_PITCH[post2], pu) + (gt === 1 ? 0.05 : gt === 2 ? 0.08 : 0) + (gt === 1 && f[F.act] === ACT.GUARD_FWD ? 0.05 : 0),
      alpha: gt > 0 ? 0.45 : lerp(POST_ALPHA[post], POST_ALPHA[post2], pu) };                  // 举架、护体时肩膀收正
  }

  // 一拳的发力进度：ld 是预备（转肩、移重心、屈膝），dr 是发力（从预备结束到打到：胯带肩，手到落点时到顶），打完以后回位；
  // 打空、被躲开的多转一截
  function punchDrive(E, tf) {
    const end = E.cut ?? E.t1, span = t => clamp((t - E.tb) / Math.max(0.5, E.t1 - E.tb), 0, 1);
    const pre = clamp((tf - E.t0) / Math.max(0.5, E.tb - E.t0), 0, 1), arm = span(tf);
    // 上步和胳膊一起出、越到最后越快，正好在拳到的那一刻落地（身体的冲劲压在拳上）；转胯带肩比手先到位
    let dr = ease(clamp(arm * 1.15, 0, 1)), lg = clamp((tf - E.ta) / Math.max(0.5, E.t1 - E.ta), 0, 1) ** 2;
    if (tf > end) {
      const r = clamp((tf - end) / Math.max(1, E.t2 - end), 0, 1);
      if (E.cut !== undefined) dr = Math.min(dr, ease(clamp(span(E.cut) * 1.15, 0, 1)));
      // 往前送的那一步、转过去的腰都跟着拳一起收：拳收回来之前身体不先退、不先转回去（不然胳膊够不着）。打空、被躲开的多转一截
      const back = 1 - retract(E, tf), over = E.oc === 3 || E.oc === 4 ? 1 + 0.15 * Math.sin(Math.PI * Math.min(1, r * 2.5)) : 1;
      lg = dr * back;
      dr *= over * back;
    }
    return { ld: ease(pre) * (1 - ease(arm)), dr, lg };
  }
  // 打到以后拳收回来的进度：打中的贴一下再收，被挡的弹回来，打空、被躲开的顺势送完再收
  function retract(E, tf) {
    const end = E.cut ?? E.t1, span = E.t2 - end;
    // 同一只手紧接着又出拳时收招被截短：贴一下、顺势送的那段跟着缩短，保证下一拳开始时拳已经回到护手位置
    const hold = Math.min(E.cut !== undefined ? 0 : E.oc === 1 ? 0.6 : E.oc === 2 ? 0.2 : 0.8, span * 0.4);
    return ease(clamp((tf - end - hold) / Math.max(0.1, span - hold), 0, 1));
  }

  // 身体（不含胳膊）：骨盆、脊柱、肩、头、腿脚，以及护手该在的位置。mode.core：不往前送步（算出拳要送多少步时用，免得两边互相引用），mode.lunge 只给这一拳试一个步幅；
  // mode.noDef：不做闪躲（算"头原来在哪"时用）；mode.noDown：不管倒地
  function bodyPose(rep, tf, k, mode = {}) {
    const fr = rep.frames, n = fr.length, t = clamp(Math.floor(tf), 0, n - 1), a = clamp(tf - t, 0, 1);
    const t2 = Math.min(n - 1, t + 1), f = fr[t].f[k], g = nextRow(fr, t, k), mo = rep._mo[k], R = rigOf(rep, k), P = rep._prep;
    if (!mode.noDown) {
      if (f[F.ph] === PH.DOWN && P.downAt[t][k] >= 0) return downPose(rep, tf, k, P.downAt[t][k], null);
      const uAt = P.upAt[t][k];
      if (uAt > 0 && tf - uAt < 12) return downPose(rep, tf, k, P.downAt[uAt - 1][k], uAt);
    }
    const [x, y, th] = placeAt(rep, tf, k), c = Math.cos(th), s = Math.sin(th);
    const W = (lf, ll, lz) => [x + c * lf - s * ll, y + s * lf + c * ll, lz];        // 局部 → 世界
    const Wv = (lf, ll, lz) => [c * lf - s * ll, s * lf + c * ll, lz];               // 局部方向 → 世界方向
    const ls = f[F.st] < 0 ? -1 : 1;
    let post = f[F.post], post2 = post, pu = 0, su = 0;
    const sw = mo.switches[mo.sptr[t]];
    if (sw && tf >= sw.t0 && tf < sw.t1) { const u = ease((tf - sw.t0) / (sw.t1 - sw.t0)); if (sw.kind === 'post') { post2 = sw.to; pu = u; } else su = u; }
    const gA = guardLocal(f, ls, post, post2, pu, su), gB = guardLocal(g, ls, post, post2, pu, su);
    // 护手位置：这一帧和下一帧、换架势前和换架势后，各算一个，最后绕着肩膀过渡（见 polar）
    const gF = [0, 1].map(w => guardLocal(f, ls, post, post2, w, su)), gG = [0, 1].map(w => guardLocal(g, ls, post, post2, w, su));
    let crouch = lerp(gA.crouch, gB.crouch, a), pitch = lerp(gA.pitch, gB.pitch, a), roll = 0, pelF = 0, pelL = 0;
    let yaw = ls * lerp(gA.alpha, gB.alpha, a) * (1 - 2 * su);       // 带符号的转腰：+ = 左肩在前（换站架时转过去）
    let headF = 0.04, headL = 0, headZ = 0, headYaw = 0, guardPush = 0, flash = false, slide = 0, lungePitch = 0;
    const heel = [0, 0], fyaw = [0, 0];                               // 按前脚、后脚
    const fat = fatigueOf(lerp(f[F.stam], g[F.stam], a)), hurt = f[F.stun] > 60;
    crouch += 0.012 * (1 + Math.sin(tf * 0.55 + k * 1.7));             // 脚下一直在弹
    if (hurt) roll += 0.05 * Math.sin(tf * 0.6 + k);                   // 被打晃了站不稳
    // 防守动作
    const D = mo.defs[mo.dptr[t]];
    let def = null;
    if (D && !mode.noDef && tf >= D.t0 && tf < D.t2) {
      const e = tf < D.ta ? easeOut((tf - D.t0) / (D.ta - D.t0)) : tf < D.t1 ? 1 : 1 - ease((tf - D.t1) / (D.t2 - D.t1));
      if (D.m === M.SLIP) { roll += 0.3 * e * D.side; crouch += 0.09 * e; pitch += 0.06 * e; headL += 0.05 * e * D.side; }   // 头偏出来拳的线 25 厘米、往下沉十来厘米
      else if (D.m === M.DUCK) { crouch += 0.22 * e; pitch += 0.25 * e; pelF -= 0.12 * e; }              // 头往下 26 厘米，勾拳从上面过去
      else if (D.m === M.PULL) { pitch -= 0.42 * e; pelF -= 0.05 * e; }                                  // 头往后 27 厘米（模拟 0.25 米）
      def = { D, e };
    }
    // 自己的拳：预备时转肩、移重心、屈膝；打出时胯带肩，后脚跟拧起来；要够到落点就往前送一步
    for (let h = 0; h < 2; h++) {
      const E = activePunch(mo, h, tf);
      if (!E) continue;
      const lead = h === 0, sideH = lead ? ls : -ls;
      if (E.feint) { const env = Math.sin(Math.PI * clamp((tf - E.t0) / (E.t2 - E.t0), 0, 1)); yaw += (lead ? ls * 0.15 : -ls * 0.3) * env; pitch += 0.04 * env; continue; }
      const { ld, dr, lg } = punchDrive(E, tf);
      if (E.typ === 0) {
        if (lead) { yaw += ls * 0.3 * dr; pitch += 0.05 * dr; }                                     // 刺拳：前肩送出去
        else { yaw += ls * 0.08 * ld - ls * 1.2 * dr; heel[1] += 0.02 * ld + 0.06 * dr; fyaw[1] += ls * 0.6 * dr; pitch += 0.08 * dr; }   // 后手直拳：拧后脚、转胯、后肩送到前面
      } else if (E.typ === 1) {
        if (lead) { yaw += ls * (0.15 * ld - 0.75 * dr); heel[0] += 0.04 * dr; fyaw[0] -= ls * 0.5 * dr; pelL += sideH * 0.03 * ld; roll -= sideH * 0.06 * dr; }   // 前手勾拳：重心压到前脚，前脚拧转
        else { yaw += ls * (0.1 * ld - 1.1 * dr); heel[1] += 0.05 * dr; fyaw[1] += ls * 0.6 * dr; }
      } else {
        crouch += 0.1 * ld - 0.03 * dr; roll += sideH * 0.12 * ld;                                  // 上勾拳：先往出拳那侧沉下去，再蹬地往上
        yaw -= ls * (lead ? 0.25 : 0.85) * dr; if (!lead) heel[1] += 0.05 * dr;
      }
      if (E.tgt === 1) { const lv = Math.max(ld, dr); crouch += 0.1 * lv; pitch += 0.1 * lv; }      // 打身体：屈膝降重心
      const Lg = mode.lunge && mode.lunge.E === E ? mode.lunge.L : mode.core ? 0 : E.lunge;
      // 上步主要靠整个人往前移、少靠前倾（前倾多了肩膀往下掉，离落点反而远）
      if (Lg) { pelF += 0.7 * Lg * lg; lungePitch += Math.asin(clamp(0.3 * Lg / R.spine, -0.5, 0.8)) * lg; slide += (lead ? 0.85 : 0.5) * Math.max(0, Lg) * lg; }
      if (Lg > 0 && E.tgt === 1) pelL += sideH * 0.6 * Lg * lg;       // 打身体上步：斜着往出拳那侧上，头从对方身体外侧过去
    }
    // 扑抱、缠抱
    const C = mo.clinches[mo.cptr[t]];
    if (C && tf >= C.t0 && tf < C.t2) {
      const e = tf < C.t1 ? ease((tf - C.t0) / (C.t1 - C.t0)) : 1 - ease((tf - C.t1) / (C.t2 - C.t1));
      pitch += 0.3 * e; pelF += 0.12 * e;
    }
    const cw = clinchWeight(mo, tf);                       // 抱住的程度：抱上去用半帧，分开后 4 帧回到站姿
    if (cw > 0) { pitch = lerp(pitch, Math.max(pitch, 0.38), cw); headL += ls * 0.12 * cw; crouch += 0.04 * cw; }
    // 挨打：直拳把头打得往后仰，勾拳把脸打偏，上勾拳把下巴顶起来；身体拳打得弯腰，肝区往右侧弯。被挡住只是护手被压回来
    for (let i = mo.hptr[t]; i >= 0; i--) {
      const H = mo.hits[i], dt = tf - H.t;
      if (dt < 0) continue;
      if (dt > 12) break;
      const E = H.E, imp = (dt / 1.3) * Math.exp(1 - dt / 1.3), A = clamp((E.dmg || 4) / 7, 0.35, 1.6) * (E.ctr ? 1.25 : 1) * (E.glance ? 0.5 : 1);
      if (E.oc === 2) { pitch -= 0.04 * A * imp; guardPush += 0.05 * A * imp; continue; }
      if (dt < 1.5 && E.tgt === 0) flash = true;
      const zone = zoneOf(E);
      if (E.tgt === 0) {
        if (E.typ === 1) { headL -= H.side * 0.07 * A * imp; headYaw -= H.side * 0.5 * A * imp; roll -= H.side * 0.07 * A * imp; }
        else if (E.typ === 2) { headZ += 0.03 * A * imp; pitch -= 0.06 * A * imp; headF -= 0.03 * A * imp; }
        else { headF -= (zone === 0 ? 0.09 : 0.07) * A * imp; pitch -= 0.1 * A * imp; }
      } else if (zone === 3) { roll -= 0.18 * A * imp; pitch += 0.15 * A * imp; crouch += 0.05 * A * imp; }
      else { pitch += 0.25 * A * imp; crouch += 0.06 * A * imp; }
    }
    // 肝区击倒的延迟里弯着腰、捂着右肋；心窝挨了一下喘不上气
    const lv = lerp(mo.liver[t], mo.liver[t2], a), wd = lerp(mo.winded[t], mo.winded[t2], a);
    pitch += 0.38 * lv + 0.12 * wd; crouch += 0.2 * lv; roll -= 0.15 * lv;
    // 上身往前折得多，胯就往后坐（重心留在两脚之间），不会一头扎进对手怀里；往前送步的前倾不算
    pelF -= 0.35 * Math.max(0, pitch - 0.3);
    pitch += lungePitch;
    // 骨架
    const pel = W(pelF, pelL, R.hipZ - crouch);
    const sp = Wv(Math.sin(pitch) * Math.cos(roll), Math.sin(roll), Math.cos(pitch) * Math.cos(roll));     // 脊柱：先前倾再侧弯
    const chest = add3(pel, mul3(sp, R.spine));
    const shAx = Wv(Math.sin(yaw), Math.cos(yaw), -Math.sin(roll) * Math.cos(yaw));                          // 左肩的方向
    const shL = add3(chest, mul3(shAx, R.shHalf)), shR = sub3(chest, mul3(shAx, R.shHalf));
    const lift = 0.04 * ((post === 3 ? 1 - pu : 0) + (post2 === 3 && post !== 3 ? pu : 0));                  // 肩卷：前肩抬起来护下巴
    (ls > 0 ? shL : shR)[2] += lift * (1 - su); (ls > 0 ? shR : shL)[2] += lift * su;                          // 换站架时换到新的前肩
    const hy = 0.55 * yaw, hipAx = Wv(Math.sin(hy), Math.cos(hy), 0);
    const hipL = add3(pel, mul3(hipAx, R.hipHalf)), hipR = sub3(pel, mul3(hipAx, R.hipHalf));
    const head = add3(add3(chest, mul3(sp, R.neckL + R.headR * 0.95)), Wv(headF, headL, headZ));
    const [ox, oy] = placeAt(rep, tf, 1 - k);
    const faceAng = th + clamp(wrapPi(Math.atan2(oy - y, ox - x) - th), -1.1, 1.1) + headYaw;               // 眼睛盯着对手
    // 脚：钉在台面上的位置，加上出拳时前脚往前滑、脚跟拧起来
    const fs = feetAt(rep, k, tf), fwd = Wv(1, 0, 0), ankles = [], knees = [], footYaw = [], heels = [], ankT = [];
    for (let i = 0; i < 2; i++) {
      const role = (i === 0) === (ls > 0) ? 0 : 1, q = fs[i], sl = role === 0 ? slide : 0.15 * slide, yawF = q.yaw + fyaw[role];
      ankT.push([q.p[0] + fwd[0] * sl, q.p[1] + fwd[1] * sl, R.ankle + q.lift + heel[role]]);
      const kn = ik3(i === 0 ? hipL : hipR, ankT[i], R.thigh, R.shin, [Math.cos(yawF), Math.sin(yawF), 0.25], fwd);
      ankles.push(kn.G); knees.push(kn.E); footYaw.push(yawF); heels.push(heel[role]);
    }
    // 护手：跟着头走；累了手往下掉，被打晃了也掉；贴身时收到脸颊两边（贴身防守的"耳罩"，两个人的拳套不会正对着搅在一起）
    const tuck = clamp((Math.hypot(ox - x, oy - y) - 0.55) / 0.45, 0.45, 1);
    const shRole = ls > 0 ? [shL, shR] : [shR, shL], reach = (R.ua + R.fa) * 0.88;
    const place = (v, h) => {
      let G = outside(outside(add3(head, Wv(v[0] * tuck - 0.03 * fat - guardPush,
        v[1] + (Math.sign(v[1]) || (h === 0 ? ls : -ls)) * 0.12 * (1 - tuck), v[2] - 0.12 * fat - (hurt ? 0.04 : 0))),
        head, head, R.headR + R.glove * 0.75), pel, chest, R.chestD + R.glove * 0.8);                       // 拳套贴着脸、贴着肚子，不陷进去
      const S = shRole[h], v2 = sub3(G, S), d = len3(v2);
      if (d > reach) G = add3(S, mul3(v2, reach / d));                                                    // 转腰把肩膀带走时，手跟着收回来，胳膊不拉直
      return outside(G, S, S, 0.18);                                                                      // 胳膊折不到拳套贴着自己肩膀：从肩膀往外推开
    };
    const guard = [0, 1].map(h => polar(shRole[h], polar(shRole[h], place(gF[0].gl[h], h), place(gF[1].gl[h], h), pu),
      polar(shRole[h], place(gG[0].gl[h], h), place(gG[1].gl[h], h), pu), a));
    return { k, R, x, y, th, c, s, ls, W, Wv, pel, chest, shL, shR, hipL, hipR, head, faceAng, ankles, knees, footYaw, heels, ankT,
      guard, sh: shRole, def, post, flash, lv, cw,
      face: { hurt, tilt: f[F.tilt], sweat: f[F.stam] < 0.3, jaw: R.jaw } };
  }

  // 胳膊：护手位置，或者正在打的那一拳的轨迹；拍挡、缠抱、捂肋另算。肘按动作往不同方向弯：护手时往下，勾拳往外抬平
  function armsOf(rep, tf, k, J, Jo) {
    if (J.armsDone) return J;
    const mo = rep._mo[k], R = J.R, f = rep.frames[clamp(Math.floor(tf), 0, rep.frames.length - 1)].f[k], fw = J.Wv(1, 0, 0);
    J.gloves = []; J.elbows = []; J.gloveT = [];          // gloveT：拳套该在的位置（胳膊够不着时和 gloves 不一样，核对用）
    for (let h = 0; h < 2; h++) {
      const S = J.sh[h], side = h === 0 ? J.ls : -J.ls, out = J.Wv(0, side, 0);
      // 护手时肘往下、略往外（往后的分量在胳膊往前上方折的时候会投影成往上，把肘顶到前上方去，所以不加）
      let G = J.guard[h], pole = norm3(add3([0, 0, -1], mul3(out, 0.3)));
      const E = activePunch(mo, h, tf), E2 = activePunch(mo, 1 - h, tf);
      // 组合拳：另一只手开始发力（转胯把这一侧的肩膀带走）时，这只手跟着加快收回来，不被拉直
      const hurry = E2 && !E2.feint && E2.t0 > (E ? E.t0 : -1e9) ? Math.min(1, punchDrive(E2, tf).dr) : 0;     // 打空时多转的那一截不算
      if (E) { const r = punchGlove(rep, k, E, tf, J, G, out, fw, hurry); G = r.G; pole = norm3(lerp3(pole, r.pole, r.w)); }
      if (J.def && J.def.D.m === M.PARRY && J.def.D.hand === h) G = parryGlove(rep, k, J.def, tf, J, G, fw);
      for (const Eo of punchesNear(rep._mo[1 - k], tf, 3)) {
        const w = ease(clamp(1 - Math.abs(tf - Eo.t1) / 3, 0, 1));
        if (Eo.oc === 2 && Eo.tgt === 1 && side === (Eo.left ? -1 : 1)) {        // 身体拳被挡：那一侧的小臂、手肘压下来盖住落点
          const K = targetOf(rep, 1 - k, Eo);
          G = lerp3(G, add3(K.C, [0, 0, 0.22]), w); pole = norm3(lerp3(pole, norm3(sub3(K.C, S)), w));
        } else if (Eo.oc === 1 && Eo.tgt === 0) {                                // 头部拳打中：挡在拳路上的护手让开（默认护架没挡住），拳从空当里进去
          const K = targetOf(rep, 1 - k, Eo), away = sub3(G, K.C), d = len3(away);
          if (d < 0.2) G = add3(G, mul3(add3(mul3(norm3(sub3(away, mul3(K.inDir, dot3(away, K.inDir)))), 0.12), [0, 0, -0.08]), w * (1 - d / 0.2)));
        }
      }
      if (J.cw > 0 && Jo) {                          // 缠抱：手搭到对方同一侧的肩背上（够得着的地方），抱上去、分开都是渐变的
        const sh = dot3(sub3(Jo.shL, J.chest), out) > dot3(sub3(Jo.shR, J.chest), out) ? Jo.shL : Jo.shR;
        let W2 = add3(add3(sh, mul3(fw, 0.05)), [0, 0, -0.05]);
        const v = sub3(W2, S), d = len3(v), m = (R.ua + R.fa) * 0.95;
        if (d > m) W2 = add3(S, mul3(v, m / d));
        G = lerp3(G, W2, J.cw); pole = norm3(lerp3(pole, norm3(add3(out, [0, 0, -0.5])), J.cw));
      }
      if (J.lv > 0 && side < 0) {                    // 肝区击倒的延迟里右手捂着右肋，跟着弯腰的程度过去
        G = lerp3(G, add3(lerp3(J.pel, J.chest, 0.35), J.Wv(0.1, -0.16, 0)), J.lv); pole = norm3(lerp3(pole, norm3(add3([0, 0, -1], out)), J.lv));
      }
      // 胳膊和"肘往下"差不多平行时的备用方向：手在肩膀下面（低手位）肘往外，手在肩膀上面（举到额前）肘往前下。
      // 护手从腹前抬到额前、拳套擦着肩膀过去的那一下，肘一直在身前，不会甩到外面再甩回来
      const uz = (G[2] - S[2]) / (len3(sub3(G, S)) || 1);
      const q = ik3(S, G, R.ua, R.fa, pole, add3(mul3(out, clamp(-uz * 2, 0, 1) + 0.1), mul3(fw, clamp(uz * 2, 0, 1))));
      J.gloves[h] = q.G; J.elbows[h] = q.E; J.gloveT[h] = G;
    }
    return J;
  }
  function punchPath(E, A, T, u, out, fw) {
    if (E.typ === 0) return lerp3(A, T, u);                                    // 直拳：直线
    const m = lerp3(A, T, 0.5), L = len3(sub3(T, A));
    const C = E.typ === 1 ? add3(add3(m, mul3(out, 0.3 * L + 0.05)), add3([0, 0, 0.04], mul3(fw, -0.05)))   // 勾拳：从外侧横着绕进来
      : add3(m, add3([0, 0, -0.22], mul3(fw, -0.08)));                          // 上勾拳：从下面兜上来
    return bez3(A, C, T, u);
  }
  function punchGlove(rep, k, E, tf, J, Gg, out, fw, hurry = 0) {
    const pole = E.typ === 1 ? norm3(add3(out, [0, 0, 0.35])) : E.typ === 2 ? norm3(add3([0, 0, -1], mul3(fw, -0.3))) : norm3(add3([0, 0, -1], mul3(out, 0.25)));
    if (E.feint) return { G: add3(Gg, mul3(fw, 0.09 * Math.sin(Math.PI * clamp((tf - E.t0) / (E.t2 - E.t0), 0, 1)))), pole, w: 0 };
    // 预备：勾拳往外引、上勾拳往下沉，直拳几乎不动（"别闪第一记刺拳"）
    const ld = ease(clamp((tf - E.t0) / Math.max(0.5, E.tb - E.t0), 0, 1));
    const load = E.typ === 1 ? add3(mul3(out, 0.04), mul3(fw, -0.03)) : E.typ === 2 ? add3([0, 0, -0.09], mul3(fw, -0.03)) : [0, 0, -0.015];
    const A = add3(Gg, mul3(load, ld)), K = targetOf(rep, k, E), end = E.cut ?? E.t1;
    const prog = tt => Math.pow(clamp((tt - E.ta) / Math.max(0.5, E.t1 - E.ta), 0, 1), 1.6);     // 越打越快，到落点时最快
    if (tf < E.ta) return { G: A, pole, w: 0.3 * ld };
    if (tf < end) return { G: punchPath(E, A, K.T, prog(tf), out, fw), pole, w: Math.min(1, 0.3 + prog(tf)) };
    // 打到以后：打中的顶进去一点再收；被挡的弹回来；被拍开的往外偏；打空、被躲开的顺势多送一截
    const r1 = clamp((tf - end) / 1.2, 0, 1);
    let P0;
    if (E.cut !== undefined) P0 = punchPath(E, A, K.T, prog(E.cut), out, fw);
    else if (E.oc === 1) P0 = add3(K.T, mul3(K.inDir, 0.03 * Math.sin(Math.PI * r1)));
    else if (E.oc === 2) P0 = K.T;
    else if (E.oc === 3 && E.ev === M.PARRY) P0 = add3(K.T, mul3(add3(mul3(K.deflect, 0.16), mul3(K.inDir, 0.04)), easeOut(r1)));
    else P0 = add3(K.T, mul3(K.inDir, Math.min(E.typ === 1 ? 0.14 : 0.08, K.gap ?? 1) * easeOut(clamp((tf - end) / 1.5, 0, 1))));   // 射程不够的，送不过差的那一截
    const rr = Math.max(retract(E, tf), hurry);
    return { G: lerp3(P0, Gg, rr), pole, w: Math.min(1, 0.3 + prog(end)) * (1 - rr) };      // 肘的方向从打到（或被打断）那一刻接着往回收
  }
  // 拍挡：来拳那一侧的手迎上去，拳到的那一下在脸前二十来厘米处碰上，把拳拍向外侧
  function parryGlove(rep, k, def, tf, J, Gg, fw) {
    const { D, e } = def, E = D.inc;
    let G = add3(Gg, add3(mul3(fw, 0.12 * e), J.Wv(0, -0.04 * (D.hand === 0 ? J.ls : -J.ls) * e, 0)));
    if (E && E.oc === 3 && E.ev === M.PARRY) {
      const K = targetOf(rep, 1 - k, E), w = ease(clamp(1 - Math.abs(tf - E.t1) / 1.5, 0, 1));
      const push = tf > E.t1 ? mul3(K.deflect, 0.12 * easeOut(clamp((tf - E.t1) / 1.2, 0, 1))) : [0, 0, 0];
      G = lerp3(G, add3(add3(K.C, mul3(K.inDir, -0.03)), push), w);
    }
    return G;
  }

  // 落点：回放里记着就用记着的；旧回放没记，按这一拳最常见的落点（把人打倒的头部拳算下颌，左勾拳打身体算肝区）
  function zoneOf(E) {
    if (E.zone !== undefined && E.zone >= 0) return E.zone;
    if (E.tgt === 0) return E.kd || E.typ === 2 ? 0 : E.typ === 1 ? 1 : 2;
    return E.typ === 1 ? (E.left ? 3 : 5) : 4;
  }
  // 模拟里这一拳够得着的中心距（和 boxing.py 的 reach_lim 同一个算法）
  function reachLim(rep, k, E) {
    const fr = rep.frames, t = clamp(E.t1, 0, fr.length - 1), f = fr[t].f[k], o = fr[t].f[1 - k], b = rep.bodies[k];
    const w = (SIM.REACH_W || [1, 0.5, 0.4])[E.typ];
    let arm = SIM.RNG[E.p] * (w * b.reach + (1 - w) * 0.75);
    if (E.typ === 0) arm *= (SIM.POSTURE_RANGE || [1, 1.05, 0.95, 1])[f[F.post]];
    if ((E.p === 2 || E.p === 3) && f[F.st] !== o[F.st]) arm *= SIM.OPEN_CROSS_RANGE || 1.05;
    return SIM.BODY_R + SIM.SHOULDER + arm;
  }
  // 拳和人碰在哪：C 是接触点，T 是那一刻拳套的中心，inDir 是来拳方向，side 是落点在对手身上的哪一侧（对着出拳那只手）
  function contactOf(rep, k, E, core) {
    const d = 1 - k, t1 = E.t1;
    // 被躲开：打的是头原来的位置（不算闪躲）；打空：按"对手站在射程边上时拳落在哪"往回退那一截，所以也不算对手的闪躲和前冲
    const Ja = bodyPose(rep, t1, k, { core: true, noDown: true }), Jd = bodyPose(rep, t1, d, { core: core || E.oc === 4, noDef: E.oc >= 3, noDown: true });
    const Ra = rigOf(rep, k), Rd = rigOf(rep, d), r = Rd.headR, S = Ja.sh[E.hand];
    const latA = Ja.Wv(0, E.hand === 0 ? Ja.ls : -Ja.ls, 0), hc = Jd.head;
    const toA = norm3([S[0] - hc[0], S[1] - hc[1], 0]);
    const sv = norm3(sub3(latA, mul3(toA, dot3(latA, toA))));
    const at = (o, a, b, z) => add3(o, add3(add3(mul3(toA, a), mul3(sv, b)), [0, 0, z]));
    const zone = zoneOf(E);
    let C, short;
    if (E.tgt === 0) {
      C = zone === 0 ? (E.typ === 2 ? at(hc, r * 0.45, 0, -r * 0.9) : E.typ === 1 ? at(hc, r * 0.45, r * 0.7, -r * 0.55) : at(hc, r * 0.8, 0, -r * 0.55))   // 下颌
        : zone === 1 ? at(hc, r * 0.25, r * 0.92, r * 0.1)                    // 头侧（太阳穴）
        : at(hc, r * 0.95, 0, r * 0.1);                                        // 额头面门
    } else {
      const mid = w => lerp3(Jd.pel, Jd.chest, w);
      C = zone === 3 ? add3(at(mid(0.42), 0.05, 0, 0), Jd.Wv(0, -(Rd.hipHalf + 0.07), 0))    // 肝区：右侧下肋
        : zone === 4 ? at(mid(0.62), Rd.chestD + 0.01, 0, 0)                   // 心窝
        : at(mid(0.45), Rd.waistD + 0.02, 0.1, 0);                             // 肋部、腹部
    }
    if (E.oc === 2) {
      // 被挡：头部拳挡在离落点最近的那只护手上（正在出拳的那只手护不了）；身体拳挡在落点前面的小臂、手肘上；肩卷用前肩卸掉
      if (E.blk === 3 && Jd.post === 3 && E.tgt === 0) C = add3(Jd.sh[0], add3(mul3(toA, 0.07), [0, 0, 0.03]));
      else if (E.tgt === 1) C = add3(C, mul3(toA, 0.07));
      else {
        const om = rep._mo[d], free = [0, 1].filter(h => !activePunch(om, h, t1)), hs = free.length ? free : [0, 1];
        const h = hs.reduce((b, q) => len3(sub3(Jd.guard[q], C)) < len3(sub3(Jd.guard[b], C)) ? q : b, hs[0]);
        C = add3(Jd.guard[h], mul3(toA, Rd.glove));
      }
    } else if (E.oc === 4) {
      // 打空：朝向偏了，拳顺着能打的最边上伸出去；射程不够，停在半空差那么一截；对手倒地、被抱住时从旁边过去
      // 差多少按画面上拳到的那一帧的两人位置算（模拟用的是前一帧；对手那一帧在轴转、横移时差得多，拳会远到送步也补不上）
      const fr = rep.frames, a0 = fr[Math.max(0, t1 - 1)].f[k], a1 = fr[t1].f[k], d1 = fr[t1].f[d];
      const gap = Math.hypot(d1[F.x] - a1[F.x], d1[F.y] - a1[F.y]) - reachLim(rep, k, E), d0 = d1;
      const off = wrapPi(Math.atan2(d0[F.y] - a0[F.y], d0[F.x] - a0[F.x]) - a0[F.th]), cone = E.typ === 1 ? SIM.ANG_HOOK || 0.785 : SIM.ANG_STRAIGHT || 0.524;
      if (Math.abs(off) > cone) {
        const ang = a0[F.th] + Math.sign(off) * cone, hd = Math.min(Math.hypot(C[0] - S[0], C[1] - S[1]), Ra.armLen * 0.8);   // 伸到胳膊能伸的地方，不用上步
        C = [S[0] + Math.cos(ang) * hd, S[1] + Math.sin(ang) * hd, C[2]];
      } else if (gap > 0) { C = add3(C, mul3(toA, gap + 0.02)); short = gap; }
      else C = add3(C, mul3(sv, 0.24));
    } else if (E.oc === 3 && E.ev === M.PARRY) C = add3(C, mul3(toA, 0.22));       // 被拍开：在脸前二十来厘米处碰上
    const toC = norm3(sub3(C, S)), inDir = E.typ === 1 ? norm3(add3(mul3(toC, 0.45), mul3(sv, -0.55))) : toC;
    return { C, T: sub3(C, mul3(inDir, Ra.glove * 1.05)), inDir, side: sv, deflect: sv, zone, gap: short };
  }
  const targetOf = (rep, k, E) => E._K || (E._K = contactOf(rep, k, E, false));
  // 要够到落点得往前送多少：胳膊伸直还差的那一截（满射程的拳约 20 厘米）
  // 太近时反过来往后让、上身直起来，给胳膊留出打出去的空间（近身短拳）
  function lungeFor(rep, k, E) {
    if (E.feint || !E.oc) return 0;
    const K = contactOf(rep, k, E, true), arm = rigOf(rep, k).armLen, rmin = arm * 0.4;
    const rmax = arm * (E.typ === 0 ? 0.96 : E.typ === 1 ? (E.tgt === 1 ? 0.95 : 0.9) : 0.86);     // 打身体的勾拳胳膊开得更大（兜着打），比打头的伸得远
    let L = 0;
    for (let it = 0; it < 5; it++) {          // 送一步肩膀大约往前走同样的距离，按实际姿势再校几次
      const Ja = bodyPose(rep, E.t1, k, { core: true, noDown: true, lunge: { E, L } }), D = sub3(K.T, Ja.sh[E.hand]), r = len3(D);
      if (r < rmin - 0.005) { L = clamp(L - (rmin - r), -0.2, 0.45); continue; }
      if (r <= rmax + 0.002) break;
      const b = dot3(D, Ja.Wv(1, 0, 0)), disc = b * b - (r * r - rmax * rmax);
      L = clamp(L + (disc >= 0 ? b - Math.sqrt(disc) : b), -0.2, 0.45);
    }
    return L;
  }



  // 倒地：从站姿过渡到躺在台上，0.6 秒（头部挨打往后倒，挨打的那一下头先甩出去；肝区击倒是跪下去、弯着腰）；站起来就倒过来放
  function downPose(rep, tf, k, dAt, uAt) {
    const tRef = uAt === null ? tf : uAt + 12;
    const J0 = armsOf(rep, tRef, k, bodyPose(rep, tRef, k, { noDown: true }));
    const knees = !!rep._mo[k].liver[Math.max(0, dAt - 1)];
    const d = uAt === null ? clamp((tf - dAt) / 12, 0, 1) : 1 - clamp((tf - uAt) / 12, 0, 1);
    return blendDown(J0, lyingPose(rep, k, dAt, J0, knees), d);
  }
  function lyingPose(rep, k, dAt, J0, knees) {
    const R = J0.R, o = rep.frames[clamp(dAt, 0, rep.frames.length - 1)].f[1 - k], x = J0.x, y = J0.y, RING = SIM.RING;
    const ang0 = Math.atan2(y - o[F.y], x - o[F.x]);
    // 往背向对手的方向倒；背后是围绳就往侧面倒
    const fits = ang => { const L = knees ? 0.7 : R.spine + 0.5; const hx = x + Math.cos(ang) * L, hy = y + Math.sin(ang) * L; return hx > 0.15 && hx < RING - 0.15 && hy > 0.15 && hy < RING - 0.15; };
    const ang = [0, Math.PI / 2, -Math.PI / 2, Math.PI].map(q => ang0 + q).find(fits) ?? ang0;
    const bx = Math.cos(ang), by = Math.sin(ang);
    const V = (b, l, z) => [bx * b - by * l, by * b + bx * l, z], P = (b, l, z) => add3([x, y, 0], V(b, l, z));   // b：往倒下的方向，l：往倒下方向的左侧
    if (!knees) {
      // 仰面倒下：面朝对手时左手在 −l 一侧，仰面躺下左右不变
      const sb = 0.15, cb = sb + R.spine;
      return { pel: P(sb, 0, 0.13), chest: P(cb, 0, 0.15), head: P(cb + R.neckL + R.headR, 0, 0.13),
        shL: P(cb, -R.shHalf, 0.13), shR: P(cb, R.shHalf, 0.13), hipL: P(sb, -R.hipHalf, 0.13), hipR: P(sb, R.hipHalf, 0.13),
        knees: [P(-0.3, -0.16, 0.3), P(-0.25, 0.14, 0.26)], ankles: [P(-0.62, -0.2, 0.07), P(-0.58, 0.18, 0.07)],
        elbows: [P(cb - 0.05, -0.36, 0.07), P(cb - 0.1, 0.35, 0.07)], gloves: [P(cb + 0.2, -0.5, 0.07), P(cb + 0.12, 0.52, 0.07)] };
    }
    // 跪下：膝盖着地，身子弯下去，左手撑地、右手捂着右肋（肝在右侧）
    const pz = 0.42, chest = P(-0.05 - R.spine * 0.9, 0, pz + R.spine * 0.36);
    return { pel: P(-0.05, 0, pz), chest, head: add3(chest, V(-0.16, 0, 0.02)), shL: add3(chest, V(0, -R.shHalf, 0)), shR: add3(chest, V(0, R.shHalf, 0)),
      hipL: P(-0.05, -R.hipHalf, pz), hipR: P(-0.05, R.hipHalf, pz), knees: [P(-0.4, -0.13, 0.06), P(-0.38, 0.13, 0.06)], ankles: [P(0.05, -0.14, 0.06), P(0.07, 0.14, 0.06)],
      elbows: [P(-0.62, -0.25, 0.3), P(-0.3, 0.26, 0.35)], gloves: [P(-0.82, -0.2, 0.07), P(-0.22, 0.17, 0.45)] };
  }
  function blendDown(J0, JL, d) {
    const zf = ease(clamp(d * 1.3, 0, 1)), rf = ease(clamp((d - 0.2) / 0.8, 0, 1));
    const mz = (a, b) => [lerp(a[0], b[0], rf), lerp(a[1], b[1], rf), lerp(a[2], b[2], zf)];          // 先腿软往下塌，再倒平
    const anat = h => (h === 0) === (J0.ls > 0) ? 0 : 1;                                                  // 前后手 → 左右手
    const J = { ...J0, armsDone: true, down: d, flash: false, face: { ...J0.face, hurt: true } };
    for (const key of ['pel', 'chest', 'head', 'shL', 'shR', 'hipL', 'hipR']) J[key] = mz(J0[key], JL[key]);
    J.ankles = [0, 1].map(i => lerp3(J0.ankles[i], JL.ankles[i], rf));
    J.knees = [0, 1].map(i => mz(J0.knees[i], JL.knees[i]));
    J.gloves = [0, 1].map(h => mz(J0.gloves[h], JL.gloves[anat(h)]));
    J.elbows = [0, 1].map(h => mz(J0.elbows[h], JL.elbows[anat(h)]));
    J.sh = J0.ls > 0 ? [J.shL, J.shR] : [J.shR, J.shL];
    return J;
  }

  // ---------------------------------------------------------------- 画三维拳手：每段肢体投影到画面，两个人的所有部件按远近一起排序再画
  // 一个人的各个部件，按画的先后排好：远侧胳膊 → 远侧腿 → 躯干 → 近侧腿 → 短裤 → 头 → 近侧胳膊。
  // 同一个人两只拳套、头和拳套离镜头差不多远时，按远近排会一帧一变、来回闪；正对镜头时两只胳膊分在画面两边，远近侧换了也看不出来。
  // 倒在地上时按远近排
  function rigParts(c, cam, J, id) {
    const L = look(id), R = J.R, ink = '#14161c', pj = p => cam.P(p[0], p[1], p[2]), dep = p => cam.W(p[0], p[1])[1];
    const lw = Math.max(1.2, 0.012 * pj(J.pel)[2]), dsh = dep(J.shL) - dep(J.shR), farL = dsh > 0;      // 左半边离镜头远
    const parts = [];
    for (let i = 0; i < 2; i++) {
      const hip = i ? J.hipR : J.hipL, kn = J.knees[i], an = J.ankles[i], far = (i === 0) === farL;
      parts.push({ k: J.k, tag: far ? 2 : 4, id: `腿${i}`, d: (dep(hip) + dep(kn) + dep(an)) / 3, f: () => leg3(c, pj, hip, kn, an, J.footYaw[i], J.heels[i], R, L, far, ink, lw) });
    }
    parts.push({ k: J.k, tag: 3, id: '躯干', d: dep(lerp3(J.pel, J.chest, 0.5)), f: () => torso3(c, cam, J, L, ink, lw) });
    parts.push({ k: J.k, tag: 5, id: '短裤', d: Math.min(dep(J.hipL), dep(J.hipR)) - 0.03, f: () => shorts3(c, cam, J, L, ink, lw) });
    parts.push({ k: J.k, tag: 6, id: '头', d: dep(J.head) - 0.02, f: () => head3(c, cam, J, L, ink, lw) });
    for (let h = 0; h < 2; h++) {
      const S = J.sh[h], E = J.elbows[h], G = J.gloves[h], far = ((h === 0) === (J.ls > 0)) === farL;
      // 上臂和小臂（连拳套）分开排：拳套绕到对方头后面时会被挡住
      parts.push({ k: J.k, tag: far ? 0 : 7, id: `上臂${h}`, d: dep(lerp3(S, E, 0.5)), f: () => arm3(c, pj, [S, lerp3(S, E, 0.4), E], [1.15, 1, 0.68], R, L, far, ink, lw) });
      parts.push({ k: J.k, tag: far ? 1 : 8, id: `小臂${h}`, d: dep(G) - 0.005, f: () => { arm3(c, pj, [E, lerp3(E, G, 0.35), G], [0.68, 0.74, 0.52], R, L, far, ink, lw); glove3(c, pj, E, G, R, L, far, ink, lw); } });
    }
    return J.down > 0.3 ? parts.sort((p, q) => q.d - p.d) : parts.sort((p, q) => p.tag - q.tag);
  }
  // 两个人的部件交错着画：每人自己的先后不变，两人之间看谁的下一个部件离镜头远就先画谁（拳套打到对方脸上、绕到对方头后面都对）
  function drawOrder(cam, Js, ids, c) {
    const [A, B] = Js.map((J, k) => rigParts(c, cam, J, ids[k])), out = [];
    let i = 0, j = 0;
    while (i < A.length || j < B.length) out.push(j >= B.length || (i < A.length && A[i].d >= B[j].d) ? A[i++] : B[j++]);
    return out;
  }
  const scr2 = q => [q[0], q[1]];
  function arm3(c, pj, pts, ws, R, L, far, ink, lw) {
    const P = pts.map(pj);
    limb(c, P.map(scr2), ws.map((m, i) => R.arm * m * P[i][2]), far ? shade(L.skin, -0.18) : L.skin, ink, lw);
  }
  function glove3(c, pj, E, G, R, L, far, ink, lw) {
    const g = pj(G), e = pj(E), r = R.glove * g[2];
    c.save(); c.translate(g[0], g[1]); c.rotate(Math.atan2(g[1] - e[1], g[0] - e[0]));
    c.fillStyle = far ? shade(L.glove, -0.22) : L.glove; c.strokeStyle = ink; c.lineWidth = lw;
    c.beginPath(); c.ellipse(0, 0, r * 1.12, r, 0, 0, Math.PI * 2); c.fill(); c.stroke();
    c.strokeStyle = far ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.32)'; c.lineWidth = lw * 1.3;      // 皮面高光
    c.beginPath(); c.ellipse(r * 0.1, -r * 0.1, r * 0.72, r * 0.58, 0, -Math.PI * 0.85, -Math.PI * 0.35); c.stroke();
    c.fillStyle = L.cuff; c.beginPath(); c.ellipse(-r * 0.82, 0, r * 0.32, r * 0.72, 0, 0, Math.PI * 2); c.fill();
    c.restore();
  }
  function leg3(c, pj, hip, kn, an, yaw, heel, R, L, far, ink, lw) {
    const P = [hip, lerp3(hip, kn, 0.4), kn, lerp3(kn, an, 0.3), an].map(pj), w = R.legW;
    limb(c, P.map(scr2), [1, 0.95, 0.62, 0.7, 0.42].map((m, i) => w * m * P[i][2]), far ? shade(L.skin, -0.18) : L.skin, ink, lw);   // 大腿 → 膝盖 → 小腿肚 → 脚踝
    const s1 = pj(lerp3(kn, an, 0.72));
    limb(c, [scr2(s1), scr2(P[4])], [w * 0.58 * s1[2], w * 0.5 * P[4][2]], far ? shade(L.sock, -0.2) : L.sock, ink, lw);
    // 鞋：沿脚尖方向；脚跟拧起来时后半截离地
    const lift = an[2] - R.ankle - heel, dx = Math.cos(yaw), dy = Math.sin(yaw);
    const toe = pj([an[0] + dx * 0.13, an[1] + dy * 0.13, 0.035 + lift]), hl = pj([an[0] - dx * 0.05, an[1] - dy * 0.05, 0.035 + lift + heel * 1.2]);
    c.fillStyle = far ? shade(L.shoe, -0.25) : L.shoe; c.strokeStyle = ink; c.lineWidth = lw;
    c.beginPath(); c.ellipse((toe[0] + hl[0]) / 2, (toe[1] + hl[1]) / 2, Math.hypot(toe[0] - hl[0], toe[1] - hl[1]) / 2 + 0.035 * toe[2], 0.045 * toe[2],
      Math.atan2(toe[1] - hl[1], toe[0] - hl[0]), 0, Math.PI * 2); c.fill(); c.stroke();
  }
  // 躯干：宽度 = 肩线（腰线）和胸厚在画面上横着的投影，侧身对镜头时窄、正对时宽
  function torsoWidths(cam, J) {
    const pj = p => cam.P(p[0], p[1], p[2]), R = J.R;
    const hpx = (p, v) => Math.abs(pj(add3(p, mul3(v, 0.1)))[0] - pj(p)[0]) / 0.1;
    const shAx = norm3(sub3(J.shL, J.shR)), fwT = norm3([shAx[1], -shAx[0], 0]);
    const hipAx = norm3(sub3(J.hipL, J.hipR)), fwH = norm3([hipAx[1], -hipAx[0], 0]);
    return { pj, shAx, fwT, hipAx, fwH, sh: p => 2 * (hpx(p, shAx) * R.shHalf * 0.85 + hpx(p, fwT) * R.chestD), hip: (p, e) => 2 * (hpx(p, hipAx) * (R.hipHalf + e) + hpx(p, fwH) * R.waistD) };
  }
  // 沿脊柱的一串点和宽度画成一个平滑的轮廓（腰细、胸宽，肩线是一道缓弧），比一串圆接起来干净
  function smoothBody(c, pts, ws, fill, ink, lw) {
    const n = pts.length, Lt = [], Rt = [];
    let dx = 0, dy = -1;
    for (let i = 0; i < n; i++) {
      const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)], d = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      dx = (b[0] - a[0]) / d; dy = (b[1] - a[1]) / d;
      Lt.push([pts[i][0] - dy * ws[i] / 2, pts[i][1] + dx * ws[i] / 2]); Rt.push([pts[i][0] + dy * ws[i] / 2, pts[i][1] - dx * ws[i] / 2]);
    }
    const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2], top = pts[n - 1], cap = [top[0] + dx * ws[n - 1] * 0.22, top[1] + dy * ws[n - 1] * 0.22];
    c.fillStyle = fill; c.strokeStyle = ink; c.lineWidth = lw; c.lineJoin = 'round';
    c.beginPath(); c.moveTo(Lt[0][0], Lt[0][1]);
    for (let i = 1; i < n - 1; i++) { const m = mid(Lt[i], Lt[i + 1]); c.quadraticCurveTo(Lt[i][0], Lt[i][1], m[0], m[1]); }
    c.lineTo(Lt[n - 1][0], Lt[n - 1][1]); c.quadraticCurveTo(cap[0], cap[1], Rt[n - 1][0], Rt[n - 1][1]);
    for (let i = n - 2; i > 0; i--) { const m = mid(Rt[i], Rt[i - 1]); c.quadraticCurveTo(Rt[i][0], Rt[i][1], m[0], m[1]); }
    c.lineTo(Rt[0][0], Rt[0][1]); c.closePath(); c.fill(); c.stroke();
  }
  function torso3(c, cam, J, L, ink, lw) {
    const T = torsoWidths(cam, J), pts = [add3(J.pel, [0, 0, 0.02]), lerp3(J.pel, J.chest, 0.45), lerp3(J.pel, J.chest, 0.8), lerp3(J.pel, J.chest, 0.97)];
    smoothBody(c, pts.map(p => scr2(T.pj(p))), [T.hip(pts[0], 0.07), T.hip(pts[1], 0.05), T.sh(pts[2]) * 1.02, T.sh(pts[3]) * 0.96], L.skin, ink, lw);
    // 胸肌、腹肌线：胸口朝着镜头时才看得到，越壮越明显
    const front = -(cam.W(J.chest[0] + T.fwT[0], J.chest[1] + T.fwT[1])[1] - cam.W(J.chest[0], J.chest[1])[1]);
    if (front > 0.2 && !J.down) {
      const R = J.R, a = T.pj(add3(lerp3(J.pel, J.chest, 0.82), mul3(T.fwT, R.chestD))), b = T.pj(add3(lerp3(J.pel, J.chest, 0.3), mul3(T.fwH, R.waistD)));
      c.strokeStyle = `rgba(0,0,0,${(0.05 + 0.07 * R.mus) * front})`; c.lineWidth = lw;
      c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke();
    }
  }
  function shorts3(c, cam, J, L, ink, lw) {
    const T = torsoWidths(cam, J), A = T.pj(add3(J.pel, [0, 0, 0.07])), B = T.pj(add3(J.pel, [0, 0, -0.17]));
    const w = T.hip(J.pel, 0.075) * Math.max(1, J.R.legW / 0.13);
    c.fillStyle = L.trunk; c.strokeStyle = ink; c.lineWidth = lw;
    if (J.down > 0.4) { c.beginPath(); c.ellipse(A[0], (A[1] + B[1]) / 2, w * 0.6, Math.abs(B[1] - A[1]) / 2 + 0.06 * A[2], 0, 0, Math.PI * 2); c.fill(); c.stroke(); return; }
    c.beginPath(); c.moveTo(A[0] - w / 2, A[1]); c.lineTo(A[0] + w / 2, A[1]); c.lineTo(B[0] + w * 0.57, B[1]); c.lineTo(B[0] - w * 0.57, B[1]); c.closePath(); c.fill(); c.stroke();
    c.fillStyle = L.stripe ? shade(L.trunk, -0.35) : L.trunk2; c.fillRect(A[0] - w / 2, A[1], w, 0.035 * A[2]);
    if (L.stripe) {                 // 短裤外侧的竖条（立绘里的紫条 / 黑金条），画在朝镜头的那条腿外侧
      const near = cam.W(J.hipL[0], J.hipL[1])[1] < cam.W(J.hipR[0], J.hipR[1])[1] ? J.hipL : J.hipR, sg = Math.sign(T.pj(near)[0] - A[0]) || 1;
      c.fillStyle = L.trunk2; c.beginPath();
      c.moveTo(A[0] + sg * w * 0.28, A[1] + 0.035 * A[2]); c.lineTo(A[0] + sg * w * 0.42, A[1] + 0.035 * A[2]);
      c.lineTo(B[0] + sg * w * 0.5, B[1]); c.lineTo(B[0] + sg * w * 0.34, B[1]); c.closePath(); c.fill();
      if (L.trim) { c.strokeStyle = L.trim; c.lineWidth = lw * 0.8; c.beginPath(); c.moveTo(A[0] + sg * w * 0.28, A[1]); c.lineTo(B[0] + sg * w * 0.34, B[1]); c.stroke(); }
    }
  }
  function head3(c, cam, J, L, ink, lw) {
    const R = J.R, pj = p => cam.P(p[0], p[1], p[2]), hp = pj(J.head);
    const nb = pj(lerp3(J.chest, J.head, 0.12)), nt = pj(lerp3(J.chest, J.head, 0.62));                  // 脖子：抗击打越好越粗
    limb(c, [scr2(nb), scr2(nt)], [R.neckW * 1.2 * nb[2], R.neckW * nt[2]], shade(L.skin, -0.06), ink, lw);
    const q0 = cam.W(J.head[0], J.head[1]), q1 = cam.W(J.head[0] + Math.cos(J.faceAng), J.head[1] + Math.sin(J.faceAng));
    drawHead3(c, hp[0], hp[1], R.headR * hp[2], Math.atan2(q1[1] - q0[1], q1[0] - q0[0]), L, J.face, lw, J.flash);
  }

  // 打击点：火花画在落点上（干净打中黄、反击红、擦到的小一号），被挡是一圈格挡的闷响，被拍开是一下白光，被躲开、打空是一道拳风
  function drawImpacts(c, rep, tf, cam, opts) {
    for (let k = 0; k < 2; k++) {
      const list = rep._mo[k].punches;
      for (let i = list.length - 1; i >= 0; i--) {
        const E = list[i];
        if (!E.oc) continue;
        const age = tf - E.t1;
        if (age < 0) continue;
        if (age > 12) break;
        const K = targetOf(rep, k, E), [px, py, sc] = cam.P(K.C[0], K.C[1], K.C[2]);
        c.save(); c.globalAlpha = clamp(1 - age / 10, 0, 1);
        if (E.oc === 1 && age < 3) {
          const rays = E.glance ? 6 : 12, r2 = (E.glance ? 0.1 : E.dmg >= 5 ? 0.3 : 0.18) * sc * (1 + age * 0.25);
          c.strokeStyle = E.ctr ? '#ff5a4f' : '#ffe066'; c.lineWidth = E.glance ? 2 : 3;
          for (let j = 0; j < rays; j++) { const an = j * 2 * Math.PI / rays + E.t1; c.beginPath(); c.moveTo(px + Math.cos(an) * 0.05 * sc, py + Math.sin(an) * 0.05 * sc); c.lineTo(px + Math.cos(an) * r2, py + Math.sin(an) * r2); c.stroke(); }
        } else if (E.oc === 2 && age < 4) {
          c.strokeStyle = 'rgba(190,220,255,0.9)'; c.lineWidth = 2.5; c.beginPath(); c.arc(px, py, (0.06 + 0.05 * age) * sc, 0, Math.PI * 2); c.stroke();
        } else if (E.oc === 3 && E.ev === M.PARRY && age < 3) {
          c.fillStyle = 'rgba(255,255,255,0.85)'; c.beginPath(); c.arc(px, py, (0.05 + 0.02 * age) * sc, 0, Math.PI * 2); c.fill();
        } else if (E.oc >= 3 && age < 3) {
          const q = cam.P(...add3(K.T, mul3(K.inDir, -0.3)));
          c.strokeStyle = 'rgba(255,255,255,0.45)'; c.lineWidth = 2;
          for (const o of [-0.035, 0.035]) { c.beginPath(); c.moveTo(q[0], q[1] + o * sc); c.lineTo(px + (px - q[0]) * 0.25, py + (py - q[1]) * 0.25 + o * sc); c.stroke(); }
        }
        if (opts.dmgNumbers && E.oc === 1 && age < 12) {
          c.globalAlpha = 1 - age / 12; c.font = `800 ${Math.max(12, 0.17 * sc)}px system-ui`; c.textAlign = 'center';
          c.fillStyle = 'rgba(0,0,0,0.5)'; c.fillText('−' + E.dmg, px + 1, py - (0.35 + age * 0.02) * sc + 1);
          c.fillStyle = E.ctr ? '#ffb3ad' : '#ffffff'; c.fillText('−' + E.dmg, px, py - (0.35 + age * 0.02) * sc);
        }
        c.restore();
      }
    }
  }

  // ---------------------------------------------------------------- 转播镜头：四个机位、跟拍、按两人距离推拉（整场预先算好，平滑）
  // 机位 rot = 0~3：绕擂台中心每次转 90°。世界俯视坐标 (x, y) → 机位坐标 (u 横向, v 纵深，越大越远)
  function toCam(rot, x, y) {
    const m = SIM.RING / 2, dx = x - m, dy = y - m;
    return rot === 1 ? [m - dy, m + dx] : rot === 2 ? [m - dx, m - dy] : rot === 3 ? [m + dy, m - dx] : [x, y];
  }
  // 这个机位下两人好不好看：连线越接近横向越好（不叠在一起），红角在左再加一点
  function camQuality(rot, f) {
    const a = toCam(rot, f[0][F.x], f[0][F.y]), b = toCam(rot, f[1][F.x], f[1][F.y]);
    const du = b[0] - a[0], d = Math.hypot(du, b[1] - a[1]) || 1;
    return Math.abs(du) / d + (du > 0 ? 0.25 : 0);
  }
  function bestRot(rep, from, to) {
    const sum = [0, 0, 0, 0];
    for (let t = from; t <= to; t++) for (let r = 0; r < 4; r++) sum[r] += camQuality(r, rep.frames[t].f);
    return sum.indexOf(Math.max(...sum));
  }
  // 导播：整场的机位预先排好（比赛是先算完再播的，导播看得到后面）。
  // 每帧每个机位一个难看分：两人连线越偏纵深越难看（叠在一起最难看），红角在右边扣一点；切一次镜头也有代价。
  // 用动态规划找总分最低的排法：一个镜头至少 1.5 秒，倒地时和比赛最后 1.5 秒不切，出拳当中切多付一点。
  const DIRECT = { minShot: 30, cut: 15, busyCut: 6, overlap: 1.5, flip: 0.05 };
  function directCams(rep) {
    const fr = rep.frames, n = fr.length, S = DIRECT.minShot + 1, NS = 4 * S;
    const roundStart = new Set(rep._prep.rounds);
    const tickCost = (r, f) => {
      const a = toCam(r, f[0][F.x], f[0][F.y]), b = toCam(r, f[1][F.x], f[1][F.y]);
      const du = b[0] - a[0], d = Math.hypot(du, b[1] - a[1]), flip = du < 0 ? DIRECT.flip : 0;
      return d < 0.4 ? flip : DIRECT.overlap * (1 - Math.abs(du) / d) ** 2 + flip;      // 贴身缠抱时哪个机位都差不多
    };
    // 状态 = 机位 r × 这个镜头已经持续的帧数 s（到 minShot 封顶）
    let cost = new Float64Array(NS).fill(Infinity), next = new Float64Array(NS);
    const from = new Int16Array(n * NS);
    for (let r = 0; r < 4; r++) cost[r * S] = tickCost(r, fr[0].f);
    for (let t = 1; t < n; t++) {
      const f = fr[t].f;
      next.fill(Infinity);
      const tc = [0, 1, 2, 3].map(r => tickCost(r, f));
      const canCut = t < n - S && !f.some(x => x[F.ph] === PH.DOWN);
      const busy = f.some(x => x[F.ph] === PH.STARTUP || x[F.ph] === PH.HITSTUN || (x[F.ph] === PH.RECOVERY && x[F.mv] < 12));
      const cutCost = roundStart.has(t) ? 0 : DIRECT.cut + (busy ? DIRECT.busyCut : 0);
      for (let r = 0; r < 4; r++) for (let s = 0; s < S; s++) {
        const i = r * S + s, c = cost[i];
        if (c === Infinity) continue;
        const j = r * S + Math.min(s + 1, S - 1), stay = c + tc[r];
        if (stay < next[j]) { next[j] = stay; from[t * NS + j] = i; }
        if (s === S - 1 && canCut) for (let r2 = 0; r2 < 4; r2++) {
          if (r2 === r) continue;
          const k = r2 * S, v = c + cutCost + tc[r2];
          if (v < next[k]) { next[k] = v; from[t * NS + k] = i; }
        }
      }
      [cost, next] = [next, cost];
    }
    const rots = new Int8Array(n);
    let i = cost.indexOf(Math.min(...cost));
    for (let t = n - 1; t >= 0; t--) { rots[t] = Math.floor(i / S); if (t) i = from[t * NS + i]; }
    return rots;
  }
  function cameraPath(rep) {
    const n = rep.frames.length, out = [], rots = directCams(rep);
    let cx = SIM.RING / 2, cy = SIM.RING / 2, zoom = 1.0;
    for (let t = 0; t < n; t++) {
      const f = rep.frames[t].f, rot = rots[t], cut = t > 0 && rot !== rots[t - 1];
      const a = toCam(rot, f[0][F.x], f[0][F.y]), b = toCam(rot, f[1][F.x], f[1][F.y]);
      const mx = (f[0][F.x] + f[1][F.x]) / 2, my = (f[0][F.y] + f[1][F.y]) / 2;
      const spread = Math.abs(a[0] - b[0]) + 0.5 * Math.abs(a[1] - b[1]);
      const z = clamp(2.2 / (spread + 1.4), 0.85, 1.2);
      const kd = f[0][F.ph] === PH.DOWN || f[1][F.ph] === PH.DOWN;
      if (t === 0 || cut) { cx = mx; cy = my; zoom = kd ? z * 1.15 : z; }               // 切镜头是硬切，新机位直接对好
      else { cx = lerp(cx, mx, 0.06); cy = lerp(cy, my, 0.06); zoom = lerp(zoom, kd ? z * 1.15 : z, 0.04); }
      out.push([cx, cy, zoom, rot]);
    }
    return out;
  }
  // 两帧之间插值；跨了切镜头就不插
  const lerpCam = (p, q, a) => p[3] !== q[3] ? p : [lerp(p[0], q[0], a), lerp(p[1], q[1], a), lerp(p[2], q[2], a), p[3]];
  // 世界 (x, y, z) → 屏幕：P 收世界坐标，Pc 收机位坐标（画擂台用，擂台是正方形，四个机位看过去一样）。z 是离台面的高度
  function makeCam(box, cam) {
    const [x0, y0, w, h] = box, [cx, cy, zoom] = cam, rot = cam[3] || 0, R = SIM.RING;
    const base = Math.min(h / 3.3, w / 5.2) * zoom;                     // 人物身高大约占画面一半
    let [cu, cv] = toCam(rot, cx, cy);
    // 镜头不拍出擂台：横向看得到的范围比台面窄时，画面边缘最多露出围绳外 0.5 米
    const half = w / 2 / (base * 0.975), edge = 0.5;
    cu = 2 * half >= R + 2 * edge ? R / 2 : clamp(cu, half - edge, R - half + edge);
    const Pc = (u, v, z = 0) => {
      const t = clamp((v - cv) / R + 0.5, -0.5, 1.5);                   // 相对镜头中心的深度：0 近、1 远
      const sc = base * (1.1 - 0.25 * t);
      const floorY = y0 + h * lerp(0.95, 0.6, t);
      return [x0 + w / 2 + (u - cu) * sc, floorY - z * sc, sc];
    };
    return { base, rot, Pc, W: (x, y) => toCam(rot, x, y), P: (x, y, z = 0) => { const [u, v] = toCam(rot, x, y); return Pc(u, v, z); } };
  }

  // ---------------------------------------------------------------- 擂台与观众席
  function drawArena(c, box, cam, tick, flash, gym, apron = '第 0 季 · SEASON 0') {
    const [x0, y0, w, h] = box, R = SIM.RING;
    if (gym) return drawGymRing(c, box, cam);
    // 背景与灯光
    const g = c.createLinearGradient(0, y0, 0, y0 + h); g.addColorStop(0, '#05070c'); g.addColorStop(0.55, '#141a26'); g.addColorStop(1, '#0b0e15');
    c.fillStyle = g; c.fillRect(x0, y0, w, h);
    for (let i = 0; i < 4; i++) {
      const lx = x0 + w * (0.15 + i * 0.23); const lg = c.createRadialGradient(lx, y0, 0, lx, y0, h * 0.7);
      lg.addColorStop(0, 'rgba(255,240,210,0.16)'); lg.addColorStop(1, 'rgba(255,240,210,0)'); c.fillStyle = lg; c.fillRect(x0, y0, w, h * 0.8);
    }
    // 观众：几排剪影，镜头移动时有视差
    const far = cam.Pc(R / 2, R + 1.6);
    const off = (cam.Pc(0, R)[0] - cam.Pc(R / 2, R)[0]) * 0.15;
    for (let row = 0; row < 5; row++) {
      const yy = far[1] - 18 - row * 22, rr = 9 - row * 0.8;
      for (let i = -2; i < 34; i++) {
        const xx = x0 + ((i + (row % 2) * 0.5) * w / 30 + off * (1 - row * 0.15)) % (w + 60) - 30;
        const hue = hash(i * 7 + row * 13);
        c.fillStyle = `rgba(${40 + hue * 50 | 0},${40 + hue * 30 | 0},${55 + hue * 40 | 0},${0.85 - row * 0.12})`;
        c.beginPath(); c.arc(xx, yy - rr, rr, 0, Math.PI * 2); c.fill();
        c.beginPath(); c.ellipse(xx, yy + rr * 0.9, rr * 1.6, rr * 1.1, 0, Math.PI, 0); c.fill();
        if (flash && hash(i * 31 + row * 17 + tick * 3) > 0.93) { c.fillStyle = 'rgba(255,255,255,0.9)'; c.beginPath(); c.arc(xx + 4, yy - rr, 3, 0, Math.PI * 2); c.fill(); }
      }
    }
    // 台面
    const q = [cam.Pc(0, 0), cam.Pc(R, 0), cam.Pc(R, R), cam.Pc(0, R)];
    const fg = c.createLinearGradient(0, q[3][1], 0, q[0][1]); fg.addColorStop(0, '#b9b2a0'); fg.addColorStop(1, '#ddd6c3');
    c.fillStyle = fg; c.beginPath(); q.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.closePath(); c.fill();
    // 台面中央标志
    const ctr = cam.Pc(R / 2, R / 2);
    c.save(); c.translate(ctr[0], ctr[1]); c.scale(1, 0.32);
    c.strokeStyle = 'rgba(140,40,30,0.35)'; c.lineWidth = 6; c.beginPath(); c.arc(0, 0, ctr[2] * 1.1, 0, Math.PI * 2); c.stroke();
    c.fillStyle = 'rgba(140,40,30,0.4)'; c.font = `900 ${ctr[2] * 0.3}px "Microsoft YaHei", system-ui`; c.textAlign = 'center'; c.fillText('AI 格斗家联赛', 0, ctr[2] * 0.18);
    c.restore();
    // 台裙
    c.fillStyle = '#7b2620'; c.beginPath(); c.moveTo(q[0][0], q[0][1]); c.lineTo(q[1][0], q[1][1]); c.lineTo(q[1][0], q[1][1] + q[1][2] * 0.35); c.lineTo(q[0][0], q[0][1] + q[0][2] * 0.35); c.closePath(); c.fill();
    c.fillStyle = 'rgba(255,255,255,0.75)'; c.font = `800 ${q[0][2] * 0.12}px "Microsoft YaHei"`; c.textAlign = 'center';
    c.fillText(apron, (q[0][0] + q[1][0]) / 2, q[0][1] + q[0][2] * 0.24);
    ropes(c, cam, 'far');
    return () => ropes(c, cam, 'near');
  }
  // 角柱颜色跟着世界坐标走：红角在 (0, 0)、蓝角在 (R, R)，换机位时颜色跟着转
  function postColor(rot, u, v) {
    const R = SIM.RING, [x, y] = toCam((4 - rot) % 4, u, v);
    return x < R / 2 && y < R / 2 ? '#d03b3b' : x > R / 2 && y > R / 2 ? '#2a78d6' : '#f4f4f4';
  }
  function ropes(c, cam, which) {
    const R = SIM.RING;
    const posts = which === 'far' ? [[0, R], [R, R]] : [[0, 0], [R, 0]];
    for (const [x, y] of posts) {
      const col = postColor(cam.rot, x, y);
      const a = cam.Pc(x, y), b = cam.Pc(x, y, 1.35);
      c.strokeStyle = '#2b2f37'; c.lineWidth = Math.max(4, 0.07 * a[2]); c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke();
      c.fillStyle = col; c.fillRect(b[0] - 0.06 * a[2], b[1], 0.12 * a[2], 1.1 * a[2]);
    }
    const strands = [[0.5, '#e8e8e8'], [0.88, '#d03b3b'], [1.25, '#e8e8e8']];
    const lines = which === 'far' ? [[[0, R], [R, R]], [[0, 0], [0, R]], [[R, 0], [R, R]]] : [[[0, 0], [R, 0]]];
    for (const [z, col] of strands) for (const [p0, p1] of lines) {
      const a = cam.Pc(p0[0], p0[1], z), b = cam.Pc(p1[0], p1[1], z);
      c.strokeStyle = which === 'near' ? col + 'cc' : col; c.lineWidth = Math.max(2, 0.025 * a[2]);
      c.beginPath(); c.moveTo(a[0], a[1]); c.quadraticCurveTo((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.03 * a[2], b[0], b[1]); c.stroke();
    }
  }

  // ---------------------------------------------------------------- 一帧比赛画面（不含转播包装）
  function drawBoutFrame(c, rep, tf, box, opts = {}) {
    prep(rep);
    const n = rep.frames.length, t = clamp(Math.floor(tf), 0, n - 1), a = tf - Math.floor(tf), tq = clamp(tf, 0, n - 1);
    const camv = opts.cam || lerpCam(rep._cam[t], rep._cam[Math.min(n - 1, t + 1)], a);
    const cam = makeCam(box, camv);
    const P = rep._prep;
    const bigNow = P.events.some(e => e.t <= t && t - e.t < 4 && e.big >= 2);
    const near = drawArena(c, box, cam, t, bigNow, opts.gym, opts.apron);
    const ids = opts.ids || [rep.bodies[0].id, rep.bodies[1].id];
    const Js = [0, 1].map(k => bodyPose(rep, tq, k));
    Js.forEach((J, k) => armsOf(rep, tq, k, J, Js[1 - k]));
    for (const J of Js) {
      const [sx, sy, sc] = cam.P(J.pel[0], J.pel[1], 0);
      c.fillStyle = 'rgba(0,0,0,0.28)'; c.beginPath(); c.ellipse(sx, sy, 0.42 * sc, 0.09 * sc, 0, 0, Math.PI * 2); c.fill();
    }
    // 两个人的所有部件按离镜头的远近一起排序：拳套打到对方脸上、绕到对方头后面，前后关系都对
    drawOrder(cam, Js, ids, c).forEach(p => p.f());
    near();
    drawImpacts(c, rep, tq, cam, opts);
    const tags = opts.labels ? Js.map((J, k) => {
      const [hx, hy, sc] = cam.P(J.head[0], J.head[1], J.head[2]);
      const du = cam.W(J.head[0] + Math.cos(J.faceAng), J.head[1] + Math.sin(J.faceAng))[0] - cam.W(J.head[0], J.head[1])[0];
      return { text: opts.labels[k], hx, hy, hr: J.R.headR * sc, side: du >= 0 ? -1 : 1, col: accent(ids[k]) };
    }) : [];
    nameTags(c, box, tags, box[1] + (opts.hud === false ? 6 : hudHeight(box) + 4));
    return { t, cam };
  }
  // 头顶名牌：主色小牌，画在最上层（不被人和围绳挡）；头顶放不下（会钻进顶栏）就挪到脑后那一侧，贴身时左右让开
  function nameTags(c, box, tags, top) {
    if (!tags.length) return;
    const fs = clamp(box[3] * 0.026, 11, 30), hh = fs * 1.4;
    c.font = `800 ${fs}px "Microsoft YaHei", system-ui`; c.textAlign = 'center'; c.textBaseline = 'middle';
    for (const g of tags) {
      g.w = c.measureText(g.text).width + fs * 1.1;
      g.x = g.hx; g.y = g.hy - g.hr * 1.9 - hh / 2;
      if (g.y < top + hh / 2) { g.y = Math.max(g.hy, top + hh / 2); g.x = g.hx + g.side * (g.hr * 1.5 + g.w / 2); }
    }
    if (tags.length === 2) {
      const [a, b] = tags[0].x <= tags[1].x ? tags : [tags[1], tags[0]];
      const need = (a.w + b.w) / 2 + fs * 0.4 - (b.x - a.x);
      if (need > 0 && Math.abs(a.y - b.y) < hh) { a.x -= need / 2; b.x += need / 2; }
    }
    for (const g of tags) {
      c.fillStyle = g.col; c.strokeStyle = 'rgba(0,0,0,0.55)'; c.lineWidth = 1.5;
      roundRect(c, g.x - g.w / 2, g.y - hh / 2, g.w, hh, hh / 2); c.fill(); c.stroke();
      c.fillStyle = '#ffffff'; c.fillText(g.text, g.x, g.y + fs * 0.04);
    }
    c.textBaseline = 'alphabetic';
  }

  // ---------------------------------------------------------------- 转播包装
  function roundRect(c, x, y, w, h, r) { c.beginPath(); c.roundRect(x, y, w, h, r); }
  const hudHeight = box => Math.max(58, box[2] * 0.062);
  function hud(c, rep, t, box, names, sub, ids = [rep.bodies[0].id, rep.bodies[1].id]) {      // ids：画哪种造型的配色（重演里陪练教练借用对手的身体）
    const [x0, y0, w] = box, fr = rep.frames[t];
    const H = hudHeight(box);
    c.fillStyle = 'rgba(6,8,14,0.78)'; c.fillRect(x0, y0, w, H);
    for (let k = 0; k < 2; k++) {
      const f = fr.f[k], right = k === 1, bw = w * 0.36, bx = right ? x0 + w - 16 - bw : x0 + 16;
      const L = look(ids[k]);
      c.fillStyle = L.accent || L.trunk; c.fillRect(right ? x0 + w - 8 : x0, y0, 8, H);
      c.font = `700 ${H * 0.24}px "Microsoft YaHei", system-ui`; c.textAlign = right ? 'right' : 'left'; c.fillStyle = '#fff';
      c.fillText(names[k], right ? bx + bw : bx, y0 + H * 0.3);
      const tags = `${SIM.postures[f[F.post]]}${f[F.st] < 0 ? '·反架' : ''}${sub ? ' · ' + SIM.plans[f[F.plan]] : ''}${f[F.tilt] ? ' · 上头' : ''}`;
      // 条从外侧往中间填，标签写在外侧：o 是从外侧量起的距离
      const O = o => right ? bx + bw - o : bx + o;
      const rect = (o1, o2, yy, hh) => c.fillRect(Math.min(O(o1), O(o2)), yy, Math.abs(o2 - o1), hh);
      const lab = H * 0.42;
      const meter = (label, o1, o2, yy, hh, v, col) => {
        c.font = `700 ${H * 0.14}px "Microsoft YaHei"`; c.fillStyle = 'rgba(255,255,255,0.6)'; c.textAlign = right ? 'right' : 'left'; c.textBaseline = 'middle';
        c.fillText(label, O(o1), yy + hh / 2); c.textBaseline = 'alphabetic';
        c.fillStyle = 'rgba(255,255,255,0.14)'; rect(o1 + lab, o2, yy, hh);
        c.fillStyle = col; rect(o1 + lab, o1 + lab + (o2 - o1 - lab) * clamp(v, 0, 1), yy, hh);
      };
      meter('体力', 0, bw, y0 + H * 0.4, H * 0.17, f[F.hp] / 100, L.accent || L.glove);
      const half = (bw - H * 0.3) / 2;
      meter('体能', 0, half, y0 + H * 0.7, H * 0.1, f[F.stam], '#cfcac0');
      meter('心态', half + H * 0.3, bw, y0 + H * 0.7, H * 0.1, f[F.comp] / 100, '#86b8ff');
      c.font = `${H * 0.16}px "Microsoft YaHei"`; c.fillStyle = 'rgba(255,255,255,0.75)'; c.textAlign = right ? 'left' : 'right';
      c.fillText(tags, right ? bx : bx + bw, y0 + H * 0.3);
    }
    const secs = fr.t * SIM.DT;
    c.fillStyle = '#e8b33c'; roundRect(c, x0 + w / 2 - H * 1.2, y0 + H * 0.12, H * 2.4, H * 0.76, 8); c.fill();
    c.fillStyle = '#111'; c.font = `800 ${H * 0.3}px system-ui`; c.textAlign = 'center';
    c.fillText(`R${fr.r + 1}  ${Math.floor(secs / 60)}:${String(Math.floor(secs % 60)).padStart(2, '0')}`, x0 + w / 2, y0 + H * 0.6);
  }
  function caption(c, box, text, sub, col, alpha = 1, opts = {}) {
    const [x0, y0, w, h] = box;
    c.globalAlpha = alpha;
    let fs = Math.round(w / 22);
    c.font = `900 ${fs}px "Microsoft YaHei", system-ui`; c.textAlign = 'center';
    const tw = c.measureText(text).width;                       // 长字幕缩小到一行放得下，不压满整个画面
    if (tw > w * 0.8) { fs = Math.floor(fs * w * 0.8 / tw); c.font = `900 ${fs}px "Microsoft YaHei", system-ui`; }
    const yy = y0 + h * (opts.y || 0.84);
    // 底下垫一条渐变暗带，字幕不跟擂台裙边上的字搅在一起
    const bh = fs * (sub ? 2.1 : 1.6), g = c.createLinearGradient(0, yy - fs * 1.1, 0, yy - fs * 1.1 + bh);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(0.25, 'rgba(0,0,0,0.55)'); g.addColorStop(0.8, 'rgba(0,0,0,0.55)'); g.addColorStop(1, 'rgba(0,0,0,0)');
    c.fillStyle = g; c.fillRect(x0, yy - fs * 1.1, w, bh);
    c.lineWidth = fs * 0.14; c.strokeStyle = 'rgba(0,0,0,0.75)'; c.strokeText(text, x0 + w / 2, yy);
    c.fillStyle = col || '#ffd24a'; c.fillText(text, x0 + w / 2, yy);
    if (sub) { c.font = `700 ${fs * 0.42}px "Microsoft YaHei"`; c.lineWidth = 4; c.strokeText(sub, x0 + w / 2, yy + fs * 0.6); c.fillStyle = '#fff'; c.fillText(sub, x0 + w / 2, yy + fs * 0.6); }
    c.globalAlpha = 1;
  }
  function banner(c, box, title, sub, prog) {
    const [x0, y0, w, h] = box, slide = easeOut(prog * 3) - ease((prog - 0.75) * 4);
    const bw = w * 0.5, bh = h * 0.16, bx = x0 + w / 2 - bw / 2 + (1 - slide) * w * 0.6, by = y0 + h * 0.4;
    c.globalAlpha = clamp(slide, 0, 1);
    c.fillStyle = 'rgba(10,12,18,0.85)'; c.fillRect(bx, by, bw, bh);
    c.fillStyle = '#e8b33c'; c.fillRect(bx, by, 10, bh);
    c.fillStyle = '#fff'; c.font = `900 ${bh * 0.42}px "Microsoft YaHei"`; c.textAlign = 'center'; c.fillText(title, bx + bw / 2, by + bh * 0.52);
    if (sub) { c.font = `${bh * 0.2}px "Microsoft YaHei"`; c.fillStyle = 'rgba(255,255,255,0.8)'; c.fillText(sub, bx + bw / 2, by + bh * 0.82); }
    c.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- 整场比赛的转播时间线：开场 → 回合 → 回合间 → 击倒回放 → 结果
  const INTRO = 4, BREAK = 4, REPLAY_SPAN = 34, REPLAY_SPEED = 0.3, RESULT = 6, DT_OUT = 1 / 20;
  function timeline(card) {
    const rep = prep(card), P = rep._prep, n = rep.frames.length;
    const segs = [{ type: 'intro', dur: INTRO }];
    const cuts = [...P.rounds.map(t => ({ t, kind: 'break' })), ...P.events.filter(e => e.big === 3).map(e => ({ t: e.t, kind: 'replay' }))].sort((a, b) => a.t - b.t);
    let cur = 0;
    for (const cut of cuts) {
      if (cut.t > cur) segs.push({ type: 'live', from: cur, to: cut.t, dur: (cut.t - cur) * DT_OUT });
      cur = cut.t;
      if (cut.kind === 'break') segs.push({ type: 'break', at: cut.t, dur: BREAK });
      else segs.push({ type: 'replay', from: Math.max(0, cut.t - REPLAY_SPAN), to: Math.min(n - 1, cut.t + 6), dur: (REPLAY_SPAN + 6) * DT_OUT / REPLAY_SPEED });
    }
    segs.push({ type: 'live', from: cur, to: n - 1, dur: (n - 1 - cur) * DT_OUT });
    segs.push({ type: 'result', dur: RESULT });
    let acc = 0; for (const s of segs) { s.start = acc; acc += s.dur; }
    return { segs, total: acc };
  }
  function segAt(tl, tOut) { for (const s of tl.segs) if (tOut < s.start + s.dur) return s; return tl.segs[tl.segs.length - 1]; }

  function roundStats(rep, upto) {
    const P = rep._prep, fr = rep.frames;
    const start = [0, ...P.rounds].filter(t => t < upto).pop() || 0;
    const st = [0, 1].map(() => ({ thrown: 0, landed: 0, power: 0, kd: 0 }));
    for (let t = start; t < upto; t++) for (let k = 0; k < 2; k++) {
      const f = fr[t].f[k];
      if (f[F.fired]) { st[k].thrown++; if (f[F.oc] === 1) { st[k].landed++; if ((f[F.fired] - 1) >= 2) st[k].power++; } }
      if (fr[t].f[1 - k][F.kd]) st[k].kd++;
    }
    return st;
  }

  // 输出时间 tOut（秒）→ 画一帧完整转播画面
  function renderFight(c, card, tOut, box, meta) {
    const rep = prep(card), tl = card._tl || (card._tl = timeline(card)), s = segAt(tl, tOut), n = rep.frames.length;
    const names = meta.names, ids = card.fighters;
    const local = tOut - s.start;
    if (s.type === 'intro') return drawIntro(c, card, box, local / s.dur, meta);
    const fopts = { ids, labels: meta.labels || ids, apron: meta.apron };
    if (s.type === 'result') {
      drawBoutFrame(c, rep, n - 1, box, fopts);
      hud(c, rep, n - 1, box, names, true);
      return drawResult(c, card, box, local / s.dur, meta);
    }
    if (s.type === 'break') {
      const t = Math.min(n - 1, s.at);
      drawBoutFrame(c, rep, t, box, fopts);
      return drawBreak(c, rep, t, box, local / s.dur, meta);
    }
    if (s.type === 'replay') {
      const tf = s.from + local / s.dur * (s.to - s.from);
      const ti = clamp(Math.floor(tf), 0, n - 1);
      const base = rep._cam[ti];
      if (s.rot === undefined) s.rot = bestRot(rep, s.from, s.to);       // 回放整段用一个看得最清楚的机位
      drawBoutFrame(c, rep, tf, box, { ...fopts, hud: false, cam: [base[0], base[1], base[2] * 1.45, s.rot] });
      const [x0, y0, w, h] = box;
      c.fillStyle = 'rgba(0,0,0,0.25)'; c.fillRect(x0, y0, w, h);
      c.fillStyle = '#d03b3b'; roundRect(c, x0 + 24, y0 + 24, 120, 40, 6); c.fill();
      c.fillStyle = '#fff'; c.font = '900 22px "Microsoft YaHei"'; c.textAlign = 'center'; c.fillText('慢 镜 回 放', x0 + 84, y0 + 52);
      return;
    }
    const tf = s.from + local / DT_OUT;
    const { t } = drawBoutFrame(c, rep, tf, box, fopts);
    hud(c, rep, t, box, names, true);
    // 回合开始横幅
    const roundStart = [0, ...rep._prep.rounds].filter(r => r <= t).pop();
    if (t - roundStart < 30) banner(c, box, `第 ${rep.frames[t].r + 1} 回合`, card.title, (t - roundStart) / 30);
    // 大事件字幕（底部，同一时刻只有一条）
    const ev = rep._prep.events.filter(e => e.t <= t && t - e.t < 28 && e.big >= 2).pop();
    if (ev) caption(c, box, `${names[ev.k]} ${ev.text}`, null, ev.big === 3 ? '#ff5a4f' : '#ffd24a', 1 - (t - ev.t) / 28);
    // 读秒：挂在倒地那一方的顶栏下面，不压在人身上
    for (let k = 0; k < 2; k++) {
      const f = rep.frames[t].f[k];
      if (f[F.ph] === PH.DOWN) countBadge(c, box, k, Math.floor((t - rep._prep.downAt[t][k]) / 6), names[k]);
    }
  }
  // 读秒数字和解说一致：倒地后每 6 帧数一个（读秒压缩成 3 秒），刚倒下那一下还没开始数
  function countBadge(c, box, k, cnt, name) {
    if (cnt < 1) return;
    const [x0, y0, w, h] = box, H = hudHeight(box), bw = w * 0.13, bh = h * 0.2;
    const bx = k === 0 ? x0 + 16 : x0 + w - 16 - bw, by = y0 + H + h * 0.02;
    c.fillStyle = 'rgba(8,10,16,0.82)'; roundRect(c, bx, by, bw, bh, 10); c.fill();
    c.fillStyle = '#ff5a4f'; c.fillRect(bx, by, bw, 4);
    c.textAlign = 'center'; c.fillStyle = '#ffffff'; c.font = `900 ${bh * 0.6}px system-ui`;
    c.fillText(String(Math.min(10, cnt)), bx + bw / 2, by + bh * 0.66);
    c.font = `700 ${bh * 0.13}px "Microsoft YaHei"`; c.fillStyle = 'rgba(255,255,255,0.75)';
    c.fillText(`${name} 倒地`, bx + bw / 2, by + bh * 0.88);
  }

  function drawIntro(c, card, box, p, meta) {
    const [x0, y0, w, h] = box, rep = card;
    const g = c.createLinearGradient(x0, y0, x0 + w, y0 + h); g.addColorStop(0, '#0b1020'); g.addColorStop(1, '#1a0d10');
    c.fillStyle = g; c.fillRect(x0, y0, w, h);
    // 两边半透明的角色色块
    for (let k = 0; k < 2; k++) {
      const L = look(card.fighters[k]);
      c.fillStyle = (L.accent || L.trunk) + '55'; c.beginPath();
      if (k === 0) { c.moveTo(x0, y0); c.lineTo(x0 + w * 0.42, y0); c.lineTo(x0 + w * 0.32, y0 + h); c.lineTo(x0, y0 + h); }
      else { c.moveTo(x0 + w, y0); c.lineTo(x0 + w * 0.58, y0); c.lineTo(x0 + w * 0.68, y0 + h); c.lineTo(x0 + w, y0 + h); }
      c.fill();
    }
    const s = h * 0.36, slide = easeOut(p * 2.5);
    for (let k = 0; k < 2; k++) {
      const id = card.fighters[k], B = build(rep.bodies[k]);
      const sx = k === 0 ? x0 + w * (0.02 + 0.16 * slide) : x0 + w * (0.98 - 0.16 * slide);
      if (meta.drawPortrait) {        // 有立绘就用立绘（试播集）
        const pw = w * 0.3, px = k === 0 ? x0 + w * (-0.06 + 0.08 * slide) : x0 + w * (0.76 - 0.08 * slide);
        meta.drawPortrait(c, id, px, y0, pw, h, slide);
      } else {
        const pose = { crouch: 0.05, lean: 0.03, twist: 0.3, head: [0, 0], lead: [0.33, B.shH + 0.02], rear: [0.17, B.shH + 0.08], feet: [0.24, -0.2, 0, 0], face: {} };
        drawBoxer(c, sx, y0 + h * 0.93, s, k === 0 ? 1 : -1, id, rep.bodies[k], pose);
      }
      c.font = `900 ${h * 0.065}px "Microsoft YaHei"`; c.textAlign = k === 0 ? 'left' : 'right'; c.fillStyle = '#fff';
      c.fillText(meta.names[k], k === 0 ? x0 + w * 0.03 : x0 + w * 0.97, y0 + h * 0.12);
    }
    // 中间的身材对比
    const rows = meta.tale || [];
    c.globalAlpha = clamp((p - 0.15) * 3, 0, 1);
    c.fillStyle = 'rgba(0,0,0,0.55)'; roundRect(c, x0 + w * 0.36, y0 + h * 0.2, w * 0.28, h * (0.08 + rows.length * 0.065), 10); c.fill();
    c.font = `800 ${h * 0.045}px "Microsoft YaHei"`; c.textAlign = 'center'; c.fillStyle = '#e8b33c';
    c.fillText(card.title + ' · 身材对比', x0 + w / 2, y0 + h * 0.26);
    rows.forEach((r, i) => {
      const yy = y0 + h * (0.33 + i * 0.065);
      c.font = `${h * 0.035}px "Microsoft YaHei"`; c.fillStyle = 'rgba(255,255,255,0.65)'; c.textAlign = 'center'; c.fillText(r[0], x0 + w / 2, yy);
      c.font = `700 ${h * 0.038}px "Microsoft YaHei"`; c.fillStyle = '#fff';
      c.textAlign = 'right'; c.fillText(r[1], x0 + w * 0.455, yy); c.textAlign = 'left'; c.fillText(r[2], x0 + w * 0.545, yy);
    });
    c.globalAlpha = 1;
    c.font = `900 ${h * 0.12}px system-ui`; c.textAlign = 'center'; c.fillStyle = '#e8b33c'; c.globalAlpha = clamp((p - 0.5) * 4, 0, 1);
    c.fillText('VS', x0 + w / 2, y0 + h * 0.86); c.globalAlpha = 1;
  }

  function drawBreak(c, rep, t, box, p, meta) {
    const [x0, y0, w, h] = box, st = roundStats(rep, t);
    c.fillStyle = `rgba(4,6,10,${0.55 * clamp(p * 4, 0, 1)})`; c.fillRect(x0, y0, w, h);
    c.globalAlpha = clamp(p * 4, 0, 1) * (1 - clamp((p - 0.85) * 7, 0, 1));
    const bw = w * 0.5, bh = h * 0.46, bx = x0 + w / 2 - bw / 2, by = y0 + h * 0.22;
    c.fillStyle = 'rgba(12,14,22,0.92)'; roundRect(c, bx, by, bw, bh, 12); c.fill();
    c.fillStyle = '#e8b33c'; c.font = `900 ${bh * 0.11}px "Microsoft YaHei"`; c.textAlign = 'center';
    c.fillText(`第 ${rep.frames[Math.max(0, t - 1)].r + 1} 回合结束`, x0 + w / 2, by + bh * 0.17);
    const rows = [['出拳', st[0].thrown, st[1].thrown], ['打中', st[0].landed, st[1].landed], ['重拳打中', st[0].power, st[1].power], ['击倒', st[0].kd, st[1].kd]];
    c.font = `700 ${bh * 0.075}px "Microsoft YaHei"`;
    c.fillStyle = '#fff'; c.textAlign = 'left'; c.fillText(meta.names[0], bx + bw * 0.06, by + bh * 0.32); c.textAlign = 'right'; c.fillText(meta.names[1], bx + bw * 0.94, by + bh * 0.32);
    rows.forEach((r, i) => {
      const yy = by + bh * (0.47 + i * 0.13);
      c.textAlign = 'center'; c.fillStyle = 'rgba(255,255,255,0.65)'; c.font = `${bh * 0.07}px "Microsoft YaHei"`; c.fillText(r[0], x0 + w / 2, yy);
      c.font = `800 ${bh * 0.09}px system-ui`; c.fillStyle = accent(rep.bodies[0].id); c.textAlign = 'left'; c.fillText(String(r[1]), bx + bw * 0.08, yy);
      c.fillStyle = accent(rep.bodies[1].id); c.textAlign = 'right'; c.fillText(String(r[2]), bx + bw * 0.92, yy);
    });
    c.globalAlpha = 1;
  }

  function drawResult(c, card, box, p, meta) {
    const [x0, y0, w, h] = box, r = card.result;
    c.fillStyle = `rgba(4,6,10,${0.6 * clamp(p * 3, 0, 1)})`; c.fillRect(x0, y0, w, h);
    const dec = r.method === '点数判定' || r.method === '平局';
    if (dec) {
      r.judges.forEach((jc, i) => {
        const q = clamp((p - 0.12 - i * 0.18) * 6, 0, 1); if (!q) return;
        const bw = w * 0.2, bx = x0 + w * (0.17 + i * 0.235), by = y0 + h * 0.24;
        c.globalAlpha = q; c.fillStyle = '#f4f1e8'; roundRect(c, bx, by, bw, h * 0.26, 8); c.fill();
        c.fillStyle = '#333'; c.font = `700 ${h * 0.035}px "Microsoft YaHei"`; c.textAlign = 'center'; c.fillText(`裁判 ${i + 1}`, bx + bw / 2, by + h * 0.06);
        c.font = `900 ${h * 0.08}px system-ui`; c.fillStyle = jc[0] > jc[1] ? look(card.fighters[0]).trunk : jc[1] > jc[0] ? look(card.fighters[1]).trunk : '#555';
        c.fillText(`${jc[0]} - ${jc[1]}`, bx + bw / 2, by + h * 0.18);
        c.globalAlpha = 1;
      });
    }
    const q = clamp((p - (dec ? 0.68 : 0.15)) * 5, 0, 1);
    if (q) {
      const text = r.winner ? `${meta.names[card.fighters.indexOf(r.winner)]}` : '平局';
      const sub = r.winner ? `${r.method} 获胜 · ${r.seconds} 秒` : '三位裁判打平';
      caption(c, box, text, sub, r.winner ? '#ffd24a' : '#ffffff', q, { y: dec ? 0.72 : 0.45 });
    }
  }

  // ---------------------------------------------------------------- 训练馆
  function gymWall(c, box) {
    const [x0, y0, w, h] = box;
    c.fillStyle = '#5b3a2e'; c.fillRect(x0, y0, w, h);
    const bh = h / 22, bw = bh * 2.6;
    c.strokeStyle = 'rgba(30,18,12,0.5)'; c.lineWidth = 1;
    for (let r = 0; r < 24; r++) for (let i = -1; i < w / bw + 1; i++) c.strokeRect(x0 + i * bw + (r % 2) * bw / 2, y0 + r * bh, bw, bh);
    const lg = c.createRadialGradient(x0 + w * 0.7, y0 + h * 0.15, 0, x0 + w * 0.7, y0 + h * 0.15, w * 0.8);
    lg.addColorStop(0, 'rgba(255,230,180,0.28)'); lg.addColorStop(1, 'rgba(0,0,0,0.35)'); c.fillStyle = lg; c.fillRect(x0, y0, w, h);
    c.fillStyle = 'rgba(232,179,60,0.85)'; c.fillRect(x0 + w * 0.06, y0 + h * 0.08, w * 0.16, h * 0.2);
    c.fillStyle = '#1b1b1b'; c.font = `900 ${h * 0.045}px "Microsoft YaHei"`; c.textAlign = 'center';
    c.fillText('AI 格斗家', x0 + w * 0.14, y0 + h * 0.16); c.font = `700 ${h * 0.03}px "Microsoft YaHei"`; c.fillText('联赛训练馆', x0 + w * 0.14, y0 + h * 0.22);
  }
  function gymFloor(c, box, floorY) {
    const [x0, y0, w, h] = box;
    const g = c.createLinearGradient(0, floorY, 0, y0 + h); g.addColorStop(0, '#2a3a4a'); g.addColorStop(1, '#1a2430');
    c.fillStyle = g; c.fillRect(x0, floorY, w, y0 + h - floorY);
    c.strokeStyle = 'rgba(255,255,255,0.06)'; c.lineWidth = 1;
    for (let i = 0; i < 12; i++) { c.beginPath(); c.moveTo(x0 + i * w / 11, floorY); c.lineTo(x0 + i * w / 11 - w * 0.08, y0 + h); c.stroke(); }
  }
  function drawGymRing(c, box, cam) {
    const R = SIM.RING;
    gymWall(c, box);
    const q = [cam.Pc(0, 0), cam.Pc(R, 0), cam.Pc(R, R), cam.Pc(0, R)];
    c.fillStyle = '#3c6e9e'; c.beginPath(); q.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.closePath(); c.fill();
    c.fillStyle = '#26476a'; c.beginPath(); c.moveTo(q[0][0], q[0][1]); c.lineTo(q[1][0], q[1][1]); c.lineTo(q[1][0], q[1][1] + q[1][2] * 0.3); c.lineTo(q[0][0], q[0][1] + q[0][2] * 0.3); c.closePath(); c.fill();
    ropes(c, cam, 'far');
    return () => ropes(c, cam, 'near');
  }

  // 训练场景用的姿势
  function guardPose(B) {
    return { crouch: 0.08, lean: 0.05, twist: 0.3, head: [0, 0], lead: [0.33, B.shH + 0.02], rear: [0.17, B.shH + 0.08], feet: [0.24, -0.2, 0, 0], face: {} };
  }
  function punchPose(B, k, e, reach) {
    const p = guardPose(B);
    const typ = SIM.TYPE[k], tgt = SIM.TGT[k], hand = SIM.HAND[k];
    const tx = reach || (B.ua + B.fa) * 0.95, ty = tgt === 0 ? B.shH + 0.1 : B.shH - 0.22, from = hand === 0 ? p.lead : p.rear;
    let g;
    if (typ === 0) g = [lerp(from[0], tx, e), lerp(from[1], ty, e)];
    else if (typ === 1) g = [from[0] + (tx * 0.8 - from[0]) * Math.sin(e * Math.PI / 2), lerp(from[1], ty, e) + 0.08 * Math.sin(e * Math.PI)];
    else g = [lerp(from[0], tx * 0.7, e), from[1] - 0.2 * Math.sin(e * Math.PI) * (1 - e) + (ty - from[1]) * e];
    if (hand === 0) { p.lead = g; p.leadBig = typ === 2; p.leadBody = tgt === 1; }
    else { p.rear = g; p.rearBig = typ >= 1; p.rearBody = tgt === 1; p.twist = 0.3 + 0.55 * e; }
    if (tgt === 1) { p.crouch += 0.25 * e; p.lean += 0.12 * e; }
    return p;
  }
  const punchEnv = ph => { const x = ph % 1; return x < 0.25 ? easeOut(x / 0.25) : 1 - ease((x - 0.25) / 0.45); };

  // 一个训练日的场景。entry：这一天的日志；ctx：{bodyOf, clips, techNames, names, plansFor(day,id), lastFight(day)}
  function drawTraining(c, id, entry, box, t, ctx) {
    const [x0, y0, w, h] = box, act = entry.activity, note = entry.note || {};
    const body = ctx.bodyOf(id), B = build(body);
    c.save(); c.beginPath(); c.rect(x0, y0, w, h); c.clip();
    const floor = y0 + h * 0.86, s = h / 2.55, cx = x0 + w * 0.42;
    const techs = (note.techs && note.techs.length ? note.techs : ['刺拳·头', '后手直拳·头']).map(n => ctx.techNames.indexOf(n)).filter(i => i >= 0 && i < 12);
    const curTech = techs[Math.floor(t * 1.2) % techs.length];
    let cap = act;
    const shadow = (x, sc = s) => { c.fillStyle = 'rgba(0,0,0,0.3)'; c.beginPath(); c.ellipse(x, floor, 0.38 * sc, 0.07 * sc, 0, 0, Math.PI * 2); c.fill(); };
    if (act.startsWith('对练')) {
      const clip = ctx.clips[`${entry.day}_${id}`];
      if (clip) {
        const rep = clip._rep || (clip._rep = prep({ frames: clip.frames, bodies: clip.bodies }));
        drawBoutFrame(c, rep, (t * 20) % rep.frames.length, box, { gym: true, hud: false, ids: [id, 'S'], labels: [ctx.names[id], clip.partner] });
      } else { gymWall(c, box); gymFloor(c, box, floor); shadow(cx); drawBoxer(c, cx, floor, s, 1, id, body, guardPose(B)); }
      if (note.iters) cap += ` · 真实训练 ${note.iters} 轮`;
    } else if (act === '看录像' || act === '实战复盘') {
      c.fillStyle = '#15171d'; c.fillRect(x0, y0, w, h);
      const tvx = x0 + w * 0.08, tvy = y0 + h * 0.08, tvw = w * 0.84, tvh = tvw * 9 / 16 * 0.85;
      c.fillStyle = '#000'; c.fillRect(tvx - 8, tvy - 8, tvw + 16, tvh + 16);
      const fg = ctx.lastFight(entry.day);
      if (fg) { c.save(); c.beginPath(); c.rect(tvx, tvy, tvw, tvh); c.clip(); prep(fg); drawBoutFrame(c, fg, (t * 20 + 200) % fg.frames.length, [tvx, tvy, tvw, tvh], { hud: false, ids: fg.fighters, labels: fg.fighters }); c.restore(); }
      c.fillStyle = 'rgba(255,255,255,0.05)'; c.fillRect(tvx, tvy, tvw, tvh * 0.5);
      c.fillStyle = '#3b2c22'; c.fillRect(cx - 0.5 * s, floor - 0.45 * s, 1.0 * s, 0.08 * s);
      const sit = guardPose(B); sit.crouch = 1.4; sit.lean = 0.05; sit.lead = [0.32, B.hipH * 0.55]; sit.rear = [0.25, B.hipH * 0.5]; sit.feet = [0.32, 0.18, 0, 0];
      drawBoxer(c, cx, floor, s * 0.8, -1, id, body, sit);
      if (act === '看录像' && note.target) cap += ` · 研究 ${ctx.names[note.target] || note.target}：了解度 ${(entry.before.knowledge[note.target] || 0).toFixed(2)} → ${(entry.after.knowledge[note.target] || 0).toFixed(2)}`;
    } else {
      gymWall(c, box); gymFloor(c, box, floor);
      const ph = t * Math.PI * 2;
      if (act === '休息') {
        c.fillStyle = '#d8dbe2'; c.fillRect(cx - 0.9 * s, floor - 0.42 * s, 1.9 * s, 0.14 * s);
        c.fillStyle = '#7a7f8a'; c.fillRect(cx - 0.95 * s, floor - 0.3 * s, 0.08 * s, 0.3 * s); c.fillRect(cx + 0.9 * s, floor - 0.3 * s, 0.08 * s, 0.3 * s);
        drawBoxer(c, cx + 0.3 * s, floor - 0.42 * s, s, 1, id, body, { down: 1 });
        c.fillStyle = 'rgba(255,255,255,0.8)'; c.font = `800 ${0.22 * s}px system-ui`; c.textAlign = 'center';
        for (let i = 0; i < 3; i++) { const q = (t * 0.5 + i / 3) % 1; c.globalAlpha = 1 - q; c.fillText('Z', cx - 0.7 * s + q * 0.4 * s, floor - (0.8 + q * 0.6) * s); }
        c.globalAlpha = 1;
        if (entry.after.injury > 0.01) { c.fillStyle = '#a8d8ff'; c.fillRect(cx - 0.1 * s, floor - 0.62 * s, 0.25 * s, 0.12 * s); }
      } else if (act === '跑步') {
        c.fillStyle = '#2a2d34'; c.fillRect(cx - 0.8 * s, floor - 0.12 * s, 1.6 * s, 0.12 * s);
        c.strokeStyle = 'rgba(255,255,255,0.25)'; c.lineWidth = 2; c.setLineDash([12, 10]); c.lineDashOffset = t * 80;
        c.beginPath(); c.moveTo(cx - 0.8 * s, floor - 0.06 * s); c.lineTo(cx + 0.8 * s, floor - 0.06 * s); c.stroke(); c.setLineDash([]);
        const sw = Math.sin(ph * 2);
        const p = guardPose(B); p.lean = 0.15; p.lead = [0.15 + 0.2 * sw, B.shH - 0.12]; p.rear = [0.1 - 0.2 * sw, B.shH - 0.12];
        p.feet = [0.05 + 0.32 * sw, 0.05 - 0.32 * sw, 0.12 * Math.max(0, Math.cos(ph * 2)), 0.12 * Math.max(0, -Math.cos(ph * 2))];
        drawBoxer(c, cx, floor - 0.12 * s - Math.abs(sw) * 0.03 * s, s, 1, id, body, p);
      } else if (act === '跳绳') {
        const jump = Math.abs(Math.sin(ph * 2)) * 0.12;
        const p = guardPose(B); p.lean = 0; p.lead = [0.2, B.hipH + 0.05]; p.rear = [-0.05, B.hipH + 0.05]; p.feet = [0.06, -0.06, 0, 0];
        shadow(cx); drawBoxer(c, cx, floor - jump * s, s, 1, id, body, p);
        const ry = Math.cos(ph * 2);
        c.strokeStyle = '#ffd24a'; c.lineWidth = 2.5; c.beginPath();
        c.ellipse(cx + 0.07 * s, floor - (B.hipH * 0.6 + jump) * s - ry * B.H * 0.42 * s, 0.32 * s, Math.abs(ry) * B.H * 0.5 * s + 4, 0, 0, Math.PI * 2); c.stroke();
      } else if (act === '器械力量' || act === '循环训练') {
        const squat = act === '循环训练' && Math.floor(t) % 2 === 1, up = (Math.sin(ph) + 1) / 2;
        const p = guardPose(B); p.lean = 0;
        if (squat) { p.crouch = 0.9 * up; p.lead = [0.3, B.shH - 0.2 * up]; p.rear = [0.25, B.shH - 0.2 * up]; }
        else { p.lead = [0.05, B.shH + 0.05 + up * 0.45]; p.rear = [-0.02, B.shH + 0.05 + up * 0.45]; }
        shadow(cx); drawBoxer(c, cx, floor, s, 1, id, body, p);
        if (!squat) {
          const by = floor - (B.shH + 0.05 + up * 0.45 - (p.crouch || 0) * 0.16 * B.H) * s;
          c.strokeStyle = '#cfd3da'; c.lineWidth = 0.045 * s; c.beginPath(); c.moveTo(cx - 0.75 * s, by); c.lineTo(cx + 0.8 * s, by); c.stroke();
          c.fillStyle = '#23262d'; for (const d of [-0.72, 0.72]) c.fillRect(cx + d * s - 0.06 * s, by - 0.22 * s, 0.12 * s, 0.44 * s);
        } else { c.fillStyle = '#23262d'; c.beginPath(); c.arc(cx + 0.3 * s, floor - (B.shH - 0.32 * up) * s, 0.1 * s, 0, Math.PI * 2); c.fill(); }
      } else if (act === '速度训练' || act === '招式训练') {
        const speed = act === '速度训练' ? 2.8 : 1.2;
        const k = act === '速度训练' ? [0, 2][Math.floor(t * speed) % 2] : curTech;
        const p = punchPose(B, k, punchEnv(t * speed));
        shadow(cx); drawBoxer(c, cx, floor, s, 1, id, body, p);
        c.fillStyle = 'rgba(190,210,240,0.12)'; c.fillRect(cx + 1.05 * s, floor - 2.05 * s, 0.1 * s, 2.05 * s);
        if (act === '速度训练') { c.strokeStyle = 'rgba(255,255,255,0.35)'; c.lineWidth = 2; for (let i = 0; i < 4; i++) { c.beginPath(); c.moveTo(cx + 0.55 * s, floor - (B.shH + 0.02 * i) * s); c.lineTo(cx + 0.9 * s, floor - (B.shH + 0.02 * i) * s); c.stroke(); } }
        else cap += ' · ' + ctx.techNames[k];
      } else if (act === '反应球') {
        const bx = cx + (0.55 + 0.3 * Math.cos(ph * 1.6)) * s, by2 = floor - (B.shH + 0.1 + 0.25 * Math.abs(Math.sin(ph * 3.2))) * s;
        c.fillStyle = '#4a4f5a'; c.fillRect(cx + 1.0 * s, floor - 2.0 * s, 0.08 * s, 2.0 * s);
        const p = guardPose(B); p.lead = [0.4 + 0.12 * Math.cos(ph * 1.6), B.shH + 0.12]; p.head = [-0.04 * Math.sin(ph * 1.6), 0];
        shadow(cx); drawBoxer(c, cx, floor, s, 1, id, body, p);
        c.fillStyle = '#ff5a4f'; c.beginPath(); c.arc(bx, by2, 0.06 * s, 0, Math.PI * 2); c.fill();
      } else if (act === '绳梯') {
        c.strokeStyle = '#ffd24a'; c.lineWidth = 2;
        for (let i = 0; i < 8; i++) { const lx = x0 + w * 0.08 + i * w * 0.105; c.strokeRect(lx, floor - 4, w * 0.1, 10); }
        const px = x0 + w * 0.12 + ((t * 0.22) % 1) * w * 0.75, q = ph * 4;
        const p = guardPose(B); p.crouch = 0.25; p.feet = [0.12 + 0.1 * Math.sin(q), -0.1 - 0.1 * Math.sin(q), 0.06 * Math.max(0, Math.cos(q)), 0.06 * Math.max(0, -Math.cos(q))];
        shadow(px); drawBoxer(c, px, floor, s, 1, id, body, p);
      } else if (act === '打沙袋' || act === '打手靶') {
        const e = punchEnv(t * 1.3), k = techs[Math.floor(t * 1.3) % techs.length];
        const p = punchPose(B, k, e, act === '打沙袋' ? 0.62 : 0.58);
        shadow(cx - 0.3 * s); drawBoxer(c, cx - 0.3 * s, floor, s, 1, id, body, p);
        if (act === '打沙袋') {
          const sw = Math.max(0, e - 0.6) * 0.15;
          c.strokeStyle = '#9aa0aa'; c.lineWidth = 2; c.beginPath(); c.moveTo(cx + 0.55 * s, y0); c.lineTo(cx + (0.55 + sw) * s, floor - 1.85 * s); c.stroke();
          c.fillStyle = '#8b2c22'; c.strokeStyle = '#14161c'; c.beginPath(); c.roundRect(cx + (0.38 + sw) * s, floor - 1.85 * s, 0.36 * s, 1.15 * s, 0.14 * s); c.fill(); c.stroke();
          c.fillStyle = 'rgba(255,255,255,0.6)'; c.fillRect(cx + (0.38 + sw) * s, floor - 1.55 * s, 0.36 * s, 0.05 * s);
        } else {
          const coach = guardPose(build(null)); coach.lead = [0.3, B.shH + (SIM.TGT[k] === 0 ? 0.1 : -0.2) + 0.03 * e]; coach.rear = [0.22, B.shH + 0.12]; coach.lean = -0.05 * e; coach.label = '教练';
          shadow(cx + 0.65 * s); drawBoxer(c, cx + 0.65 * s, floor, s, -1, 'S', null, coach);
        }
        cap += ' · ' + ctx.techNames[k];
      } else if (act === '战术会议') {
        const wbx = x0 + w * 0.32, wby = y0 + h * 0.12, wbw = w * 0.6, wbh = h * 0.55;
        c.fillStyle = '#f2f0ea'; c.fillRect(wbx, wby, wbw, wbh); c.strokeStyle = '#8a8f99'; c.lineWidth = 4; c.strokeRect(wbx, wby, wbw, wbh);
        const rank = ctx.plansFor(entry.day, id);
        c.fillStyle = '#333'; c.font = `800 ${h * 0.045}px "Microsoft YaHei"`; c.textAlign = 'left'; c.fillText('战术排名（模拟对手）', wbx + wbw * 0.05, wby + wbh * 0.12);
        rank.slice(0, 6).forEach(([pl, sc], i) => {
          const yy = wby + wbh * (0.2 + i * 0.13), grow = clamp(t / 1.5 - i * 0.1, 0, 1);
          c.font = `${h * 0.035}px "Microsoft YaHei"`; c.fillStyle = '#333'; c.fillText(SIM.plans[pl], wbx + wbw * 0.05, yy + h * 0.03);
          c.fillStyle = i === 0 ? '#1baf7a' : '#8aa4c8'; c.fillRect(wbx + wbw * 0.36, yy, Math.max(4, (sc + 1) / 2 * wbw * 0.55 * grow), h * 0.04);
        });
        const p = guardPose(B); p.lead = [0.55, B.shH + 0.25]; p.rear = [0.05, B.hipH + 0.1]; p.lean = 0;
        shadow(x0 + w * 0.18); drawBoxer(c, x0 + w * 0.18, floor, s, 1, id, body, p);
      } else { shadow(cx); drawBoxer(c, cx, floor, s, 1, id, body, guardPose(B)); }
    }
    // 夜里：兑现卡片（每 6 秒循环的后 35%）
    const night = entry.night, nt = (t % 6) / 6;
    if (night && nt > 0.65) {
      const q = clamp((nt - 0.65) * 8, 0, 1);
      c.globalAlpha = q; c.fillStyle = 'rgba(10,14,30,0.85)'; c.fillRect(x0, y0, w, h);
      c.fillStyle = '#f4e9b0'; c.beginPath(); c.arc(x0 + w * 0.14, y0 + h * 0.18, h * 0.07, 0, Math.PI * 2); c.fill();
      c.fillStyle = 'rgb(10,14,30)'; c.beginPath(); c.arc(x0 + w * 0.16, y0 + h * 0.16, h * 0.065, 0, Math.PI * 2); c.fill();
      c.fillStyle = '#fff'; c.font = `800 ${h * 0.06}px "Microsoft YaHei"`; c.textAlign = 'left';
      c.fillText(`夜里 · 睡 ${night.sleep} 小时`, x0 + w * 0.25, y0 + h * 0.2);
      c.font = `${h * 0.04}px "Microsoft YaHei"`; c.fillStyle = 'rgba(255,255,255,0.8)';
      c.fillText(`睡眠质量 ${Math.round(night.sleep_q * 100)}% · 饮食 ${night.diet} · 体重 ${night.mass} 公斤`, x0 + w * 0.25, y0 + h * 0.28);
      const gains = Object.entries(night.gained || {}).filter(([, v]) => v > 0.0005);
      if (night.skill_gain > 0.0005) gains.push(['熟练度合计', night.skill_gain]);
      c.fillStyle = '#9fe0b0'; c.font = `700 ${h * 0.045}px "Microsoft YaHei"`;
      if (!gains.length) { c.fillStyle = 'rgba(255,255,255,0.6)'; c.fillText('今晚没有可兑现的收益', x0 + w * 0.1, y0 + h * 0.45); }
      gains.slice(0, 6).forEach(([k, v], i) => c.fillText(`${k} +${v.toFixed(3)}`, x0 + w * (0.1 + (i % 2) * 0.42), y0 + h * (0.45 + Math.floor(i / 2) * 0.09)));
      c.globalAlpha = 1;
    }
    c.fillStyle = 'rgba(0,0,0,0.6)'; c.fillRect(x0, y0 + h - 34, w, 34);
    c.fillStyle = look(id).trunk; c.fillRect(x0, y0 + h - 34, 7, 34);
    c.fillStyle = '#fff'; c.font = `700 ${Math.max(13, h * 0.032)}px "Microsoft YaHei"`; c.textAlign = 'left';
    c.fillText(`${ctx.names[id]} · ${cap}`, x0 + 16, y0 + h - 11);
    if (note.injured) { c.fillStyle = '#ff6b5e'; c.font = `800 ${h * 0.04}px "Microsoft YaHei"`; c.textAlign = 'right'; c.fillText('训练中受伤！', x0 + w - 14, y0 + h * 0.07); }
    c.restore();
  }

  // ---------------------------------------------------------------- 声音事件表（导出视频时由 tools/make_audio.py 合成）
  // 时间都是这一条目内的输出时间（秒）。pan：-1 左 ~ 1 右，按命中位置在画面上的左右算
  const CN_NUM = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十'];
  function panAt(rep, tick, x, y) {
    const cam = rep._cam[clamp(tick, 0, rep._cam.length - 1)], rot = cam[3] || 0;
    return clamp((toCam(rot, x, y)[0] - toCam(rot, cam[0], cam[1])[0]) * 0.34 * cam[2], -0.85, 0.85);
  }
  function tickEvents(rep, tt, tOut, ev, extra = {}) {
    const fr = rep.frames[tt];
    for (let k = 0; k < 2; k++) {
      const f = fr.f[k], o = fr.f[1 - k];
      if (f[F.fired]) {
        const p = f[F.fired] - 1, pan = panAt(rep, tt, o[F.x], o[F.y]);
        const power = p >= 2, body = SIM.TGT[p] === 1;
        if (f[F.oc] === 1) ev.push({ t: tOut, type: 'hit', dmg: f[F.dmg], power, body, counter: !!f[F.ctr], pan, ...extra });
        else if (f[F.oc] === 2) ev.push({ t: tOut, type: 'block', power, pan, ...extra });
        else if (f[F.oc] === 3) ev.push({ t: tOut, type: 'evade', pan, ...extra });
        else ev.push({ t: tOut, type: 'whiff', power, pan, ...extra });
      }
      if (f[F.kd]) ev.push({ t: tOut, type: 'kd', pan: panAt(rep, tt, f[F.x], f[F.y]), ...extra });
    }
  }
  function fightAudio(card, from, to, meta) {
    const rep = prep(card), tl = card._tl || (card._tl = timeline(card)), ev = [], n = rep.frames.length;
    const names = meta.names, P = rep._prep, roundStarts = new Set([0, ...P.rounds]);
    const segs = tl.segs;
    segs.forEach((s, si) => {
      const s0 = s.start, s1 = s.start + s.dur;
      if (s1 < from || s0 > to) return;
      const inWin = t => t >= from && t < to;
      if (s.type === 'intro') {
        ev.push({ t: s0, type: 'bed', to: s1, kind: 'crowd', level: 0.55 });
        ev.push({ t: s0 + 0.1, type: 'sting' });
        ev.push({ t: s0 + 0.15, type: 'cheer', level: 0.6 });
        ev.push({ t: s0 + 0.5, type: 'voice', text: `${card.title}！红角，${names[0]}！蓝角，${names[1]}！` });
      } else if (s.type === 'live') {
        ev.push({ t: s0, type: 'bed', to: s1, kind: 'crowd', level: 0.6 });
        if (roundStarts.has(s.from)) {
          ev.push({ t: s0 + 0.02, type: 'bell', kind: 'start' });
          ev.push({ t: s0 + 0.35, type: 'voice', text: `第${CN_NUM[rep.frames[Math.min(n - 1, s.from)].r + 1]}回合` });
        }
        for (let tt = s.from; tt < s.to; tt++) {
          const tOut = s0 + (tt - s.from) * DT_OUT;
          if (!inWin(tOut)) continue;
          tickEvents(rep, tt, tOut, ev);
          const e = P.events.find(e => e.t === tt && e.big >= 2);
          if (e) ev.push({ t: tOut + 0.05, type: 'cheer', level: e.big === 3 ? 1.0 : 0.6 });
          if (e && e.big === 3) ev.push({ t: tOut + 0.25, type: 'voice', text: '击倒！' });
          for (let k = 0; k < 2; k++) {
            const f = rep.frames[tt].f[k];
            if (f[F.ph] === PH.DOWN) {
              const el = tt - P.downAt[tt][k];
              if (el > 0 && el % 6 === 0 && el / 6 <= 10) ev.push({ t: tOut, type: 'voice', text: CN_NUM[el / 6], count: true });
            }
          }
        }
        const next = segs[si + 1];
        if (next && (next.type === 'break' || next.type === 'result')) ev.push({ t: s1 - 0.02, type: 'bell', kind: next.type === 'result' ? 'final' : 'end' });
      } else if (s.type === 'break') {
        ev.push({ t: s0, type: 'bed', to: s1, kind: 'crowd', level: 0.42 });
      } else if (s.type === 'replay') {
        ev.push({ t: s0, type: 'bed', to: s1, kind: 'crowd', level: 0.25, muffled: true });
        ev.push({ t: s0, type: 'swoosh' });
        for (let tt = s.from; tt < s.to; tt++) {
          const tOut = s0 + (tt - s.from) / (s.to - s.from) * s.dur;
          if (inWin(tOut)) tickEvents(rep, tt, tOut, ev, { slow: true });
        }
      } else if (s.type === 'result') {
        ev.push({ t: s0, type: 'bed', to: s1, kind: 'crowd', level: 0.7 });
        ev.push({ t: s0 + 0.1, type: 'cheer', level: 1.0 });
        const r = card.result;
        const who = r.winner ? names[card.fighters.indexOf(r.winner)] : null;
        ev.push({ t: s0 + 0.8, type: 'voice', text: who ? `胜者，${who}！${r.method === '点数判定' ? '点数判定' : r.method}获胜！` : '三位裁判打平，平局！' });
      }
    });
    return ev.filter(e => e.t >= from - 0.01 && e.t < to);
  }

  // 训练日的声音：节奏和画面里的动作循环一一对应
  function trainingAudio(entry, id, from, to, pan, ctx) {
    const act = entry.activity, ev = [], note = entry.note || {};
    const every = (period, phase, type, extra = {}) => { for (let t = Math.ceil((from - phase) / period) * period + phase; t < to; t += period) if (t >= from) ev.push({ t, type, pan, ...extra }); };
    ev.push({ t: from, type: 'bed', to, kind: act.startsWith('看录像') || act === '实战复盘' ? 'tv' : 'gym', level: 0.5, pan });
    if (act === '跑步') { every(0.25, 0.0, 'step'); ev.push({ t: from, type: 'bed', to, kind: 'treadmill', level: 0.5, pan }); }
    else if (act === '跳绳') every(0.5, 0.0, 'rope');
    else if (act === '器械力量') every(1.0, 0.25, 'clank');
    else if (act === '循环训练') { every(2.0, 0.25, 'clank'); every(2.0, 1.25, 'thud', { soft: true }); }
    else if (act === '速度训练') every(1 / 2.8, 0.25 / 2.8, 'whiff', { power: false });
    else if (act === '招式训练') every(1 / 1.2, 0.25 / 1.2, 'whiff', { power: true });
    else if (act === '打沙袋') every(1 / 1.3, 0.25 / 1.3, 'bag');
    else if (act === '打手靶') every(1 / 1.3, 0.25 / 1.3, 'mitt');
    else if (act === '反应球') every(1 / 3.2, 0.0, 'ball');
    else if (act === '绳梯') every(0.125, 0.0, 'tap');
    else if (act === '战术会议') every(0.4, 0.2, 'marker');
    else if (act.startsWith('对练')) {
      const clip = ctx.clips[`${entry.day}_${id}`];
      if (clip) {
        const rep = clip._rep || (clip._rep = prep({ frames: clip.frames, bodies: clip.bodies }));
        const n = rep.frames.length;
        for (let t = Math.ceil(from * 20) / 20; t < to; t += 1 / 20) {
          const tt = Math.floor(t * 20) % n;
          const sub = [];
          tickEvents(rep, tt, t, sub, { gym: true });
          sub.forEach(e => { e.pan = pan + 0.3 * e.pan; ev.push(e); });
        }
      }
    }
    // 夜里的兑现卡片（6 秒循环的 3.9 秒处）
    for (let k = Math.floor(from / 6); k * 6 < to; k++) { const t = k * 6 + 3.9; if (t >= from && t < to && entry.night) ev.push({ t, type: 'night', pan }); }
    if (note.injured && from < 0.5) ev.push({ t: Math.max(from, 0.3), type: 'hurt', pan });
    return ev;
  }

  return { accent, tickEvents, fightAudio, trainingAudio, drawTraining, guardPose, punchPose, F, PH, LOOK, look, build, drawBoxer, drawHead, prep, bodyPose, armsOf, targetOf, rigOf, feetAt, reachLim, drawOrder, drawBoutFrame, makeCam, drawArena, hud, hudHeight, caption, banner, timeline, segAt, renderFight, setSim: s => { SIM = s; }, ease, easeOut, lerp, clamp, shade, mixHex, hash };
})();
if (typeof window !== 'undefined') window.VIZ = VIZ;
