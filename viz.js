// AI 格斗家联赛 · 转播画面渲染。观看页和导出视频共用这一份代码，所以网页里看到的就是导出的视频。
// 输入只有回放档重算出来的逐帧数据（只存动作的回放 → 重新模拟），画面全部由代码画出。
'use strict';
const VIZ = (() => {
  const F = { x: 0, y: 1, th: 2, hp: 3, stun: 4, stam: 5, comp: 6, ph: 7, mv: 8, g: 9, fired: 10, oc: 11, dmg: 12, act: 13, kd: 14, ev: 15, ctr: 16, fk: 17, post: 18, st: 19, tilt: 20, plan: 21 };
  const PH = { FREE: 0, STARTUP: 1, RECOVERY: 2, HITSTUN: 3, BLOCKSTUN: 4, DEFEND: 5, CLINCH: 6, DOWN: 7, MOVING: 8 };
  const M = { PARRY: 13, SLIP: 14, DUCK: 15, PULL: 16 };
  let SIM = null;                       // 由页面设置：{RING, DT, TYPE, TGT, HAND, ROUND_TICKS, actions, plans, postures, methods}
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
  // 体型：按身体参数算
  function build(body) {
    const reach = body ? body.reach : 0.75, mass = body ? body.mass : 72;
    const H = 1.72 + (reach - 0.75) * 0.75;
    return {
      H, hipH: 0.5 * H, shH: 0.81 * H, headR: 0.064 * H, neck: 0.03 * H,
      ua: 0.165 * H * (0.85 + 0.6 * (reach - 0.75)), fa: 0.15 * H * (0.85 + 0.6 * (reach - 0.75)),
      thigh: 0.25 * H, shin: 0.25 * H, chest: 0.14 + (mass - 70) * 0.004, waist: 0.12 + (mass - 70) * 0.0035,
      limb: 0.055 + (mass - 70) * 0.0012, glove: 0.072 + (mass - 70) * 0.0005,
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
    const leg = (fx, lift, far) => {
      const k = ik(hip[0] + (far ? -0.03 : 0.03), hipH, fx, 0.06 + lift, B.thigh, B.shin, 1);      // 膝盖往前弯
      const col = far ? shade(L.skin, -0.18) : L.skin;
      tube(c, [[X(hip[0] + (far ? -0.03 : 0.03)), Y(hipH)], [X(k.jx), Y(k.jy)], [X(fx), Y(0.06 + lift)]], B.limb * 1.15 * s, col, ink, lw);
      // 袜子和鞋
      tube(c, [[X(k.jx + (fx - k.jx) * 0.72), Y(k.jy + (0.06 + lift - k.jy) * 0.72)], [X(fx), Y(0.07 + lift)]], B.limb * 1.2 * s, far ? shade(L.sock, -0.2) : L.sock, ink, lw);
      c.fillStyle = far ? shade(L.shoe, -0.25) : L.shoe; c.strokeStyle = ink; c.lineWidth = lw;
      c.beginPath(); c.ellipse(X(fx + 0.04), Y(0.035 + lift), 0.11 * s, 0.045 * s, 0, 0, Math.PI * 2); c.fill(); c.stroke();
    };
    const arm = (sh, g, big, bodyShot, far, glove) => {
      const k = ik(sh[0], sh[1], g[0], g[1], B.ua, B.fa, -1);                                     // 手肘往下
      const col = far ? shade(L.skin, -0.18) : L.skin;
      tube(c, [[X(sh[0]), Y(sh[1])], [X(k.jx), Y(k.jy)], [X(g[0]), Y(g[1])]], B.limb * s, col, ink, lw);
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
    // 腹肌线
    c.strokeStyle = 'rgba(0,0,0,0.12)'; c.beginPath(); c.moveTo(X(shC[0] + tw * 0.55), Y(shC[1] - 0.15)); c.quadraticCurveTo(X(hip[0] + B.waist * 1.1), Y((hipH + shC[1]) / 2), X(hip[0] + B.waist * 0.8), Y(hipH + 0.08)); c.stroke();
    // 短裤
    c.fillStyle = L.trunk; c.strokeStyle = ink;
    c.beginPath(); c.moveTo(X(hip[0] - B.waist * 1.15), Y(hipH + 0.08)); c.lineTo(X(hip[0] + B.waist * 1.15), Y(hipH + 0.08));
    c.lineTo(X(hip[0] + B.waist * 1.3), Y(hipH - 0.2)); c.lineTo(X(hip[0] - B.waist * 1.3), Y(hipH - 0.2)); c.closePath(); c.fill(); c.stroke();
    c.fillStyle = L.stripe ? shade(L.trunk, -0.35) : L.trunk2; c.fillRect(Math.min(X(hip[0] - B.waist * 1.15), X(hip[0] + B.waist * 1.15)), Y(hipH + 0.08), Math.abs(X(hip[0] + B.waist * 1.15) - X(hip[0] - B.waist * 1.15)), 0.035 * s);
    if (L.stripe) {                 // 短裤外侧的竖条（立绘里的紫条 / 黑金条）
      c.fillStyle = L.trunk2; c.beginPath();
      c.moveTo(X(hip[0] + B.waist * 0.78), Y(hipH + 0.045)); c.lineTo(X(hip[0] + B.waist * 1.02), Y(hipH + 0.045));
      c.lineTo(X(hip[0] + B.waist * 1.18), Y(hipH - 0.2)); c.lineTo(X(hip[0] + B.waist * 0.9), Y(hipH - 0.2)); c.closePath(); c.fill();
      if (L.trim) { c.strokeStyle = L.trim; c.lineWidth = lw * 0.8; c.beginPath(); c.moveTo(X(hip[0] + B.waist * 0.78), Y(hipH + 0.045)); c.lineTo(X(hip[0] + B.waist * 0.9), Y(hipH - 0.2)); c.stroke(); c.strokeStyle = ink; }
    }
    // 近侧的腿
    leg(nearIsRear ? feet[1] : feet[0], nearIsRear ? feet[3] : feet[2], false);
    // 头
    drawHead(c, X(headC[0]), Y(headC[1]), B.headR * s, dir, L, pose.face || {}, lw, pose.flash);
    // 近侧的手最后画
    if (nearIsRear) arm(shRear, rear, pose.rearBig, pose.rearBody, false, L.glove); else arm(shLead, lead, pose.leadBig, pose.leadBody, false, L.glove);
    if (pose.label) {
      c.font = `700 ${Math.max(11, 0.15 * s)}px system-ui, "Microsoft YaHei"`; c.textAlign = 'center';
      const ly = Y(headC[1] + B.headR * 2.4), lx = X(headC[0]) + (pose.labelDx || 0);
      c.fillStyle = 'rgba(0,0,0,0.45)'; c.fillText(pose.label, lx + 1, ly + 1);
      c.fillStyle = '#ffffff'; c.fillText(pose.label, lx, ly);
    }
  }

  function drawHead(c, hx, hy, r, dir, L, face, lw, flash) {
    const ink = '#14161c';
    c.fillStyle = flash ? '#ffffff' : (face.tilt ? mixHex(L.skin, '#e0503a', 0.45) : L.skin);
    c.strokeStyle = ink; c.lineWidth = lw;
    c.beginPath(); c.ellipse(hx, hy, r * 0.95, r, 0, 0, Math.PI * 2); c.fill(); c.stroke();
    // 头发
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
    // 耳朵
    c.fillStyle = shade(L.skin, -0.1); c.beginPath(); c.ellipse(hx - dir * r * 0.15, hy + r * 0.05, r * 0.18, r * 0.26, 0, 0, Math.PI * 2); c.fill();
    // 眼睛、眉毛
    const ex = hx + dir * r * 0.55, ey = hy - r * 0.05;
    if (face.hurt) {
      c.strokeStyle = ink; c.lineWidth = lw * 1.2; c.beginPath(); c.moveTo(ex - r * 0.15, ey - r * 0.1); c.lineTo(ex + r * 0.15, ey + r * 0.1); c.moveTo(ex - r * 0.15, ey + r * 0.1); c.lineTo(ex + r * 0.15, ey - r * 0.1); c.stroke();
    } else {
      c.fillStyle = '#ffffff'; c.beginPath(); c.ellipse(ex, ey, r * 0.16, r * 0.13, 0, 0, Math.PI * 2); c.fill();
      c.fillStyle = ink; c.beginPath(); c.arc(ex + dir * r * 0.06, ey, r * 0.075, 0, Math.PI * 2); c.fill();
    }
    const br = (L.brow || 0) + (face.tilt ? -0.35 : 0) + (face.hurt ? 0.3 : 0);
    c.strokeStyle = ink; c.lineWidth = lw * 1.6;
    c.beginPath(); c.moveTo(ex - dir * r * 0.2, ey - r * 0.25 - br * r * 0.15); c.lineTo(ex + dir * r * 0.22, ey - r * 0.25 + br * r * 0.15); c.stroke();
    if (L.scar) {                   // 脸上的旧伤疤
      c.strokeStyle = '#a3262b'; c.lineWidth = lw * 1.1; c.beginPath();
      c.moveTo(hx + dir * r * 0.32, hy + r * 0.05); c.lineTo(hx + dir * r * 0.62, hy + r * 0.38); c.stroke();
      c.lineWidth = lw * 0.7; c.beginPath(); c.moveTo(hx + dir * r * 0.4, hy + r * 0.24); c.lineTo(hx + dir * r * 0.52, hy + r * 0.15); c.stroke();
    }
    if (L.smirk && !face.hurt) {    // 带着笑意打量对手
      c.strokeStyle = ink; c.lineWidth = lw; c.beginPath();
      c.moveTo(hx + dir * r * 0.5, hy + r * 0.5); c.quadraticCurveTo(hx + dir * r * 0.66, hy + r * 0.58, hx + dir * r * 0.84, hy + r * 0.4); c.stroke();
    }
    // 护齿
    c.fillStyle = L.trunk2; c.beginPath(); c.ellipse(hx + dir * r * 0.72, hy + r * 0.45, r * 0.2, r * 0.1, 0, 0, Math.PI * 2); c.fill();
    if (face.sweat) { c.fillStyle = 'rgba(160,210,255,0.85)'; c.beginPath(); c.ellipse(hx - dir * r * 0.6, hy - r * 0.9, r * 0.1, r * 0.16, 0, 0, Math.PI * 2); c.fill(); }
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
    tube(c, [[X(hipX), Y(hipY)], [X(0.25 - 0.2 * d), Y(lerp(0.3, 0.18, d))], [X(0.45 - 0.05 * d), Y(0.06)]], B.limb * 1.15 * s, L.skin, ink, lw);
    tube(c, [[X(hipX), Y(hipY)], [X(shx), Y(shy)]], B.chest * 1.6 * s, L.skin, ink, lw);
    c.fillStyle = L.trunk; c.beginPath(); c.ellipse(X(hipX), Y(hipY), B.waist * 1.3 * s, 0.1 * s, -dir * ang, 0, Math.PI * 2); c.fill();
    tube(c, [[X(shx), Y(shy)], [X(shx + 0.25), Y(shy + 0.15 - 0.2 * d)]], B.limb * s, L.skin, ink, lw);
    c.fillStyle = L.glove; c.beginPath(); c.arc(X(shx + 0.25), Y(shy + 0.15 - 0.2 * d), B.glove * s, 0, Math.PI * 2); c.fill(); c.stroke();
    drawHead(c, X(hx), Y(hy), B.headR * s, dir, L, { hurt: true }, lw);
  }

  function shade(hex, k) { return mixHex(hex, k < 0 ? '#000000' : '#ffffff', Math.abs(k)); }
  function mixHex(a, b, t) {
    const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    const r = Math.round(lerp(pa >> 16, pb >> 16, t)), g = Math.round(lerp((pa >> 8) & 255, (pb >> 8) & 255, t)), bl = Math.round(lerp(pa & 255, pb & 255, t));
    return '#' + ((1 << 24) + (r << 16) + (g << 8) + bl).toString(16).slice(1);
  }

  // ---------------------------------------------------------------- 回放预处理：每帧每人的动画参数
  function prep(rep) {
    if (rep._prep) return rep;
    const n = rep.frames.length, fr = rep.frames;
    const P = { since: [], fmove: [], suStart: [], recLen: [], hitAt: [], hitBody: [], downAt: [], upAt: [], travel: [], rounds: [] };
    const last = [-999, -999], lm = [0, 0], su = [-1, -1], hit = [-999, -999], hitB = [0, 0], dn = [-999, -999], up = [-999, -999], trav = [0, 0];
    for (let t = 0; t < n; t++) {
      const a = [], b = [], c2 = [], h = [], hb = [], d = [], u = [], tr = [];
      for (let k = 0; k < 2; k++) {
        const f = fr[t].f[k], pf = t ? fr[t - 1].f[k] : f;
        if (f[F.fired] > 0) { last[k] = t; lm[k] = f[F.fired] - 1; }
        if (f[F.ph] === PH.STARTUP && pf[F.ph] !== PH.STARTUP) su[k] = t;
        const o = fr[t].f[1 - k];
        if (o[F.fired] > 0 && o[F.oc] === 1) { hit[k] = t; hitB[k] = SIM.TGT[o[F.fired] - 1]; }
        if (f[F.ph] === PH.DOWN && pf[F.ph] !== PH.DOWN) dn[k] = t;
        if (f[F.ph] !== PH.DOWN && pf[F.ph] === PH.DOWN) up[k] = t;
        trav[k] += Math.hypot(f[F.x] - pf[F.x], f[F.y] - pf[F.y]);
        a.push(t - last[k]); b.push(lm[k]); c2.push(su[k]); h.push(hit[k]); hb.push(hitB[k]); d.push(dn[k]); u.push(up[k]); tr.push(trav[k]);
      }
      P.since.push(a); P.fmove.push(b); P.suStart.push(c2); P.hitAt.push(h); P.hitBody.push(hb); P.downAt.push(d); P.upAt.push(u); P.travel.push(tr);
    }
    // 每次出拳的收招长度（出拳到回到自由状态）
    for (let k = 0; k < 2; k++) {
      let t = 0;
      while (t < n) {
        if (fr[t].f[k][F.fired] > 0) { let e = t + 1; while (e < n && fr[e].f[k][F.ph] === PH.RECOVERY) e++; for (let q = t; q < e; q++) (P.recLen[q] = P.recLen[q] || [6, 6])[k] = Math.max(2, e - t); t = e; } else t++;
      }
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
    rep._cam = cameraPath(rep);
    return rep;
  }

  // ---------------------------------------------------------------- 从回放算姿势（连续时间）
  function poseAt(rep, tf, k) {
    const P = rep._prep, fr = rep.frames, n = fr.length;
    const t = clamp(Math.floor(tf), 0, n - 1), a = tf - Math.floor(tf);
    const f = fr[t].f[k], o = fr[t].f[1 - k];
    const B = build(rep.bodies[k]);
    const ph = f[F.ph], mv = f[F.mv], post = f[F.post];
    const dist = Math.hypot(f[F.x] - o[F.x], f[F.y] - o[F.y]);
    const sh = B.shH;
    // 待机：上下弹动、呼吸
    const bob = Math.sin((t + a) * 0.45 + k * 1.7) * 0.012;
    let crouch = 0.08, lean = 0.05, twist = 0.3, head = [0, bob];
    let lead = [0.33, sh + 0.02], rear = [0.17, sh + 0.08];
    if (post === 1) lead = [0.5, sh - 0.02];
    if (post === 2) { crouch = 0.45; lean = 0.18; lead = [0.27, sh + 0.06]; rear = [0.18, sh + 0.1]; }
    if (post === 3) { lead = [0.24, B.hipH + 0.12]; rear = [0.1, B.hipH + 0.18]; lean = 0.0; }
    if (f[F.g] === 1) { lead = [0.2, sh + 0.17]; rear = [0.18, sh + 0.15]; crouch += 0.08; }
    if (f[F.g] === 2) { lead = [0.2, sh - 0.12]; rear = [0.15, sh - 0.16]; crouch += 0.12; lean += 0.08; }
    // 头部闪躲
    if (ph === PH.DEFEND || (ph === PH.RECOVERY && mv >= 13 && mv <= 16)) {
      const k2 = ph === PH.DEFEND ? 1 : 0.4;
      if (mv === M.SLIP) { head = [-0.05 * k2, -0.08 * k2]; lean += 0.12 * k2; }
      if (mv === M.DUCK) { crouch += 0.8 * k2; lean += 0.32 * k2; }
      if (mv === M.PULL) { lean -= 0.3 * k2; }
      if (mv === M.PARRY) { lead = [0.4, sh + 0.12 + 0.05 * Math.sin((t + a) * 2)]; }
    }
    // 出拳：起手蓄力 → 出拳（快）→ 收招（慢）
    const pose = { crouch, lean, twist, head, lead, rear, face: {} };
    const since = P.since[t][k] + a, fm = P.fmove[t][k];
    const recLen = (P.recLen[t] && P.recLen[t][k]) || 6;
    let ext = 0;
    if (ph === PH.RECOVERY && mv < 12 && since < recLen + 1) ext = since < 1.2 ? easeOut(since / 1.2) : 1 - ease((since - 1.2) / Math.max(1, recLen - 1.2));
    if (ph === PH.STARTUP && mv < 12) {
      const prog = clamp((t + a - P.suStart[t][k]) / 4, 0, 1);
      const hand = SIM.HAND[mv];
      if (hand === 0) pose.lead = [lead[0] - 0.06 * prog, lead[1] + 0.02 * prog]; else { pose.rear = [rear[0] - 0.1 * prog, rear[1] - 0.02 * prog]; pose.twist = twist - 0.15 * prog; }
      if (SIM.TYPE[mv] === 2) pose.crouch += 0.15 * prog;
    }
    if (ext > 0.001) {
      const typ = SIM.TYPE[fm], tgt = SIM.TGT[fm], hand = SIM.HAND[fm];
      const reachMax = (B.ua + B.fa) * 0.98 + (hand === 1 ? 0.12 : 0.05);
      const tx = Math.min(reachMax + 0.15, Math.max(0.35, dist - 0.2)), ty = tgt === 0 ? sh + 0.12 : sh - 0.22;
      const from = hand === 0 ? lead : rear;
      let g;
      if (typ === 0) g = [lerp(from[0], tx, ext), lerp(from[1], ty, ext)];
      else if (typ === 1) g = [from[0] + (tx * 0.82 - from[0]) * Math.sin(ext * Math.PI / 2), lerp(from[1], ty, ext) + 0.08 * Math.sin(ext * Math.PI)];
      else g = [lerp(from[0], tx * 0.7, ext), from[1] - 0.22 * Math.sin(ext * Math.PI) * (1 - ext) + (ty - from[1]) * ext];
      if (hand === 0) { pose.lead = g; pose.leadBig = typ === 2; pose.leadBody = tgt === 1; }
      else { pose.rear = g; pose.rearBig = typ >= 1; pose.rearBody = tgt === 1; pose.twist = twist + 0.55 * ext; pose.lean = lean + 0.1 * ext; }
      if (tgt === 1) { pose.crouch += 0.25 * ext; pose.lean += 0.12 * ext; }
      if (typ === 2) pose.crouch += 0.1 * (1 - ext);
    }
    // 挨打：头往后甩、身体回弹；打身体就弯腰
    const sinceHit = t + a - P.hitAt[t][k];
    if (sinceHit >= 0 && sinceHit < 8) {
      const amp = Math.exp(-sinceHit / 2.5) * Math.cos(sinceHit * 0.9);
      if (P.hitBody[t][k]) { pose.lean += 0.25 * amp; pose.crouch += 0.2 * Math.max(0, amp); }
      else { pose.head = [pose.head[0] - 0.09 * amp, pose.head[1] + 0.02 * amp]; pose.lean -= 0.22 * amp; }
      pose.flash = sinceHit < 1.5 && o[F.oc] === 1;
    }
    // 缠抱：抱在一起
    if (ph === PH.CLINCH) { pose.lean = 0.35; pose.lead = [dist + 0.05, sh - 0.05]; pose.rear = [dist - 0.05, sh - 0.12]; pose.crouch = 0.15; }
    // 倒地与站起
    const sinceDown = t + a - P.downAt[t][k], sinceUp = t + a - P.upAt[t][k];
    if (ph === PH.DOWN) pose.down = clamp(sinceDown / 7, 0.02, 1);
    else if (sinceUp >= 0 && sinceUp < 12) pose.down = 1 - ease(sinceUp / 12);
    // 步伐：按走过的距离迈步
    const tr = P.travel[t][k] + (P.travel[Math.min(n - 1, t + 1)][k] - P.travel[t][k]) * a;
    const cyc = tr / 0.32 * Math.PI * 2;
    const moving = t > 0 && Math.abs(P.travel[Math.min(n - 1, t + 1)][k] - P.travel[Math.max(0, t - 1)][k]) > 0.01;
    const stride = moving ? 0.07 : 0.0;
    const wide = post === 2 ? 0.04 : 0;
    pose.feet = [0.25 + wide + stride * Math.sin(cyc), -0.22 - wide + stride * Math.sin(cyc + Math.PI),
      moving ? 0.035 * Math.max(0, Math.cos(cyc)) : 0, moving ? 0.035 * Math.max(0, Math.cos(cyc + Math.PI)) : 0];
    // 表情
    pose.face = { hurt: f[F.stun] > 60, tilt: f[F.tilt], sweat: f[F.stam] < 0.3 };
    return pose;
  }

  // ---------------------------------------------------------------- 转播镜头：跟拍、按两人距离推拉（整场预先算好，平滑）
  function cameraPath(rep) {
    const n = rep.frames.length, out = [];
    let cx = SIM.RING / 2, cy = SIM.RING / 2, zoom = 1.0;
    for (let t = 0; t < n; t++) {
      const f = rep.frames[t].f;
      const mx = (f[0][F.x] + f[1][F.x]) / 2, my = (f[0][F.y] + f[1][F.y]) / 2;
      const spread = Math.abs(f[0][F.x] - f[1][F.x]) + 0.5 * Math.abs(f[0][F.y] - f[1][F.y]);
      const z = clamp(2.2 / (spread + 1.4), 0.85, 1.2);
      const kd = f[0][F.ph] === PH.DOWN || f[1][F.ph] === PH.DOWN;
      cx = lerp(cx, mx, 0.06); cy = lerp(cy, my, 0.05); zoom = lerp(zoom, kd ? z * 1.15 : z, 0.04);
      out.push([cx, cy, zoom]);
    }
    return out;
  }
  // 世界 (x, y, z) → 屏幕。y 越大越远（俯视坐标），z 是离台面的高度
  function makeCam(box, cam) {
    const [x0, y0, w, h] = box, [cx, cy, zoom] = cam;
    const base = Math.min(h / 3.3, w / 5.2) * zoom;                     // 人物身高大约占画面一半
    return {
      base,
      P: (x, y, z = 0) => {
        const t = clamp((y - cy) / SIM.RING + 0.5, -0.5, 1.5);          // 相对镜头中心的深度：0 近、1 远
        const sc = base * (1.1 - 0.25 * t);
        const floorY = y0 + h * lerp(0.95, 0.6, t);
        return [x0 + w / 2 + (x - cx) * sc, floorY - z * sc, sc];
      },
    };
  }

  // ---------------------------------------------------------------- 擂台与观众席
  function drawArena(c, box, cam, tick, flash, gym) {
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
    const far = cam.P(R / 2, R + 1.6);
    const off = (cam.P(0, R)[0] - cam.P(R / 2, R)[0]) * 0.15;
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
    const q = [cam.P(0, 0), cam.P(R, 0), cam.P(R, R), cam.P(0, R)];
    const fg = c.createLinearGradient(0, q[3][1], 0, q[0][1]); fg.addColorStop(0, '#b9b2a0'); fg.addColorStop(1, '#ddd6c3');
    c.fillStyle = fg; c.beginPath(); q.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.closePath(); c.fill();
    // 台面中央标志
    const ctr = cam.P(R / 2, R / 2);
    c.save(); c.translate(ctr[0], ctr[1]); c.scale(1, 0.32);
    c.strokeStyle = 'rgba(140,40,30,0.35)'; c.lineWidth = 6; c.beginPath(); c.arc(0, 0, ctr[2] * 1.1, 0, Math.PI * 2); c.stroke();
    c.fillStyle = 'rgba(140,40,30,0.4)'; c.font = `900 ${ctr[2] * 0.3}px "Microsoft YaHei", system-ui`; c.textAlign = 'center'; c.fillText('AI 格斗家联赛', 0, ctr[2] * 0.18);
    c.restore();
    // 台裙
    c.fillStyle = '#7b2620'; c.beginPath(); c.moveTo(q[0][0], q[0][1]); c.lineTo(q[1][0], q[1][1]); c.lineTo(q[1][0], q[1][1] + q[1][2] * 0.35); c.lineTo(q[0][0], q[0][1] + q[0][2] * 0.35); c.closePath(); c.fill();
    c.fillStyle = 'rgba(255,255,255,0.75)'; c.font = `800 ${q[0][2] * 0.12}px "Microsoft YaHei"`; c.textAlign = 'center';
    c.fillText('第 0 季 · SEASON 0', (q[0][0] + q[1][0]) / 2, q[0][1] + q[0][2] * 0.24);
    ropes(c, cam, 'far');
    return () => ropes(c, cam, 'near');
  }
  function ropes(c, cam, which) {
    const R = SIM.RING;
    const posts = which === 'far' ? [[0, R, '#f4f4f4'], [R, R, '#2a78d6']] : [[0, 0, '#d03b3b'], [R, 0, '#f4f4f4']];
    for (const [x, y, col] of posts) {
      const a = cam.P(x, y), b = cam.P(x, y, 1.35);
      c.strokeStyle = '#2b2f37'; c.lineWidth = Math.max(4, 0.07 * a[2]); c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke();
      c.fillStyle = col; c.fillRect(b[0] - 0.06 * a[2], b[1], 0.12 * a[2], 1.1 * a[2]);
    }
    const strands = [[0.5, '#e8e8e8'], [0.88, '#d03b3b'], [1.25, '#e8e8e8']];
    const lines = which === 'far' ? [[[0, R], [R, R]], [[0, 0], [0, R]], [[R, 0], [R, R]]] : [[[0, 0], [R, 0]]];
    for (const [z, col] of strands) for (const [p0, p1] of lines) {
      const a = cam.P(p0[0], p0[1], z), b = cam.P(p1[0], p1[1], z);
      c.strokeStyle = which === 'near' ? col + 'cc' : col; c.lineWidth = Math.max(2, 0.025 * a[2]);
      c.beginPath(); c.moveTo(a[0], a[1]); c.quadraticCurveTo((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.03 * a[2], b[0], b[1]); c.stroke();
    }
  }

  // ---------------------------------------------------------------- 一帧比赛画面（不含转播包装）
  function drawBoutFrame(c, rep, tf, box, opts = {}) {
    prep(rep);
    const n = rep.frames.length, t = clamp(Math.floor(tf), 0, n - 1), a = tf - Math.floor(tf);
    const camv = opts.cam || lerp3(rep._cam[t], rep._cam[Math.min(n - 1, t + 1)], a);
    const cam = makeCam(box, camv);
    const P = rep._prep;
    const bigNow = P.events.some(e => e.t <= t && t - e.t < 4 && e.big >= 2);
    const near = drawArena(c, box, cam, t, bigNow, opts.gym);
    const fr = rep.frames[t], nx = rep.frames[Math.min(n - 1, t + 1)];
    const pos = [0, 1].map(k => [lerp(fr.f[k][F.x], nx.f[k][F.x], a), lerp(fr.f[k][F.y], nx.f[k][F.y], a)]);
    const ids = opts.ids || [rep.bodies[0].id, rep.bodies[1].id];
    const order = [0, 1].sort((i, j) => pos[j][1] - pos[i][1]);
    for (const k of order) {
      const [sx, sy, sc] = cam.P(pos[k][0], pos[k][1]);
      const [ox] = cam.P(pos[1 - k][0], pos[1 - k][1]);
      const dir = Math.abs(ox - sx) > 3 ? Math.sign(ox - sx) : (Math.cos(fr.f[k][F.th]) >= 0 ? 1 : -1);
      c.fillStyle = 'rgba(0,0,0,0.28)'; c.beginPath(); c.ellipse(sx, sy, 0.4 * sc, 0.09 * sc, 0, 0, Math.PI * 2); c.fill();
      const pose = poseAt(rep, tf, k);
      if (opts.labels) pose.label = opts.labels[k];
      // 贴身时两人名字会叠在一起：各往自己那侧横向让开（往上错会钻进顶栏）
      if (Math.abs(ox - sx) < 0.9 * sc) pose.labelDx = (sx < ox || (sx === ox && k === 0) ? -1 : 1) * (0.45 * sc - Math.abs(ox - sx) / 2);
      drawBoxer(c, sx, sy, sc, dir, ids[k], rep.bodies[k], pose, fr.f[k][F.st] < 0 ? -1 : 1);
    }
    near();
    // 命中火花和伤害数字
    for (let back = 0; back < 12; back++) {
      const tt = t - back; if (tt < 1) break;
      for (let k = 0; k < 2; k++) {
        const f = rep.frames[tt].f[k]; if (!(f[F.fired] && f[F.oc] === 1)) continue;
        const tg = rep.frames[tt].f[1 - k];
        const B = build(rep.bodies[1 - k]);
        const [px, py, sc] = cam.P(tg[F.x], tg[F.y], SIM.TGT[f[F.fired] - 1] === 0 ? B.shH + 0.15 : B.shH - 0.2);
        c.globalAlpha = 1 - back / 12;
        if (back < 3) {
          c.strokeStyle = f[F.ctr] ? '#ff5a4f' : '#ffe066'; c.lineWidth = 3;
          for (let i = 0; i < 12; i++) { const an = i * Math.PI / 6 + back; const r1 = 0.06 * sc, r2 = (f[F.dmg] >= 5 ? 0.3 : 0.18) * sc * (1 + back * 0.2); c.beginPath(); c.moveTo(px + Math.cos(an) * r1, py + Math.sin(an) * r1); c.lineTo(px + Math.cos(an) * r2, py + Math.sin(an) * r2); c.stroke(); }
        }
        c.font = `800 ${Math.max(12, 0.17 * sc)}px system-ui`; c.textAlign = 'center';
        c.fillStyle = 'rgba(0,0,0,0.5)'; c.fillText('−' + f[F.dmg], px + 1, py - (0.35 + back * 0.02) * sc + 1);
        c.fillStyle = f[F.ctr] ? '#ffb3ad' : '#ffffff'; c.fillText('−' + f[F.dmg], px, py - (0.35 + back * 0.02) * sc);
        c.globalAlpha = 1;
      }
    }
    return { t, cam };
  }
  const lerp3 = (p, q, a) => [lerp(p[0], q[0], a), lerp(p[1], q[1], a), lerp(p[2], q[2], a)];

  // ---------------------------------------------------------------- 转播包装
  function roundRect(c, x, y, w, h, r) { c.beginPath(); c.roundRect(x, y, w, h, r); }
  function hud(c, rep, t, box, names, sub) {
    const [x0, y0, w] = box, fr = rep.frames[t];
    const H = Math.max(54, w * 0.055);
    c.fillStyle = 'rgba(6,8,14,0.78)'; c.fillRect(x0, y0, w, H);
    const ids = [rep.bodies[0].id, rep.bodies[1].id];
    for (let k = 0; k < 2; k++) {
      const f = fr.f[k], right = k === 1, bw = w * 0.36, bx = right ? x0 + w - 16 - bw : x0 + 16;
      const L = look(ids[k]);
      c.fillStyle = L.accent || L.trunk; c.fillRect(right ? x0 + w - 8 : x0, y0, 8, H);
      c.font = `700 ${H * 0.27}px "Microsoft YaHei", system-ui`; c.textAlign = right ? 'right' : 'left'; c.fillStyle = '#fff';
      c.fillText(names[k], right ? bx + bw : bx, y0 + H * 0.34);
      const tags = `${SIM.postures[f[F.post]]}${f[F.st] < 0 ? '·反架' : ''}${sub ? ' · ' + SIM.plans[f[F.plan]] : ''}${f[F.tilt] ? ' · 上头' : ''}`;
      const bar = (yy, v, col, hh) => { c.fillStyle = 'rgba(255,255,255,0.14)'; c.fillRect(bx, yy, bw, hh); c.fillStyle = col; const ww = bw * clamp(v, 0, 1); c.fillRect(right ? bx + bw - ww : bx, yy, ww, hh); };
      bar(y0 + H * 0.46, f[F.hp] / 100, L.accent || L.glove, H * 0.16);
      bar(y0 + H * 0.68, f[F.stam], '#cfcac0', H * 0.08);
      bar(y0 + H * 0.82, f[F.comp] / 100, '#86b8ff', H * 0.08);
      c.font = `${H * 0.17}px "Microsoft YaHei"`; c.fillStyle = 'rgba(255,255,255,0.75)'; c.textAlign = right ? 'left' : 'right';
      c.fillText(tags, right ? bx : bx + bw, y0 + H * 0.34);
    }
    const secs = fr.t * SIM.DT;
    c.fillStyle = '#e8b33c'; roundRect(c, x0 + w / 2 - H * 1.2, y0 + H * 0.12, H * 2.4, H * 0.76, 8); c.fill();
    c.fillStyle = '#111'; c.font = `800 ${H * 0.3}px system-ui`; c.textAlign = 'center';
    c.fillText(`R${fr.r + 1}  ${Math.floor(secs / 60)}:${String(Math.floor(secs % 60)).padStart(2, '0')}`, x0 + w / 2, y0 + H * 0.6);
  }
  function caption(c, box, text, sub, col, alpha = 1, opts = {}) {
    const [x0, y0, w, h] = box;
    c.globalAlpha = alpha;
    const fs = Math.round(w / 20);
    c.font = `900 ${fs}px "Microsoft YaHei", system-ui`; c.textAlign = 'center';
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
    if (s.type === 'result') {
      drawBoutFrame(c, rep, n - 1, box, { ids, labels: meta.labels || ids });
      hud(c, rep, n - 1, box, names, true);
      return drawResult(c, card, box, local / s.dur, meta);
    }
    if (s.type === 'break') {
      const t = Math.min(n - 1, s.at);
      drawBoutFrame(c, rep, t, box, { ids, labels: meta.labels || ids });
      return drawBreak(c, rep, t, box, local / s.dur, meta);
    }
    if (s.type === 'replay') {
      const tf = s.from + local / s.dur * (s.to - s.from);
      const ti = clamp(Math.floor(tf), 0, n - 1);
      const base = rep._cam[ti];
      drawBoutFrame(c, rep, tf, box, { ids, labels: meta.labels || ids, cam: [base[0], base[1], base[2] * 1.45] });
      const [x0, y0, w, h] = box;
      c.fillStyle = 'rgba(0,0,0,0.25)'; c.fillRect(x0, y0, w, h);
      c.fillStyle = '#d03b3b'; roundRect(c, x0 + 24, y0 + 24, 120, 40, 6); c.fill();
      c.fillStyle = '#fff'; c.font = '900 22px "Microsoft YaHei"'; c.textAlign = 'center'; c.fillText('慢 镜 回 放', x0 + 84, y0 + 52);
      return;
    }
    const tf = s.from + local / DT_OUT;
    const { t } = drawBoutFrame(c, rep, tf, box, { ids, labels: meta.labels || ids });
    hud(c, rep, t, box, names, true);
    // 回合开始横幅
    const roundStart = [0, ...rep._prep.rounds].filter(r => r <= t).pop();
    if (t - roundStart < 30) banner(c, box, `第 ${rep.frames[t].r + 1} 回合`, card.title, (t - roundStart) / 30);
    // 大事件字幕
    const ev = rep._prep.events.filter(e => e.t <= t && t - e.t < 28 && e.big >= 2).pop();
    if (ev) caption(c, box, `${names[ev.k]} ${ev.text}`, null, ev.big === 3 ? '#ff5a4f' : '#ffd24a', 1 - (t - ev.t) / 28);
    // 读秒
    for (let k = 0; k < 2; k++) {
      const f = rep.frames[t].f[k];
      if (f[F.ph] === PH.DOWN) {
        const el = t - rep._prep.downAt[t][k], cnt = Math.min(10, Math.floor(el / 60 * 10) + 1);
        const [x0, y0, w, h] = box;
        c.font = `900 ${w / 7}px system-ui`; c.textAlign = 'center'; c.lineWidth = 10; c.strokeStyle = 'rgba(0,0,0,0.7)';
        c.strokeText(String(cnt), x0 + w / 2, y0 + h * 0.62); c.fillStyle = '#ffffff'; c.fillText(String(cnt), x0 + w / 2, y0 + h * 0.62);
        c.font = `700 ${w / 40}px "Microsoft YaHei"`; c.fillText(`${names[k]} 倒地，裁判读秒`, x0 + w / 2, y0 + h * 0.7);
      }
    }
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
    const q = [cam.P(0, 0), cam.P(R, 0), cam.P(R, R), cam.P(0, R)];
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
        drawBoutFrame(c, rep, (t * 20) % rep.frames.length, box, { gym: true, ids: [id, 'S'], labels: [ctx.names[id], clip.partner] });
      } else { gymWall(c, box); gymFloor(c, box, floor); shadow(cx); drawBoxer(c, cx, floor, s, 1, id, body, guardPose(B)); }
      if (note.iters) cap += ` · 真实训练 ${note.iters} 轮`;
    } else if (act === '看录像' || act === '实战复盘') {
      c.fillStyle = '#15171d'; c.fillRect(x0, y0, w, h);
      const tvx = x0 + w * 0.08, tvy = y0 + h * 0.08, tvw = w * 0.84, tvh = tvw * 9 / 16 * 0.85;
      c.fillStyle = '#000'; c.fillRect(tvx - 8, tvy - 8, tvw + 16, tvh + 16);
      const fg = ctx.lastFight(entry.day);
      if (fg) { c.save(); c.beginPath(); c.rect(tvx, tvy, tvw, tvh); c.clip(); prep(fg); drawBoutFrame(c, fg, (t * 20 + 200) % fg.frames.length, [tvx, tvy, tvw, tvh], { ids: fg.fighters, labels: fg.fighters }); c.restore(); }
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
  function panAt(rep, tick, x) {
    const cam = rep._cam[clamp(tick, 0, rep._cam.length - 1)];
    return clamp((x - cam[0]) * 0.34 * cam[2], -0.85, 0.85);
  }
  function tickEvents(rep, tt, tOut, ev, extra = {}) {
    const fr = rep.frames[tt];
    for (let k = 0; k < 2; k++) {
      const f = fr.f[k], o = fr.f[1 - k];
      if (f[F.fired]) {
        const p = f[F.fired] - 1, pan = panAt(rep, tt, o[F.x]);
        const power = p >= 2, body = SIM.TGT[p] === 1;
        if (f[F.oc] === 1) ev.push({ t: tOut, type: 'hit', dmg: f[F.dmg], power, body, counter: !!f[F.ctr], pan, ...extra });
        else if (f[F.oc] === 2) ev.push({ t: tOut, type: 'block', power, pan, ...extra });
        else if (f[F.oc] === 3) ev.push({ t: tOut, type: 'evade', pan, ...extra });
        else ev.push({ t: tOut, type: 'whiff', power, pan, ...extra });
      }
      if (f[F.kd]) ev.push({ t: tOut, type: 'kd', pan: panAt(rep, tt, f[F.x]), ...extra });
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

  return { accent, tickEvents, fightAudio, trainingAudio, drawTraining, guardPose, punchPose, F, PH, LOOK, look, build, drawBoxer, drawHead, prep, poseAt, drawBoutFrame, makeCam, drawArena, hud, caption, banner, timeline, segAt, renderFight, setSim: s => { SIM = s; }, ease, easeOut, lerp, clamp, shade, mixHex, hash };
})();
if (typeof window !== 'undefined') window.VIZ = VIZ;
