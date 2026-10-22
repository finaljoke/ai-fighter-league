// AI 生成的全身模型（GLB，一个网格、上百万面）减到页面画得动的面数，给 tools/build_skin.py 绑骨骼用。规范见 docs/角色皮肤规范.md
// 用法：node tools/reduce_model.mjs <原模型.glb> characters/<id>.model.glb [三角面数，默认 60000]
// 贴图接缝（同一位置的几个顶点贴图坐标不同）原样保留，节点上的缩放烘进坐标；只留底色贴图，原样带上（build_skin.py 再缩小）
import { readFileSync, writeFileSync, statSync } from 'node:fs';
import { MeshoptSimplifier as S } from 'meshoptimizer/simplifier';

const [inp, out, target = '60000'] = process.argv.slice(2);
if (!out) { console.log('用法：node tools/reduce_model.mjs <原模型.glb> <输出.glb> [三角面数]'); process.exit(1); }
await S.ready;

const buf = readFileSync(inp);
let off = 12, js, bin;
while (off < buf.length) {
  const len = buf.readUInt32LE(off), type = buf.readUInt32LE(off + 4), chunk = buf.subarray(off + 8, off + 8 + len);
  if (type === 0x4e4f534a) js = JSON.parse(chunk.toString('utf8'));
  else if (type === 0x004e4942) bin = chunk;
  off += 8 + len;
}
const view = i => { const v = js.bufferViews[i], o = v.byteOffset || 0; return bin.subarray(o, o + v.byteLength); };
const read = (ai, Ctor, comps) => {
  const a = js.accessors[ai], b = view(a.bufferView), o = b.byteOffset + (a.byteOffset || 0);
  return new Ctor(b.buffer.slice(o, o + a.count * comps * Ctor.BYTES_PER_ELEMENT));
};
const nodes = js.nodes.filter(n => n.mesh !== undefined);
if (nodes.length !== 1 || js.meshes[nodes[0].mesh].primitives.length !== 1) throw new Error('只认一个网格、一个图元的模型');
const node = nodes[0], prim = js.meshes[node.mesh].primitives[0];
const P = read(prim.attributes.POSITION, Float32Array, 3), N = read(prim.attributes.NORMAL, Float32Array, 3), T = read(prim.attributes.TEXCOORD_0, Float32Array, 2);
const I = js.accessors[prim.indices].componentType === 5125 ? read(prim.indices, Uint32Array, 1) : Uint32Array.from(read(prim.indices, Uint16Array, 1));
const m = node.matrix || [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
for (let i = 0; i < P.length; i += 3) {
  const [x, y, z] = [P[i], P[i + 1], P[i + 2]];
  for (let r = 0; r < 3; r++) P[i + r] = m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r];
}

// 贴图坐标也算进误差（权重 0.5），贴图上一块块的不会滑开。面数到 4 万多时短裤会破洞，6 万是干净的
const [idx, err] = S.simplifyWithAttributes(I, P, 3, T, 2, [0.5, 0.5], null, Math.min(I.length, Number(target) * 3), 0.05, []);
const [remap, nv] = S.compactMesh(idx);
const P2 = new Float32Array(nv * 3), N2 = new Float32Array(nv * 3), T2 = new Float32Array(nv * 2);
remap.forEach((dst, src) => {
  if (dst === 0xffffffff) return;
  P2.set(P.subarray(src * 3, src * 3 + 3), dst * 3); N2.set(N.subarray(src * 3, src * 3 + 3), dst * 3); T2.set(T.subarray(src * 2, src * 2 + 2), dst * 2);
});

const tex = js.materials[prim.material].pbrMetallicRoughness.baseColorTexture.index, im = js.images[js.textures[tex].source];
const parts = [], views = [];
let len = 0;
const add = (a, target) => {
  views.push({ buffer: 0, byteOffset: len, byteLength: a.byteLength, ...(target ? { target } : {}) });
  parts.push(Buffer.from(a.buffer, a.byteOffset, a.byteLength));
  len += a.byteLength;
  if (len % 4) { parts.push(Buffer.alloc(4 - len % 4)); len += 4 - len % 4; }
  return views.length - 1;
};
const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
P2.forEach((x, i) => { lo[i % 3] = Math.min(lo[i % 3], x); hi[i % 3] = Math.max(hi[i % 3], x); });
const gltf = {
  asset: { version: '2.0', generator: 'fighter_league tools/reduce_model.mjs' },
  scene: 0, scenes: [{ nodes: [0] }], nodes: [{ name: 'body', mesh: 0 }],
  meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: 0 }] }],
  materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicFactor: 0, roughnessFactor: 0.8 }, doubleSided: true }],
  textures: [{ source: 0 }], images: [{ bufferView: 4, mimeType: im.mimeType }],
  accessors: [
    { bufferView: add(P2, 34962), componentType: 5126, count: nv, type: 'VEC3', min: lo, max: hi },
    { bufferView: add(N2, 34962), componentType: 5126, count: nv, type: 'VEC3' },
    { bufferView: add(T2, 34962), componentType: 5126, count: nv, type: 'VEC2' },
    { bufferView: add(idx, 34963), componentType: 5125, count: idx.length, type: 'SCALAR' },
  ],
  bufferViews: views, buffers: [],
};
add(view(im.bufferView));
gltf.buffers.push({ byteLength: len });
let jb = Buffer.from(JSON.stringify(gltf), 'utf8');
jb = Buffer.concat([jb, Buffer.alloc((4 - jb.length % 4) % 4, 0x20)]);
const bb = Buffer.concat(parts), head = Buffer.alloc(12), ch = (b, t) => { const h = Buffer.alloc(8); h.writeUInt32LE(b.length, 0); h.writeUInt32LE(t, 4); return h; };
head.writeUInt32LE(0x46546c67, 0); head.writeUInt32LE(2, 4); head.writeUInt32LE(12 + 8 + jb.length + 8 + bb.length, 8);
writeFileSync(out, Buffer.concat([head, ch(jb, 0x4e4f534a), jb, ch(bb, 0x004e4942), bb]));
console.log(`${out}：${I.length / 3} → ${idx.length / 3} 个三角面，${P.length / 3} → ${nv} 个顶点，误差 ${(err * 100).toFixed(2)}%，${(statSync(out).size / 1e6).toFixed(1)} MB`);
