// 把赛季观看页里的画面导出成 mp4（和网页里看到的完全一样，因为用的是同一份渲染代码）。
// 用法：
//   node tools/export_video.mjs runs/season_s0.html out/第1战.mp4 --items 0
//   node tools/export_video.mjs runs/season_s0.html out/训练营.mp4 --items 1-35 --day-len 3
//   node tools/export_video.mjs runs/season_s0.html out/整季.mp4 --items all --day-len 2
//   选项：--w 1920 --h 1080 --fps 30 --from 秒 --to 秒（只对单个条目生效，截一段）--mute 1（不要声音）--keep 1（保留中间文件）
// 声音：页面给出声音事件表（EXPORT.audio），tools/make_audio.py 合成音轨，最后和画面合并
import puppeteer from 'puppeteer-core';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync, unlinkSync, existsSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const [page_, out, ...rest] = process.argv.slice(2);
if (!page_ || !out) { console.error('用法：node tools/export_video.mjs <season.html> <out.mp4> [--items 0,3-10|all] [--w 1280] [--h 720] [--fps 30]'); process.exit(1); }
const opt = { items: 'all', w: 1280, h: 720, fps: 30, 'day-len': null, from: null, to: null, mute: null, keep: null };
for (let i = 0; i < rest.length; i += 2) opt[rest[i].replace(/^--/, '')] = rest[i + 1];
const W = +opt.w, H = +opt.h, FPS = +opt.fps;
mkdirSync(path.dirname(path.resolve(out)), { recursive: true });

const browser = await puppeteer.launch({ executablePath: EDGE, headless: 'new', args: ['--allow-file-access-from-files'] });
const page = await browser.newPage();
page.on('pageerror', e => { console.error('[页面错误]', e.message); process.exitCode = 1; });
await page.setViewport({ width: 640, height: 480 });
await page.goto(pathToFileURL(path.resolve(page_)).href);
await page.evaluate(() => document.fonts.ready);
// 页面里有图片（试播集的立绘）时，等它们加载完再开始导出
await page.waitForFunction(() => !window.EXPORT.ready || window.EXPORT.ready(), { timeout: 30000 });
const all = await page.evaluate(() => EXPORT.items());

const pick = [];
if (opt.items === 'all') all.forEach(it => pick.push(it.i));
else for (const part of String(opt.items).split(',')) {
  const [a, b] = part.split('-').map(Number);
  for (let i = a; i <= (Number.isFinite(b) ? b : a); i++) if (all[i]) pick.push(i);
}
const plan = pick.map(i => {
  const it = all[i];
  let from = 0, to = it.type === 'day' && opt['day-len'] ? +opt['day-len'] : it.dur;
  if (pick.length === 1) { if (opt.from !== null) from = +opt.from; if (opt.to !== null) to = Math.min(to, +opt.to); }
  return { ...it, from, to };
});
const totalSec = plan.reduce((s, p) => s + (p.to - p.from), 0);
console.log(`导出 ${plan.length} 段，共 ${totalSec.toFixed(1)} 秒，${W}×${H} @ ${FPS}fps → ${out}`);

const OUT = path.resolve(out);
const SOUND = !opt.mute;
const VID = SOUND ? OUT.replace(/\.mp4$/i, '') + '.video.mp4' : OUT;
const ff = spawn('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
  '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium', '-movflags', '+faststart', VID]);
ff.stderr.on('data', d => process.stderr.write(d));
const done = new Promise(res => ff.on('close', res));
const write = buf => new Promise(res => (ff.stdin.write(buf) ? res() : ff.stdin.once('drain', res)));

const t0 = Date.now();
let frames = 0;
for (const p of plan) {
  const n = Math.round((p.to - p.from) * FPS);
  for (let f = 0; f < n; f++) {
    const url = await page.evaluate((i, t, w, h) => EXPORT.frame(i, t, w, h), p.i, p.from + f / FPS, W, H);
    await write(Buffer.from(url.split(',')[1], 'base64'));
    frames++;
    if (frames % (FPS * 10) === 0) process.stdout.write(`  ${(frames / FPS).toFixed(0)} / ${totalSec.toFixed(0)} 秒（${((Date.now() - t0) / 1000).toFixed(0)}s）\n`);
  }
}
ff.stdin.end();
const code = await done;
if (code !== 0) { await browser.close(); console.error(`ffmpeg 出错，退出码 ${code}`); process.exit(1); }

if (SOUND) {
  // 每段的声音事件按"这段在成片里从第几秒开始"平移，拼成一张总表
  const events = [];
  let at = 0;
  for (const p of plan) {
    const ev = await page.evaluate((i, a, b) => EXPORT.audio(i, a, b), p.i, p.from, p.to);
    for (const e of ev) {
      const shift = { ...e, t: e.t - p.from + at };
      if (e.to !== undefined) shift.to = Math.min(e.to, p.to) - p.from + at;
      events.push(shift);
    }
    at += p.to - p.from;
  }
  const base = OUT.replace(/\.mp4$/i, '');
  const evFile = base + '.events.json', wav = base + '.wav';
  writeFileSync(evFile, JSON.stringify({ duration: frames / FPS, events }));
  const here = path.dirname(fileURLToPath(import.meta.url));
  const py = spawnSync('py', ['-X', 'utf8', path.join(here, 'make_audio.py'), evFile, wav], { stdio: 'inherit' });
  if (py.status !== 0) { console.error('音轨合成失败，保留无声版：' + VID); process.exit(1); }
  const mux = spawnSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', VID, '-i', wav,
    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', OUT], { stdio: 'inherit' });
  if (mux.status !== 0) { console.error('合并音轨失败，保留无声版：' + VID); process.exit(1); }
  // 用 unlinkSync 不用 rmSync：Node 24 的 rmSync 在 Windows 上遇到中文路径会静默删不掉
  if (!opt.keep) for (const f of [VID, evFile, wav]) if (existsSync(f)) unlinkSync(f);
}
await browser.close();
console.log(`完成：${out}（${frames} 帧${SOUND ? '，含声音' : ''}，用时 ${((Date.now() - t0) / 1000).toFixed(0)} 秒）`);
