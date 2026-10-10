// 预览页面里的几帧：node tools/preview_frames.mjs <页面.html> <输出目录> "条目@秒,条目@秒"
import puppeteer from 'puppeteer-core';
import { pathToFileURL } from 'node:url';
import { writeFileSync } from 'node:fs';
const [page_, outDir, spec] = process.argv.slice(2);
const b = await puppeteer.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: 'new', args: ['--allow-file-access-from-files'] });
const p = await b.newPage(); p.on('pageerror', e => console.error('[页面错误]', e.message));
await p.goto(pathToFileURL(page_).href); await p.evaluate(() => document.fonts.ready);
await p.waitForFunction(() => window.EXPORT.ready(), { timeout: 30000 });
const items = await p.evaluate(() => EXPORT.items());
console.log(items.map(x => `${x.i}:${x.title}(${x.dur.toFixed(1)}s)`).join(' '));
for (const s of spec.split(',')) {
  const [i, t] = s.split('@').map(Number);
  const url = await p.evaluate((i, t) => EXPORT.frame(i, t, 1280, 720), i, t);
  writeFileSync(`${outDir}/f_${i}_${t}.jpg`, Buffer.from(url.split(',')[1], 'base64'));
}
const au = await p.evaluate(() => EXPORT.audio(1, 0, 10).length);
console.log('profile audio events', au);
await b.close();
