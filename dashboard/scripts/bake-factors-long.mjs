#!/usr/bin/env node

/**
 * bake-factors-long.mjs — refresh lib/data/factorsLong.json, the BACKUP tier of the
 * 🧬 factor row's 20Y / 30Y / 40Y windows (the live tier downloads the same six Ken
 * French zips at request time, cached 7 days; see lib/factorsLong.js).
 *
 * The bake only matters when Dartmouth is unreachable, and then it is shown as
 * "through <month>" like always — so it stays useful for years. Re-bake whenever
 * convenient (e.g. yearly): the live tier keeps the row current on its own.
 *
 * REFUSES to write a bake that ends EARLIER than the existing one (a broken download
 * must never replace good history with less).
 *
 * Usage (from dashboard/): node scripts/bake-factors-long.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { KF_BASE, LONG_SOURCES, LONG_KEYS, longFromZips } from '../lib/factorsLong.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, '..', 'lib', 'data', 'factorsLong.json');

async function download(file) {
    const res = await fetch(`${KF_BASE}/${file}_CSV.zip`, { headers: { 'User-Agent': 'financial-dashboard-bake/1.0' } });
    if (!res.ok) throw new Error(`${res.status} ${file}`);
    return Buffer.from(await res.arrayBuffer());
}

const zips = {};
for (const k of LONG_KEYS) {
    zips[k] = await download(LONG_SOURCES[k].file);
    console.log(`  ${k.padEnd(9)} ${LONG_SOURCES[k].file} (${zips[k].length} bytes)`);
}
const long = longFromZips(zips);

let prev = null;
try { prev = JSON.parse(fs.readFileSync(OUT, 'utf8')); } catch { /* first bake */ }
if (prev?.through && long.through < prev.through) {
    console.error(`REFUSING: new data ends ${long.through}, existing bake ends ${prev.through}`);
    process.exit(1);
}

const round4 = (x) => Math.round(x * 1e4) / 1e4;
const out = {
    bakedAt: new Date().toISOString().slice(0, 10),
    provider: 'Ken French Data Library (Fama/French, CRSP) — value-weighted monthly % returns',
    start: long.start,
    through: long.through,
    months: long.months,
    series: Object.fromEntries(LONG_KEYS.map((k) => [k, long.series[k].map(round4)])),
};
fs.writeFileSync(OUT, `${JSON.stringify(out)}\n`);
console.log(`wrote ${path.relative(process.cwd(), OUT)}: ${long.months.length} months, ${long.start} → ${long.through}`);
