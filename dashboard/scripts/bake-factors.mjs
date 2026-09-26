#!/usr/bin/env node

/**
 * bake-factors.mjs — refresh lib/data/factorsBaked.json, the LAST tier of the 🧬
 * factor row (/api/factors).
 *
 * What the bake is for: when every live source is down for a ticker, the route
 * still has ~10 years of WEEKLY closes to draw from. While any live DAILY tier
 * works, only the part of the bake OLDER than that daily series (~2y) is used, so
 * a bake stays useful for about two years. As a total-outage floor it shows data
 * "through <bake date>" in orange — honest, never blank.
 *
 * Source: Nasdaq historical (keyless, 10y daily) thinned to the last trading day of
 * each week; CNBC weekly (Sunday-dated, shifted to Friday) if Nasdaq fails.
 * REFUSES to write a bake with fewer weeks per ticker than the existing one (a
 * degraded source must never gut history no live tier can rebuild).
 *
 * Usage (from dashboard/): node scripts/bake-factors.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, '..', 'lib', 'data', 'factorsBaked.json');
const TICKERS = ['SPY', 'VLUE', 'MTUM', 'QUAL', 'IWM', 'USMV'];
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';
const DAY = 864e5;
const iso = (ms) => new Date(ms).toISOString().slice(0, 10);

async function getJson(url) {
    const res = await fetch(url, { headers: { 'User-Agent': UA, Accept: 'application/json' } });
    if (!res.ok) throw new Error(`${res.status} ${url}`);
    return res.json();
}

async function nasdaq(t) {
    const now = Date.now();
    const from = iso(now - (10 * 365.25 + 14) * DAY);
    const d = await getJson(`https://api.nasdaq.com/api/quote/${t}/historical?assetclass=etf&fromdate=${from}&todate=${iso(now)}&limit=9999`);
    const rows = d?.data?.tradesTable?.rows || [];
    const out = [];
    for (const r of rows) {
        const m = /^(\d{2})\/(\d{2})\/(\d{4})$/.exec(r.date || '');
        const p = parseFloat(String(r.close).replace(/[$,]/g, ''));
        if (m && p > 0) out.push({ date: `${m[3]}-${m[1]}-${m[2]}`, price: p });
    }
    out.sort((a, b) => (a.date < b.date ? -1 : 1));
    return thinToWeekly(out);
}

async function cnbcWeekly(t) {
    const d = await getJson(`https://ts-api.cnbc.com/harmony/app/charts/5Y.json?symbol=${t}`);
    // Skip the last 7 days: CNBC's newest weekly bar is an unreliable snapshot (see lib/factors.js weeklyToFriday).
    const cutoff = iso(Date.now() - 7 * DAY);
    const out = [];
    for (const b of d?.barData?.priceBars || []) {
        const tt = String(b.tradeTime || '');
        const p = parseFloat(b.close);
        if (tt.length < 8 || !(p > 0)) continue;
        const fri = iso(Date.parse(`${tt.slice(0, 4)}-${tt.slice(4, 6)}-${tt.slice(6, 8)}T00:00:00Z`) + 5 * DAY);
        if (fri <= cutoff) out.push({ date: fri, price: p });
    }
    return out;
}

function thinToWeekly(h) {
    const out = [];
    let last = null;
    for (const p of h) {
        const ms = Date.parse(`${p.date}T00:00:00Z`);
        const wk = iso(ms - ((new Date(ms).getUTCDay() + 6) % 7) * DAY);
        if (wk === last) out[out.length - 1] = p;
        else { out.push(p); last = wk; }
    }
    return out;
}

const prev = fs.existsSync(OUT) ? JSON.parse(fs.readFileSync(OUT, 'utf8')) : null;
const tickers = {};
const sources = {};
for (const t of TICKERS) {
    let h = [];
    try { h = await nasdaq(t); sources[t] = 'nasdaq'; } catch (e) { console.error(`${t} nasdaq failed: ${e.message}`); }
    if (h.length < 400) {
        try { h = await cnbcWeekly(t); sources[t] = 'cnbc-weekly'; } catch (e) { console.error(`${t} cnbc failed: ${e.message}`); }
    }
    const prevLen = prev?.tickers?.[t]?.length || 0;
    if (h.length < 400 || h.length < prevLen) {
        console.error(`REFUSING to bake: ${t} has ${h.length} weeks (previous bake ${prevLen}).`);
        process.exit(1);
    }
    tickers[t] = h.map((p) => [p.date, Math.round(p.price * 100) / 100]);
    console.log(`${t}: ${h.length} weeks ${h[0].date} → ${h[h.length - 1].date} (${sources[t]})`);
}
fs.writeFileSync(OUT, JSON.stringify({ bakedAt: iso(Date.now()), basis: 'price', sources, tickers }) + '\n');
console.log(`wrote ${OUT}`);
