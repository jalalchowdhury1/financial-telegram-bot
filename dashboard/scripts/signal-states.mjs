/**
 * Prints every /v2 signal's state (lib/signalStudy.js signalStates) as JSON, from the live site's baked
 * files with the sheet series carrying each one up to today, the way the dashboard joins them. Read by the
 * Mac's daily ~/.local/bin/signal-ping (one Telegram line + a Now item per new fire). Exits 1 on any failure.
 *   node dashboard/scripts/signal-states.mjs        (DASH_BASE=http://localhost:4100 for a local build)
 */
import { SIGNALS, signalStates } from '../lib/signalStudy.js';

const BASE = process.env.DASH_BASE || 'https://financial-telegram-bot-beryl.vercel.app';
const DAY = 86400000;
const iso = (ms) => new Date(ms).toISOString().slice(0, 10);
const get = async (path) => {
    const r = await fetch(BASE + path, { signal: AbortSignal.timeout(20000) });
    if (!r.ok) throw new Error(`${path} → ${r.status}`);
    return r.json();
};
const decode = (b) => b.t.flatMap((d, i) => (Number.isFinite(b.v[i]) ? [{ date: iso(Date.parse(`${b.from}T00:00:00Z`) + d * DAY), value: b.v[i] }] : []));
const sheet = (s, key) => (s.v[key] || []).flatMap((x, i) => (Number.isFinite(x) ? [{ date: iso(Date.parse(`${s.from}T00:00:00Z`) + i * DAY), value: x }] : []));
const tail = (pts, more) => [...pts, ...more.filter((p) => p.date > pts[pts.length - 1].date)];

const keys = [...new Set(SIGNALS.flatMap((g) => (g.combo ? g.combo.legs.map((l) => l.key) : [g.key])))];
const [series, spy, ...baked] = await Promise.all([get('/api/history').then((h) => h.series), get('/history/spy-daily.json'), ...keys.map((k) => get(`/history/${k}.json`))]);
if (!series?.v) throw new Error('/api/history has no series');
const v3 = new Map(sheet(series, 'vix3m').map((p) => [p.date, p.value]));
const live = { 'vix-ratio-daily': sheet(series, 'vixCurrent').filter((p) => v3.get(p.date) > 0).map((p) => ({ date: p.date, value: p.value / v3.get(p.date) })) };
const files = Object.fromEntries(keys.map((k, i) => [k, tail(decode(baked[i]), live[k] || sheet(series, k))]));
console.log(JSON.stringify(signalStates(files, decode(spy).map((p) => ({ date: p.date, price: p.value }))), null, 1));
