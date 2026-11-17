/**
 * bake-pill-record.mjs — the pills' report card: replay every trading day through the
 * site's own ruleVerdicts (lib/jevBrief.js), then score what SPY did next.
 *
 *   node scripts/bake-pill-record.mjs        → lib/data/pillRecord.json (bundled by the popup)
 *
 * Inputs are dated by when they were PUBLISHED (public/history/* already are; FRED daily
 * series are shifted one day), computed the way the live routes compute them:
 *   SPY vs 200d / from 52w high: raw closes (200 incl. today / 252-day closing high)
 *   RSP/SPY, IWM/SPY: raw-close ratios, lib/breadth.js pairStats
 *   IV: VIX close, percentile over the last 252 closes incl. today (lib/vol.js); RV = 21 log returns
 *   junk spread: FRED BAMLH0A0HYM2, 20 rows back (FRED only serves it from Nov 2023)
 *   Fear & Greed: public/history/cnnFearGreed.json (from 2021)
 * Forward returns use Yahoo ADJUSTED closes (dividends in). A verdict's "edge" is tested by
 * circularly shifting the verdict series against the returns (keeps both streaks intact),
 * so a long run of one colour can't fake a result.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { register } from 'node:module';

// lib/ files import siblings without '.js' (Next resolves them); teach Node the same
register('data:text/javascript,' + encodeURIComponent(`export async function resolve(s, c, next) {
    try { return await next(s, c); } catch (e) { if (s.startsWith('.') && !s.endsWith('.js')) return next(s + '.js', c); throw e; }
}`));
const { ruleVerdicts, PILLS } = await import('../lib/jevBrief.js');
const { pairStats } = await import('../lib/breadth.js');
const { realizedVol, ivPercentile } = await import('../lib/vol.js');

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const START = '2000-01-03';
const H1 = 21, H3 = 63;
// a day counts for a pill only when EVERY input that pill reads exists (else it is a weaker rule)
const NEEDS = {
    regime: (x) => [x.spy.ma200Pct, x.fg.score, x.credit.oasChg20],
    recession: (x) => [x.fred.sahmRule, x.fred.yieldCurve, x.fred.claims, x.fred.nfci],
    breadth: (x) => [x.breadth.rspSpy.chg20Pct, x.breadth.rspSpy.vs50dPct, x.breadth.iwmSpy.chg20Pct],
    hedging: (x) => [x.vol.spy.ivPctile1y, x.vol.spy.vrp],
    conflict: (x) => [x.fg.score, x.spy.ma200Pct, x.fred.yieldCurve, x.t10y3m, x.credit.oasChg20, x.spy.high52Pct, x.breadth.rspSpy.chg20Pct],
};

async function yahoo(t) {
    const r = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${t}?period1=725846400&period2=${Math.floor(Date.now() / 1000)}&interval=1d`, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    const j = (await r.json()).chart.result[0];
    if (j.meta.dataGranularity !== '1d') throw new Error(`${t}: Yahoo sent ${j.meta.dataGranularity} bars`);
    const q = j.indicators.quote[0], adj = j.indicators.adjclose[0].adjclose;
    const out = new Map();
    j.timestamp.forEach((ts, i) => {
        if (q.close[i] != null) out.set(new Date(ts * 1000).toISOString().slice(0, 10), { c: q.close[i], a: adj[i] });
    });
    return out;
}

async function fred(id) {
    const r = await fetch(`https://fred.stlouisfed.org/graph/fredgraph.csv?id=${id}`);
    return (await r.text()).trim().split('\n').slice(1).map((l) => l.split(','))
        .filter(([, v]) => v && v !== '.').map(([d, v]) => ({ d, v: +v }));
}

function hist(key) {
    const h = JSON.parse(fs.readFileSync(path.join(ROOT, 'public/history', `${key}.json`), 'utf8'));
    const t0 = Date.parse(`${h.from}T00:00:00Z`);
    return h.t.map((t, i) => ({ d: new Date(t0 + t * 864e5).toISOString().slice(0, 10), v: h.v[i] }));
}

/** as-of reader over ascending [{d, v}]: last value dated ≤ day (or < day when lag). */
function asOf(series, lagOneDay = false, maxAgeDays = 45) {
    let i = -1;
    return (day) => {
        while (i + 1 < series.length && (lagOneDay ? series[i + 1].d < day : series[i + 1].d <= day)) i++;
        if (i < 0) return null;
        return (Date.parse(day) - Date.parse(series[i].d)) / 864e5 > maxAgeDays ? null : series[i].v;
    };
}

const [SPY, RSP, IWM] = await Promise.all(['SPY', 'RSP', 'IWM'].map(yahoo));
const [VIX, T3M, OAS] = await Promise.all(['VIXCLS', 'T10Y3M', 'BAMLH0A0HYM2'].map(fred));
const days = [...SPY.keys()].sort();
const closes = days.map((d) => SPY.get(d).c);
const adj = days.map((d) => SPY.get(d).a);

const fg = asOf(hist('cnnFearGreed'), false, 10);
const curve = asOf(hist('yieldCurve'), false, 10);
const sahm = asOf(hist('sahmRule'), false, 70);
const claims = asOf(hist('claims'), false, 21);
const nfci = asOf(hist('nfci'), false, 21);
const t3m = asOf(T3M, true, 10);
const vixAt = new Map(VIX.map((p) => [p.d, p.v]));
const oasPrev = asOf(OAS.map((p, i) => ({ d: p.d, v: i })), true, 10); // index of the last print before today

const ratio = (leg, i) => days.slice(Math.max(0, i - 80), i + 1)
    .filter((d) => leg.has(d)).map((d) => ({ date: d, ratio: leg.get(d).c / SPY.get(d).c }));

const rows = [];
for (let i = 252; i < days.length; i++) {
    const d = days[i];
    if (d < START) continue;
    const c = closes[i];
    const ma200 = closes.slice(i - 199, i + 1).reduce((a, b) => a + b, 0) / 200;
    const hi52 = Math.max(...closes.slice(i - 251, i + 1));
    const vixWin = days.slice(0, i + 1).filter((x) => vixAt.has(x)).slice(-252).map((x) => vixAt.get(x));
    const iv = vixAt.get(d) ?? null;
    const rv = realizedVol(closes.slice(i - 21, i + 1), 21);
    const oi = oasPrev(d);
    const rsp = RSP.has(d) ? pairStats(ratio(RSP, i)) : {};
    const iwm = IWM.has(d) ? pairStats(ratio(IWM, i)) : {};
    const data = {
        spy: { ma200Pct: (c / ma200 - 1) * 100, high52Pct: (c / hi52 - 1) * 100 },
        fg: { score: fg(d) },
        fred: { yieldCurve: curve(d), sahmRule: sahm(d), claims: claims(d), nfci: nfci(d) },
        breadth: { rspSpy: { chg20Pct: rsp.chg20Pct ?? null, vs50dPct: rsp.vs50dPct ?? null }, iwmSpy: { chg20Pct: iwm.chg20Pct ?? null } },
        vol: { spy: { ivPctile1y: iv != null && vixWin.length >= 200 ? ivPercentile(vixWin, iv) : null, vrp: iv != null && rv != null ? iv - rv : null } },
        t10y3m: t3m(d),
        credit: { oasChg20: oi != null && oi >= 20 ? Math.round((OAS[oi].v - OAS[oi - 20].v) * 100) : null },
    };
    const v = ruleVerdicts(data);
    const f = (h) => (i + h < days.length ? (adj[i + h] / adj[i] - 1) * 100 : null);
    const worst = i + H3 < days.length ? (Math.min(...adj.slice(i + 1, i + H3 + 1)) / adj[i] - 1) * 100 : null;
    rows.push({ d, v: Object.fromEntries(PILLS.map((p) => [p, NEEDS[p](data).every((n) => n != null && Number.isFinite(n)) ? v[p].verdict : 'unknown'])), r1: f(H1), r3: f(H3), w3: worst });
}

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const round1 = (x) => Math.round(x * 10) / 10;

/** circular-shift test: how often does a random shift give a gap this big? */
function shiftP(labels, vals, target) {
    const n = labels.length, gap = (off) => {
        let s1 = 0, n1 = 0, s0 = 0, n0 = 0;
        for (let k = 0; k < n; k++) {
            const hit = labels[(k + off) % n] === target;
            if (hit) { s1 += vals[k]; n1++; } else { s0 += vals[k]; n0++; }
        }
        return n1 && n0 ? s1 / n1 - s0 / n0 : 0;
    };
    const obs = gap(0);
    let more = 0, tries = 0;
    for (let off = 252; off < n - 252; off += 7) { tries++; if (Math.abs(gap(off)) >= Math.abs(obs)) more++; }
    return tries ? more / tries : null;
}

const out = { built: new Date().toISOString().slice(0, 10), pills: {} };
for (const p of PILLS) {
    const scored = rows.filter((r) => r.r3 != null && r.v[p] !== 'unknown' && r.v[p] !== 'n/a');
    if (!scored.length) continue;
    const by = {};
    for (const r of scored) (by[r.v[p]] ||= []).push(r);
    const verdicts = {};
    for (const [verdict, rs] of Object.entries(by)) {
        let runs = 0;
        scored.forEach((r, k) => { if (r.v[p] === verdict && (k === 0 || scored[k - 1].v[p] !== verdict)) runs++; });
        verdicts[verdict] = {
            days: rs.length, runs,
            avg3m: round1(mean(rs.map((r) => r.r3))),
            up3m: Math.round((rs.filter((r) => r.r3 > 0).length / rs.length) * 100),
            worst3m: round1(mean(rs.map((r) => r.w3))),
            avg1m: round1(mean(rs.map((r) => r.r1))),
            pRet: shiftP(scored.map((r) => r.v[p]), scored.map((r) => r.r3), verdict),
            pFall: shiftP(scored.map((r) => r.v[p]), scored.map((r) => r.w3), verdict),
        };
    }
    out.pills[p] = {
        from: scored[0].d, to: scored[scored.length - 1].d, days: scored.length,
        all: { avg3m: round1(mean(scored.map((r) => r.r3))), up3m: Math.round((scored.filter((r) => r.r3 > 0).length / scored.length) * 100), worst3m: round1(mean(scored.map((r) => r.w3))) },
        verdicts,
    };
    // one plain takeaway: under 3 years = too short; else the clearest gap, only if a random
    // shift beats it less than 1 time in 10; else "no clear edge"
    const rec = out.pills[p], all = rec.all;
    const years = (Date.parse(rec.to) - Date.parse(rec.from)) / 3.156e10;
    const best = Object.entries(verdicts).flatMap(([k, x]) => [[k, 'fall', x.pFall, x], [k, 'ret', x.pRet, x]])
        .filter(([, , pv, x]) => pv != null && x.runs >= 5).sort((a, b) => a[2] - b[2])[0];
    rec.note = years < 5 ? `Too short to judge: all inputs only since ${rec.from.slice(0, 4)}`
        : best && best[2] < 0.1
            ? best[1] === 'fall'
                ? `${best[0]} came before ${best[3].worst3m < all.worst3m ? 'deeper' : 'smaller'} dips (${best[3].worst3m}% vs ${all.worst3m}%)`
                : `${best[0]} came before ${best[3].avg3m < all.avg3m ? 'weaker' : 'stronger'} 3 months (${best[3].avg3m}% vs ${all.avg3m}%)`
            : 'No clear edge: every colour did about the same';
}

fs.mkdirSync(path.join(ROOT, 'lib/data'), { recursive: true });
fs.writeFileSync(path.join(ROOT, 'lib/data/pillRecord.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
