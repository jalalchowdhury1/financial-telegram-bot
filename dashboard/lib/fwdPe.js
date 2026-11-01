/**
 * S&P 500 forward P/E — FactSet's free weekly "Earnings Insight" report (Fridays, ~1:30 pm ET).
 *
 * Each report says "The forward 12-month P/E ratio for the S&P 500 is 18.7 … below the 5-year average
 * (19.8) and below the 10-year average (19.1)". The ratio uses the close of the trading day BEFORE the
 * report (the 2017 reports say "based on Thursday's closing price"), so a row is keyed by its report date
 * and the chart anchors it on the last SPY close before that date (lib/valuation.js).
 *
 * public/history/fwdPE.json (scripts/bake-fwd-pe.mjs): every report since Apr 2011 (2011–2016 from the
 * Wayback Machine, which kept ~220 of them), `{from, t, v, a5, a10}` = decodeLong's shape plus FactSet's
 * own 5- and 10-year averages that week (null when the report did not state one). /api/fwd-pe adds the
 * reports since the bake. Pure, so the bake script imports it with plain Node.
 */

import { parseEnvelope } from './kv.js'; // .js: scripts/bake-fwd-pe.mjs imports this file with plain Node

const PE_RE = /(?:forward 12-month|12-month forward) P\/E ratio (?:for the S&P 500 )?(?:is|of)(?: now)? (\d{1,2}\.\d)\b/gi;
// "the 5-year average (19.8)", "the 10-year average of 14.4", "ten-year average forward 12-month P/E ratio of 15.4"
// (2011), "the average forward 12-Month P/E ratio over the past 10 years (14.4)" (2012); \b: never the 15-year average
const avgRe = (n) => new RegExp(`(?:\\b(?:${n})-year average(?: forward 12-month P/E ratio)?(?: of)?|average forward 12-month P/E ratio over the (?:past|prior) (?:${n}) years) \\(?(\\d{1,2}\\.\\d)\\)?`, 'i');
const A5 = avgRe('5|five'), A10 = avgRe('10|ten');
const inBand = (v, lo, hi) => (Number.isFinite(v) && v >= lo && v <= hi ? v : null);
// 2013–2016 often name the averages only on the sector page: "The P/E ratio of 15.7 for the index as a whole is above
// the prior 5-year average forward 12-month P/E ratio of 13.3, and above the prior 10-year average … of 13.8."
const WHOLE = /for the index as a whole is (.{0,240}?\d\.\d)\./i;

/**
 * `{pe, a5, a10}` from a report's text, or null when it has no forward P/E. The averages come from the
 * first statement of the ratio that names them (page 1 since 2016; the valuation page before that), else
 * the sector page's index-as-a-whole sentence. 2011–2012 and most of 2013 print no 5-year average.
 */
export function parseFactSet(text) {
    const flat = typeof text === 'string' ? text.replace(/\s+/g, ' ') : '';
    let pe = null, a5 = null, a10 = null;
    for (const m of flat.matchAll(PE_RE)) {
        pe ??= inBand(Number(m[1]), 5, 40);
        const after = flat.slice(m.index, m.index + 420).split(/trailing/i)[0]; // never the trailing P/E's averages
        const f = after.match(A5), t = after.match(A10);
        if (a5 == null && f) a5 = inBand(Number(f[1]), 8, 30);
        if (a10 == null && t) a10 = inBand(Number(t[1]), 8, 30);
        if (a5 != null && a10 != null) break;
    }
    const whole = flat.match(WHOLE)?.[1] || '';
    a5 ??= inBand(Number(whole.match(A5)?.[1]), 8, 30);
    a10 ??= inBand(Number(whole.match(A10)?.[1]), 8, 30);
    return pe == null ? null : { pe, a5, a10 };
}

/** The report PDF for a date (YYYY-MM-DD → …EarningsInsight_MMDDYY.pdf; some weeks carry an 'A': …MMDDYYA.pdf). */
export function reportUrl(iso, suffix = '') {
    const [y, m, d] = iso.split('-');
    return `https://advantage.factset.com/hubfs/Website/Resources%20Section/Research%20Desk/Earnings%20Insight/EarningsInsight_${m}${d}${y.slice(2)}${suffix}.pdf`;
}

/**
 * Dates a new report could carry, newest first: today back to the day after `after`. Reports land Fridays
 * (Thursday or Wednesday in a holiday week), so only Wednesdays to Fridays.
 */
export function reportCandidates(today, after) {
    const out = [];
    for (let t = Date.parse(`${today}T12:00:00Z`); new Date(t).toISOString().slice(0, 10) > after; t -= 864e5) {
        if ([3, 4, 5].includes(new Date(t).getUTCDay())) out.push(new Date(t).toISOString().slice(0, 10));
    }
    return out;
}

const DAY = 864e5;
const ms = (iso) => Date.parse(`${iso}T00:00:00Z`);

/** The baked file → ascending `[{date, pe, a5, a10}]` (date = the report's), or null when malformed. */
export function decodeFwd(body) {
    if (!body || !/^\d{4}-\d{2}-\d{2}$/.test(body.from) || !Array.isArray(body.t) || !Array.isArray(body.v) || body.t.length !== body.v.length) return null;
    const t0 = ms(body.from), out = [];
    body.t.forEach((d, i) => {
        const pe = body.v[i];
        if (!Number.isInteger(d) || !Number.isFinite(pe)) return;
        const n = (a) => (Number.isFinite(a?.[i]) ? a[i] : null);
        out.push({ date: new Date(t0 + d * DAY).toISOString().slice(0, 10), pe, a5: n(body.a5), a10: n(body.a10) });
    });
    return out.length ? out : null;
}

/** Rows → the baked file's shape (the inverse of decodeFwd). */
export function encodeFwd(rows, meta = {}) {
    const from = rows[0].date, t0 = ms(from);
    return {
        ...meta, from, to: rows[rows.length - 1].date,
        t: rows.map((r) => Math.round((ms(r.date) - t0) / DAY)),
        v: rows.map((r) => r.pe), a5: rows.map((r) => r.a5 ?? null), a10: rows.map((r) => r.a10 ?? null),
    };
}

/** The baked rows plus the newer route rows (a date already baked keeps its baked row). */
export function joinFwd(baked, newer) {
    const last = baked?.length ? baked[baked.length - 1].date : '';
    return [...(baked || []), ...(newer || []).filter((r) => r.date > last)].sort((a, b) => (a.date < b.date ? -1 : 1));
}

export const FWD_KV_KEY = 'ftb:fwdpe';
export const PROBE_GAP_MS = 3 * 3600e3;
export const PROBE_BUDGET_MS = 25000; // no new FactSet fetch after this: /api/fwd-pe answers well inside /api/freshness's wait

/**
 * The reports since the bake (`bakedTo`): KV's list, plus any FactSet posted since, looked for at most every
 * 3 h and only once the newest is 5+ days old. `fetchPdf(url)` → bytes, or null on 404; `pdfText(bytes)` → its
 * text. `seen` = the newest report found even if it would not parse (/api/freshness turns that red). Newest day
 * first, until PROBE_BUDGET_MS; the bake fills any week skipped. Never throws.
 */
export async function resolveFwdPe({ fetchPdf, pdfText, kv = null, bakedTo, today, now = Date.now(), faults = new Set() }) {
    let doc = null;
    const messages = [];
    if (kv && !faults.has('fwdpe_kv')) {
        try { doc = parseEnvelope(await kv.get(FWD_KV_KEY)); } catch { messages.push('KV read failed'); }
    }
    let rows = (Array.isArray(doc?.rows) ? doc.rows : []).filter((r) => r?.date > bakedTo && r.pe > 0);
    let seen = doc?.seen || null, checkedAt = doc?.checkedAt || 0;
    const newest = rows.length ? rows[rows.length - 1].date : bakedTo;
    if ((ms(today) - ms(newest)) / DAY < 5 || now - checkedAt < PROBE_GAP_MS) return { rows, seen, checkedAt, messages };
    const found = [], until = Date.now() + PROBE_BUDGET_MS;
    for (const day of reportCandidates(today, newest)) {
        let buf = null;
        for (const suffix of ['', 'A']) {
            if (buf || Date.now() > until) break;
            try { buf = faults.has('fwdpe_http') ? null : await fetchPdf(reportUrl(day, suffix)); } catch (e) { messages.push(`${day}: ${String(e?.message || e).slice(0, 80)}`); }
        }
        if (!buf && Date.now() > until) { messages.push(`time budget hit at ${day}`); break; }
        if (!buf) continue;
        if (!seen || day > seen) seen = day;
        let got = null;
        try { got = parseFactSet(await pdfText(buf)); } catch { /* counted below */ }
        if (got) found.push({ date: day, ...got });
        else messages.push(`${day}: report has no forward P/E line`);
    }
    rows = joinFwd(rows, found);
    checkedAt = now;
    if (kv && !faults.has('fwdpe_kv')) {
        try { await kv.set(FWD_KV_KEY, { rows, seen, checkedAt }); } catch { messages.push('KV write failed'); }
    }
    return { rows, seen, checkedAt, messages };
}
