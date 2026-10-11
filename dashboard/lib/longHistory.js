/**
 * 📈 Long look-back for the tap-a-number popover: decades of history BEFORE the sheet.
 *
 * The popover's own series is the history sheet (daily snapshots, which start 2026-03-12).
 * scripts/bake_long_history.py bakes what came before, one static file per stat in
 * public/history/<key>.json, plus the small bundled index imported here (which stats have
 * one, from when, the source). The file is fetched only when that stat's popover opens, once
 * per page (a failed fetch is retried on the next open), and the chart simply stays
 * sheet-only until it lands — or if it never does.
 *
 * The join never mixes the two: baked points are drawn only BEFORE the cut — the sheet's
 * first point, or the index's `sheetFrom` for a stat whose sheet history is a different
 * number (P/E switched source on 2026-09-05; BEA revised the whole savings history on
 * 2026-09-30). From the cut on, every point is the sheet's.
 */
import INDEX from './data/longHistoryIndex.json';

const DAY = 86400000;
const t = (iso) => Date.parse(`${iso}T00:00:00Z`);
const ISO = /^\d{4}-\d{2}-\d{2}$/;

/** `{from, to, n, source, sheetFrom?}` for a stat with a baked file, else null. */
export function longInfo(key) {
    return (key && INDEX.keys?.[key]) || null;
}

/** A baked file `{from, t:[day offsets], v:[values]}` → `[{date, value}]`, or null when malformed. */
export function decodeLong(body) {
    if (!body || !ISO.test(body.from) || !Array.isArray(body.t) || !Array.isArray(body.v) || body.t.length !== body.v.length) return null;
    const t0 = t(body.from);
    const out = [];
    for (let i = 0; i < body.t.length; i++) {
        const v = body.v[i], d = body.t[i];
        if (typeof v !== 'number' || !Number.isFinite(v) || !Number.isInteger(d)) continue;
        out.push({ date: new Date(t0 + d * DAY).toISOString().slice(0, 10), value: v });
    }
    return out.length ? out : null;
}

const cache = new Map(); // key → Promise<points|null>; a failure is evicted so the next open retries
const done = new Map();  // key → points, for a synchronous first render once loaded

/** Fetch + decode one stat's baked history. Never rejects: resolves null on any failure. */
export function loadLong(key) {
    if (!longInfo(key)) return Promise.resolve(null);
    if (done.has(key)) return Promise.resolve(done.get(key));
    if (!cache.has(key)) {
        const p = Promise.resolve()
            .then(() => fetch(`/history/${encodeURIComponent(key)}.json`))
            .then((r) => (r?.ok ? r.json() : null))
            .then(decodeLong)
            .catch(() => null)
            .then((pts) => {
                if (pts) done.set(key, pts); else cache.delete(key);
                return pts;
            });
        cache.set(key, p);
    }
    return cache.get(key);
}

/** The loaded points, if this stat's file has already arrived (undefined otherwise). */
export function peekLong(key) {
    return done.get(key);
}

/** Test hook. */
export function resetLong() { cache.clear(); done.clear(); }

/** The date from which the sheet's points are drawn (baked points stay strictly before it). */
export function cutFor(sheetPoints, info) {
    const first = sheetPoints?.[0]?.date || null;
    const from = info?.sheetFrom && ISO.test(info.sheetFrom) ? info.sheetFrom : null;
    if (!first) return from;
    return from && from > first ? from : first;
}

/** Baked points before the cut, then the sheet's points from the cut. Baked ones carry `long: true`. */
export function joinLong(sheetPoints, longPoints, info) {
    const sheet = sheetPoints || [];
    if (!Array.isArray(longPoints) || !longPoints.length) return sheet;
    const cut = cutFor(sheet, info);
    if (!cut) return sheet;
    const old = [];
    for (const p of longPoints) {
        if (p.date >= cut) break;
        old.push({ date: p.date, value: p.value, long: true });
    }
    if (!old.length) return sheet;
    return old.concat(sheet.filter((p) => p.date >= cut));
}

/** Days from the oldest point this stat can show (baked or sheet) to its newest sheet point. */
export function spanDays(sheetPoints, info) {
    const pts = sheetPoints || [];
    if (!pts.length) return 0;
    const last = pts[pts.length - 1].date;
    const first = info?.from && ISO.test(info.from) && info.from < pts[0].date ? info.from : pts[0].date;
    return Math.round((t(last) - t(first)) / DAY);
}
