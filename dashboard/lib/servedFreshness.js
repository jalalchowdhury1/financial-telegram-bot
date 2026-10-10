/**
 * servedFreshness.js — fleet contract v1 for GET /api/freshness.
 *
 * Measures what the dashboard SERVES, not whether a writer ran. Each item reports ages in
 * hours only (no dates, values or content):
 *   inputAgeH   hours since the newest input the served thing should reflect (null = none)
 *   servedAgeH  hours since the input the CURRENTLY served thing reflects (null = unknown)
 *   graceH      allowed producer lag
 *   maxAgeH     optional absolute cap for feeds with no separate input
 * The fleet monitor (not this app) judges red/green; see AGENTS.md "Freshness endpoint".
 *
 * Pipelines (producer -> screen):
 *   rubber-band    Mac mini launchd (weekdays 18:30 ET) -> secret gist -> /api/rubber-band
 *                  -> Rubber Band card + the 🪢 Telegram line. Input = the newest NYSE close.
 *   history-sheet  financial-dashboard-history scraper (GHA cron 02:00 + 14:00 UTC) -> Sheet1
 *                  -> /api/history -> print marks, "What moved" σ, 90-day tap charts.
 */
import { sessionOf, etParts, etWallToMs } from './marketClock';

const H = 3600e3;
const round1 = (x) => Math.round(x * 10) / 10;
const addDays = (date, n) => new Date(Date.parse(`${date}T12:00:00Z`) + n * 864e5).toISOString().slice(0, 10);
const isDate = (s) => typeof s === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(s) && !Number.isNaN(Date.parse(`${s}T00:00:00Z`));

// Close is 16:00 ET (13:00 on early-close days); the Mac runs at 18:30 ET, so an early
// close legitimately waits 5.5 h. +0.5 h edge cache (lib/cdn.js rubber-band ≤ 30 min).
export const RUBBER_BAND_GRACE_H = 6;
// Scraper runs every 12 h; one dropped GHA run (seen ~1×/fortnight in Sheet1) + GHA queue
// delay must stay green, two in a row must not.
export const HISTORY_GRACE_H = 12;
export const HISTORY_MAX_AGE_H = 30;
// Scraper cron hours (UTC) — the Date column is the runner's UTC date.
const HISTORY_RUN_HOURS_UTC = [2, 14];

/** Epoch ms of a trading day's close (16:00 ET, or 13:00 on early closes); null if not a session. */
export function closeMsOf(date) {
    const s = sessionOf(date);
    return s ? etWallToMs(date, s.close) : null;
}

/** Epoch ms of the newest NYSE close at or before `now`. */
export function latestCloseMs(now = Date.now()) {
    let d = etParts(now).date;
    for (let i = 0; i < 15; i++, d = addDays(d, -1)) {
        const c = closeMsOf(d);
        if (c != null && c <= now) return c;
    }
    return null;
}

/** Rubber band: payload = the /api/rubber-band JSON (producer: scripts/rubber_band.py build_snapshot). */
export function rubberBandItem(payload, now = Date.now()) {
    const newest = latestCloseMs(now);
    const asOf = payload && payload.asOf;
    let served = null;
    if (isDate(asOf)) served = closeMsOf(asOf) ?? etWallToMs(asOf, 16 * 60);
    return {
        name: 'rubber-band',
        inputAgeH: newest == null ? null : round1((now - newest) / H),
        servedAgeH: served == null ? null : round1((now - served) / H),
        graceH: RUBBER_BAND_GRACE_H,
    };
}

/**
 * Newest Sheet1 row: its UTC date and how many rows carry it (1 = only the 02:00 run so
 * far, ≥2 = the 14:00 run too). Rows are raw CSV cells; non-date first cells are skipped.
 */
export function newestRowOf(rows) {
    let date = null;
    let count = 0;
    for (const r of rows || []) {
        const d = r && typeof r[0] === 'string' ? r[0].trim() : '';
        if (!isDate(d)) continue;
        if (date == null || d > date) { date = d; count = 1; } else if (d === date) count++;
    }
    return date ? { date, rows: count } : null;
}

/** History sheet: payload = the /api/history JSON; reads `_meta.newestRow`. */
export function historyItem(payload, now = Date.now()) {
    const nr = payload && payload._meta && payload._meta.newestRow;
    let served = null;
    if (nr && isDate(nr.date)) {
        const hour = nr.rows >= 2 ? HISTORY_RUN_HOURS_UTC[1] : HISTORY_RUN_HOURS_UTC[0];
        served = Date.parse(`${nr.date}T${String(hour).padStart(2, '0')}:00:00Z`);
    }
    return {
        name: 'history-sheet',
        inputAgeH: null,
        servedAgeH: served == null ? null : round1(Math.max(0, now - served) / H),
        graceH: HISTORY_GRACE_H,
        maxAgeH: HISTORY_MAX_AGE_H,
    };
}

/** Same rule the fleet monitor applies — exported so tests prove fresh=green / stale=red. */
export function isRed(item) {
    const { inputAgeH, servedAgeH, graceH, maxAgeH } = item;
    if (inputAgeH != null && inputAgeH > graceH && (servedAgeH == null || servedAgeH > inputAgeH + 0.25)) return true;
    if (maxAgeH != null && servedAgeH != null && servedAgeH > maxAgeH) return true;
    return false;
}

// Market stats behind serve()/route tiers (added 2026-10-09 with the KV last-good tier):
// a route can now quietly serve a saved copy for days. Live payload → reflects the
// newest close (servedAgeH = inputAgeH). Saved copy → servedAgeH = hours since it was
// saved (a copy saved after the close still reflects it; one saved before goes red
// once the input is > grace old). Nothing served (Unavailable / error) → null → red.
export const SERVED_COPY_GRACE_H = 6;
export const SERVED_ROUTES = ['spy', 'spy-daily-move', 'market-extra', 'fred', 'sheets', 'fear-greed'];
const ISO_RE = /(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)/;

/** null = nothing usable served; { copy:false } = live; { copy:true, savedMs } = a saved copy. */
export function servedCopyOf(payload) {
    if (!payload || typeof payload !== 'object' || payload.error) return null;
    const m = payload._meta || {};
    const src = String(m.source || payload.source || '');
    if (/^(Unavailable|Failed|Static Defaults|none)$/i.test(src)) return null;
    const copy = !!m.lastGoodAt || /last-good|last-known-good|^Stale/i.test(src);
    if (!copy) return { copy: false };
    const t = Date.parse(m.lastGoodAt || (ISO_RE.exec(src) || [])[1] || '');
    return { copy: true, savedMs: Number.isFinite(t) ? t : null };
}

export function servedCopyItem(name, payload, now = Date.now()) {
    const newest = latestCloseMs(now);
    const inputAgeH = newest == null ? null : round1((now - newest) / H);
    const s = servedCopyOf(payload);
    let servedAgeH = null;
    if (s && !s.copy) servedAgeH = inputAgeH;
    else if (s && s.savedMs != null) servedAgeH = round1((now - s.savedMs) / H);
    return { name: `served:${name}`, inputAgeH, servedAgeH, graceH: SERVED_COPY_GRACE_H };
}

export function buildFreshness(raw, now = Date.now()) {
    return {
        app: 'financial-telegram-bot',
        v: 1,
        items: [
            rubberBandItem(raw.rubberBand, now), historyItem(raw.history, now),
            ...SERVED_ROUTES.map((r) => servedCopyItem(r, raw.served?.[r], now)),
        ],
    };
}
