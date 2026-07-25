/**
 * Four Horsemen — independent-source cascades for the three FRED-fed lines.
 *
 * WHY THIS EXISTS. Three of the four horsemen (initial claims, unemployment,
 * the 10Y−2Y spread) originally had exactly ONE live provider: the FRED API,
 * behind ONE api key. Copper/Gold gets 4-5 providers per leg and the vol table
 * gets 3-4 per cell, but the recession-watch card — the one actually used for
 * decisions — had none. A revoked/quota-burned FRED key or an api.stlouisfed.org
 * outage took all three lines down at once.
 *
 * Each horseman now cascades through genuinely independent providers. Crucially
 * these are ORIGIN sources, not FRED mirrors: BLS publishes the unemployment
 * rate and Treasury publishes the yield curve; FRED merely republishes them. So
 * a total FRED failure is survivable, not just a slow api key.
 *
 *   SPREAD (10Y−2Y) : FRED T10Y2Y → US Treasury daily yield curve CSV (keyless,
 *                     one request per calendar year) → FRED keyless graph CSV
 *   UNEMPLOYMENT    : FRED UNRATE → BLS API v2 LNS14000000 (keyless) → FRED
 *                     keyless graph CSV
 *   CLAIMS          : FRED ICSA → FRED keyless graph CSV
 *
 * The keyless `fredgraph.csv` tier shares FRED's servers but NOT its api key, so
 * it survives the likeliest FRED failure mode (key revoked, quota burned, key
 * env var lost on a redeploy) even though it won't survive a full FRED outage.
 * It is deliberately last.
 *
 * COST ON THE HAPPY PATH IS ZERO. The route only builds these cascades when its
 * primary FRED series came back empty, so a healthy load makes no extra calls.
 *
 * DEGRADED HISTORY IS EXPECTED AND FINE. Fallback providers return less history
 * than FRED's full run (Treasury ~2 years, BLS ~10 years). The stat chips, the
 * "N of 4 riding" badge and the 12-month trend notes all still work; the ALL/20Y
 * tabs just draw a shorter line. A short real chart beats a blank card.
 *
 * Fault gates for end-to-end testing:
 *   ?_fail=hm_treasury, hm_bls, hm_fredcsv   (disable a fallback provider)
 *   ?_fail=fred                              (kills the primary — existing gate)
 * e.g. `?_fail=fred` alone should show all three lines resolving from the
 * fallback tiers; `?_fail=fred,hm_treasury,hm_bls,hm_fredcsv` should show them
 * degrade to the /tmp last-known-good.
 *
 * Everything here is pure given injected fetchers, so it is unit-testable
 * without network. Parsers are COLUMN/FIELD-ANCHORED (never positional) for the
 * same reason the F-2 bankruptcy parser is: an upstream reshuffle must produce
 * no data rather than wrong data.
 */

import { isStale, withFreshness } from './freshness';

/** Split one CSV line, honoring double-quoted fields. */
export function splitCsvLine(line) {
    const out = [];
    let cur = '', inQ = false;
    for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (inQ) {
            if (c === '"') {
                if (line[i + 1] === '"') { cur += '"'; i++; } else { inQ = false; }
            } else cur += c;
        } else if (c === '"') inQ = true;
        else if (c === ',') { out.push(cur); cur = ''; }
        else cur += c;
    }
    out.push(cur);
    return out;
}

const finite = (n) => Number.isFinite(n);

/** Sort ascending by date and drop duplicate dates (last one wins). */
function ascendUnique(points) {
    const byDate = new Map();
    for (const p of points) if (p && p.date && finite(p.value)) byDate.set(p.date, p.value);
    return [...byDate.entries()]
        .map(([date, value]) => ({ date, value }))
        .sort((a, b) => (a.date < b.date ? -1 : 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * US Treasury "daily treasury yield curve" CSV → ascending 10Y−2Y spread.
 *
 * Shape: `Date,"1 Mo","1.5 Month","2 Mo",...,"2 Yr",...,"10 Yr",...` with rows
 * newest-first and US-format dates (`07/24/2026`). The 2Y and 10Y columns are
 * located BY HEADER NAME — Treasury has added tenor columns before (the "1.5
 * Month" column is recent), so a positional read would silently shift.
 */
export function parseTreasuryCsv(csv) {
    if (!csv || typeof csv !== 'string') return [];
    const lines = csv.trim().split(/\r?\n/).filter((l) => l.trim());
    if (lines.length < 2) return [];

    const header = splitCsvLine(lines[0]).map((h) => h.trim().toLowerCase());
    const dateIdx = header.findIndex((h) => h === 'date');
    // "2 Yr" / "10 Yr" exactly — never match "20 Yr" for "2 Yr".
    const idxOf = (n) => header.findIndex((h) => new RegExp(`^${n}\\s*yr$`).test(h));
    const i2 = idxOf(2), i10 = idxOf(10);
    if (dateIdx < 0 || i2 < 0 || i10 < 0) throw new Error('Treasury CSV: 2 Yr / 10 Yr column not found');

    const out = [];
    for (const line of lines.slice(1)) {
        const f = splitCsvLine(line);
        const m = (f[dateIdx] || '').trim().match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
        if (!m) continue;
        const [, mm, dd, yyyy] = m;
        const y2 = parseFloat(f[i2]), y10 = parseFloat(f[i10]);
        if (!finite(y2) || !finite(y10)) continue;
        out.push({
            date: `${yyyy}-${mm.padStart(2, '0')}-${dd.padStart(2, '0')}`,
            // Round to FRED's own 2dp so a Treasury-sourced value is byte-identical
            // to the FRED one it replaces (verified: 4.69 − 4.33 = 0.36 = T10Y2Y).
            value: Math.round((y10 - y2) * 100) / 100,
        });
    }
    return ascendUnique(out);
}

/**
 * BLS API v2 response → ascending monthly series.
 *
 * Monthly points are periods M01-M12; M13 is the ANNUAL AVERAGE and must be
 * dropped or it lands as a 13th month and corrupts the trend fit. Dates are
 * stamped on the 1st, matching how FRED dates monthly series.
 */
export function parseBlsSeries(json) {
    const data = json?.Results?.series?.[0]?.data;
    if (!Array.isArray(data)) {
        throw new Error(`BLS: unexpected payload (${json?.status || 'no status'})`);
    }
    const out = [];
    for (const d of data) {
        const m = /^M(\d{2})$/.exec(d?.period || '');
        if (!m || m[1] === '13') continue; // M13 = annual average, not a month
        const value = parseFloat(d.value);
        if (!finite(value)) continue;
        out.push({ date: `${d.year}-${m[1]}-01`, value });
    }
    return ascendUnique(out);
}

/**
 * FRED's keyless graph CSV → ascending series. Header is
 * `observation_date,<SERIES_ID>` (older vintages: `DATE,<SERIES_ID>`); missing
 * observations are a bare `.` exactly like the JSON API.
 */
export function parseFredGraphCsv(csv) {
    if (!csv || typeof csv !== 'string') return [];
    const lines = csv.trim().split(/\r?\n/).filter((l) => l.trim());
    if (lines.length < 2) return [];
    const header = splitCsvLine(lines[0]).map((h) => h.trim().toLowerCase());
    if (!/^(observation_)?date$/.test(header[0] || '')) {
        throw new Error('fredgraph CSV: unexpected header');
    }
    const out = [];
    for (const line of lines.slice(1)) {
        const f = splitCsvLine(line);
        const date = (f[0] || '').trim();
        if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) continue;
        const value = parseFloat((f[1] || '').trim());
        if (!finite(value)) continue; // covers the '.' missing marker
        out.push({ date, value });
    }
    return ascendUnique(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cascade
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Try each source in order; return the first that yields a usable, fresh series.
 *
 * A source descriptor is:
 *   { name, freshnessDays, fetch: async () => [{date:'YYYY-MM-DD', value:Number}] }
 * with the array ascending. Sources are skipped when fault-injected (`hm_<name>`),
 * when they throw, when they return fewer than 2 points (a one-point series can
 * neither draw a line nor fit a trend), or when their newest point is already
 * past the series' staleness deadline — a fallback that is itself stale is worse
 * than falling through to the next provider.
 *
 * Never throws. Returns { history, current, currentDate, source, tried }.
 */
export async function resolveHorseman(sources, faults, now = new Date()) {
    const tried = [];
    for (const s of sources) {
        if (faults && faults.has(`hm_${s.name}`)) { tried.push(`${s.name}:off`); continue; }
        try {
            const history = await s.fetch();
            if (!Array.isArray(history) || history.length < 2) { tried.push(`${s.name}:empty`); continue; }
            const last = history[history.length - 1];
            if (!last || !finite(last.value) || !last.date) { tried.push(`${s.name}:empty`); continue; }
            if (isStale(last.date, s.freshnessDays, now)) { tried.push(`${s.name}:stale(${last.date})`); continue; }
            tried.push(`${s.name}:ok(${history.length})`);
            return { history, current: last.value, currentDate: last.date, source: s.name, tried };
        } catch (e) {
            tried.push(`${s.name}:err`);
            console.warn(`[horsemen ${s.name}] ${e?.message}`);
        }
    }
    return { history: [], current: null, currentDate: null, source: null, tried };
}

/**
 * Should we try the independent providers for this line?
 *
 * Yes when the primary gave us too little to draw (fewer than 2 points), or gave
 * us nothing current (`unavailable`), or gave us a value that is already past its
 * staleness deadline. That last case is the one an independent ORIGIN source
 * uniquely fixes: when FRED quietly stops updating a single series, the /tmp
 * last-known-good is equally stale, but BLS/Treasury are still publishing.
 */
export function needsRepair(metric) {
    if (!metric) return true;
    if (!Array.isArray(metric.history) || metric.history.length < 2) return true;
    return metric.unavailable === true || metric.stale === true;
}

/**
 * Never let a fallback move a line BACKWARDS in time. A provider with a shorter
 * history but an older newest-point is a downgrade, not a repair — e.g. BLS
 * lagging a month behind a freshly-printed FRED UNRATE.
 */
export function isUpgrade(candidate, incumbent) {
    if (!candidate || !candidate.currentDate) return false;
    if (!incumbent || !incumbent.asOf) return true;
    return candidate.currentDate > incumbent.asOf;
}

/**
 * Overlay freshly-resolved horsemen onto a cached base payload.
 *
 * The case this exists for: live FRED is entirely dead (revoked key, outage) but
 * Treasury and BLS — who ORIGINATE two of these series — are answering fine.
 * Without this, the route would judge its payload "not good" on loadedCount 0 and
 * discard the live data in favour of a wholly cached one, which would make the
 * independent providers useless in exactly the scenario they were added for.
 *
 * So: take everything from the cached base (so the other cards keep their last
 * known values instead of blanking), then overwrite ONLY the lines we just
 * fetched live. The result is honestly labelled — `_meta.stale` and `hasErrors`
 * stay set, because most of the payload really is cached — and the caller must
 * not store it back as last-known-good (see serve's `shouldStore`).
 *
 * Returns null when there is nothing live to overlay.
 */
export function mergeHorsemenOverBase(base, live) {
    const hasLive = live && (live.horsemen && Object.keys(live.horsemen).length > 0 || live.yieldCurve);
    if (!hasLive) return null;

    const b = base && typeof base === 'object' ? base : {};
    const meta = b._meta || {};
    const merged = {
        ...b,
        horsemen: { ...(b.horsemen || {}), ...(live.horsemen || {}) },
    };
    if (live.yieldCurve) merged.yieldCurve = live.yieldCurve;

    const names = [...Object.keys(live.horsemen || {}), ...(live.yieldCurve ? ['spread'] : [])];
    merged._meta = {
        ...meta,
        stale: true,
        hasErrors: true,
        messages: [
            ...(meta.messages || []),
            `live FRED unavailable; Four Horsemen served LIVE from independent sources (${names.join(', ')}), rest from cache`,
            // Carry the per-line cascade trail through the merge — it is the only
            // way to see WHICH provider answered when verifying tiers on prod.
            ...(live.messages || []),
        ],
    };
    return merged;
}

/**
 * Shape a resolved horseman into the `horsemen.<key>` payload the card reads.
 * Mirrors the freshness contract of every other FRED metric so the UI's orange
 * 🕐 stale treatment and the health check's `staleDays > 3` rule work unchanged.
 */
export function buildHorseman(resolved, freshnessDays, now = new Date()) {
    const f = withFreshness(resolved.current, resolved.currentDate, freshnessDays, now);
    return {
        current: f.value,
        asOf: f.asOf,
        stale: f.stale,
        staleDays: f.staleDays,
        unavailable: f.unavailable,
        history: resolved.history || [],
        source: resolved.source,
        tried: resolved.tried || [],
    };
}
