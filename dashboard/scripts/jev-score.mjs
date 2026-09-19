#!/usr/bin/env node

/**
 * jev-score.mjs — Score Jev pills predictions against actual market outcomes.
 *
 * Reads the logged daily verdicts from Upstash KV, fetches SPY and RSP daily
 * closes from Polygon, and scores:
 *   - Regime verdict vs the sign of SPY's 5-trading-day forward return
 *   - Breadth verdict vs the sign of the RSP/SPY ratio's 5-trading-day forward change
 *
 * Prints a hit-rate table per pill and per `by` (jev vs rule).
 *
 * Requires env vars:
 *   KV_REST_API_URL, KV_REST_API_TOKEN — Upstash KV credentials
 *   POLYGON_KEY — Polygon.io API key for daily closes
 *
 * Usage: node scripts/jev-score.mjs
 */

const KV_REST_API_URL = process.env.KV_REST_API_URL || '';
const KV_REST_API_TOKEN = process.env.KV_REST_API_TOKEN || '';
const POLYGON_KEY = process.env.POLYGON_KEY || '';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtPct(v) {
    if (v == null || !Number.isFinite(v)) return '  N/A  ';
    return (v >= 0 ? ' +' : ' ') + v.toFixed(1).padStart(5, '0') + '%';
}

function pad(s, n = 12) {
    return String(s).padEnd(n);
}

// ---------------------------------------------------------------------------
// Upstash KV helpers (same pattern as lib/jevLog.js)
// ---------------------------------------------------------------------------

async function kvGet(key) {
    const url = `${KV_REST_API_URL}/get/${encodeURIComponent(key)}`;
    const res = await fetch(url, {
        headers: { Authorization: `Bearer ${KV_REST_API_TOKEN}` },
    });
    if (!res.ok) return null;
    const data = await res.json();
    if (data.error) return null;
    return data.result ?? null;
}

async function kvLrange(key) {
    const url = `${KV_REST_API_URL}/lrange/${encodeURIComponent(key)}/0/-1`;
    const res = await fetch(url, {
        headers: { Authorization: `Bearer ${KV_REST_API_TOKEN}` },
    });
    if (!res.ok) return null;
    const data = await res.json();
    if (data.error) return null;
    return Array.isArray(data.result) ? data.result : null;
}

// ---------------------------------------------------------------------------
// Polygon daily close
// ---------------------------------------------------------------------------

async function polygonClose(ticker, dateStr) {
    // Fetch a single day's close from Polygon's grouped-daily or aggs endpoint.
    // The simplest approach: GET /v2/aggs/ticker/<T>/range/1/day/<date>/<date>
    const url = `https://api.polygon.io/v2/aggs/ticker/${ticker}/range/1/day/${dateStr}/${dateStr}?adjusted=true&apiKey=${POLYGON_KEY}`;
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    if (data.status !== 'OK' || !data.results || !data.results.length) return null;
    return data.results[0].c; // close price
}

async function polygonClosesRange(ticker, fromDate, toDate) {
    const url = `https://api.polygon.io/v2/aggs/ticker/${ticker}/range/1/day/${fromDate}/${toDate}?adjusted=true&apiKey=${POLYGON_KEY}`;
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    if (data.status !== 'OK' || !data.results) return null;
    // Build a map: date(YYYY-MM-DD) → close
    const map = {};
    for (const r of data.results) {
        const ms = r.t; // Unix ms
        const d = new Date(ms).toISOString().slice(0, 10);
        map[d] = r.c;
    }
    return map;
}

// ---------------------------------------------------------------------------
// Determine the sign of forward return over N trading days
// ---------------------------------------------------------------------------

function findCloseNdaysLater(closes, dateStr, n) {
    // closes: map of date → close
    const dates = Object.keys(closes).sort();
    const idx = dates.indexOf(dateStr);
    if (idx === -1) return null;
    const targetIdx = idx + n;
    if (targetIdx >= dates.length) return null;
    return closes[dates[targetIdx]];
}

function sign(v) {
    if (v == null || !Number.isFinite(v)) return null;
    if (v > 0) return 'up';
    if (v < 0) return 'down';
    return 'flat';
}

// ---------------------------------------------------------------------------
// Scoring logic
// ---------------------------------------------------------------------------

const PILL_SCORE = {
    regime: {
        goodWhen: { 'risk-on': 'up', neutral: null, 'risk-off': 'down' },
        forwardDays: 5,
        ticker: 'SPY',
    },
    breadth: {
        goodWhen: { broad: 'up', narrow: null, 'rolling-over': 'down' },
        forwardDays: 5,
        // RSP/SPY ratio direction vs itself
        ticker: 'RSP/SPY',
    },
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
    console.log('=== Jev Pills Scoring ===\n');

    if (!KV_REST_API_URL || !KV_REST_API_TOKEN) {
        console.error('❌ KV_REST_API_URL and KV_REST_API_TOKEN are required.');
        process.exit(1);
    }

    if (!POLYGON_KEY) {
        console.error('❌ POLYGON_KEY is required.');
        process.exit(1);
    }

    // 1. Read logged days
    const days = await kvLrange('ftb:jev:days');
    if (!days || !days.length) {
        console.log('No logged days found.');
        return;
    }

    const sortedDays = [...days].sort();
    console.log(`Found ${sortedDays.length} logged day(s): ${sortedDays[0]} → ${sortedDays[sortedDays.length - 1]}\n`);

    // 2. Read each day's verdicts
    const dayVerdicts = [];
    for (const d of sortedDays) {
        const raw = await kvGet(`ftb:jev:${d}`);
        if (!raw) {
            console.warn(`  ⚠ skipping ${d}: no data`);
            continue;
        }
        // raw may be a string (JSON) or already parsed
        const payload = typeof raw === 'string' ? JSON.parse(raw) : raw;
        dayVerdicts.push({ date: d, payload });
    }

    if (dayVerdicts.length < 2) {
        console.log('Need at least 2 logged days to score (one for baseline, one for outcome).');
        return;
    }

    // 3. Fetch SPY closes for the entire range + 5 extra trading days
    const rangeStart = sortedDays[0];
    const rangeEnd = sortedDays[sortedDays.length - 1];
    // Add ~10 calendar days buffer for the forward window
    const fetchEnd = new Date(new Date(rangeEnd).getTime() + 15 * 86400000).toISOString().slice(0, 10);

    console.log(`Fetching SPY closes ${rangeStart} → ${fetchEnd}...`);
    const spyCloses = await polygonClosesRange('SPY', rangeStart, fetchEnd);
    if (!spyCloses) {
        console.error('❌ Failed to fetch SPY closes from Polygon.');
        process.exit(1);
    }
    console.log(`  Got ${Object.keys(spyCloses).length} SPY trading days\n`);

    // Fetch RSP closes too for breadth ratio
    console.log(`Fetching RSP closes ${rangeStart} → ${fetchEnd}...`);
    const rspCloses = await polygonClosesRange('RSP', rangeStart, fetchEnd);
    if (!rspCloses) {
        console.error('❌ Failed to fetch RSP closes from Polygon.');
        process.exit(1);
    }
    console.log(`  Got ${Object.keys(rspCloses).length} RSP trading days\n`);

    // 4. Score each day (except the last few where forward window is unavailable)
    const results = { regime: { total: 0, hit: 0, by: { jev: { total: 0, hit: 0 }, rule: { total: 0, hit: 0 } } } };
    // Add breadth scorer
    results.breadth = { total: 0, hit: 0, by: { jev: { total: 0, hit: 0 }, rule: { total: 0, hit: 0 } } };

    for (let i = 0; i < dayVerdicts.length - 1; i++) {
        const { date, payload } = dayVerdicts[i];
        const pills = payload.pills || {};
        if (!pills) continue;

        // --- Regime score ---
        const reg = pills.regime || {};
        const spyPriceToday = spyCloses[date];
        // Need close 5 trading days later
        const spyCloseFwd = findCloseNdaysLater(spyCloses, date, 5);
        if (spyPriceToday != null && spyCloseFwd != null) {
            const forwardReturn = ((spyCloseFwd - spyPriceToday) / spyPriceToday) * 100;
            const fwdSign = sign(forwardReturn);
            const expected = PILL_SCORE.regime.goodWhen[reg.verdict];
            const isHit = expected ? (fwdSign === expected) : true; // neutral is always "correct"
            const by = reg.by || 'rule';

            results.regime.total++;
            results.regime.by[by].total++;
            if (isHit) {
                results.regime.hit++;
                results.regime.by[by].hit++;
            }

            if (i < 5 || i === dayVerdicts.length - 2) {
                // Print first 5 and last one
                console.log(`  ${date} regime=${reg.verdict}(${by}) SPY=${spyPriceToday} → 5d=${spyCloseFwd} (${fmtPct(forwardReturn)}) ${isHit ? '✅' : '❌'}`);
            }
        }

        // --- Breadth score ---
        const brd = pills.breadth || {};
        const rspPriceToday = rspCloses[date];
        const rspCloseFwd = findCloseNdaysLater(rspCloses, date, 5);
        // RSP/SPY ratio direction
        if (spyPriceToday != null && rspPriceToday != null && spyCloseFwd != null && rspCloseFwd != null) {
            const ratioToday = rspPriceToday / spyPriceToday;
            const ratioFwd = rspCloseFwd / spyCloseFwd;
            const ratioChange = ((ratioFwd - ratioToday) / ratioToday) * 100;
            const ratioSign = sign(ratioChange);
            const expected = PILL_SCORE.breadth.goodWhen[brd.verdict];
            const isHit = expected ? (ratioSign === expected) : true;
            const by = brd.by || 'rule';

            results.breadth.total++;
            results.breadth.by[by].total++;
            if (isHit) {
                results.breadth.hit++;
                results.breadth.by[by].hit++;
            }
        }
    }

    // 5. Print summary
    console.log('\n─── Hit Rate Summary ───\n');

    // Per pill
    console.log(`${pad('Pill', 14)} ${pad('Total', 8)} ${pad('Hits', 8)} ${pad('Rate', 8)}`);
    console.log('-'.repeat(40));
    for (const pill of ['regime', 'breadth']) {
        const r = results[pill];
        const rate = r.total > 0 ? ((r.hit / r.total) * 100).toFixed(1) + '%' : 'N/A';
        console.log(`${pad(pill, 14)} ${pad(String(r.total), 8)} ${pad(String(r.hit), 8)} ${pad(rate, 8)}`);
    }

    // Per by (jev vs rule)
    console.log(`\n${pad('By', 14)} ${pad('Pill', 14)} ${pad('Total', 8)} ${pad('Hits', 8)} ${pad('Rate', 8)}`);
    console.log('-'.repeat(55));
    for (const by of ['jev', 'rule']) {
        for (const pill of ['regime', 'breadth']) {
            const r = results[pill].by[by];
            if (r.total === 0) continue;
            const rate = ((r.hit / r.total) * 100).toFixed(1) + '%';
            console.log(`${pad(by, 14)} ${pad(pill, 14)} ${pad(String(r.total), 8)} ${pad(String(r.hit), 8)} ${pad(rate, 8)}`);
        }
    }

    console.log('\nDone.');
}

main().catch((err) => {
    console.error('Fatal error:', err);
    process.exit(1);
});