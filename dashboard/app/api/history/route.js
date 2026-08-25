/**
 * /api/history — what every markable number WAS, so the dashboard can show which
 * ones changed in a way that is news. See lib/marks.js and
 * docs/superpowers/specs/2026-08-25-fresh-print-marks-design.md.
 *
 * Reads the PUBLIC CSV export of Sheet1 in the financial-dashboard-history sheet
 * (the same doc /api/fred's last-resort tier reads, different tab). No credentials:
 * the sheet is published, so there is nothing here to leak.
 *
 * Returns a DIGEST, not the sheet — a ~2KB `{today, metrics:{key: {baseline, heldFrom,
 * runs, sigma}}}` against a ~100KB CSV that grows two rows a day. The digest carries only
 * what history can know; the live-vs-baseline comparison happens on the client, where the
 * current value is known (see lib/marks.js `markFor`).
 *
 * Dashboard-only, no Lambda hop. Never-throw via serve(): on any failure the digest comes
 * back empty, which renders NO marks and leaves every number exactly as it is today. This
 * feature must fail invisible — a wrong mark is far worse than a missing one.
 *
 * Fault gate: `?_fail=history_sheet`.
 */
import { serve } from '../../../lib/store';
import { faultsFrom, trip } from '../../../lib/faults';
import { parseCsvLine } from '../../../lib/sheetLkg';
import { buildDigest } from '../../../lib/marks';

const SHEET_CSV =
    'https://docs.google.com/spreadsheets/d/1lA-_yjLMc3qDTt9sogSPQrCohNULIk5wwJYfb5wIHfc/export?format=csv&gid=0';

const FETCH_TIMEOUT_MS = 12000;
/** Guard against a runaway sheet: ~2 rows/day, so this is many years of headroom. */
const MAX_ROWS = 20000;

/** Fetch the CSV with a hard deadline; the caller turns any failure into an empty digest. */
async function fetchSheetCsv(faults) {
    trip('history_sheet', faults);
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), FETCH_TIMEOUT_MS);
    try {
        const res = await fetch(SHEET_CSV, {
            signal: ctl.signal,
            // The scraper appends twice daily; a 30-min cache matches the rest of
            // the dashboard and keeps this off the request path most of the time.
            next: { revalidate: 1800 },
            headers: { 'user-agent': 'financial-dashboard/1.0' },
        });
        if (!res.ok) throw new Error(`sheet HTTP ${res.status}`);
        return await res.text();
    } finally {
        clearTimeout(timer);
    }
}

/** CSV text → rows of raw string cells. Header row is dropped by the date check downstream. */
export function parseRows(text) {
    if (!text || typeof text !== 'string') return [];
    const out = [];
    for (const line of text.split(/\r?\n/)) {
        if (!line.trim()) continue;
        out.push(parseCsvLine(line));
        if (out.length >= MAX_ROWS) break;
    }
    return out;
}

export async function GET(request) {
    // Touch the request so Next renders this handler per-request. Without it the route
    // is STATICALLY PRERENDERED at build time — the payload freezes and ?_fail= is
    // silently ignored on production. faultsFrom alone does NOT make a route dynamic
    // (its try/catch swallows Next's DynamicServerError probe). AGENTS.md §3.
    request.headers.get('user-agent');

    const faults = faultsFrom(request);

    return serve('history', async () => {
        const csv = await fetchSheetCsv(faults);
        const rows = parseRows(csv);
        const digest = buildDigest(rows);
        const count = Object.keys(digest.metrics).length;
        return {
            ...digest,
            _meta: {
                source: 'Google Sheet (financial-dashboard-history, Sheet1)',
                hasErrors: false,
                messages: [`Baselines for ${count} metrics`],
                fetchedAt: new Date().toISOString(),
            },
        };
    }, {
        // A digest with no metrics is not servable — fall through to the last-known-good
        // copy rather than telling the client "nothing changed", which is a claim we
        // cannot actually make when the sheet did not load.
        isGood: (p) => !!p && !!p.metrics && Object.keys(p.metrics).length > 0,
        fallback: { today: null, metrics: {} },
    });
}
