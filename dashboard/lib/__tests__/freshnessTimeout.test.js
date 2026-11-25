/**
 * @jest-environment node
 *
 * Tripwire (21 Oct 2026): /api/freshness gave /api/fred 20 s while fred may take up to
 * FRED_BUDGET_MS (40 s) on a cold cache, so a slow-but-fine fred read as "served nothing"
 * and the fleet row went red. The probe must outlast the route and fit in its own maxDuration.
 */
import fs from 'fs';
import path from 'path';

const fresh = require('../../app/api/freshness/route');
const fredSrc = fs.readFileSync(path.join(__dirname, '../../app/api/fred/route.js'), 'utf8');
const fredBudget = Number(fredSrc.match(/const FRED_BUDGET_MS = (\d+)/)[1]);

test('freshness waits longer than /api/fred budget', () => {
    expect(fresh.TIMEOUT_MS).toBeGreaterThan(fredBudget + 5000);
});

test('freshness waits longer than /api/fwd-pe can take', () => {
    const { PROBE_BUDGET_MS } = require('../fwdPe');
    const fwdSrc = fs.readFileSync(path.join(__dirname, '../../app/api/fwd-pe/route.js'), 'utf8');
    const fetchMs = Number(fwdSrc.match(/AbortSignal\.timeout\((\d+)\)/)[1]);
    expect(fresh.TIMEOUT_MS).toBeGreaterThan(PROBE_BUDGET_MS + fetchMs + 5000); // last fetch starts just inside the budget
});

// Tripwire (25 Nov 2026): served:sheets went red "(none)" with no reason; the probe's own
// fetch of /api/sheets failed once and soft() threw the error away.
describe('servedSibling: retry once, then name the failure', () => {
    const { servedCopyItem } = require('../servedFreshness');
    const now = Date.parse('2026-11-25T16:18:00Z');

    test('a one-off failure is retried and the real answer graded', async () => {
        let n = 0;
        const get = async () => { if (n++ === 0) throw new Error('/api/sheets HTTP 504'); return { _meta: { source: 'Google Sheets (Live)', hasErrors: false } }; };
        const p = await fresh.servedSibling(get, '/api/sheets', 0, () => 1000);
        expect(n).toBe(2);
        expect(servedCopyItem('sheets', p, now).servedAgeH).not.toBeNull();
    });

    test('two failures → item.why names the HTTP error, not a bare (none)', async () => {
        const get = async () => { throw new Error('/api/sheets HTTP 504'); };
        const item = servedCopyItem('sheets', await fresh.servedSibling(get, '/api/sheets', 0, () => 1000), now);
        expect(item.servedAgeH).toBeNull();
        expect(item.why).toBe('probe fetch failed: /api/sheets HTTP 504');
    });

    test('no retry when a second full wait would overrun maxDuration; a timeout is named', async () => {
        let n = 0;
        const get = async () => { n++; const e = new Error('aborted'); e.name = 'AbortError'; throw e; };
        const p = await fresh.servedSibling(get, '/api/sheets', 0, () => fresh.TIMEOUT_MS);
        expect(n).toBe(1);
        expect(p.error).toMatch(/\/api\/sheets no answer in 50 s/);
    });
});

test('freshness timeout fits inside its own maxDuration', () => {
    expect(fresh.TIMEOUT_MS).toBeLessThan(fresh.maxDuration * 1000);
});
