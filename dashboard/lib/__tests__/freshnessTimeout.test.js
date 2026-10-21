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

test('freshness timeout fits inside its own maxDuration', () => {
    expect(fresh.TIMEOUT_MS).toBeLessThan(fresh.maxDuration * 1000);
});
