/**
 * @jest-environment node
 *
 * scripts/signal-states.mjs end to end (the Mac's daily signal ping reads its JSON): the real script against a
 * FAKE site on 127.0.0.1. The sheet series carries each baked file up to today. A feed that fails exits 1 with
 * nothing on stdout; a signal whose data is empty or stopped keeps an old or null asOf. Either way the ping's
 * log ends red instead of reading as a calm day.
 */
const http = require('http');
const path = require('path');
const { execFile } = require('child_process');

const SCRIPT = path.join(__dirname, '..', 'signal-states.mjs');
const KEYS = ['aaiiDiff', 'vix-ratio-daily', 'cl', 'claims', 'usdcad', 'vixCurrent', 'housing', 'peRatio', 'tnx', 'spy-daily'];
// Baked: a reading every 7 days from 2024-01-01 to 2026-06-30, all calm.
const baked = (v) => ({ from: '2024-01-01', t: Array.from({ length: 131 }, (_, i) => i * 7), v: Array(131).fill(v) });
// Sheet: one value a day from 2026-08-01 for 60 days (to 2026-09-29); a null day is skipped, not a zero.
const day = (n, v, last) => Array.from({ length: n }, (_, i) => (i === n - 1 ? last : i === 10 ? null : v));
const sheet = { from: '2026-08-01', v: { aaiiDiff: day(60, 5, 25), vixCurrent: day(60, 15, 21), vix3m: day(60, 20, 20) } };

let server, base, routes;
beforeAll(async () => {
    server = http.createServer((req, res) => {
        const r = routes[req.url];
        res.writeHead(r ? r[0] : 404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(r ? r[1] : {}));
    });
    await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
    base = `http://127.0.0.1:${server.address().port}`;
});
afterAll(() => new Promise((ok) => server.close(ok)));
beforeEach(() => {
    routes = { '/api/history': [200, { series: sheet }], ...Object.fromEntries(KEYS.map((k) => [`/history/${k}.json`, [200, baked(k === 'vix-ratio-daily' ? 0.8 : k === 'spy-daily' ? 500 : 5)]])) };
});

const run = () => new Promise((resolve) => {
    execFile(process.execPath, [SCRIPT], { env: { ...process.env, DASH_BASE: base, NODE_NO_WARNINGS: '1' }, timeout: 30000 },
        (err, stdout, stderr) => resolve({ code: err ? err.code : 0, stdout, stderr }));
});

it('carries each baked file up to the sheet\'s newest day, and fires on it', async () => {
    const { code, stdout } = await run();
    expect(code).toBe(0);
    const st = Object.fromEntries(JSON.parse(stdout).map((g) => [g.id, g]));
    expect(Object.keys(st)).toEqual(['aaii', 'vix', 'fo', 'cu', 'hv', 'pt']);
    expect(st.aaii).toMatchObject({ fired: '2026-09-29', asOf: '2026-09-29', share: 1.25 });
    expect(st.vix).toMatchObject({ fired: '2026-09-29', asOf: '2026-09-29', share: 1.05 }); // 21 / 20, from the sheet's own VIX and VIX 3M
    // Housing isn't in this sheet: its pair stops 45 days after the last baked reading. The ping sees the old
    // asOf and ends its run "stale: hv …" (~/.local/bin/signal-ping) instead of calling it a calm day.
    expect(st.hv).toMatchObject({ fired: null, asOf: '2026-08-13' });
});

it('a baked file that comes back empty gives its signal no data (asOf null), never a reading', async () => {
    routes['/history/tnx.json'] = [200, { from: '2024-01-01', t: [], v: [] }];
    const { code, stdout } = await run();
    expect(code).toBe(0);
    expect(JSON.parse(stdout).find((g) => g.id === 'pt')).toMatchObject({ fired: null, asOf: null, share: null });
});

it.each([
    ['a baked file fails', '/history/housing.json', [500, {}], /housing\.json → 500/],
    ['the sheet has no series', '/api/history', [200, {}], /no series/],
])('%s: exits 1 and prints nothing', async (_, url, answer, why) => {
    routes[url] = answer;
    const { code, stdout, stderr } = await run();
    expect(code).toBe(1);
    expect(stdout).toBe('');
    expect(stderr).toMatch(why);
});
