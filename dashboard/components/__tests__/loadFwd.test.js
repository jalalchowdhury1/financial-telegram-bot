/**
 * loadFwd's layers (components/ValuationLens.js): the baked file gets a second try, and /api/fwd-pe gets 8 s, since
 * the bake alone draws every view. A fresh module per test: loadFwd keeps its first answer.
 */
const fwdFile = { from: '2026-01-02', to: '2026-09-18', t: [0, 259], v: [22, 19], a5: [19.8, 19.8], a10: [19.1, 19.1] };
const ok = (body) => Promise.resolve({ ok: true, status: 200, json: async () => body });
let loadFwd;
beforeEach(() => { jest.resetModules(); ({ loadFwd } = require('../ValuationLens')); });
afterEach(() => jest.useRealTimers());

test('a failed first fetch of the baked file is tried again', async () => {
    let n = 0;
    global.fetch = jest.fn((url) => (url === '/history/fwdPE.json' ? (n++ ? ok(fwdFile) : Promise.reject(new Error('blip'))) : ok({ rows: [] })));
    expect((await loadFwd()).map((r) => r.pe)).toEqual([22, 19]);
});

test('a route still busy at FactSet after 8 s never holds the card: the bake answers', async () => {
    jest.useFakeTimers();
    global.fetch = jest.fn((url, init) => (url === '/history/fwdPE.json' ? ok(fwdFile)
        : new Promise((_, no) => init.signal.addEventListener('abort', () => no(new Error('aborted'))))));
    let rows = null;
    loadFwd().then((r) => { rows = r; });
    await jest.advanceTimersByTimeAsync(7900);
    expect(rows).toBeNull();
    await jest.advanceTimersByTimeAsync(200);
    expect(rows.map((r) => r.date)).toEqual(['2026-01-02', '2026-09-18']);
});

test('the route\'s newer reports join the bake when it answers in time', async () => {
    global.fetch = jest.fn((url) => (url === '/history/fwdPE.json' ? ok(fwdFile) : ok({ rows: [{ date: '2026-09-25', pe: 18.5, a5: 19.8, a10: 19.1 }] })));
    expect((await loadFwd()).map((r) => r.pe)).toEqual([22, 19, 18.5]);
});
