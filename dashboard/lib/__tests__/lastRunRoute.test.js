/**
 * @jest-environment node
 *
 * /api/last-run (audit B13): every GitHub call has a timeout, the route declares
 * maxDuration, and a fallback layer (a failed run, another workflow, a commit time) is
 * never reported as the report's clean last run.
 */
jest.mock('fs', () => ({ ...jest.requireActual('fs'), writeFileSync: jest.fn(), existsSync: jest.fn(() => false) }));   // no /tmp writes from a test
const route = require('../../app/api/last-run/route');

const json = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
afterEach(() => { delete global.fetch; });

test('declares maxDuration and passes a timeout signal to every GitHub fetch', async () => {
    expect(route.maxDuration).toBeGreaterThan(0);
    global.fetch = jest.fn(async () => json({ workflow_runs: [{ updated_at: '2026-10-31T12:00:00Z' }] }));
    const b = await (await route.GET()).json();
    expect(b._meta.hasErrors).toBeFalsy();
    expect(global.fetch.mock.calls[0][1].signal).toBeInstanceOf(AbortSignal);
});

test('layer 2 (any status: maybe a FAILED run) is flagged hasErrors', async () => {
    global.fetch = jest.fn(async (url) => (url.includes('status=success') ? json({}, 403) : json({ workflow_runs: [{ updated_at: '2026-10-31T12:00:00Z', conclusion: 'failure' }] })));
    const b = await (await route.GET()).json();
    expect(b._meta.source).toBe('GitHub Actions (any status)');
    expect(b._meta.hasErrors).toBe(true);
});

test('layer 4 (a commit time as proxy) is flagged hasErrors', async () => {
    global.fetch = jest.fn(async (url) => (url.includes('/commits') ? json([{ commit: { committer: { date: '2026-10-30T09:00:00Z' } } }]) : json({}, 500)));
    const b = await (await route.GET()).json();
    expect(b._meta.source).toBe('GitHub Commits (proxy)');
    expect(b._meta.hasErrors).toBe(true);
});
