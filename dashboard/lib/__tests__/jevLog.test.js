/**
 * Tests for jevLog — Upstash KV wrapper.
 *
 * Tests focus on the contract: return types, error handling, KV-unavailable paths.
 * Mock-based tests verify the real Upstash REST protocol:
 *   - SET sends raw JSON.stringify(value) as body, no wrapper
 *   - NX guard is path-form: POST /set/<key>/<val>/EX/<ttl>/NX, no body
 *   - LPUSH is path-form: POST /lpush/<key>/<val>, no body
 *   - logVerdicts claims NX first; if already claimed -> no further writes
 *   - readDay/yesterday parse a JSON STRING result from GET
 */

import {
    readDay,
    yesterday,
    logVerdicts,
    listDays,
    isConfigured,
} from '../jevLog';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Saves original env values and restores after each test. Works with async fn. */
async function withEnv(envVars, fn) {
    const allKeys = ['KV_REST_API_URL', 'KV_REST_API_TOKEN', 'TYPESAFE_API_KEY'];
    const originals = {};
    for (const k of allKeys) {
        originals[k] = process.env[k];
        if (k in envVars) {
            process.env[k] = envVars[k];
        } else {
            delete process.env[k];
        }
    }
    try {
        return await fn();
    } finally {
        for (const [k, v] of Object.entries(originals)) {
            if (v === undefined) delete process.env[k];
            else process.env[k] = v;
        }
    }
}

const KV_BASE = 'https://test.upstash.io';
const KV_TOKEN = 'test-token';
const ENV_OK = { KV_REST_API_URL: KV_BASE, KV_REST_API_TOKEN: KV_TOKEN };

/** Create a mock fetch that returns the given JSON response. */
function mockFetchOk(responseJson) {
    return jest.fn(() =>
        Promise.resolve({
            ok: true,
            json: async () => responseJson,
        }),
    );
}

/** Create a mock fetch that returns different responses in sequence. */
function mockFetchSequence(responses) {
    let idx = 0;
    return jest.fn(() => {
        const resp = responses[idx];
        idx++;
        return Promise.resolve({
            ok: true,
            json: async () => resp,
        });
    });
}

// ---------------------------------------------------------------------------
// isConfigured
// ---------------------------------------------------------------------------

describe('isConfigured', () => {
    test('returns false when KV env vars are missing', async () => {
        await withEnv({}, () => {
            expect(isConfigured()).toBe(false);
        });
    });

    test('returns false when only URL is set', async () => {
        await withEnv({ KV_REST_API_URL: 'https://example.com' }, () => {
            expect(isConfigured()).toBe(false);
        });
    });

    test('returns false when only TOKEN is set', async () => {
        await withEnv({ KV_REST_API_TOKEN: 'abc123' }, () => {
            expect(isConfigured()).toBe(false);
        });
    });

    test('returns true when both URL and TOKEN are set', async () => {
        await withEnv(ENV_OK, () => {
            expect(isConfigured()).toBe(true);
        });
    });
});

// ---------------------------------------------------------------------------
// logVerdicts — basic failure paths
// ---------------------------------------------------------------------------

describe('logVerdicts basics', () => {
    test('returns false when KV is not configured', async () => {
        const result = await logVerdicts('2026-09-20', { pills: {} });
        expect(result).toBe(false);
    });
});

// ---------------------------------------------------------------------------
// readDay — basic failure paths
// ---------------------------------------------------------------------------

describe('readDay basics', () => {
    test('returns null when KV is not configured', async () => {
        const result = await readDay('2026-09-20');
        expect(result).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// yesterday — basic failure paths
// ---------------------------------------------------------------------------

describe('yesterday basics', () => {
    test('returns null when KV is not configured', async () => {
        const result = await yesterday();
        expect(result).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// listDays — basic failure paths
// ---------------------------------------------------------------------------

describe('listDays basics', () => {
    test('returns null when KV is not configured', async () => {
        const result = await listDays();
        expect(result).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// Mock-fetch tests — Upstash REST protocol
// ---------------------------------------------------------------------------

describe('logVerdicts — Upstash protocol', () => {
    let originalFetch;

    beforeEach(() => {
        originalFetch = global.fetch;
    });

    afterEach(() => {
        global.fetch = originalFetch;
    });

    test('NX guard uses path-form URL ending with /1/EX/3456000/NX', async () => {
        const fetch = mockFetchOk({ result: 'OK' });
        global.fetch = fetch;

        await withEnv(ENV_OK, () =>
            logVerdicts('2026-09-20', { pills: { regime: { verdict: 'risk-on' } } }),
        );

        // First call is the NX guard
        expect(fetch).toHaveBeenCalledTimes(3);
        const nxUrl = fetch.mock.calls[0][0];
        expect(nxUrl).toContain('/1/EX/3456000/NX');
        // NX guard has no body
        const nxOpts = fetch.mock.calls[0][1];
        expect(nxOpts.body).toBeUndefined();
    });

    test('SET call body equals JSON.stringify(payload) with no wrapper', async () => {
        const fetch = mockFetchSequence([
            { result: 'OK' },     // NX guard (path-form, no body)
            { result: 'OK' },     // SET
            { result: 1 },         // LPUSH (path-form, no body)
        ]);
        global.fetch = fetch;

        const payload = { pills: { regime: { verdict: 'risk-on' } } };
        await withEnv(ENV_OK, () =>
            logVerdicts('2026-09-20', payload),
        );

        // Second call is the SET
        const setOpts = fetch.mock.calls[1][1];
        expect(JSON.parse(setOpts.body)).toEqual(payload);
        expect(setOpts.body).toBe(JSON.stringify(payload));
    });

    test('LPUSH URL ends with /lpush/ftb%3Ajev%3Adays/2026-09-20', async () => {
        const fetch = mockFetchSequence([
            { result: 'OK' },     // NX guard
            { result: 'OK' },     // SET
            { result: 1 },         // LPUSH
        ]);
        global.fetch = fetch;

        await withEnv(ENV_OK, () =>
            logVerdicts('2026-09-20', { pills: {} }),
        );

        // Third call is LPUSH
        const lpushUrl = fetch.mock.calls[2][0];
        expect(lpushUrl).toContain('/lpush/ftb%3Ajev%3Adays/2026-09-20');
        // LPUSH has no body
        const lpushOpts = fetch.mock.calls[2][1];
        expect(lpushOpts.body).toBeUndefined();
    });

    test('when NX guard responds { result: null }, no SET and no LPUSH, returns true', async () => {
        const fetch = mockFetchOk({ result: null }); // NX: key already exists
        global.fetch = fetch;

        const result = await withEnv(ENV_OK, () =>
            logVerdicts('2026-09-20', { pills: {} }),
        );

        expect(result).toBe(true);
        // Only one call - the NX guard
        expect(fetch).toHaveBeenCalledTimes(1);
    });

    test('returns false when NX guard fetch fails (500)', async () => {
        global.fetch = jest.fn(() =>
            Promise.resolve({
                ok: false,
                status: 500,
                json: async () => ({ error: 'error' }),
            }),
        );

        const result = await withEnv(ENV_OK, () =>
            logVerdicts('2026-09-20', { pills: {} }),
        );
        expect(result).toBe(false);
    });

    test('returns false when NX guard fetch throws', async () => {
        global.fetch = jest.fn(() => Promise.reject(new Error('Network error')));

        const result = await withEnv(ENV_OK, () =>
            logVerdicts('2026-09-20', { pills: {} }),
        );
        expect(result).toBe(false);
    });

    test('returns false when SET fails after NX guard claims', async () => {
        const fetch = mockFetchSequence([
            { result: 'OK' },            // NX guard claimed
            { result: null, error: 'error' },  // SET fails
        ]);
        global.fetch = fetch;

        const result = await withEnv(ENV_OK, () =>
            logVerdicts('2026-09-20', { pills: {} }),
        );
        expect(result).toBe(false);
    });

    test('returns true on successful full log (NX + SET + LPUSH)', async () => {
        const fetch = mockFetchSequence([
            { result: 'OK' },     // NX guard claimed
            { result: 'OK' },     // SET succeeds
            { result: 1 },         // LPUSH succeeds
        ]);
        global.fetch = fetch;

        const result = await withEnv(ENV_OK, () =>
            logVerdicts('2026-09-20', { pills: { regime: { verdict: 'risk-on' } } }),
        );
        expect(result).toBe(true);
        expect(fetch).toHaveBeenCalledTimes(3);
    });
});

// ---------------------------------------------------------------------------
// readDay with mock fetch - parses JSON STRING result
// ---------------------------------------------------------------------------

describe('readDay with mock fetch', () => {
    let originalFetch;

    beforeEach(() => {
        originalFetch = global.fetch;
    });

    afterEach(() => {
        global.fetch = originalFetch;
    });

    test('parses a JSON string result from /get', async () => {
        const expected = { pills: { regime: { verdict: 'risk-on' } } };
        global.fetch = mockFetchOk({ result: JSON.stringify(expected), error: null });

        const result = await withEnv(ENV_OK, () => readDay('2026-09-20'));
        expect(result).toEqual(expected);
    });

    test('returns null on empty result', async () => {
        global.fetch = mockFetchOk({ result: null, error: null });

        const result = await withEnv(ENV_OK, () => readDay('2026-09-20'));
        expect(result).toBeNull();
    });

    test('returns null on invalid JSON string', async () => {
        global.fetch = mockFetchOk({ result: '{not-json}', error: null });

        const result = await withEnv(ENV_OK, () => readDay('2026-09-20'));
        expect(result).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// yesterday with mock fetch
// ---------------------------------------------------------------------------

describe('yesterday with mock fetch', () => {
    let originalFetch;

    beforeEach(() => {
        originalFetch = global.fetch;
    });

    afterEach(() => {
        global.fetch = originalFetch;
    });

    test('returns null when no days are logged', async () => {
        global.fetch = mockFetchOk({ result: [], error: null });

        const result = await withEnv(ENV_OK, () => yesterday());
        expect(result).toBeNull();
    });

    test('returns a logged day when days exist (GET returns JSON string)', async () => {
        global.fetch = mockFetchSequence([
            { result: ['2026-09-18'], error: null },   // LRANGE
            { result: JSON.stringify({ pills: { regime: { verdict: 'risk-on', p: 0.8, by: 'jev' } } }), error: null },  // GET
        ]);

        const result = await withEnv(ENV_OK, () => yesterday());

        expect(result).not.toBeNull();
        expect(result.date).toBe('2026-09-18');
        expect(result.pills.regime.verdict).toBe('risk-on');
    });
});