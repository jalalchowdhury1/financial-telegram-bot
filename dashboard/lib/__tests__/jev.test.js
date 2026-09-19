import { judgeMany, defaultPoster, JEV_TIMEOUT_MS, JEV_P_FLOOR } from '../jev';

// ---------------------------------------------------------------------------
// A light-weight fake poster for deterministic testing
// ---------------------------------------------------------------------------
function fakePoster(result) {
    return async () => result;
}

function fakeReject(errMessage) {
    return async () => { throw new Error(errMessage); };
}

const sampleQuestions = {
    regime: {
        type: 'choice',
        instructions: 'Assess regime',
        criteria: { 'risk-on': 'Risk on', neutral: 'Neutral', 'risk-off': 'Risk off' },
    },
    recession: {
        type: 'choice',
        instructions: 'Assess recession',
        criteria: { low: 'Low', rising: 'Rising', high: 'High' },
    },
};

const sampleState = '1. SPY 500 +2% ...';

describe('judgeMany', () => {
    // Helper: patch env and call
    async function callWithKey(key, post) {
        const orig = process.env.TYPESAFE_API_KEY;
        // Setting to undefined coerces to the string "undefined" in Node.js,
        // which is truthy. Delete the key for the "no key" case instead.
        if (key === undefined) {
            delete process.env.TYPESAFE_API_KEY;
        } else {
            process.env.TYPESAFE_API_KEY = key;
        }
        try {
            return await judgeMany(sampleState, sampleQuestions, post || fakePoster({ answers: {} }));
        } finally {
            if (orig === undefined) delete process.env.TYPESAFE_API_KEY; else process.env.TYPESAFE_API_KEY = orig;
        }
    }

    test('returns null when TYPESAFE_API_KEY is missing (no key) — poster not called', async () => {
        const poster = jest.fn();
        const result = await callWithKey(undefined, poster);
        expect(result).toBeNull();
        expect(poster).not.toHaveBeenCalled();
    });

    test('returns null when TYPESAFE_API_KEY is empty string', async () => {
        const result = await callWithKey('');
        expect(result).toBeNull();
    });

    test('returns parsed verdicts when Jev responds correctly', async () => {
        const post = fakePoster({
            answers: {
                regime: { choice: 'risk-on', confidence: 0.83, probabilities: {} },
                recession: { choice: 'low', confidence: 0.72, probabilities: {} },
            },
            usage: {},
        });
        const result = await callWithKey('sk-abc123', post);
        expect(result).toEqual({
            regime: { verdict: 'risk-on', p: 0.83 },
            recession: { verdict: 'low', p: 0.72 },
        });
    });

    test('drops unknown verdict (not in criteria)', async () => {
        const post = fakePoster({
            answers: {
                regime: { choice: 'super-risk-on', confidence: 0.9, probabilities: {} },
                recession: { choice: 'low', confidence: 0.7, probabilities: {} },
            },
            usage: {},
        });
        const result = await callWithKey('sk-abc123', post);
        // regime not in criteria keys → dropped; recession present
        expect(result).toEqual({
            recession: { verdict: 'low', p: 0.7 },
        });
    });

    test('returns null when all verdicts are unknown (all dropped)', async () => {
        const post = fakePoster({
            answers: {
                regime: { choice: 'invalid1', confidence: 0.9 },
                recession: { choice: 'invalid2', confidence: 0.9 },
            },
        });
        const result = await callWithKey('sk-abc123', post);
        expect(result).toBeNull();
    });

    test('returns null on network error (post throws)', async () => {
        const post = fakeReject('Network failure');
        const result = await callWithKey('sk-abc123', post);
        expect(result).toBeNull();
    });

    test('returns null on missing answers in response', async () => {
        const post = fakePoster({ usage: {} });
        const result = await callWithKey('sk-abc123', post);
        expect(result).toBeNull();
    });

    test('returns null on null/undefined response', async () => {
        for (const bad of [null, undefined]) {
            const post = fakePoster(bad);
            const result = await callWithKey('sk-abc123', post);
            expect(result).toBeNull();
        }
    });

    test('handles confidence=0 correctly', async () => {
        const post = fakePoster({
            answers: {
                regime: { choice: 'risk-on', confidence: 0, probabilities: {} },
            },
            usage: {},
        });
        const result = await callWithKey('sk-abc123', post);
        expect(result).toEqual({
            regime: { verdict: 'risk-on', p: 0 },
        });
    });

    test('handles missing confidence field gracefully', async () => {
        const post = fakePoster({
            answers: {
                regime: { choice: 'risk-on', probabilities: {} },
            },
            usage: {},
        });
        const result = await callWithKey('sk-abc123', post);
        expect(result).toEqual({
            regime: { verdict: 'risk-on', p: null },
        });
    });
});

describe('defaultPoster', () => {
    test('uses AbortController with JEV_TIMEOUT_MS', () => {
        // Just check the constants are what we expect
        expect(JEV_TIMEOUT_MS).toBe(6000);
        expect(JEV_P_FLOOR).toBe(0.6);
    });
});