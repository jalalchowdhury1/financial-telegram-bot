import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseLatestStats, surveyWeek, toPayload, shouldPush } from './parse.mjs';

const LIVE = `Latest Stats
AAII Investor Sentiment Survey: Bearish
2026-10-08
38.98%
46.49%
AAII Investor Sentiment Survey: Neutral
2026-10-08
20.76%
18.86%
AAII Investor Sentiment Survey: Bullish
2026-10-08
40.25%
34.65%
Related Collections`;

test('parses the real 8 Oct 2026 page text', () => {
    assert.deepEqual(parseLatestStats(LIVE), { bull: 40.25, neutral: 20.76, bear: 38.98, released: '2026-10-08' });
});

test('rejects the loading placeholder, partial text and nonsense sums', () => {
    assert.equal(parseLatestStats(LIVE.replaceAll('2026-10-08', '1980-01-01')), null);
    assert.equal(parseLatestStats('Just a moment...'), null);
    assert.equal(parseLatestStats(LIVE.replace('40.25%', '90.25%')), null);
    assert.equal(parseLatestStats(LIVE.replace('Bullish\n2026-10-08', 'Bullish\n2026-10-01')), null);
});

test('release Thursday → survey week Wednesday (matches AAII and the dashboard)', () => {
    assert.equal(surveyWeek('2026-10-08'), '2026-10-07');
    assert.equal(surveyWeek('2026-10-01'), '2026-09-30');
    assert.equal(surveyWeek('2026-10-07'), '2026-10-07');
});

test('payload = AAII one-decimal numbers and bear − bull diff (-1.30%, as the history sheet has)', () => {
    assert.deepEqual(toPayload(parseLatestStats(LIVE)), {
        bull: 40.3, neutral: 20.8, bear: 39, diff: '-1.30%', as_of: '2026-10-07', source: 'macromicro', stale: false,
    });
});

test('only a strictly newer week is pushed', () => {
    const p = { as_of: '2026-10-07' };
    assert.equal(shouldPush(null, p), true);
    assert.equal(shouldPush({ data: { as_of: '2026-09-30' } }, p), true);
    assert.equal(shouldPush({ data: { as_of: '2026-10-07' } }, p), false);
    assert.equal(shouldPush({ data: { as_of: '2026-10-14' } }, p), false);
});
