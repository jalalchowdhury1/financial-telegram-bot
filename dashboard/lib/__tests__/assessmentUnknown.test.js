/**
 * @jest-environment node
 *
 * /api/assessment (audit B14): a missing input reaches the model (and the rule-based
 * answer) as "unknown", never as a confident label ("Positive (no inversion)", "Safe",
 * "Extreme Greed") or the string "undefined".
 */
jest.mock('../claudeAssessment', () => ({
    CLAUDE_KEY_ENV: 'TEST_CLAUDE_KEY', CLAUDE_LABEL: 'claude',
    claudeBudgetOk: jest.fn(async () => true),
    claudeAssessment: jest.fn(async () => 'ok'),
    assessmentAuthorized: jest.fn(() => true),
    fallbackBudgetOk: jest.fn(async () => false),
}));
const { claudeAssessment } = require('../claudeAssessment');
const { POST } = require('../../app/api/assessment/route');

const post = (data) => POST(new Request('https://x.test/api/assessment', { method: 'POST', body: JSON.stringify(data) }));
const KEYS = ['GROQ_API_KEY', 'OPENAI_API_KEY', 'TEST_CLAUDE_KEY'];
afterEach(() => { for (const k of KEYS) delete process.env[k]; });

test('null inputs are "unknown" in the prompt', async () => {
    process.env.TEST_CLAUDE_KEY = 'k';
    await post({ yieldCurve: null, sahmRule: null, fearGreed: null, claims: 230 });
    const prompt = claudeAssessment.mock.calls[0][0];
    expect(prompt).toMatch(/Yield Curve \(10Y-2Y\): unknown/);
    expect(prompt).toMatch(/Sahm Rule Recession Indicator: unknown/);
    expect(prompt).toMatch(/Fear & Greed Index: unknown/);
    expect(prompt).not.toMatch(/Positive \(no inversion\)|→ Safe|Extreme Greed|undefined/);
    expect(prompt).toMatch(/Claims \(4wk avg\): 230K/);
});

test('rule-based answer with null inputs says unknown, not "inverted" / "safe"', async () => {
    const b = await (await post({ yieldCurve: null, sahmRule: null, fearGreed: null })).json();
    expect(b.assessment).not.toMatch(/inverted|safe|positive|fear detected/i);
    expect(b.assessment).toMatch(/unknown/i);
});
