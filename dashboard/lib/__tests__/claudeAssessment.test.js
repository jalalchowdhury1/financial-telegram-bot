/**
 * @jest-environment node
 *
 * /api/assessment: Claude first (capped per day in KV), the old free-model
 * cascade as the fallback, and no key = the old path. The real key is never
 * read here: the SDK and KV are mocked and the env var is a fake.
 */
import fs from 'fs';
import path from 'path';

const create = jest.fn();
const ctorArgs = [];
jest.mock('@anthropic-ai/sdk', () => ({
    __esModule: true,
    default: jest.fn().mockImplementation((opts) => {
        ctorArgs.push(opts);
        return { messages: { create } };
    }),
}));

let kvCount = 1;   // what INCR answers; null = KV unavailable
jest.mock('../kv', () => ({
    kvCall: jest.fn(async (p) => (p.startsWith('/incr/') ? (kvCount === null ? null : { result: kvCount }) : { result: 1 })),
}));

const { POST } = require('../../app/api/assessment/route');
const lib = require('../claudeAssessment');

const DATA = { yieldCurve: 0.45, fearGreed: 31, sahmRule: 0.2, claims: 230, sentiment: 58, copperGold: 1.4, creditSpread: 1.1, realYields: 1.9 };
const post = () => POST(new Request('https://x.test/api/assessment', { method: 'POST', body: JSON.stringify(DATA) }));
const msg = (text, stop_reason = 'end_turn') => ({ stop_reason, content: [{ type: 'thinking', thinking: '' }, { type: 'text', text }] });

const KEYS = ['CLAUDE_CREDITS_API_KEY', 'OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'GROQ_API_KEY', 'MOONSHOT_API_KEY'];
const saved = {};
const realFetch = global.fetch;

beforeEach(() => {
    KEYS.forEach((k) => { saved[k] = process.env[k]; delete process.env[k]; });
    create.mockReset();
    ctorArgs.length = 0;
    kvCount = 1;
    global.fetch = jest.fn(async () => new Response(JSON.stringify({ choices: [{ message: { content: 'old cascade text' } }] }), { status: 200 }));
    jest.spyOn(console, 'log').mockImplementation(() => {});
    jest.spyOn(console, 'warn').mockImplementation(() => {});
});
afterEach(() => {
    KEYS.forEach((k) => { if (saved[k] === undefined) delete process.env[k]; else process.env[k] = saved[k]; });
    global.fetch = realFetch;
    jest.restoreAllMocks();
});

const withClaude = () => { process.env.CLAUDE_CREDITS_API_KEY = 'sk-ant-test-not-real'; process.env.GROQ_API_KEY = 'gsk-test'; process.env.OPENROUTER_API_KEY = 'sk-or-test'; };

test('Claude answers first and the old cascade is never called', async () => {
    withClaude();
    create.mockResolvedValue(msg('Macro is fine. 🎯 CAUTIOUS'));
    const res = await post();
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.assessment).toContain('Macro is fine.');
    expect(body.assessment).toContain('Provider: Claude Haiku 5.5 (claude-haiku-5-5)');
    expect(global.fetch).not.toHaveBeenCalled();
});

test('Claude request shape: exact model id, low effort, no disabled thinking / sampling / prefill, tight timeout', async () => {
    withClaude();
    create.mockResolvedValue(msg('ok'));
    await post();
    const sent = create.mock.calls[0][0];
    expect(sent.model).toBe('claude-haiku-5-5');
    expect(sent.output_config).toEqual({ effort: 'low' });
    expect(sent).not.toHaveProperty('thinking');
    expect(sent).not.toHaveProperty('temperature');
    expect(sent.messages[sent.messages.length - 1].role).toBe('user');
    expect(sent.messages[0].content).toContain('Yield Curve (10Y-2Y): 0.45%');
    expect(ctorArgs[0]).toEqual({ apiKey: 'sk-ant-test-not-real', timeout: lib.CLAUDE_TIMEOUT_MS, maxRetries: 0 });
});

test.each([
    ['API error', () => create.mockRejectedValue(new Error('529 overloaded'))],
    ['refusal', () => create.mockResolvedValue(msg('', 'refusal'))],
    ['empty text', () => create.mockResolvedValue(msg('   '))],
])('any Claude failure (%s) falls back to the old cascade', async (_name, arrange) => {
    withClaude();
    arrange();
    const body = await (await post()).json();
    expect(body.assessment).toContain('old cascade text');
    expect(body.assessment).toContain('Provider: OpenRouter DeepSeek V4 Flash');
    expect(global.fetch).toHaveBeenCalled();
});

test('over the daily cap: Claude is skipped, the old cascade serves', async () => {
    withClaude();
    kvCount = lib.CLAUDE_DAILY_CAP + 1;
    const body = await (await post()).json();
    expect(create).not.toHaveBeenCalled();
    expect(body.assessment).toContain('Provider: OpenRouter DeepSeek V4 Flash');
});

test('KV unavailable: fails closed (no Claude), the old cascade serves', async () => {
    withClaude();
    kvCount = null;
    const body = await (await post()).json();
    expect(create).not.toHaveBeenCalled();
    expect(body.assessment).toContain('old cascade text');
});

test('no Claude key = the old path exactly (SDK never built)', async () => {
    process.env.GROQ_API_KEY = 'gsk-test';
    process.env.OPENROUTER_API_KEY = 'sk-or-test';
    const body = await (await post()).json();
    expect(ctorArgs).toHaveLength(0);
    expect(body.assessment).toContain('Provider: OpenRouter DeepSeek V4 Flash');
});

test('no keys at all = rule-based, as before', async () => {
    const body = await (await post()).json();
    expect(body.assessment).toContain('Rule-based assessment');
    expect(ctorArgs).toHaveLength(0);
    expect(global.fetch).not.toHaveBeenCalled();
});

test('the key is never named ANTHROPIC_API_KEY (flips Claude Code billing on the Mac)', () => {
    const route = fs.readFileSync(path.join(__dirname, '../../app/api/assessment/route.js'), 'utf8');
    const helper = fs.readFileSync(path.join(__dirname, '../claudeAssessment.js'), 'utf8');
    expect(route + helper).not.toMatch(/ANTHROPIC_API_KEY/);
    expect(lib.CLAUDE_KEY_ENV).toBe('CLAUDE_CREDITS_API_KEY');
});
