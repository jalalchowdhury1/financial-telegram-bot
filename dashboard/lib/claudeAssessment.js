/**
 * Claude leg of /api/assessment (2026-10-09). See the header of
 * app/api/assessment/route.js. Kept out of the route file because Next only
 * allows its own named exports from a route module.
 */
import Anthropic from '@anthropic-ai/sdk';
import { kvCall } from './kv';

export const CLAUDE_MODEL = 'claude-haiku-5-5';
export const CLAUDE_LABEL = `Claude Haiku 5.5 (${CLAUDE_MODEL})`;
export const CLAUDE_KEY_ENV = 'CLAUDE_CREDITS_API_KEY'; // never the SDK's default key var
export const CLAUDE_TIMEOUT_MS = 12000;
export const CLAUDE_DAILY_CAP = 100;

export const FALLBACK_DAILY_CAP = 50; // the paid OpenRouter/OpenAI/Moonshot cascade (19 Oct 2026)
export const ASSESSMENT_SECRET_ENV = 'ASSESSMENT_SECRET';
export const ASSESSMENT_SECRET_HEADER = 'x-assessment-secret';

/**
 * The route spends money, so a caller must send `x-assessment-secret` equal to env
 * ASSESSMENT_SECRET. Env unset = nobody gets in (fails closed). Never throws.
 */
export function assessmentAuthorized(request) {
    try {
        const want = process.env[ASSESSMENT_SECRET_ENV] || '';
        const got = request.headers.get(ASSESSMENT_SECRET_HEADER) || '';
        if (!want || got.length !== want.length) return false;
        let diff = 0;
        for (let i = 0; i < want.length; i++) diff |= want.charCodeAt(i) ^ got.charCodeAt(i);
        return diff === 0;
    } catch {
        return false;
    }
}

/** Count one call against `prefix`'s per-ET-day cap. true = under the cap. KV unreadable = false. Never throws. */
export async function dailyBudgetOk(prefix, cap) {
    try {
        const day = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
        const key = `${prefix}:${day}`;
        const d = await kvCall(`/incr/${encodeURIComponent(key)}`, { method: 'POST' });
        const n = Number(d?.result);
        if (!Number.isFinite(n)) return false;          // KV unreadable → fail closed
        if (n === 1) await kvCall(`/expire/${encodeURIComponent(key)}/172800`, { method: 'POST' });
        return n <= cap;
    } catch {
        return false;
    }
}

/** Count this call against today's (ET) Claude cap. true = under the cap. Never throws. */
export const claudeBudgetOk = () => dailyBudgetOk('ftb:assessment:claude', CLAUDE_DAILY_CAP);

/** Same for the fallback cascade (one count per request that reaches it). */
export const fallbackBudgetOk = () => dailyBudgetOk('ftb:assessment:fallback', FALLBACK_DAILY_CAP);

/** One Claude call. Returns the assessment text or throws (caller falls back). */
export async function claudeAssessment(prompt) {
    const client = new Anthropic({
        apiKey: process.env[CLAUDE_KEY_ENV],
        timeout: CLAUDE_TIMEOUT_MS,
        maxRetries: 0, // the old cascade IS the retry
    });
    // No thinking:{type:'disabled'} / budget_tokens (400 on Haiku 5.5), no prefill.
    const msg = await client.messages.create({
        model: CLAUDE_MODEL,
        max_tokens: 2048,
        output_config: { effort: 'low' },
        messages: [{ role: 'user', content: prompt }],
    });
    if (msg.stop_reason === 'refusal') throw new Error('claude refused');
    const text = (msg.content || [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text)
        .join('')
        .trim();
    if (!text) throw new Error('claude returned no text');
    return text;
}
