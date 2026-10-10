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

/** Count this call against today's (ET) cap. true = under the cap. Never throws. */
export async function claudeBudgetOk() {
    try {
        const day = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
        const key = `ftb:assessment:claude:${day}`;
        const d = await kvCall(`/incr/${encodeURIComponent(key)}`, { method: 'POST' });
        const n = Number(d?.result);
        if (!Number.isFinite(n)) return false;          // KV unreadable → fail closed
        if (n === 1) await kvCall(`/expire/${encodeURIComponent(key)}/172800`, { method: 'POST' });
        return n <= CLAUDE_DAILY_CAP;
    } catch {
        return false;
    }
}

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
