import { fallbackMove } from '../../../lib/spyTiers';
import { serve } from '../../../lib/store';
import { faultsFrom } from '../../../lib/faults';

export const fetchCache = 'default-cache';

// A hung Lambda must not hold the fallback tiers back until API Gateway's ~29 s cap.
const LAMBDA_TIMEOUT_MS = 10000;
// The Lambda reads Sheet B12 raw: '#N/A', 'Loading...' or a blank cell is not a move.
const isMove = (v) => typeof v === 'string' && /^[-+]?\d/.test(v.trim()) && Number.isFinite(parseFloat(v));

async function lambdaMove(messages) {
    const lambdaUrl = process.env.LAMBDA_URL;
    if (!lambdaUrl) { messages.push('LAMBDA_URL not configured'); return null; }
    try {
        const res = await fetch(`${lambdaUrl}/api/spy-daily-move`, { cache: 'no-store', signal: AbortSignal.timeout(LAMBDA_TIMEOUT_MS) });
        if (!res.ok) { messages.push(`Lambda HTTP ${res.status}`); return null; }
        const j = await res.json();
        if (j && isMove(j.value) && !j.error) return j;
        messages.push(`Lambda returned no usable value (${j?.error || 'null'})`);
    } catch (e) { messages.push(`Lambda failed: ${e.message}`); }
    return null;
}

export async function GET(request) {
    request.headers.get('user-agent');
    const debug = new URL(request.url).searchParams.get('debug');
    const messages = [];

    if (debug === 'compare') {
        const lam = await lambdaMove(messages);
        const fb = await fallbackMove(messages).catch((e) => ({ value: null, _err: e.message }));
        return Response.json({ lambda: lam, fallback: fb, messages });
    }

    // Never-throws: Lambda -> Finnhub -> CNBC -> Polygon (latest session only) -> Yahoo
    // -> last-known-good -> null. Tiers + fault names live in lib/spyTiers.js.
    const faults = faultsFrom(request);
    return serve('spy-daily-move', async () => {
        const lam = faults.has('lambda') ? null : await lambdaMove(messages);
        if (lam) return lam;
        const fb = await fallbackMove(messages, faults);
        return { ...fb, _meta: { messages } };
    }, {
        isGood: (x) => x && isMove(x.value),
        fallback: { value: null, source: 'Unavailable' },
        faults,
    });
}
