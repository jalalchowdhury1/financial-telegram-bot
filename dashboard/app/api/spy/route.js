import { fallbackSpy, spyPreferNewer } from '../../../lib/spyTiers';
import { serve } from '../../../lib/store';
import { faultsFrom } from '../../../lib/faults';

// default-cache lets the fallback source fetches use the Data Cache even though
// the handler is dynamic (Lambda call stays no-store). See fred/route.js note.
export const fetchCache = 'default-cache';

async function lambdaSpy(messages) {
    const lambdaUrl = process.env.LAMBDA_URL;
    if (!lambdaUrl) { messages.push('LAMBDA_URL not configured'); return null; }
    try {
        const res = await fetch(`${lambdaUrl}/api/spy`, { cache: 'no-store' });
        if (!res.ok) { messages.push(`Lambda HTTP ${res.status}`); return null; }
        const j = await res.json();
        if (j && j.current != null && !j.error) return j;
        messages.push('Lambda returned no usable SPY data');
    } catch (e) { messages.push(`Lambda failed: ${e.message}`); }
    return null;
}

const isGood = (x) => x && x.current != null && x.ma200 && x.week52High;


export async function GET(request) {
    request.headers.get('user-agent'); // keep handler dynamic
    const debug = new URL(request.url).searchParams.get('debug');
    const messages = [];

    if (debug === 'compare') {
        const [lam, fb] = [await lambdaSpy(messages), await fallbackSpy(messages).catch((e) => ({ _err: e.message }))];
        const pick = (o) => o && !o._err ? { current: o.current, ma200: o.ma200?.value, week52High: o.week52High?.value, rsi: o.rsi, return3y: o.return3y, dailyChangePct: o.dailyChange?.pct, source: o._meta?.source, stale: o._meta?.stale || undefined } : o;
        return Response.json({ lambda: pick(lam), fallback: pick(fb), messages });
    }

    // Never-throws: Lambda -> Polygon(+Finnhub/CNBC spot) -> Nasdaq(+spot) -> Yahoo
    // -> flagged older close -> last-known-good -> error skeleton (lib/spyTiers.js).
    const faults = faultsFrom(request);
    return serve('spy', async () => {
        const lam = faults.has('lambda') ? null : await lambdaSpy(messages);
        // A Lambda answer from its Google Sheet layer carries the sheet's own RSI/MA method
        // (RSI 63 vs 57 from bars, 2026-10-09): our bar-computed tiers go first, the sheet
        // answer is kept only if every one of them fails.
        const fromSheet = lam && /^Google Sheet/.test(lam._meta?.source || '');
        if (lam && !fromSheet) return lam;
        if (fromSheet) messages.push('Lambda answered from its Google Sheet layer; trying bar-computed tiers first');
        try {
            const fb = await fallbackSpy(messages, faults);
            fb._meta.messages = [...messages, ...fb._meta.messages];
            if (!fromSheet || !fb._meta?.stale) return fb;
        } catch (e) {
            if (!fromSheet) throw e;
            messages.push(`bar-computed tiers failed: ${e.message}`);
        }
        return lam;
    }, {
        // A flagged-stale build (no tier had the latest session) is NOT "good": serve()
        // then prefers the last-known-good when it is newer and only falls through to the
        // flagged build when there is none. Never saved as LKG either — that would stamp
        // an old close with a fresh savedAt.
        isGood: (x) => isGood(x) && !x._meta?.stale,
        // ...unless that last-known-good is OLDER than the flagged build's close.
        preferNewer: spyPreferNewer,
        fallback: { error: 'SPY temporarily unavailable' },
        faults,
    });
}
