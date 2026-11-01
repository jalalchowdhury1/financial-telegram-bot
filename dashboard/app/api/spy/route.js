import { fallbackSpy, spyPreferNewer, latestSessionDate } from '../../../lib/spyTiers';
import { serve } from '../../../lib/store';
import { faultsFrom } from '../../../lib/faults';
import { assessmentAuthorized } from '../../../lib/claudeAssessment';

// default-cache lets the fallback source fetches use the Data Cache even though
// the handler is dynamic (Lambda call stays no-store). See fred/route.js note.
export const fetchCache = 'default-cache';

// A hung Lambda must not hold the fallback tiers back until API Gateway's ~29 s cap.
const LAMBDA_TIMEOUT_MS = 10000;

async function lambdaSpy(messages) {
    const lambdaUrl = process.env.LAMBDA_URL;
    if (!lambdaUrl) { messages.push('LAMBDA_URL not configured'); return null; }
    try {
        const res = await fetch(`${lambdaUrl}/api/spy`, { cache: 'no-store', signal: AbortSignal.timeout(LAMBDA_TIMEOUT_MS) });
        if (!res.ok) { messages.push(`Lambda HTTP ${res.status}`); return null; }
        const j = await res.json();
        if (j && j.current != null && !j.error) return j;
        messages.push('Lambda returned no usable SPY data');
    } catch (e) { messages.push(`Lambda failed: ${e.message}`); }
    return null;
}

const isGood = (x) => x && x.current != null && x.ma200 && x.week52High;
const near = (a, b, tol) => Number.isFinite(a / b) && Math.abs(a / b - 1) < tol;

/**
 * Why a Lambda answer can't go first (null = it can). Its FRED layer is S&P 500 index
 * points, not SPY dollars, and a SPY spot on top of them read "−89% vs 200-day" (audit
 * 1 Nov 2026): neither is ever served. Its Google Sheet layer and an older session wait
 * behind our bar-computed tiers; an older session is flagged stale at its newest bar.
 */
export function lambdaDoubt(lam, now = Date.now()) {
    const src = lam._meta?.source || '';
    const sheet = /^Google Sheet/.test(src);
    const last = lam.chartHistory?.at(-1);
    if (/FRED/.test(src)) return { never: true, why: 'Lambda answered in S&P 500 index points (FRED), not SPY dollars' };
    if (!near(lam.current, lam.ma200?.value, 0.5) || (!sheet && !near(lam.current, last?.price, 0.15))) {
        return { never: true, why: 'Lambda price, 200-day and chart are on different scales' };
    }
    if (sheet) return { why: 'Lambda answered from its Google Sheet layer' };
    const expected = latestSessionDate(now);
    if (last.date < expected) return { why: `Lambda newest bar ${last.date} is not the latest session ${expected}`, asOf: last.date };
    return null;
}

export async function GET(request) {
    request.headers.get('user-agent'); // keep handler dynamic
    const debug = new URL(request.url).searchParams.get('debug');
    const messages = [];

    // Two Polygon calls per hit (5/min free quota): secret-gated like market-extra's.
    if (debug === 'compare' && assessmentAuthorized(request)) {
        const [lam, fb] = [await lambdaSpy(messages), await fallbackSpy(messages).catch((e) => ({ _err: e.message }))];
        const pick = (o) => o && !o._err ? { current: o.current, ma200: o.ma200?.value, week52High: o.week52High?.value, rsi: o.rsi, return3y: o.return3y, dailyChangePct: o.dailyChange?.pct, source: o._meta?.source, stale: o._meta?.stale || undefined } : o;
        return Response.json({ lambda: pick(lam), fallback: pick(fb), messages });
    }

    // Never-throws: Lambda -> Polygon(+Finnhub/CNBC spot) -> Nasdaq(+spot) -> Yahoo
    // -> flagged older close -> last-known-good -> error skeleton (lib/spyTiers.js).
    const faults = faultsFrom(request);
    return serve('spy', async () => {
        const lam = faults.has('lambda') ? null : await lambdaSpy(messages);
        const doubt = lam && lambdaDoubt(lam);
        if (lam && !doubt) return lam;
        // A doubted answer is kept only if every bar-computed tier fails or holds an older
        // session (the sheet's own formulas count as current); never one on mixed scales.
        let spare = null;
        if (doubt) {
            messages.push(`${doubt.why}; trying bar-computed tiers first`);
            if (!doubt.never) spare = doubt.asOf ? { ...lam, _meta: { ...lam._meta, stale: true, hasErrors: true, asOf: doubt.asOf } } : lam;
        }
        try {
            const fb = await fallbackSpy(messages, faults);
            fb._meta.messages = [...messages, ...fb._meta.messages];
            if (!spare || !fb._meta?.stale || (spare._meta?.stale && !(spare._meta.asOf > fb._meta.asOf))) return fb;
        } catch (e) {
            if (!spare) throw e;
            messages.push(`bar-computed tiers failed: ${e.message}`);
        }
        return { ...spare, _meta: { ...spare._meta, messages: [...messages, ...(spare._meta?.messages || [])] } };
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
