import { yahooChart, polygonDaily, finnhubQuote, dailyChange } from '../../../lib/sources';
import { calculateRSI } from '../../../lib/finance';
import { serve } from '../../../lib/store';
import { faultsFrom } from '../../../lib/faults';

// default-cache lets the fallback source fetches use the Data Cache even though
// the handler is dynamic (Lambda call stays no-store). See fred/route.js note.
export const fetchCache = 'default-cache';

const r2 = (x) => (x == null ? null : Math.round(x * 100) / 100);
const sma = (arr, i, p) => (i >= p - 1 ? arr.slice(i - p + 1, i + 1).reduce((s, v) => s + v, 0) / p : null);

/** Build the exact /api/spy shape from an oldest->newest [{date,price}] history.
 *  Requires enough history (>=220 trading days) so MA200/52w/RSI are real —
 *  otherwise throws, so the card never receives a broken partial object. */
function buildSpy(history, current, prevClose, source) {
    const prices = history.map((h) => h.price);
    const n = prices.length;
    if (n < 220) throw new Error(`${source}: insufficient history (${n} rows)`);
    const ma200v = sma(prices, n - 1, 200);
    const dc = dailyChange(current, prevClose);
    const last252 = prices.slice(-252);
    const wkHigh = Math.max(...last252);
    const px3y = prices[Math.max(0, n - 756)];
    const return3y = px3y ? ((current - px3y) / px3y) * 100 : null;
    const rsi = calculateRSI(history, 9);

    const chartHistory = [];
    for (let i = Math.max(0, n - 302); i < n; i++) {
        chartHistory.push({ date: history[i].date, price: r2(prices[i]), ma50: r2(sma(prices, i, 50)), ma200: r2(sma(prices, i, 200)) });
    }

    return {
        current: r2(current),
        dailyChange: { value: dc.value, pct: dc.pct },
        ma200: { value: r2(ma200v), pct: ma200v ? ((current - ma200v) / ma200v) * 100 : 0 },
        week52High: { value: r2(wkHigh), pct: wkHigh ? ((current - wkHigh) / wkHigh) * 100 : 0 },
        rsi,
        return3y,
        chartHistory,
        _meta: { source, hasErrors: true, messages: [`Served from ${source}`] },
    };
}

async function fallbackSpy(messages, faults = new Set()) {
    const poly = (process.env.POLYGON_KEY || '') && !faults.has('polygon') ? process.env.POLYGON_KEY : '';
    const finnhub = (process.env.FINNHUB_KEY || '') && !faults.has('finnhub') ? process.env.FINNHUB_KEY : '';
    // 1) Polygon (server-friendly, 5y daily) — the reliable path.
    if (poly) {
        try {
            const p = await polygonDaily('SPY', poly, { years: 5, revalidate: 1800 });
            // Prefer a real-time current price from Finnhub if available.
            let current = p.current, prevClose = p.prevClose;
            if (finnhub) { try { const q = await finnhubQuote('SPY', finnhub); current = q.current; prevClose = q.prevClose; } catch {} }
            return buildSpy(p.history, current, prevClose, finnhub ? 'Polygon + Finnhub (fallback)' : 'Polygon (fallback)');
        } catch (e) { messages.push(`Polygon fallback failed: ${e.message}`); }
    } else {
        messages.push(faults.has('polygon') ? 'Polygon disabled (injected)' : 'POLYGON_KEY not configured');
    }
    // 2) Yahoo 5y — best effort (often rate-limited from cloud IPs).
    if (faults.has('yahoo')) throw new Error('[injected fault: yahoo]');
    const y = await yahooChart('SPY', { range: '5y', interval: '1d', revalidate: 300 });
    return buildSpy(y.history, y.current, y.prevClose, 'Yahoo Finance (fallback)');
}

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

export async function GET(request) {
    request.headers.get('user-agent'); // keep handler dynamic
    const debug = new URL(request.url).searchParams.get('debug');
    const messages = [];

    if (debug === 'compare') {
        const [lam, fb] = [await lambdaSpy(messages), await fallbackSpy(messages).catch((e) => ({ _err: e.message }))];
        const pick = (o) => o && !o._err ? { current: o.current, ma200: o.ma200?.value, week52High: o.week52High?.value, rsi: o.rsi, return3y: o.return3y, dailyChangePct: o.dailyChange?.pct, source: o._meta?.source } : o;
        return Response.json({ lambda: pick(lam), fallback: pick(fb), messages });
    }

    // Never-throws: Lambda -> Polygon(+Finnhub) -> Yahoo -> last-known-good -> error skeleton.
    const faults = faultsFrom(request);
    return serve('spy', async () => {
        const lam = faults.has('lambda') ? null : await lambdaSpy(messages);
        if (lam) return lam;
        const fb = await fallbackSpy(messages, faults);
        fb._meta.messages = [...messages, ...fb._meta.messages];
        return fb;
    }, {
        isGood: (x) => x && x.current != null && x.ma200 && x.week52High,
        fallback: { error: 'SPY temporarily unavailable' },
        faults,
    });
}
