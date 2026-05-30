import { yahooChart, finnhubQuote, polygonDaily, dailyChange } from '../../../lib/sources';

export const fetchCache = 'default-cache';

const fmtPct = (p) => `${p >= 0 ? '+' : ''}${p.toFixed(2)}%`;

async function lambdaMove(messages) {
    const lambdaUrl = process.env.LAMBDA_URL;
    if (!lambdaUrl) { messages.push('LAMBDA_URL not configured'); return null; }
    try {
        const res = await fetch(`${lambdaUrl}/api/spy-daily-move`, { cache: 'no-store' });
        if (!res.ok) { messages.push(`Lambda HTTP ${res.status}`); return null; }
        const j = await res.json();
        if (j && j.value != null && !j.error) return j;
        messages.push(`Lambda returned no usable value (${j?.error || 'null'})`);
    } catch (e) { messages.push(`Lambda failed: ${e.message}`); }
    return null;
}

async function fallbackMove(messages) {
    const finnhub = process.env.FINNHUB_KEY || '';
    const poly = process.env.POLYGON_KEY || '';
    if (finnhub) {
        try { const q = await finnhubQuote('SPY', finnhub); return { value: fmtPct(dailyChange(q.current, q.prevClose).pct), source: 'Finnhub (fallback)' }; }
        catch (e) { messages.push(`Finnhub failed: ${e.message}`); }
    }
    if (poly) {
        try { const p = await polygonDaily('SPY', poly, { years: 1, revalidate: 600 }); return { value: fmtPct(dailyChange(p.current, p.prevClose).pct), source: 'Polygon (fallback)' }; }
        catch (e) { messages.push(`Polygon failed: ${e.message}`); }
    }
    const y = await yahooChart('SPY', { range: '5d', interval: '1d', revalidate: 300 });
    return { value: fmtPct(dailyChange(y.current, y.prevClose).pct), source: 'Yahoo Finance (fallback)' };
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

    const lam = await lambdaMove(messages);
    if (lam) return Response.json(lam);
    try {
        const fb = await fallbackMove(messages);
        return Response.json({ ...fb, messages });
    } catch (e) {
        return Response.json({ value: null, source: 'Failed', error: [...messages, e.message].join(' | ') });
    }
}
