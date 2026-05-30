import { fetchJson } from '../../../lib/fetcher';

export const fetchCache = 'default-cache';

async function lambdaPoly(messages) {
    const lambdaUrl = process.env.LAMBDA_URL;
    if (!lambdaUrl) { messages.push('LAMBDA_URL not configured'); return null; }
    try {
        const res = await fetch(`${lambdaUrl}/api/polymarket`, { cache: 'no-store' });
        if (!res.ok) { messages.push(`Lambda HTTP ${res.status}`); return null; }
        const j = await res.json();
        if (j && Array.isArray(j.bets) && j.bets.length && !j.error) return j;
        messages.push('Lambda returned no usable bets');
    } catch (e) { messages.push(`Lambda failed: ${e.message}`); }
    return null;
}

/** Polymarket public Gamma API -> top-10 active markets by 24h volume. */
async function fallbackPoly(messages) {
    const url = 'https://gamma-api.polymarket.com/markets?active=true&closed=false&order=volume24hr&ascending=false&limit=10';
    const data = await fetchJson(url, { revalidate: 300 });
    const arr = Array.isArray(data) ? data : data?.markets || [];
    const bets = arr.map((m) => {
        let odds = null;
        try { const p = JSON.parse(m.outcomePrices || '[]'); odds = p.length ? parseFloat(p[0]) : null; } catch {}
        if (odds == null && m.lastTradePrice != null) odds = parseFloat(m.lastTradePrice);
        const volume = parseFloat(m.volume24hr ?? m.volumeNum ?? m.volume ?? 0);
        return { name: m.question || m.title || 'Unknown market', odds, volume };
    }).filter((b) => b.name && b.odds != null);
    if (!bets.length) throw new Error('Gamma API returned no usable markets');
    return { bets: bets.slice(0, 10), source: 'Polymarket Gamma API (fallback)' };
}

export async function GET(request) {
    request.headers.get('user-agent');
    const debug = new URL(request.url).searchParams.get('debug');
    const messages = [];

    if (debug === 'compare') {
        const lam = await lambdaPoly(messages);
        const fb = await fallbackPoly(messages).catch((e) => ({ _err: e.message }));
        return Response.json({
            lambda: lam && { count: lam.bets.length, top: lam.bets.slice(0, 3) },
            fallback: fb && (fb._err ? fb : { count: fb.bets.length, top: fb.bets.slice(0, 3) }),
            messages,
        });
    }

    const lam = await lambdaPoly(messages);
    if (lam) return Response.json(lam);
    try {
        const fb = await fallbackPoly(messages);
        return Response.json({ bets: fb.bets, source: fb.source, timestamp: new Date().toISOString(), messages });
    } catch (e) {
        return Response.json({ bets: [], error: [...messages, e.message].join(' | ') }, { status: 500 });
    }
}
