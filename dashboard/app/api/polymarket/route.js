import { fetchJson } from '../../../lib/fetcher';
import { serve } from '../../../lib/store';
import { faultsFrom } from '../../../lib/faults';

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

/** Polymarket public Gamma API -> top-10 active markets by 24h volume
 *  (highest real trading activity; a reasonable trending-bets backup). */
// Sports keyword sweep (mirrors bot/fetchers.py — used when the tag is absent).
const SPORTS_KW = ['nfl', 'nba', 'nhl', 'mlb', 'ncaa', 'super bowl', 'world series', 'fifa',
    'world cup', 'champions league', 'premier league', 'tennis', 'golf', 'cricket', 'rugby',
    'formula 1', ' f1', 'motogp', 'mma', 'ufc', 'boxing', 'wwe', 'esports', 'basketball',
    'soccer', 'football', 'baseball', 'hockey', ' vs ', 'finals'];
const TOPICS = [
    ['Crypto', '🪙', ['bitcoin', 'btc', 'ethereum', 'crypto', 'microstrategy', 'solana', 'coinbase', 'xrp']],
    ['Geopolitics', '🌍', ['iran', 'israel', 'gaza', 'ukraine', 'russia', 'china', 'taiwan', 'ceasefire', 'nuclear', 'hormuz', 'north korea', 'peace deal', 'missile', 'sanction', 'cuba', 'venezuela']],
    ['Politics', '🏛️', ['trump', 'biden', 'election', 'president', 'senate', 'congress', 'governor', 'democrat', 'republican', 'nominee', 'primary', 'impeach', 'vance', 'mayor']],
    ['Tech', '🤖', ['openai', 'anthropic', 'gpt', 'nvidia', 'spacex', 'tesla', 'apple', 'google', 'ipo', 'chatgpt', 'claude', 'agi', 'starship']],
    ['Economy', '📉', ['recession', 'rate cut', 'inflation', 'gdp', 'unemployment', 's&p', 'interest rate', 'valuation']],
];
function topicOf(t) {
    for (const [name, emoji, kws] of TOPICS) if (kws.some((k) => t.includes(k))) return [name, emoji];
    return ['World', '🌐'];
}

/** Curated "market sentiment" board (mirrors bot/fetchers.fetch_polymarket_trending):
 *  paginate -> binary, non-sports, 8-92%, de-duped by event, topic-diverse, spread. */
const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
function cleanTitle(t) {
    return t.replace('Democratic', 'Dem').replace('Republican', 'GOP')
        .replace('Presidential Election Winner', 'US President')
        .replace('Presidential Nominee', 'Nominee')
        .replace('Presidential Election', 'Election').replace(/\s+/g, ' ').trim();
}
function isCandidateName(git) {
    const g = (git || '').trim().toLowerCase();
    if (!g || '<>↑↓$0123456789'.includes(g[0])) return false;
    return !MONTHS.some((mo) => g.startsWith(mo));
}

async function fallbackPoly(messages) {
    const base = 'https://gamma-api.polymarket.com/markets?active=true&closed=false&order=volume1wk&ascending=false&limit=100';
    let raw = [];
    for (let off = 0; off < 500; off += 100) {
        const page = await fetchJson(`${base}&offset=${off}`, { revalidate: 300 });
        const arr = Array.isArray(page) ? page : page?.markets || [];
        if (!arr.length) break;
        raw = raw.concat(arr);
    }
    const horizon = Date.now() + 86400000;
    // Group by event so multi-candidate races collapse to "Event: favorite".
    const byEvent = new Map();
    for (const m of raw) {
        const ev = (m.events && m.events[0]) || {};
        const key = ev.ticker || ev.slug || m.slug || m.question || `_${byEvent.size}`;
        if (!byEvent.has(key)) byEvent.set(key, []);
        byEvent.get(key).push([m, ev]);
    }
    const cands = [];
    for (const [key, members] of byEvent) {
        try {
            const firstEv = members[0][1];
            const evTitle = (firstEv.title || '').trim();
            const parsed = [];
            for (const [m] of members) {
                const endIso = m.endDate || null;
                if (endIso) { const ed = Date.parse(endIso); if (!Number.isNaN(ed) && ed < horizon) continue; }
                let outs = []; try { outs = JSON.parse(m.outcomes || '[]').map((o) => String(o).toLowerCase()); } catch {}
                if (outs.join(',') !== 'yes,no') continue;
                let odds = null; try { const p = JSON.parse(m.outcomePrices || '[]'); odds = p.length ? parseFloat(p[0]) : null; } catch {}
                if (odds == null) continue;
                const change = m.oneMonthPriceChange != null ? parseFloat(m.oneMonthPriceChange) : null;
                parsed.push({ odds, vol: parseFloat(m.volumeNum ?? m.volume ?? 0), change, end: endIso, git: (m.groupItemTitle || '').trim(), q: m.question || '' });
            }
            if (!parsed.length) continue;
            const text = `${evTitle} ${parsed.map((p) => p.q).join(' ')} ${firstEv.slug || ''}`.toLowerCase();
            const sportTag = members.some(([m]) => (m.tags || []).some((t) => (t.label || '').toLowerCase().includes('sport')));
            if (sportTag || SPORTS_KW.some((k) => text.includes(k))) continue;
            const [topic, topicEmoji] = topicOf(text);
            const isMulti = members.length >= 2 && parsed.some((p) => p.git);
            let name, chosen, vol, isEvent;
            if (isMulti) {
                const fav = parsed.reduce((a, b) => (b.odds > a.odds ? b : a));
                if (fav.odds < 0.05 || fav.odds > 0.85) continue;
                vol = parsed.reduce((s, p) => s + p.vol, 0);
                if (vol < 25000) continue;
                name = (evTitle && isCandidateName(fav.git)) ? `${cleanTitle(evTitle)}: ${fav.git}` : (fav.q || evTitle || 'Unknown');
                if (name.includes('__')) continue;
                chosen = fav; isEvent = true;
            } else {
                const p = parsed[0];
                if (p.odds < 0.08 || p.odds > 0.92 || p.vol < 25000) continue;
                name = p.q || 'Unknown'; chosen = p; vol = p.vol; isEvent = false;
            }
            cands.push({
                name, odds: Math.round(chosen.odds * 100) / 100, volume: vol,
                change: chosen.change != null ? Math.round(chosen.change * 1e4) / 1e4 : null,
                topic, topicEmoji, endDate: chosen.end || null, eventSlug: firstEv.slug || null,
                _event: key, _isEvent: isEvent,
            });
        } catch { /* skip event */ }
    }
    cands.sort((a, b) => b.volume - a.volume);
    const LIMIT = 8, MAX_LONG = 4, seen = new Set(), perTopic = {}, bets = [];
    let nLong = 0;
    for (const allowLong of [false, true]) {
        for (const c of cands) {
            if (bets.length >= LIMIT) break;
            if (seen.has(c._event) || (perTopic[c.topic] || 0) >= 2) continue;
            const isLong = c.odds < 0.30 && !c._isEvent;
            if (isLong && !allowLong && nLong >= MAX_LONG) continue;
            seen.add(c._event); perTopic[c.topic] = (perTopic[c.topic] || 0) + 1; if (isLong) nLong++;
            const { _event, _isEvent, ...bet } = c; bets.push(bet);
        }
        if (bets.length >= LIMIT) break;
    }
    if (!bets.length) throw new Error('Gamma API returned no usable markets');
    return { bets, source: 'Polymarket Gamma API (fallback)' };
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

    // Never-throws: Lambda -> Gamma -> last-known-good -> empty list.
    const faults = faultsFrom(request);
    return serve('polymarket', async () => {
        const lam = faults.has('lambda') ? null : await lambdaPoly(messages);
        if (lam) return lam;
        if (faults.has('gamma')) throw new Error('[injected fault: gamma]');
        const fb = await fallbackPoly(messages);
        return { bets: fb.bets, source: fb.source, timestamp: new Date().toISOString(), _meta: { messages } };
    }, {
        isGood: (x) => x && Array.isArray(x.bets) && x.bets.length > 0,
        fallback: { bets: [], timestamp: new Date().toISOString() },
        faults,
    });
}
