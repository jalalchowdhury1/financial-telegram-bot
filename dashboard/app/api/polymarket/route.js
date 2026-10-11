import { fetchJson } from '../../../lib/fetcher';
import { serve } from '../../../lib/store';
import { faultsFrom } from '../../../lib/faults';

export const fetchCache = 'default-cache';
export const maxDuration = 30;

// The "Market Sentiment" board: three lists, each mirroring a polymarket.com page.
//   trending = the front page's hand-picked cards (Gamma events/keyset by featuredOrder)
//   breaking = the "Breaking News" page (polymarket.com/api/biggest-movers), backed up by
//              Gamma markets ordered by oneDayPriceChange
//   macro    = the Macro dashboard (events tagged macro-*), with ≈30-day CLOB sparklines
// PRIMARY = the Lambda (bot/fetchers.py fetch_polymarket_board). This file is the
// fallback and holds the SAME curation (AGENTS.md checklist): change both together.
//
// Timing budget (maxDuration 30 s): the Lambda hop gets LAMBDA_TIMEOUT_MS; every direct
// fetch after it is clamped to what is left of DEADLINE_MS, which leaves serve() room for
// its last-good read (a KV GET, ≤ 3 s) before Vercel's cap.
const LAMBDA_TIMEOUT_MS = 10000;
const DEADLINE_MS = 24000;

const GAMMA = 'https://gamma-api.polymarket.com';
const SITE = 'https://polymarket.com';
const CLOB = 'https://clob.polymarket.com';
const MIN_MOVER_VOLUME = 50000;   // a 24h move on a thinner market is noise, not news
const MIN_MOVE = 0.05;            // 5 points
const MACRO_TAGS = ['macro-graph', 'macro-single', 'macro-fed', 'macro-inflation', 'macro-jobs',
    'macro-unemployment', 'macro-geopolitics'];
const LIMITS = { trending: 12, breaking: 10, macro: 6 };
export const FALLBACK_SOURCE = 'Polymarket Gamma API (fallback)';

// Untagged text (movers carry no tags). Word-bounded on purpose: a substring list matched
// "inter" inside "interest rates" and "america" inside "Latin American".
const SPORTS_RE = new RegExp(
    "\\b(nfl|nba|wnba|nhl|mlb|mls|ncaa|cfb|ufc|mma|wwe|nascar|f1|formula 1|motogp|grand prix|" +
    "premier league|champions league|europa league|la liga|serie a|bundesliga|ligue 1|" +
    "world cup|super bowl|world series|stanley cup|ballon d'?or|heisman|wimbledon|" +
    "french open|australian open|atp|wta|pga|lpga|golf|tennis|cricket|ipl|rugby|boxing|" +
    "wrestling|esports?|league of legends|lol|dota|valorant|counter-strike|cs2|overwatch|" +
    "esl|bo[1357]|o/u|over/under|moneyline|touchdowns?|playoffs?|fc|vs\\.?)\\b|spread:", 'i');

// First match wins. Elections are claimed before geopolitics so "Prime Minister of Israel
// after the next election?" reads as politics, while "US-Iran ceasefire" stays geopolitics.
const TOPICS = [
    ['Crypto', '🪙', /\b(crypto|bitcoin|btc|ethereum|eth|solana|xrp|dogecoin|stablecoins?|airdrop|fdv|token|coinbase|binance|microstrategy)\b/i],
    ['Tech', '🤖', /\b(tech|ai|openai|anthropic|chatgpt|gpt|gemini|grok|claude|fable|llm|nvidia|spacex|spacexai|starship|tesla|apple|google|microsoft|meta|science|fda|vaccines?|ipo)\b/i],
    // not "macro": Polymarket tags elections "Macro Election"
    ['Economy', '📉', /\b(economy|finance|fed|fomc|rate (?:cut|hike)s?|interest rates?|inflation|cpi|gdp|recession|unemployment|jobs report|tariffs?|s&p|stocks?|consumer sentiment)\b/i],
    ['Politics', '🏛️', /\b(elections?|midterms?|senate|house|congress|governor|mayor(?:al)?|nominee|primary|president(?:ial)?|prime minister|parliament)\b/i],
    ['Geopolitics', '🌍', /\b(geopolitics|military|war|ceasefire|invad\w*|invasion|iran|israel|gaza|ukraine|russia|putin|china|xi jinping|taiwan|nato|nuclear|hormuz|houthi|yemen|blockade|missiles?|sanctions?|middle east)\b/i],
    ['Politics', '🏛️', /\b(politics|trump|ambassador|cabinet|impeach\w*|supreme court)\b/i],
    ['Culture', '🎬', /\b(culture|pop culture|movies?|box office|music|album|songs?|youtube|mrbeast|views|netflix|oscars?|grammys?|emmys?|rotten tomatoes|spotify|billboard|celebrit\w+|tweets?)\b/i],
];

/** Gamma sends outcomes/outcomePrices/clobTokenIds as JSON strings; accept either. */
function list(raw) {
    if (Array.isArray(raw)) return raw;
    try { const v = JSON.parse(raw || '[]'); return Array.isArray(v) ? v : []; } catch { return []; }
}
function num(raw) {
    if (typeof raw === 'string' ? !raw.trim() : typeof raw !== 'number') return null;
    const v = Number(raw);
    return Number.isFinite(v) ? v : null;
}
const round3 = (v) => (v == null ? null : Math.round(v * 1000) / 1000);
const tagText = (tags) => (Array.isArray(tags) ? tags : [])
    .filter((t) => t && typeof t === 'object')
    .map((t) => `${t.label || ''} ${(t.slug || '').replace(/-/g, ' ')}`).join(' ');
function isSports(text, tags) {
    if ((Array.isArray(tags) ? tags : []).some((t) => t && typeof t === 'object'
        && `${t.label || ''} ${t.slug || ''}`.toLowerCase().includes('sport'))) return true;
    return SPORTS_RE.test(text || '');
}
function topicOf(text) {
    for (const [name, emoji, rx] of TOPICS) if (rx.test(text || '')) return [name, emoji];
    return ['World', '🌐'];
}
/** A market's parent event (its slug is the one polymarket.com links to), or {}. */
function eventOf(m) {
    const ev = Array.isArray(m.events) ? m.events[0] : null;
    return ev && typeof ev === 'object' ? ev : {};
}
/** True when the market's end date has passed (it only awaits resolution: old news). */
function ended(iso) {
    const t = Date.parse(iso);
    return Number.isFinite(t) && t < Date.now();
}

/** The event's still-open outcomes, likeliest first (how Polymarket's cards order them). */
function outcomesOf(ev) {
    const rows = [];
    for (const m of Array.isArray(ev.markets) ? ev.markets : []) {
        if (!m || typeof m !== 'object' || m.closed || m.archived || m.active === false) continue;
        const prices = list(m.outcomePrices);
        const odds = prices.length ? num(prices[0]) : null;
        if (odds == null) continue;
        const outs = list(m.outcomes);
        const tokens = list(m.clobTokenIds);
        rows.push({
            label: String(m.groupItemTitle || '').trim() || (outs.length ? String(outs[0]) : 'Yes'),
            odds,
            change: num(m.oneDayPriceChange),
            change30: num(m.oneMonthPriceChange),
            token: tokens.length ? String(tokens[0]) : null,
        });
    }
    return rows.sort((a, b) => b.odds - a.odds);
}

/** [{t, p}] price history -> at most `max` prices, oldest first. */
function spark(points, max = 48) {
    let p = (Array.isArray(points) ? points : [])
        .filter((x) => x && typeof x === 'object').map((x) => num(x.p)).filter((v) => v != null);
    if (p.length > max) {
        const step = (p.length - 1) / (max - 1);
        p = Array.from({ length: max }, (_, i) => p[Math.round(i * step)]);
    }
    return p.map(round3);
}

/** A fetcher whose timeouts never run past the request's deadline. fetchJson's own timeout
 *  stops once the headers arrive, and Next does not pre-read a revalidate-0 body (the 2.5 MB
 *  keyset), so a second timer caps the whole call, body included: a stalled body must not
 *  outlive the function and turn serve()'s last-good answer into a 504. */
function makeGet(deadlineAt) {
    return (url, ms, revalidate = 120) => {
        const budget = Math.max(1000, Math.min(ms, deadlineAt - Date.now()));
        let timer;
        const cap = new Promise((_, reject) => {
            timer = setTimeout(() => reject(new Error(`timed out after ${budget}ms: ${url}`)), budget);
        });
        return Promise.race([fetchJson(url, { revalidate, timeout: budget }), cap])
            .finally(() => clearTimeout(timer));
    };
}

/** CLOB price history for one outcome token -> sparkline; [] on any failure. */
async function history(get, token, interval, fidelity, max) {
    if (!token) return [];
    try {
        const d = await get(`${CLOB}/prices-history?market=${encodeURIComponent(token)}&interval=${interval}&fidelity=${fidelity}`, 6000);
        return spark(d?.history, max);
    } catch { return []; }
}

/** Front-page cards. Throws when the feed itself fails (vs. [] = nothing to show). */
async function trendingRows(get, limit) {
    // revalidate 0: the feed is ~2.5 MB, over Next's 2 MB data-cache item limit.
    const data = await get(`${GAMMA}/events/keyset?active=true&archived=false&closed=false&order=featuredOrder&ascending=true&featured_order=true`, 15000, 0);
    const events = Array.isArray(data) ? data : data?.events;
    if (!Array.isArray(events)) throw new Error('featured keyset returned no events list');
    const rows = [];
    for (const ev of events) {
        try {
            const title = String(ev.title || '').trim();
            const slug = String(ev.slug || '');
            const text = `${title} ${slug.replace(/-/g, ' ')} ${tagText(ev.tags)}`;
            // Sports, and the 5-minute "Up or Down" coin flips (pure intraday churn).
            if (!title || isSports(text, ev.tags) || slug.includes('updown') || title.toLowerCase().includes('up or down')) continue;
            const outs = outcomesOf(ev);
            if (!outs.length) continue;
            const [topic, topicEmoji] = topicOf(text);
            rows.push({
                title, slug, topic, topicEmoji,
                volume: num(ev.volume) || 0,
                volume24h: num(ev.volume24hr) || 0,
                endDate: ev.endDate ?? null,
                outcomes: outs.slice(0, 6).map((o) => ({ label: o.label, odds: round3(o.odds), change: round3(o.change) })),
                nOutcomes: outs.length,
            });
        } catch { /* skip this event */ }
        if (rows.length >= limit) break;
    }
    return [rows, 'featured'];
}

/** Biggest moves first, one per event (a date ladder can't fill the list). */
function pickMovers(cands, limit) {
    cands.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));
    const seen = new Set(), out = [];
    for (const c of cands) {
        if (seen.has(c.slug)) continue;
        seen.add(c.slug);
        out.push(c);
        if (out.length >= limit) break;
    }
    return out;
}
function moverRow(question, slug, odds, change, volume, sparkline) {
    const [topic, topicEmoji] = topicOf(`${question} ${slug.replace(/-/g, ' ')}`);
    return { question, slug, odds: round3(odds), change: round3(change), volume, spark: sparkline, topic, topicEmoji };
}

/** polymarket.com's own Breaking News feed (comes with each market's 24h history). */
async function moversSite(get, limit) {
    const [data, sports] = await Promise.all([
        get(`${SITE}/api/biggest-movers`, 6000),
        // The site's own sports bucket: anything in it is dropped from "all".
        get(`${SITE}/api/biggest-movers?category=sports`, 4000).catch(() => null),
    ]);
    const movers = data && !Array.isArray(data) ? data.markets : null;
    if (!Array.isArray(movers) || !movers.length) throw new Error('biggest-movers returned no markets');
    const sportsIds = new Set((Array.isArray(sports?.markets) ? sports.markets : []).map((m) => String(m?.id)));
    const cands = [];
    for (const m of movers) {
        if (!m || typeof m !== 'object') continue;
        try {
            const ev = eventOf(m);
            const question = String(m.question || '').trim();
            const slug = String(ev.slug || m.slug || '');
            let odds = num(m.currentPrice);
            if (odds == null) { const p = list(m.outcomePrices); odds = p.length ? num(p[0]) : null; }
            const live = num(m.livePriceChange);   // points, e.g. -54
            const change = live != null ? live / 100 : num(m.oneDayPriceChange);
            const volume = num(ev.volume) || 0;
            if (!question || odds == null || change == null || m.closed
                || sportsIds.has(String(m.id))
                || isSports(`${question} ${slug.replace(/-/g, ' ')}`)
                || volume < MIN_MOVER_VOLUME || Math.abs(change) < MIN_MOVE) continue;
            cands.push(moverRow(question, slug, odds, change, volume, spark(m.history)));
        } catch { /* skip this market */ }
    }
    // 25 markets and not one usable: more likely a changed shape than a quiet day.
    if (!cands.length) throw new Error(`biggest-movers: none of ${movers.length} markets passed the filters`);
    return pickMovers(cands, limit);
}

/** Backup: Gamma markets by 24h price change, both directions, plus a 24h CLOB sparkline
 *  so the list looks the same when this path is the one serving. */
async function moversGamma(get, limit) {
    const page = (asc) => get(`${GAMMA}/markets?active=true&closed=false&order=oneDayPriceChange&ascending=${asc}&limit=50&volume_num_min=${MIN_MOVER_VOLUME}`, 8000);
    const pages = await Promise.all([page(false), page(true)]);
    const cands = [];
    for (const m of pages.flatMap((pg) => (Array.isArray(pg) ? pg : []))) {
        if (!m || typeof m !== 'object') continue;
        try {
            const ev = eventOf(m);
            const question = String(m.question || '').trim();
            const slug = String(ev.slug || m.slug || '');
            const prices = list(m.outcomePrices);
            const odds = prices.length ? num(prices[0]) : null;
            const change = num(m.oneDayPriceChange);
            const volume = num(m.volumeNum || m.volume) || 0;
            if (!question || odds == null || change == null
                || m.sportsMarketType || m.gameStartTime || ended(m.endDate)
                || isSports(`${question} ${slug.replace(/-/g, ' ')}`, m.tags)
                || volume < MIN_MOVER_VOLUME || Math.abs(change) < MIN_MOVE) continue;
            const tokens = list(m.clobTokenIds);
            cands.push({ ...moverRow(question, slug, odds, change, volume, []), _token: tokens.length ? String(tokens[0]) : null });
        } catch { /* skip this market */ }
    }
    const rows = pickMovers(cands, limit);
    const sparks = await Promise.all(rows.map((r) => history(get, r._token, '1d', 30, 48)));
    return rows.map(({ _token, ...r }, i) => ({ ...r, spark: sparks[i] }));
}

/** Breaking News: the site's feed, else Gamma. Throws only when BOTH fail, so an empty
 *  list with a source means a genuinely quiet day. */
async function breakingRows(get, limit, messages = []) {
    try {
        return [await moversSite(get, limit), 'biggest-movers'];
    } catch (e) {
        messages.push(`biggest-movers unavailable (${String(e?.message).slice(0, 100)}); using Gamma 24h change`);
    }
    return [await moversGamma(get, limit), 'gamma'];
}

/** Macro dashboard tiles. Throws when every macro tag failed to load, or when they all
 *  answered with nothing open: the dashboard always has tiles, so an empty strip means the
 *  tags changed, not a quiet day (an outage the card and health check must see). */
async function macroRows(get, limit) {
    const groups = await Promise.all(MACRO_TAGS.map((tag) =>
        get(`${GAMMA}/events?tag_slug=${tag}&active=true&closed=false&archived=false&limit=10`, 8000).catch(() => null)));
    if (groups.every((g) => g == null)) throw new Error('every macro tag failed');
    const tiles = [], seen = new Set();
    for (const events of groups) {
        for (const ev of Array.isArray(events) ? events : []) {
            if (!ev || typeof ev !== 'object') continue;
            try {
                const title = String(ev.title || '').trim();
                const slug = String(ev.slug || '');
                const text = `${title} ${slug.replace(/-/g, ' ')} ${tagText(ev.tags)}`;
                if (!title || seen.has(slug) || isSports(text, ev.tags)) continue;
                const outs = outcomesOf(ev);
                if (!outs.length) continue;
                seen.add(slug);
                const lead = outs[0];
                tiles.push({ title, slug, label: lead.label, odds: round3(lead.odds), change: round3(lead.change30),
                    volume: num(ev.volume) || 0, spark: [], _token: lead.token });
            } catch { /* skip this event */ }
        }
    }
    if (!tiles.length) throw new Error('macro tags answered but held no open events');
    const top = tiles.slice(0, limit);
    const sparks = await Promise.all(top.map((t) => history(get, t._token, '1m', 720, 40)));
    return [top.map(({ _token, ...t }, i) => {
        const s = sparks[i];
        // the change the sparkline draws, so the two agree
        return s.length >= 2 ? { ...t, spark: s, change: round3(t.odds - s[0]) } : { ...t, spark: s };
    }), 'macro-tags'];
}

const BUILDERS = { trending: trendingRows, breaking: breakingRows, macro: macroRows };

/** Build some (default all) lists straight from Polymarket, in parallel. Never throws:
 *  a list whose sources all failed comes back as [[], null]. */
export async function directBoard(get, messages, names = Object.keys(BUILDERS)) {
    const got = await Promise.all(names.map(async (n) => {
        try { return await BUILDERS[n](get, LIMITS[n], messages); }
        catch (e) { messages.push(`${n}: ${String(e?.message).slice(0, 120)}`); return [[], null]; }
    }));
    return Object.fromEntries(names.map((n, i) => [n, got[i]]));
}

/** Normalise any board (Lambda or direct) into the payload the card reads. */
function finish(board, messages) {
    const sources = { trending: null, breaking: null, macro: null, ...(board.sources || {}) };
    const macro = Array.isArray(board.macro) ? board.macro : [];
    const macroSlugs = new Set(macro.map((m) => m.slug));
    const failed = Object.keys(sources).filter((n) => !sources[n]);
    if (failed.length) messages.push(`no source answered for: ${failed.join(', ')}`);
    return {
        trending: (Array.isArray(board.trending) ? board.trending : []).filter((t) => !macroSlugs.has(t.slug)),
        breaking: Array.isArray(board.breaking) ? board.breaking : [],
        macro,
        sources,
        source: board.source,
        timestamp: board.timestamp || new Date().toISOString(),
        // A list with no source is an outage, not a quiet day: flag it (never edge-cached,
        // never saved as last-good, and the health check warns).
        _meta: { hasErrors: failed.length > 0, messages },
    };
}

async function lambdaPoly(messages) {
    const lambdaUrl = process.env.LAMBDA_URL;
    if (!lambdaUrl) { messages.push('LAMBDA_URL not configured'); return null; }
    try {
        const res = await fetch(`${lambdaUrl}/api/polymarket`, { cache: 'no-store', signal: AbortSignal.timeout(LAMBDA_TIMEOUT_MS) });
        if (!res.ok) { messages.push(`Lambda HTTP ${res.status}`); return null; }
        const j = await res.json();
        // `trending`, not the old `bets`: a Lambda still on the old code falls through to
        // the direct build instead of handing the new card a shape it can't draw.
        if (j && Array.isArray(j.trending) && j.trending.length && !j.error) return j;
        messages.push('Lambda returned no usable board');
    } catch (e) { messages.push(`Lambda failed: ${e.message}`); }
    return null;
}

const isGood = (x) => !!x && Array.isArray(x.trending) && x.trending.length > 0;

export async function GET(request) {
    request.headers.get('user-agent');
    const debug = new URL(request.url).searchParams.get('debug');
    const messages = [];
    const get = makeGet(Date.now() + DEADLINE_MS);

    if (debug === 'compare') {
        const head = (b) => b && Object.fromEntries(['trending', 'breaking', 'macro'].map((n) =>
            [n, { count: (b[n] || []).length, top: (b[n] || []).slice(0, 3).map((r) => r.title || r.question) }]));
        const [lam, direct] = await Promise.all([lambdaPoly(messages), directBoard(get, messages)]);
        const fb = Object.fromEntries(Object.entries(direct).map(([n, [rows, src]]) => [n, rows]));
        return Response.json({
            lambda: lam && { ...head(lam), sources: lam.sources || null },
            fallback: { ...head(fb), sources: Object.fromEntries(Object.entries(direct).map(([n, [, src]]) => [n, src])) },
            messages,
        });
    }

    // Never-throws: Lambda -> direct Polymarket -> last-known-good -> empty board.
    const faults = faultsFrom(request);
    return serve('polymarket', async () => {
        const lam = faults.has('lambda') ? null : await lambdaPoly(messages);
        if (lam) {
            // Fill a list the Lambda could not load at all from here. An empty list WITH a
            // source is a quiet day: leave it.
            const missing = ['breaking', 'macro'].filter((n) =>
                !(Array.isArray(lam[n]) && lam[n].length) && !(lam.sources && lam.sources[n]));
            // The Lambda settled for the Gamma backup for Breaking (polymarket.com may refuse
            // an AWS address): try the site's own feed from here before keeping the backup.
            const upgrade = lam.sources?.breaking === 'gamma';
            if ((missing.length || upgrade) && !faults.has('gamma')) {
                const [got, site] = await Promise.all([
                    missing.length ? directBoard(get, messages, missing) : null,
                    upgrade ? moversSite(get, LIMITS.breaking).catch((e) => {
                        messages.push(`biggest-movers (direct): ${String(e?.message).slice(0, 100)}`);
                        return null;
                    }) : null,
                ]);
                for (const n of missing) {
                    const [rows, src] = got[n];
                    if (src) { lam[n] = rows; lam.sources = { ...(lam.sources || {}), [n]: `${src} (direct)` }; }
                }
                if (site) { lam.breaking = site; lam.sources = { ...lam.sources, breaking: 'biggest-movers (direct)' }; }
            }
            return finish(lam, messages);
        }
        if (faults.has('gamma')) throw new Error('[injected fault: gamma]');
        const got = await directBoard(get, messages);
        return finish({
            trending: got.trending[0], breaking: got.breaking[0], macro: got.macro[0],
            sources: { trending: got.trending[1], breaking: got.breaking[1], macro: got.macro[1] },
            source: FALLBACK_SOURCE,
            timestamp: new Date().toISOString(),
        }, messages);
    }, {
        isGood,
        // Only a complete board becomes last-good, so a later outage serves a whole card.
        shouldStore: (x) => isGood(x) && !x._meta?.hasErrors,
        fallback: { trending: [], breaking: [], macro: [], timestamp: new Date().toISOString() },
        faults,
    });
}
