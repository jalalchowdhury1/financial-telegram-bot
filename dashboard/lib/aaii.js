/**
 * aaii.js — AAII Sentiment Survey, read straight from AAII (no Google Sheet).
 *
 * Replaces the old chain sentiment-scraper (GitHub Actions) → Google Sheet →
 * /api/sheets. The sheet writer's service-account key leaked, so the sheet is
 * retired as a live source (it stays as frozen history). See AGENTS.md §3
 * "AAII direct".
 *
 * Tiers (first one that parses wins):
 *   1. aaii.com   GET https://www.aaii.com/sentimentsurvey/sent_results — the
 *                 server-rendered results table (port of sentiment-scraper's
 *                 Tier 0 regex, sources/aaii-http.ts). Needs a browser UA:
 *                 a bare curl UA gets 403.
 *   2. substack   AAII's own Substack (insights.aaii.com): archive JSON → post
 *                 JSON, then RSS (port of sentiment-scraper's Tier 2,
 *                 sources/aaii-substack.ts).
 *
 * The weekly survey closes Wednesday and prints Thursday, so a healthy answer
 * is at most ~8 days old. Older than STALE_AFTER_DAYS (9) = a missed week →
 * `stale: true`, never shown as fresh.
 *
 * Answer shape (the fixed contract other repos read — do not rename):
 *   { bull, neutral, bear, diff, as_of, source, stale }
 *   diff  = bear − bull as "15.40%" — the exact string the sheet's E2 held
 *           (sentiment-scraper wrote `${(bearish - bullish).toFixed(2)}%`).
 *   as_of = the survey week date, YYYY-MM-DD.
 *   source = 'aaii.com' | 'substack'.
 */

export const AAII_URL = 'https://www.aaii.com/sentimentsurvey/sent_results';
export const SUBSTACK_BASE = 'https://insights.aaii.com';
export const STALE_AFTER_DAYS = 9;
export const FRESH_CACHE_MS = 3 * 3600e3;       // re-fetch AAII at most every 3 h per instance
export const LAST_GOOD_MAX_MS = 21 * 864e5;     // a real past answer, flagged stale, beats a 503

export const BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
};

const MONTHS = { jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5, jul: 6, aug: 7, sep: 8, oct: 9, nov: 10, dec: 11 };
const DAY_MS = 864e5;

/** "Sep 23" / "September 23" / "Sep 23 2026" / "2026-09-23" → UTC Date (most recent past
 *  occurrence when no year: a "Dec 31" read in January is last year's). null if unparseable. */
export function parseSurveyDate(text, now = new Date()) {
    const t = String(text ?? '').trim();
    const iso = /^(\d{4})-(\d{2})-(\d{2})/.exec(t);
    if (iso) return new Date(Date.UTC(+iso[1], +iso[2] - 1, +iso[3]));
    const m = /^([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?$/.exec(t);
    if (!m) return null;
    const month = MONTHS[m[1].slice(0, 3).toLowerCase()];
    const day = +m[2];
    if (month === undefined || day < 1 || day > 31) return null;
    if (m[3]) return new Date(Date.UTC(+m[3], month, day));
    let d = new Date(Date.UTC(now.getUTCFullYear(), month, day));
    if (d.getTime() > now.getTime() + DAY_MS) d = new Date(Date.UTC(now.getUTCFullYear() - 1, month, day));
    return d;
}

const isoDay = (d) => d.toISOString().slice(0, 10);

function validRow(bull, neutral, bear) {
    if ([bull, neutral, bear].some((v) => typeof v !== 'number' || Number.isNaN(v) || v < 0 || v > 100)) return false;
    return Math.abs(bull + neutral + bear - 100) <= 1;
}

/** aaii.com results table → {bull, neutral, bear, as_of} of the NEWEST row, or null.
 *  Strict cell regex first; a tag-stripped loose pass (sum-validated) if AAII reshuffles
 *  attributes. Rows are newest-first on the page. */
export function parseAaiiHtml(html, now = new Date()) {
    if (typeof html !== 'string' || !html) return null;
    const strict = /<td[^>]*class="tableTxt"[^>]*>([A-Z][a-z]{2}\s+\d{1,2})<\/td>\s*<td[^>]*class="tableTxt"[^>]*>([\d.]+)%\s*<\/td>\s*<td[^>]*class="tableTxt"[^>]*>([\d.]+)%\s*<\/td>\s*<td[^>]*class="tableTxt"[^>]*>([\d.]+)%\s*<\/td>/g;
    const candidates = [...html.matchAll(strict)];
    if (!candidates.length) {
        const text = html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
        candidates.push(...text.matchAll(/([A-Z][a-z]{2}\s+\d{1,2})\s+([\d.]+)%?\s+([\d.]+)%?\s+([\d.]+)%?/g));
    }
    for (const m of candidates) {
        const [, date, b, n, r] = m;
        const bull = parseFloat(b), neutral = parseFloat(n), bear = parseFloat(r);
        if (!validRow(bull, neutral, bear)) continue;
        const d = parseSurveyDate(date.replace(/\s+/g, ' '), now);
        if (!d) continue;
        return { bull, neutral, bear, as_of: isoDay(d) };
    }
    return null;
}

const MONTH_LONG = { january: 'Jan', february: 'Feb', march: 'Mar', april: 'Apr', may: 'May', june: 'Jun', july: 'Jul', august: 'Aug', september: 'Sep', october: 'Oct', november: 'Nov', december: 'Dec' };

/** AAII's weekly Substack prose → {bull, neutral, bear, as_of} or null. Each label binds
 *  to its own sentence (negative lookahead), so "…neutral sentiment increased." can't
 *  steal the next label's number. Survey date: "week ending <Month D>" if present, else
 *  the last Wednesday before the post date (the survey week closes Wednesday). */
export function parseSubstackBody(bodyHtml, postDateISO, now = new Date()) {
    const text = String(bodyHtml ?? '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
    const grab = (label) => {
        const m = new RegExp(`${label} sentiment(?:(?!sentiment)[^%]){0,260}?(?:to|at)\\s+([\\d.]+)%`, 'i').exec(text);
        return m ? parseFloat(m[1]) : NaN;
    };
    const bull = grab('Bullish'), neutral = grab('Neutral'), bear = grab('Bearish');
    if (!validRow(bull, neutral, bear)) return null;
    let d = null;
    const we = /week ending ([A-Za-z]+) (\d{1,2})/i.exec(text);
    if (we && MONTH_LONG[we[1].toLowerCase()]) d = parseSurveyDate(`${MONTH_LONG[we[1].toLowerCase()]} ${+we[2]}`, now);
    if (!d) {
        // The survey week closes on a Wednesday (the aaii.com table labels every row with
        // one). Posts go out Thursday or later (Sep 23 survey → post Sat Sep 26 2026), so
        // the survey date is the last Wednesday strictly before the post day.
        const post = new Date(postDateISO);
        if (Number.isNaN(post.getTime())) return null;
        d = new Date(Date.UTC(post.getUTCFullYear(), post.getUTCMonth(), post.getUTCDate()) - DAY_MS);
        while (d.getUTCDay() !== 3) d = new Date(d.getTime() - DAY_MS);
    }
    return { bull, neutral, bear, as_of: isoDay(d) };
}

/** Bear − bull as the sheet's E2 string: "15.40%", "-3.10%". */
export function formatDiff(bull, bear) {
    return `${(bear - bull).toFixed(2)}%`;
}

export function ageDays(asOf, now = new Date()) {
    const d = parseSurveyDate(asOf, now);
    return d ? Math.floor((now.getTime() - d.getTime()) / DAY_MS) : Infinity;
}

/** A parsed row + its tier → the contract payload. `stale` is always recomputed from
 *  as_of against `now`, so a cached copy can never pass for fresh. */
export function toPayload(row, source, now = new Date()) {
    return {
        bull: row.bull,
        neutral: row.neutral,
        bear: row.bear,
        diff: formatDiff(row.bull, row.bear),
        as_of: row.as_of,
        source,
        stale: ageDays(row.as_of, now) > STALE_AFTER_DAYS,
    };
}

const SENTIMENT_RE = /sentiment[- ]survey/i;

/**
 * Run the tiers. `fetchText(url, opts)` must throw on non-2xx (lib/fetcher.js does).
 * `trip(name)` throws for an injected fault (?_fail=aaii_http / aaii_substack / aaii_rss).
 * Returns { payload, messages } — payload null when every tier failed.
 */
export async function fetchAaiiLive({ fetchText, trip = () => {}, now = new Date() }) {
    const messages = [];
    const get = (url, accept) => fetchText(url, { timeout: 12000, headers: { ...BROWSER_HEADERS, Accept: accept } });

    // Tier 1: aaii.com results table
    try {
        trip('aaii_http');
        const row = parseAaiiHtml(await get(AAII_URL, BROWSER_HEADERS.Accept), now);
        if (row) return { payload: toPayload(row, 'aaii.com', now), messages: [...messages, 'aaii.com: ok'] };
        messages.push('aaii.com: no parseable row');
    } catch (e) { messages.push(`aaii.com: ${String(e?.message).slice(0, 120)}`); }

    // Tier 2a: Substack archive JSON → post JSON
    try {
        trip('aaii_substack');
        const posts = JSON.parse(await get(`${SUBSTACK_BASE}/api/v1/archive?sort=new&limit=30`, 'application/json'));
        const post = Array.isArray(posts) ? posts.find((p) => SENTIMENT_RE.test(`${p?.title ?? ''} ${p?.slug ?? ''}`)) : null;
        if (!post?.slug) throw new Error('no sentiment post in the archive');
        const full = JSON.parse(await get(`${SUBSTACK_BASE}/api/v1/posts/${encodeURIComponent(post.slug)}`, 'application/json'));
        const row = parseSubstackBody(full?.body_html ?? '', post.post_date ?? now.toISOString(), now);
        if (row) return { payload: toPayload(row, 'substack', now), messages: [...messages, `substack api: ok (${post.slug})`] };
        messages.push('substack api: no parseable numbers');
    } catch (e) { messages.push(`substack api: ${String(e?.message).slice(0, 120)}`); }

    // Tier 2b: Substack RSS
    try {
        trip('aaii_rss');
        const rss = await get(`${SUBSTACK_BASE}/feed`, 'application/rss+xml, application/xml');
        for (const item of rss.split(/<item>/i).slice(1)) {
            const title = /<title>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/title>/i.exec(item)?.[1] ?? '';
            if (!SENTIMENT_RE.test(title) && !/bullish|bearish/i.test(item)) continue;
            const content = /<content:encoded>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/content:encoded>/i.exec(item)?.[1] ?? item;
            const pub = /<pubDate>([\s\S]*?)<\/pubDate>/i.exec(item)?.[1];
            const row = parseSubstackBody(content, pub ? new Date(pub).toISOString() : now.toISOString(), now);
            if (row) return { payload: toPayload(row, 'substack', now), messages: [...messages, 'substack rss: ok'] };
        }
        messages.push('substack rss: no parseable item');
    } catch (e) { messages.push(`substack rss: ${String(e?.message).slice(0, 120)}`); }

    return { payload: null, messages };
}

export const AAII_KV_KEY = 'ftb:aaii:newest';

/** The KV copy of the newest survey ever seen, or null. Never throws. */
async function loadKv(kv, now) {
    if (!kv) return null;
    try {
        const raw = await kv.get(AAII_KV_KEY);
        const p = typeof raw === 'string' ? JSON.parse(raw) : raw;
        if (!p?.data?.as_of) return null;
        if (!(now.getTime() - Date.parse(p.savedAt || '') <= LAST_GOOD_MAX_MS)) return null;
        return p;
    } catch { return null; }
}

/**
 * Cached resolver used by /api/aaii and /api/sheets.
 *   fresh cache (< 3 h, this instance) → live tiers → last good (≤ 21 d, real past
 *   numbers; `stale` recomputed, `_meta.lastGood` set) → { payload: null }.
 * NEVER GOES BACKWARDS (9 Oct 2026): aaii.com 503'd the day after printing the 8 Oct
 * survey (−1.3); the Substack tier still had the 30 Sep one (11.9) and the pill silently
 * reverted to it, while the history sheet kept −1.3. So the newest survey ever served is
 * kept in Upstash KV (`ftb:aaii:newest`, survives cold instances) and in this instance's
 * /tmp copy; a live tier answering with an OLDER survey week loses to it.
 * Fault-test calls (`testMode`) neither read the fresh cache nor write anything; the
 * `aaii_lastgood` fault disables the last-good and KV reads.
 * `store` = { load(key, maxAgeMs) → {data, savedAt}|null, save(key, data) } (lib/store.js).
 * `kv` = { get, set } (lib/factorStore defaultKv) or null.
 */
export async function resolveAaii({ fetchText, store, kv = null, faults = new Set(), now = new Date() }) {
    const testMode = faults.size > 0;
    const trip = (name) => { if (faults.has(name)) throw new Error(`[injected fault: ${name}]`); };

    const useBackups = !faults.has('aaii_lastgood');
    // read first: a newer survey pushed to KV (e.g. by the Mac MacroMicro job) must beat
    // this instance's 3 h cache too, or a warm instance keeps the old week for hours
    const kvCopy = useBackups ? await loadKv(kv, now) : null;

    if (!testMode) {
        const fresh = store.load('aaii-live', FRESH_CACHE_MS);
        if (fresh?.data?.as_of && !(kvCopy && kvCopy.data.as_of > fresh.data.as_of)) {
            return { payload: toPayload(fresh.data, fresh.data.source, now), cachedAt: fresh.savedAt, messages: [`cached from ${fresh.savedAt}`] };
        }
    }

    const { payload, messages } = await fetchAaiiLive({ fetchText, trip, now });
    const lg = useBackups ? store.load('aaii-live', LAST_GOOD_MAX_MS) : null;
    // the newest saved survey week across both backups
    const saved = [lg, kvCopy].filter((x) => x?.data?.as_of)
        .sort((a, b) => (a.data.as_of < b.data.as_of ? 1 : -1))[0] || null;

    if (payload) {
        if (saved && saved.data.as_of > payload.as_of) {
            if (!testMode) store.save('aaii-live', saved.data); // keep the newer copy warm here
            return {
                payload: toPayload(saved.data, saved.data.source, now),
                cachedAt: saved.savedAt,
                messages: [...messages, `live tier had an older survey (${payload.as_of}); keeping ${saved.data.as_of} from ${saved.savedAt}`],
            };
        }
        if (!testMode) {
            store.save('aaii-live', payload);
            if (kv && kvCopy?.data?.as_of !== payload.as_of) {
                try { await kv.set(AAII_KV_KEY, { data: payload, savedAt: now.toISOString() }); } catch { /* KV down: /tmp still has it */ }
            }
        }
        return { payload, cachedAt: null, messages };
    }

    if (saved) {
        return {
            payload: toPayload(saved.data, saved.data.source, now),
            cachedAt: saved.savedAt,
            lastGood: true,
            messages: [...messages, `every tier failed; serving last good from ${saved.savedAt}`],
        };
    }
    return { payload: null, cachedAt: null, messages };
}
