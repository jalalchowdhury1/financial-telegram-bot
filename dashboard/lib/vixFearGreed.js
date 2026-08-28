/**
 * VIX "Fear/Greed" tag for the CustomIndicatorBar VIX pill (e.g. "GREED13").
 *
 * This used to be written into the VIX sheet's cell C2 once a day by a
 * separate repo (vix-fear-greed), read verbatim by app/api/sheets/route.js.
 * That repo is being deleted, so the computation is folded in here — the
 * dashboard now computes the tag itself from FRED's VIXCLS series, with the
 * sheet's C2 value kept as a transition fallback (see resolveVixFearGreedTag)
 * for as long as vix-fear-greed keeps writing it.
 *
 * THE FORMULA (must match vix-fear-greed/fear_greed.py exactly):
 *   vix = ~1y of VIX daily closes, missing values dropped
 *   sma50   = 50-day rolling mean, most recent value
 *   latest  = most recent close
 *   pct_diff = (latest - sma50) / sma50          # fraction of SMA
 *   score = round(abs(pct_diff) * 100), capped at 99, zero-padded to 2 digits
 *   tag = "FEAR"+score   if pct_diff > 0   (VIX running hot vs its own trend)
 *       = "GREED"+score  if pct_diff < 0   (VIX running cool vs its own trend)
 *       = "NEUTRAL00"    if pct_diff == 0
 * Only the most recent 50 valid closes matter (rolling(50).mean().iloc[-1] +
 * iloc[-1] depend on nothing earlier), so fetching FRED's newest ~280
 * observations (~1 trading year, mirroring the Python script's period="1y")
 * is more than enough buffer for missing prints.
 */
import { gate } from './faults';

const VIXCLS_LIMIT = 280;

/** FRED `series/observations` URL for VIXCLS, newest-first, `limit` rows. */
export function fredVixUrl(apiKey, limit = VIXCLS_LIMIT) {
    return `https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key=${apiKey}&file_type=json&sort_order=desc&limit=${limit}`;
}

/**
 * Parse a FRED `series/observations` payload into ascending (oldest -> newest)
 * `{date, value}` pairs. VIXCLS occasionally emits "." for a missing print
 * (matches fear_greed.py's `vix.dropna()`) — those rows are dropped. Sorts by
 * date explicitly rather than trusting the API's declared sort order.
 */
export function parseVixObservations(data) {
    const obs = data?.observations;
    if (!Array.isArray(obs)) return [];
    return obs
        .filter((o) => o && typeof o.date === 'string' && o.value !== '.' && o.value !== undefined && o.value !== null)
        .map((o) => ({ date: o.date, value: parseFloat(o.value) }))
        .filter((o) => Number.isFinite(o.value))
        .sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
}

/**
 * Python 3's round() is banker's rounding (round-half-to-even), not
 * round-half-away-from-zero like JS's Math.round. The score input here is
 * always >= 0 (it's abs(pct_diff) * 100), so this only needs to handle that
 * case, but it's written to match fear_greed.py's `round()` call exactly in
 * case a real deviation ever lands precisely on a .5 boundary.
 */
function pyRound(x) {
    const floor = Math.floor(x);
    const diff = x - floor;
    if (diff < 0.5) return floor;
    if (diff > 0.5) return floor + 1;
    return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * The pure formula. `closesAsc` is ascending (oldest -> newest) numeric VIX
 * closes; only the trailing 50 are used. Returns null (rather than throwing)
 * when there isn't enough data for a 50-day SMA — mirrors fear_greed.py
 * needing >= 50 valid closes before `.rolling(50)` produces a real value.
 */
export function fearGreedTag(closesAsc) {
    if (!Array.isArray(closesAsc) || closesAsc.length < 50) return null;
    const last50 = closesAsc.slice(-50);
    const sma50 = last50.reduce((sum, v) => sum + v, 0) / 50;
    const latest = closesAsc[closesAsc.length - 1];
    if (!Number.isFinite(sma50) || sma50 === 0 || !Number.isFinite(latest)) return null;

    const pctDiff = (latest - sma50) / sma50;
    const score = Math.min(pyRound(Math.abs(pctDiff) * 100), 99);
    const scoreStr = String(score).padStart(2, '0');

    if (pctDiff > 0) return `FEAR${scoreStr}`;
    if (pctDiff < 0) return `GREED${scoreStr}`;
    return 'NEUTRAL00';
}

/**
 * Fetch VIXCLS from FRED and compute the tag. Throws on any failure (missing
 * key, network/HTTP error, insufficient valid data) — callers decide the
 * fallback; see resolveVixFearGreedTag below. `fetchJson` is injected (same
 * pattern as lib/spEps.js / lib/copperGold.js source descriptors) so this is
 * unit-testable without a network call.
 */
export async function computeVixFearGreedTag(apiKey, { fetchJson, limit = VIXCLS_LIMIT } = {}) {
    if (!apiKey) throw new Error('FRED_API_KEY not configured');
    const data = await fetchJson(fredVixUrl(apiKey, limit), { revalidate: 0 });
    const parsed = parseVixObservations(data);
    const tag = fearGreedTag(parsed.map((o) => o.value));
    if (!tag) throw new Error(`Insufficient VIXCLS data (${parsed.length} valid closes, need >= 50)`);
    return tag;
}

/**
 * The VIX pill's fear/greed source cascade: FRED-computed (primary) -> the
 * Google Sheet's C2 value already read by the sheets cascade (transition
 * fallback, while the vix-fear-greed repo that writes it still runs) ->
 * 'N/A'. Returns `{tag, message}` — `message` ALWAYS states which source won
 * (or that the fallback fired and why), so a silent slide onto the sheet
 * value is visible in `_meta.messages` rather than looking identical to a
 * healthy computed reading (AGENTS.md §7's governing principle: a fallback
 * that silently satisfies the caller is a false negative).
 *
 * `faults` (optional Set) supports the repo's `?_fail=vix_fred` convention
 * for forcing the fallback path on prod to re-verify it.
 */
export async function resolveVixFearGreedTag({ fredApiKey, fetchJson, sheetValue = 'N/A', faults, limit } = {}) {
    try {
        const tag = await gate('vix_fred', faults, () => computeVixFearGreedTag(fredApiKey, { fetchJson, limit }));
        return { tag, message: 'VIX fear/greed: computed from FRED VIXCLS (formula matches the retired vix-fear-greed repo)' };
    } catch (e) {
        return {
            tag: sheetValue,
            message: `VIX fear/greed: FRED computation failed (${e.message}) — using sheet value (${sheetValue})`,
        };
    }
}
