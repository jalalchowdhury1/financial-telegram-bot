/**
 * 🔝 Glance bar (components/GlanceBar.js) — what it carries. Pure: it reads the payloads the
 * page already holds, so the bar can never show a number the cards below don't.
 */
import { savedLabel } from './snapshot';

const fin = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * {price: '769.64', move: {up, text: '▲0.74%', when: 'Fri' | null} | null, fg: 31 | null, fgScore}
 * — or null when SPY is missing, errored or has no price. F&G down only drops F&G: the bar's
 * ↻ is most wanted exactly when a feed has failed.
 * The move is spy-daily-move's %, else SPY's own daily change; neither = no move (never 0.00%).
 * `when` = the page's spyWhen (the SPY card's badge): shown unless it is "today", so a Friday
 * move never passes for a live one on Sunday.
 */
export function glanceNumbers({ spy, spyDailyMove, fg, when } = {}) {
    if (!spy || spy.error || !fin(spy.current)) return null;
    const raw = spyDailyMove?.value;
    let pct = fin(raw) ? raw : typeof raw === 'string' ? parseFloat(raw) : NaN;
    if (!Number.isFinite(pct)) pct = fin(spy.dailyChange?.pct) ? spy.dailyChange.pct : NaN;
    const session = typeof when === 'string' && when && when !== 'today' ? when : null;
    const move = Number.isFinite(pct)
        ? { up: pct >= 0, text: `${pct >= 0 ? '▲' : '▼'}${Math.abs(pct).toFixed(2)}%`, when: session }
        : null;
    const score = fg && !fg.error ? fg.score : null;
    return { price: spy.current.toFixed(2), move, fg: fin(score) ? Math.round(score) : null, fgScore: score };
}

/**
 * The bar's age follows its OWN numbers, not the page's: SPY served from the server's
 * last-good copy → {stale: 'Thu 16:00'}; SPY or F&G still this device's saved copy →
 * {saved: '10:42'} (even after other feeds landed live, so a 3-day-old price never reads
 * "just now"); else {at: ms}; else null.
 */
export function glanceAge({ updatedAt, saved, stale } = {}) {
    if (stale) return { stale };
    if (saved) return { saved };
    return fin(updatedAt) ? { at: updatedAt } : null;
}

/** Show once the anchor (Market Pulse) has scrolled wholly above the top of the screen. */
/**
 * A body the server flagged stale (a VIX proxy is a different gauge, not an old one):
 * {ms, label} — "Thu 16:00" from its last-good copy, "Wed close" from the close it carries
 * (lib/spyTiers.js's older-close build has asOf, no lastGoodAt), else "copy". Null = not stale.
 */
export function staleOf(body, now = Date.now()) {
    const m = body?._meta;
    if (!m?.stale || m.proxy) return null;
    const at = Date.parse(m.lastGoodAt);
    if (Number.isFinite(at)) return { ms: at, label: savedLabel(at, now) };
    const day = /^\d{4}-\d{2}-\d{2}/.exec(m.asOf || '')?.[0];
    if (!day) return { ms: -Infinity, label: 'copy' };
    const noon = Date.parse(`${day}T12:00:00Z`); // a calendar day: read at UTC noon so no device timezone moves it
    const name = new Date(noon).toLocaleDateString('en-US', now - noon > 6 * 864e5 ? { month: 'short', day: 'numeric', timeZone: 'UTC' } : { weekday: 'short', timeZone: 'UTC' });
    return { ms: noon, label: `${name} close` };
}

export const shouldShowGlance = (anchorBottom) => fin(anchorBottom) && anchorBottom < 0;
