/**
 * Freshness helpers for FRED metrics.
 *
 * Two clocks matter for every number:
 *   - when we last fetched it (handled by the 30-min server cache), and
 *   - the date of the actual data point (handled here).
 *
 * `isStale` answers: is the newest data point older than what's normal for this
 * series? If so, we refuse to show the value (render N/A) rather than pass off
 * an old number as current.
 */

const MS_PER_DAY = 86400000;

/**
 * @param {string|null|undefined} asOfDate  ISO date of the latest observation (e.g. '2026-05-28')
 * @param {number} deadlineDays             max acceptable age in days
 * @param {Date}   [now]                     current time (injectable for tests)
 * @returns {boolean} true when missing or older than the deadline
 */
export function isStale(asOfDate, deadlineDays, now = new Date()) {
    if (!asOfDate) return true;
    const then = new Date(asOfDate);
    if (Number.isNaN(then.getTime())) return true;
    const ageDays = (now.getTime() - then.getTime()) / MS_PER_DAY;
    return ageDays > deadlineDays;
}

/**
 * Build a freshness-aware metric. Nulls the value when the data point is too
 * old or absent, so the UI shows N/A instead of a stale/garbage number.
 *
 * @returns {{ value: number|null, asOf: string|null, stale: boolean, unavailable: boolean }}
 */
export function withFreshness(value, asOfDate, deadlineDays, now = new Date()) {
    const unavailable = value === undefined || value === null || Number.isNaN(value);
    const stale = isStale(asOfDate, deadlineDays, now) && !unavailable;
    let staleDays = 0;
    if (stale && asOfDate) {
        const then = new Date(asOfDate);
        if (!Number.isNaN(then.getTime())) {
            const ageDays = (now.getTime() - then.getTime()) / MS_PER_DAY;
            staleDays = Math.max(0, Math.floor(ageDays - deadlineDays));
        }
    }
    return {
        value: unavailable ? null : value, // keep stale values; null ONLY when truly missing
        asOf: asOfDate ?? null,
        stale,
        unavailable,
        staleDays,
    };
}

// ── Client-safe display helpers (used by dashboard components) ──

/** Format an ISO date/timestamp as e.g. "May 28, 2026" in local time. */
export function formatAsOf(asOf) {
    if (!asOf) return null;
    // Treat bare dates ('2026-05-28') as local, not UTC, to avoid off-by-one.
    const iso = /^\d{4}-\d{2}-\d{2}$/.test(asOf) ? `${asOf}T00:00:00` : asOf;
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return null;
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

/**
 * Given a metric ({ value, asOf, stale, unavailable }), return the tooltip suffix and a
 * display `tone`: 'fresh' | 'stale' | 'unavailable'. `amber` is kept for back-compat.
 */
export function freshnessNote(metric) {
    if (!metric) return { suffix: '', tone: 'fresh', amber: false };
    const dateStr = formatAsOf(metric.asOf);
    const isUnavailable = metric.unavailable === true || metric.value === null || metric.value === undefined;
    if (metric.stale && !isUnavailable) {
        return {
            suffix: dateStr ? ` • 🕐 As of ${dateStr} (stale)` : ' • 🕐 Stale',
            tone: 'stale',
            amber: true,
        };
    }
    if (isUnavailable) {
        return { suffix: ' • ⚠ Unavailable — source busy, try again shortly', tone: 'unavailable', amber: true };
    }
    return { suffix: dateStr ? ` • As of ${dateStr}` : '', tone: 'fresh', amber: false };
}
