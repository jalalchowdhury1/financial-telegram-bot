// Pure helpers for the MacroMicro AAII backup (unit-tested in parse.test.mjs).

/**
 * MacroMicro "Latest Stats" text → { bull, neutral, bear, released } or null.
 * The block reads: "…: Bearish\n2026-10-08\n38.98%\n46.49%" (latest, then previous).
 * Placeholders ("1980-01-01 0.0000%") render before the data loads: rejected.
 */
export function parseLatestStats(text) {
    if (!text) return null;
    const grab = (label) => {
        const m = new RegExp(`Survey:\\s*${label}\\s+(\\d{4}-\\d{2}-\\d{2})\\s+([\\d.]+)%`, 'i').exec(text);
        return m ? { date: m[1], value: Number(m[2]) } : null;
    };
    const bull = grab('Bullish'), neutral = grab('Neutral'), bear = grab('Bearish');
    if (!bull || !neutral || !bear) return null;
    if (!(bull.date === neutral.date && neutral.date === bear.date)) return null;
    if (bull.date < '2020-01-01') return null; // still the placeholder
    const sum = bull.value + neutral.value + bear.value;
    if (!(sum > 97 && sum < 103)) return null; // three shares of one survey add to ~100
    return { bull: bull.value, neutral: neutral.value, bear: bear.value, released: bull.date };
}

/**
 * MacroMicro dates the RELEASE (Thursday); AAII and the dashboard date the survey WEEK by its
 * closing Wednesday ("Sep 30" = released Thu Oct 1). → the latest Wednesday on or before.
 */
export function surveyWeek(released) {
    const d = new Date(`${released}T12:00:00Z`);
    const back = (d.getUTCDay() - 3 + 7) % 7; // Wednesday = 3
    d.setUTCDate(d.getUTCDate() - back);
    return d.toISOString().slice(0, 10);
}

const r1 = (x) => Math.round(x * 10) / 10;

/** The dashboard's AAII payload shape (lib/aaii.js toPayload), AAII's one-decimal convention. */
export function toPayload(stats) {
    const bull = r1(stats.bull), neutral = r1(stats.neutral), bear = r1(stats.bear);
    return {
        bull, neutral, bear,
        diff: `${(bear - bull).toFixed(2)}%`,
        as_of: surveyWeek(stats.released),
        source: 'macromicro',
        stale: false,
    };
}

/** Push only a strictly newer survey week; never overwrite a newer or equal one. */
export function shouldPush(kvValue, payload) {
    const cur = kvValue?.data?.as_of;
    return !cur || payload.as_of > cur;
}
