/**
 * One plain-English "why" line per Jev pill, shown under the verdict (JevPills.js).
 *
 * The reason used to live only in a hover `title`, which an iPhone never shows. This line
 * is built ONLY from the payload's own factor rows (`factors[pill].rows`, from
 * lib/jevBrief.js: pillFactors): the rows that fired, with their real values. No prose is
 * hard-coded that could disagree with the numbers. Phrases are keyed by row label, so a
 * label rename in pillFactors must be mirrored here (jevWhy.test.js fails until it is).
 * A fired row with no phrase falls back to factors[pill].summary, then pills[pill].reason,
 * then nothing. Missing rows or an n/a number -> null, never a guess.
 */

const num = (v) => {
    const n = parseFloat(String(v ?? '').replace('−', '-'));
    return Number.isFinite(n) ? n : null;
};
// Display with a true minus sign, like the rest of the page.
const minus = (v) => String(v).replace(/^-/, '−');
const vote = (e) => (e === '+1' ? 1 : e === '−1' || e === '-1' ? -1 : 0);
const ordinal = (n) => {
    const r = Math.round(n);
    const t = r % 100;
    const suffix = t >= 11 && t <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' })[r % 10] || 'th';
    return `${r}${suffix}`;
};
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

// Phrase per row label. Each returns a string, or null when its number is missing.
const PHRASES = {
    regime: {
        'SPY vs 200-day avg': (r) => ({ 1: 'uptrend', [-1]: 'downtrend' })[vote(r.effect)] || null,
        'Fear & Greed': (r) => ({ 1: 'greedy crowd', [-1]: 'fearful crowd' })[vote(r.effect)] || null,
        'HYG/LQD 20d': (r) => ({ 1: 'junk bonds firm', [-1]: 'junk bonds slipping' })[vote(r.effect)] || null,
    },
    recession: {
        'Sahm rule': (r) => (num(r.value) == null ? null : `Sahm ${r.value}`),
        'Yield curve (2s10s)': (r) => (num(r.value) == null ? null : `curve inverted (${minus(r.value)})`),
        'Jobless claims': (r) => (num(r.value) == null ? null : `claims ${r.value}`),
        NFCI: (r) => (num(r.value) == null ? null : `money tight (NFCI ${minus(r.value)})`),
    },
    breadth: {
        'RSP/SPY 20d': (r) => (num(r.value) == null ? null : `average stock ${minus(r.value)} vs SPY in 20 days`),
        'RSP/SPY vs 50d avg': () => 'below its 50-day trend',
        'IWM/SPY 20d': (r) => (num(r.value) == null ? null : `small caps ${minus(r.value)}`),
    },
    hedging: {
        'IV percentile (1y)': (r) => {
            const v = num(r.value);
            if (v == null) return null;
            if (r.effect === 'cheap') return `options cheap: bottom ${v}% of the year`;
            if (r.effect === 'expensive') return `options pricey: top ${100 - v}% of the year`;
            return `options mid-priced: ${ordinal(v)} percentile of the year`;
        },
        VRP: (r) => (num(r.value) == null ? null : `options cost ${r.value} pts over real moves`),
    },
    conflict: {
        'sentiment vs price': (r) => {
            const fg = num((String(r.value).match(/F&G\s*(-?[\d.]+)/) || [])[1]);
            if (fg == null) return 'crowd mood fights the trend';
            return fg < 50 ? 'fearful crowd in an uptrend' : 'greedy crowd in a downtrend';
        },
        '2s10s vs 3m10y': () => 'two yield curves disagree',
        'credit vs equities': () => 'junk bonds slipping in an uptrend',
        'breadth vs index': () => 'near the high, average stock slipping',
    },
};

/** Row labels each pill has a phrase for (the test pins these to pillFactors). */
export const WHY_LABELS = Object.fromEntries(Object.entries(PHRASES).map(([k, v]) => [k, Object.keys(v)]));

// Per-pill composition from the rows. `say(row)` = that row's phrase (undefined = no template).
const COMPOSE = {
    regime(rows, say) {
        const up = rows.filter((r) => vote(r.effect) === 1).map(say);
        const down = rows.filter((r) => vote(r.effect) === -1).map(say);
        return `${up.length} of ${rows.length} votes${up.length ? `: ${up.join(' + ')}` : ''}`
            + `${down.length ? ` · against: ${down.join(' + ')}` : ''}`;
    },
    recession(rows, say) {
        const fired = rows.filter((r) => r.hit).map(say);
        return `${fired.length} of ${rows.length} warnings tripped${fired.length ? `: ${fired.join(' + ')}` : ''}`;
    },
    breadth(rows, say) {
        const fired = rows.filter((r) => r.hit);
        // Nothing fired ("narrow"): the lead fact alone, if its number exists.
        if (!fired.length) return say(rows.find((r) => r.label === 'RSP/SPY 20d')) || null;
        return fired.map(say).join(' · ');
    },
    hedging(rows, say) {
        const iv = rows.find((r) => r.label === 'IV percentile (1y)');
        const vrp = rows.find((r) => r.label === 'VRP');
        // Cheap needs both rows; the IV line says it. Expensive fires on either.
        if (iv?.hit) return say(iv);
        if (vrp?.hit) return say(vrp);
        const unknown = rows.find((r) => r.hit);
        if (unknown) return say(unknown);
        return iv ? say(iv) : null;     // "fair": where prices sit in their year
    },
    conflict(rows, say) {
        const fired = rows.filter((r) => r.hit);
        if (!fired.length) return `all ${rows.length} pairs agree`;
        return fired.map(say).join(' · ');
    },
};

export function pillWhy(data, pill) {
    const f = data?.factors?.[pill];
    const rows = Array.isArray(f?.rows) ? f.rows.filter((r) => r && typeof r.label === 'string') : [];
    const compose = COMPOSE[pill];
    if (!rows.length || !compose) return null;
    const phrases = PHRASES[pill];
    let missing = false;
    const say = (r) => {
        if (!r) return null;
        const fn = phrases[r.label];
        if (!fn) { missing = true; return ''; }
        const s = fn(r);
        if (s == null && r.hit) missing = true;   // a fired row we cannot word honestly
        return s;
    };
    const line = compose(rows, say);
    if (missing) {
        const fallback = f.summary || data?.pills?.[pill]?.reason;
        return typeof fallback === 'string' && fallback ? fallback : null;
    }
    return line ? cap(line) : null;
}
