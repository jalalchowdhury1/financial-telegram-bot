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
 *
 * Two honesty rules (review, QoL ship 6):
 *  - An input that is n/a is NOT a calm reading. Counts use only the rows that were measured
 *    and name the rest ("0 of 3 warnings tripped (NFCI n/a)"); nothing measured -> null.
 *  - The rows are the RULE's. If the badge shows a different verdict (Jev at p >= 0.6 overrode
 *    it, lib/jevBrief.js: mergeVerdicts), the line says so: "Rule says risk-on · …". A rule
 *    badge the rows do not support (mismatched payload) gets no line at all.
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
        // A ratio: HYG can fall and still "beat" LQD, so never say junk bonds rose or fell.
        'HYG/LQD 20d': (r) => ({ 1: 'junk bonds beating safe bonds', [-1]: 'junk bonds lagging safe bonds' })[vote(r.effect)] || null,
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
            // "fair" can still sit low or high in the year (cheap also needs VRP < 6): state the fact.
            return `options at the ${ordinal(v)} percentile of the year`;
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
        'credit vs equities': () => 'junk bonds lagging safe bonds in an uptrend',
        'breadth vs index': () => 'near the high, average stock slipping',
    },
};

// Short names for an n/a input inside the line ("(F&G n/a)"). Unlisted -> the row label.
const SHORT = {
    'SPY vs 200-day avg': 'SPY trend', 'Fear & Greed': 'F&G', 'HYG/LQD 20d': 'HYG/LQD',
    'Sahm rule': 'Sahm', 'Yield curve (2s10s)': 'curve', 'Jobless claims': 'claims',
};

// A row was measured when it fired, or when its shown value has no 'n/a' in it
// (pillFactors prints 'n/a' for every missing number, including inside the conflict pairs).
const measured = (r) => r.hit || (r.value != null && String(r.value).trim() !== '' && !/n\/a/i.test(String(r.value)));
const naNote = (rows) => {
    const gone = rows.filter((r) => !measured(r)).map((r) => SHORT[r.label] || r.label);
    return gone.length ? ` (${gone.join(', ')} n/a)` : '';
};

// The verdict the rows themselves give — the same thresholds as pillFactors / ruleVerdicts.
const RULE = {
    regime(rows) {
        const score = rows.reduce((n, r) => n + vote(r.effect), 0);
        return score >= 2 ? 'risk-on' : score <= -1 ? 'risk-off' : 'neutral';
    },
    recession(rows) {
        const fired = rows.filter((r) => r.hit);
        return fired.some((r) => r.effect === 'high') ? 'high' : fired.length ? 'rising' : 'low';
    },
    breadth(rows) {
        const e = rows.filter((r) => r.hit).map((r) => r.effect);
        return e.includes('rolling-over') ? 'rolling-over' : e.includes('broad') ? 'broad' : 'narrow';
    },
    hedging(rows) {
        const e = rows.filter((r) => r.hit).map((r) => r.effect);
        return e.includes('expensive') ? 'expensive' : e.includes('cheap') ? 'cheap' : 'fair';
    },
    conflict(rows) {
        const n = rows.filter((r) => r.hit).length;
        return n === 0 ? 'aligned' : n === 1 ? 'mild-divergence' : 'major-divergence';
    },
};
const spoken = (v) => String(v).replace(/^(rolling|mild|major)-/, '$1 ');

/** Row labels each pill has a phrase for (the test pins these to pillFactors). */
export const WHY_LABELS = Object.fromEntries(Object.entries(PHRASES).map(([k, v]) => [k, Object.keys(v)]));

// Per-pill composition from the rows. `say(row)` = that row's phrase (undefined = no template).
const COMPOSE = {
    regime(rows, say) {
        const known = rows.filter(measured).length;
        if (!known) return null;
        const up = rows.filter((r) => vote(r.effect) === 1).map(say);
        const down = rows.filter((r) => vote(r.effect) === -1).map(say);
        return `${up.length} of ${known} votes${up.length ? `: ${up.join(' + ')}` : ''}`
            + `${down.length ? ` · against: ${down.join(' + ')}` : ''}${naNote(rows)}`;
    },
    recession(rows, say) {
        const known = rows.filter(measured).length;
        if (!known) return null;
        const fired = rows.filter((r) => r.hit).map(say);
        return `${fired.length} of ${known} warnings tripped${fired.length ? `: ${fired.join(' + ')}` : ''}${naNote(rows)}`;
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
        if (fired.length) return fired.map(say).join(' · ');
        const known = rows.filter(measured).length;
        const gone = rows.length - known;
        if (!known) return null;
        return `all ${known} pairs agree${gone ? ` (${gone} pair${gone > 1 ? 's' : ''} n/a)` : ''}`;
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
    // Does the badge show the verdict these rows give? If not, only a Jev override may keep
    // the line, and then it is labelled as the rule's.
    const shown = data?.pills?.[pill];
    const ruleSays = RULE[pill](rows);
    const overridden = !!shown?.verdict && shown.verdict !== ruleSays;
    if (overridden && shown.by !== 'jev') return null;

    const line = compose(rows, say);
    if (missing) {
        if (overridden) return null;            // the summary/reason are the rule's prose too
        const fallback = f.summary || shown?.reason;
        return typeof fallback === 'string' && fallback ? fallback : null;
    }
    if (!line) return null;
    return overridden ? `Rule says ${spoken(ruleSays)} · ${line}` : cap(line);
}
