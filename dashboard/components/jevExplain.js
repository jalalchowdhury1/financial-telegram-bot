/**
 * Plain-English explanations for the Jev pills popup — what each pill measures
 * and what every input means. Pure data; keyed by pill key and by the factor
 * row label emitted by lib/jevBrief.js: pillFactors (keep the labels in sync).
 */

// ask = the one question the pill answers; key = what each verdict needs (best → worst).
// Keep every line short enough to stay on one line on a phone.
export const PILL_EXPLAIN = {
    regime: { ask: 'Is the market in a go mood?', key: { 'risk-on': 'score 2+', 'neutral': 'score 0 or 1', 'risk-off': 'score −1 or less' } },
    recession: { ask: 'How close is a recession?', key: { 'low': 'no warning', 'rising': 'any warning', 'high': 'Sahm 0.5+, or curve + claims' } },
    breadth: { ask: 'Most stocks rising, or just giants?', key: { 'broad': 'avg stock + small caps ahead', 'narrow': 'in between', 'rolling-over': 'avg over 1.5% behind, under 50d' } },
    hedging: { ask: 'How pricey are hedges?', key: { 'cheap': '%ile under 20, markup under 6', 'fair': 'in between', 'expensive': '%ile over 70 or markup over 10' } },
    conflict: { ask: 'Do the signals agree?', key: { 'aligned': 'no splits', 'mild-divergence': 'one split', 'major-divergence': '2+ splits' } },
};

// Row names on screen (the label is the data key; tap a row to see it).
export const INPUT_NAMES = {
    'SPY vs 200-day avg': 'S&P vs 200-day',
    'HYG/LQD 20d': 'Junk vs safe bonds',
    'Yield curve (2s10s)': 'Yield curve',
    'NFCI': 'Money tightness',
    'RSP/SPY 20d': 'Avg stock vs S&P',
    'RSP/SPY vs 50d avg': 'Avg stock vs 50-day',
    'IWM/SPY 20d': 'Small caps vs S&P',
    'IV percentile (1y)': 'Option price %ile',
    'VRP': 'Hedge markup',
    'sentiment vs price': 'Mood vs price',
    '2s10s vs 3m10y': 'Two yield curves',
    'credit vs equities': 'Junk bonds vs stocks',
    'breadth vs index': 'Avg stock vs index',
};

// Shown when a row is tapped.
export const INPUT_EXPLAIN = {
    'SPY vs 200-day avg': 'Above = uptrend.',
    'Fear & Greed': 'CNN crowd mood, 0–100.',
    'HYG/LQD 20d': 'Junk ahead = risk appetite.',
    'Sahm rule': 'Jobless rate vs its 1-year low.',
    'Yield curve (2s10s)': '10y minus 2y. Below 0 = inverted.',
    'Jobless claims': 'New weekly claims.',
    'NFCI': 'Chicago Fed. Above 0 = tight.',
    'RSP/SPY 20d': 'Equal-weight vs normal S&P, 20 days.',
    'RSP/SPY vs 50d avg': 'That ratio vs its 50-day.',
    'IWM/SPY 20d': 'Russell 2000 vs S&P, 20 days.',
    'IV percentile (1y)': 'vs the past year, 0–100.',
    'VRP': 'Expected minus actual moves.',
    'sentiment vs price': 'Mood fighting the trend.',
    '2s10s vs 3m10y': 'One inverted, one not.',
    'credit vs equities': 'S&P up, junk lagging safe bonds.',
    'breadth vs index': 'Index near high, avg stock lagging.',
};

// Every feed a pill's rule actually reads (keep in sync with ruleVerdicts / conflictPairs).
export const PILL_FEEDS = {
    regime: ['spy', 'fg', 'breadth'],
    recession: ['fred'],
    breadth: ['breadth'],
    hedging: ['vol'],
    conflict: ['spy', 'fg', 'fred', 'breadth'],
};
