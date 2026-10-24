/**
 * Plain-English explanations for the Jev pills popup — what each pill measures
 * and what every input means. Pure data; keyed by pill key and by the factor
 * row label emitted by lib/jevBrief.js: pillFactors (keep the labels in sync).
 */

// ask = the one question the pill answers; key = what each verdict needs (best → worst).
export const PILL_EXPLAIN = {
    regime: {
        ask: 'Is the market in a go or no-go mood? Each input votes +1 or −1.',
        key: { 'risk-on': 'score 2 or more', 'neutral': 'score 0 or 1', 'risk-off': 'score −1 or less' },
    },
    recession: {
        ask: 'How close is a recession? Four early warnings; the strongest wins.',
        key: { 'low': 'no warning tripped', 'rising': 'any one warning tripped', 'high': 'Sahm 0.5+, or curve below 0 with claims 260k+' },
    },
    breadth: {
        ask: 'Are most stocks rising, or only a few giants?',
        key: { 'broad': 'average stock and small caps both beat SPY', 'narrow': 'anything in between', 'rolling-over': 'average stock trails SPY by 1.5%+ and is below its 50-day' },
    },
    hedging: {
        ask: 'How pricey is portfolio insurance (options) right now?',
        key: { 'cheap': 'IV under 20 and VRP under 6', 'fair': 'anything in between', 'expensive': 'IV over 70 or VRP over 10' },
    },
    conflict: {
        ask: 'Do the signals agree? Splits often come before a regime change.',
        key: { 'aligned': 'no pair disagrees', 'mild-divergence': '1 pair disagrees', 'major-divergence': '2 or more disagree' },
    },
};

export const INPUT_EXPLAIN = {
    'SPY vs 200-day avg': 'Above its 200-day = uptrend.',
    'Fear & Greed': 'CNN crowd mood, 0–100.',
    'HYG/LQD 20d': 'Junk vs safe bonds. Junk ahead = appetite for risk.',
    'Sahm rule': 'Jobless rate vs its 1-year low.',
    'Yield curve (2s10s)': '10y minus 2y yield. Below 0 = inverted.',
    'Jobless claims': 'New weekly unemployment claims.',
    'NFCI': 'Chicago Fed gauge. Above 0 = tight money.',
    'RSP/SPY 20d': 'Average stock vs the giants, 20 days.',
    'RSP/SPY vs 50d avg': 'Same ratio vs its 50-day average.',
    'IWM/SPY 20d': 'Small caps vs the S&P, 20 days.',
    'IV percentile (1y)': 'Option prices vs the past year, 0–100.',
    'VRP': 'Option-implied minus actual moves. High = hedges overpriced.',
    'sentiment vs price': 'Crowd mood fighting the trend.',
    '2s10s vs 3m10y': 'Two yield curves split on inversion.',
    'credit vs equities': 'Stocks up while junk bonds sell off.',
    'breadth vs index': 'Index near its high, average stock lagging.',
};

/** Friendly names + which feeds each pill depends on, for the "data through" line. */
export const FEED_NAMES = { spy: 'SPY', fg: 'Fear & Greed', breadth: 'ETF ratios', vol: 'volatility', fred: 'FRED' };
// Every feed a pill's rule actually reads (keep in sync with ruleVerdicts / conflictPairs).
export const PILL_FEEDS = {
    regime: ['spy', 'fg', 'breadth'],
    recession: ['fred'],
    breadth: ['breadth'],
    hedging: ['vol'],
    conflict: ['spy', 'fg', 'fred', 'breadth'],
};
