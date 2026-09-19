/**
 * Plain-English explanations for the Jev pills popup — what each pill measures
 * and what every input means. Pure data; keyed by pill key and by the factor
 * row label emitted by lib/jevBrief.js: pillFactors (keep the labels in sync).
 */

export const PILL_EXPLAIN = {
    regime:
        'Is the market in a "go" (risk-on) or "no-go" (risk-off) mood? Three yes/no votes: is SPY above its 200-day average, is the Fear & Greed crowd optimistic, and are junk bonds beating safe bonds. Add the votes: 2 or more = risk-on, −1 or worse = risk-off, in between = neutral.',
    recession:
        'How close the economy is to a recession, using four classic early warnings: the Sahm rule (unemployment rising fast), an inverted yield curve, weekly jobless claims, and the Fed’s financial-conditions index. Checked in order; the first warning that trips sets the verdict. None tripped = low.',
    breadth:
        'Is the rally carried by many stocks or just a few giants? Compares the equal-weight S&P (RSP) and small caps (IWM) with the cap-weighted S&P (SPY). Average stock and small caps both gaining ground = broad. Average stock losing ground and below its 50-day trend = rolling over. Anything else = narrow.',
    hedging:
        'What portfolio insurance costs right now. Uses SPY implied volatility (the price of options) against its own 1-year range, plus the VRP: how much more options cost than the market actually moves. Low on both = cheap to hedge; high on either = expensive.',
    conflict:
        'Do the signals agree with each other? Four pairs are checked for contradictions, like a fearful crowd while price trends up. 0 contradictions = aligned, 1 = mild divergence, 2 or more = major divergence. Divergences often show up before a regime change.',
};

export const INPUT_EXPLAIN = {
    'SPY vs 200-day avg': 'How far SPY sits above or below its 200-day average price. Above = uptrend.',
    'Fear & Greed': 'CNN’s 0–100 crowd-mood gauge. Under 30 = fear, 50 and over = greed.',
    'HYG/LQD 20d': 'Junk bonds (HYG) vs high-quality bonds (LQD), change over 20 trading days. Junk winning = investors comfortable with risk; junk lagging = credit getting nervous. The two usually move together, so small numbers are normal.',
    'Sahm rule': 'Unemployment rate now vs its low of the past year. 0.5 or more has marked every US recession since 1970.',
    'Yield curve (2s10s) + claims': '10-year minus 2-year Treasury yield, together with weekly jobless claims. Inverted curve plus rising claims = the classic recession pair.',
    'Yield curve (2s10s)': '10-year minus 2-year Treasury yield. Below zero (inverted) has preceded most recessions.',
    'Jobless claims': 'Weekly first-time unemployment claims, in thousands. 260k and up = layoffs picking up.',
    'NFCI': 'Chicago Fed financial-conditions index. Above zero = money is tighter than average.',
    'RSP/SPY 20d': 'Equal-weight S&P vs cap-weight S&P over 20 trading days. Rising = the average stock is keeping up with the giants.',
    'RSP/SPY vs 50d avg': 'The same ratio vs its own 50-day average. Below = the average stock has been losing ground for weeks.',
    'IWM/SPY 20d': 'Small caps vs the S&P over 20 trading days. Rising = small companies are joining the rally.',
    'IV percentile (1y)': 'Where today’s SPY implied volatility ranks within the past year (0–100). Low = options are cheap by recent standards.',
    'VRP': 'Volatility risk premium: implied volatility minus realized volatility. High = option sellers are charging a fat premium, so hedges are overpriced.',
    'sentiment vs price': 'Crowd mood vs the price trend: a fearful crowd in an uptrend, or a greedy crowd in a downtrend.',
    '2s10s vs 3m10y': 'Two yield-curve measures disagreeing on whether the curve is inverted.',
    'credit vs equities': 'Stocks in an uptrend while junk bonds sell off. Credit often sees trouble first.',
    'breadth vs index': 'Index near its high while the average stock falls behind. A rally getting thin.',
};

/** Friendly names + which feeds each pill depends on, for the "data through" line. */
export const FEED_NAMES = { breadth: 'ETF ratios', vol: 'volatility', fred: 'FRED' };
export const PILL_FEEDS = {
    regime: ['breadth'],
    recession: ['fred'],
    breadth: ['breadth'],
    hedging: ['vol'],
    conflict: ['fred', 'breadth'],
};
