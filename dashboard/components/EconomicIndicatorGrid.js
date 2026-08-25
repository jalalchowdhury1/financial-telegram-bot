'use client';
import ErrorBoundary from './ErrorBoundary';
import Skeleton from './Skeleton';
import Delta from './Delta';
import { useMark } from './MarkProvider';
import { freshnessNote } from '../lib/freshness';

// Copper/Gold sub-line: show the TREND (what the owner cares about) — the ~1-month
// and ~3-month change with a direction arrow — falling back to the static benchmark
// when no change could be computed.
const cgArrow = (v) => (v > 0 ? '▲' : v < 0 ? '▼' : '→');
const cgPct = (v) => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`;
function copperGoldBenchmark(m) {
    if (!m || m.value == null) return '↑ = growth/risk-on';
    const parts = [];
    if (m.changePct != null) parts.push(`${cgArrow(m.changePct)} ${cgPct(m.changePct)} 1mo`);
    if (m.changePct3mo != null) parts.push(`${cgArrow(m.changePct3mo)} ${cgPct(m.changePct3mo)} 3mo`);
    return parts.length ? parts.join(' · ') : '↑ = growth/risk-on';
}
function copperGoldTooltip(base, m) {
    if (!m || m.value == null) return base;
    const bits = [];
    if (m.copper != null) bits.push(`Copper $${m.copper.toFixed(2)}/lb`);
    if (m.gold != null) bits.push(`Gold $${Math.round(m.gold)}/oz`);
    const src = m.source ? ` Sources — ${m.source}.` : '';
    return bits.length ? `${base} (${bits.join(', ')}).${src}` : base;
}

/**
 * One indicator row. Split out of the map so it can call `useMark` — hooks cannot
 * run inside a callback. `markKey` is the lib/marks.js key; rows without one are
 * never marked (P/E is a move metric, the rest of the grid is print or none).
 */
function IndicatorRow({ ind, statusColor }) {
    const mark = useMark(ind.markKey, ind.raw);
    const note = freshnessNote(ind.metric);
    const staleStyle = note.tone === 'stale' ? { color: 'var(--orange)' }
        : note.tone === 'unavailable' ? { color: 'var(--yellow)' }
        : undefined;
    const shownValue = note.tone === 'stale' ? `\u{1F550} ${ind.value}` : ind.value;
    return (
        <div className="stat-row">
            <span className="stat-label">
                {ind.icon} <span className="tooltip-trigger" data-tooltip={`${ind.tooltip}${note.suffix}`}>{ind.label}</span>
            </span>
            <span className="stat-right">
                <Delta mark={note.tone === 'fresh' ? mark : null} format={ind.fmtPrev}
                    className={`stat-value ${statusColor(ind.status)}`}>
                    <span style={staleStyle}>{shownValue}</span>
                </Delta>
                <span className="stat-benchmark">{ind.benchmark}</span>
            </span>
        </div>
    );
}

export default function EconomicIndicatorGrid({ fred, loading, statusColor }) {
    return (
        <div className="card" style={{ animationDelay: '0.5s' }}>
            <div className="card-header">
                <h2>📊 Economic Indicators</h2>
            </div>
            <ErrorBoundary>
                {loading || !fred || fred.error ? <Skeleton count={6} /> : (
                    <>
                        {[
                            { icon: fred.indicators.sahmRule?.status === 'safe' ? '✅' : '🔴', label: 'Sahm Rule', tooltip: "Recession indicator: triggers if 3mo average unemployment rises 0.5% above its 12mo low.", value: fred.indicators.sahmRule?.value?.toFixed(2) ?? 'N/A', status: fred.indicators.sahmRule?.status, benchmark: '< 0.50', metric: fred.indicators.sahmRule, markKey: 'sahmRule', raw: fred.indicators.sahmRule?.value, fmtPrev: (v) => v.toFixed(2) },
                            { icon: '🛒', label: 'Consumer Sentiment', tooltip: "University of Michigan survey assessing consumer confidence.", value: fred.indicators.sentiment?.value?.toFixed(1) ?? 'N/A', status: fred.indicators.sentiment?.status, benchmark: '> 80 strong', metric: fred.indicators.sentiment, markKey: 'sentiment', raw: fred.indicators.sentiment?.value, fmtPrev: (v) => v.toFixed(1) },
                            { icon: '📋', label: 'Initial Claims (4wk)', tooltip: "4-week moving average of initial jobless claims. A critical real-time labor market gauge.", value: fred.indicators.claims?.value ? `${fred.indicators.claims.value.toFixed(0)}K` : 'N/A', status: fred.indicators.claims?.status, benchmark: '< 250K healthy', metric: fred.indicators.claims, markKey: 'claims', raw: fred.indicators.claims?.value, fmtPrev: (v) => `${v.toFixed(0)}K` },
                            { icon: '🏦', label: 'BBB Credit Spread', tooltip: "The premium corporations pay over Treasuries to borrow. Widening indicates market stress.", value: fred.indicators.creditSpread?.value ? `${fred.indicators.creditSpread.value.toFixed(2)}%` : 'N/A', status: fred.indicators.creditSpread?.status, benchmark: '< 1.5% tight', metric: fred.indicators.creditSpread, markKey: 'creditSpread', raw: fred.indicators.creditSpread?.value, fmtPrev: (v) => `${v.toFixed(2)}%` },
                            { icon: '💵', label: 'Real Yields (10Y TIPS)', tooltip: "10-Year Treasury Inflation-Indexed Security. Shows the true inflation-adjusted cost of capital.", value: fred.indicators.realYields?.value ? `${fred.indicators.realYields.value.toFixed(2)}%` : 'N/A', status: fred.indicators.realYields?.status, benchmark: '< 0% easy', metric: fred.indicators.realYields, markKey: 'realYields', raw: fred.indicators.realYields?.value, fmtPrev: (v) => `${v.toFixed(2)}%` },
                            { icon: '🟠', label: 'Copper/Gold Ratio', tooltip: copperGoldTooltip("Dr. Copper vs. safe-haven gold — a leading gauge of growth & rate expectations. Rising = risk-on/expansion; falling = risk-off.", fred.indicators.copperGold), value: fred.indicators.copperGold?.value != null ? fred.indicators.copperGold.value.toFixed(2) : 'N/A', status: fred.indicators.copperGold?.status === 'rising' ? 'strong' : (fred.indicators.copperGold?.status === 'falling' ? 'weak' : 'neutral'), benchmark: copperGoldBenchmark(fred.indicators.copperGold), metric: fred.indicators.copperGold, markKey: 'copperGold', raw: fred.indicators.copperGold?.value, fmtPrev: (v) => v.toFixed(2) },
                            { icon: '💎', label: 'Market Valuation', tooltip: "Current S&P 500 P/E Ratio. A measure of how expensive the market is historically.", value: fred.peRatio ? `P/E ~${fred.peRatio.toFixed(1)}` : 'P/E N/A', status: fred.peRatio > 25 ? 'restrictive' : 'neutral', benchmark: 'Fair at ~20', metric: { value: fred.peRatio ?? null, asOf: fred.peRatioAsOf, stale: false }, markKey: 'peRatio', raw: fred.peRatio, fmtPrev: (v) => `P/E ~${v.toFixed(1)}` },
                        ].map(ind => <IndicatorRow key={ind.label} ind={ind} statusColor={statusColor} />)}
                    </>
                )}
            </ErrorBoundary>
        </div>
    );
}
