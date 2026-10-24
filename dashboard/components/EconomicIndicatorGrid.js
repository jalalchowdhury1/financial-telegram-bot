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
    // LME copper can sit well apart from COMEX (tariff spread): say so on the card itself.
    if (m.copperSource === 'westmetall') parts.push('copper: LME');
    return parts.length ? parts.join(' · ') : '↑ = growth/risk-on';
}
function copperGoldTooltip(base, m) {
    if (!m || m.value == null) return base;
    const bits = [];
    if (m.copper != null) bits.push(`Copper $${m.copper.toFixed(2)}/lb${m.copperLabel ? ` (${m.copperLabel})` : ''}`);
    if (m.gold != null) bits.push(`Gold $${Math.round(m.gold)}/oz`);
    const src = m.source ? ` Sources — ${m.source}.` : '';
    return bits.length ? `${base} (${bits.join(', ')}).${src}` : base;
}

/**
 * The Market Valuation row. When the P/E cascade fell all the way to FRED PE10
 * (`peIsCape`), the number is Shiller CAPE — a 10-yr smoothed ratio that runs ~40%
 * above trailing P/E — so it is LABELLED CAPE, judged against CAPE's own norm, and
 * kept out of the since-last-visit mark and the P/E history chart (comparing a
 * CAPE against yesterday's P/E would invent a huge "move").
 */
export function valuationRow(fred) {
    const pe = fred?.peRatio;
    if (fred?.peIsCape && pe) {
        return {
            icon: '💎', label: 'Market Valuation',
            tooltip: 'Shiller CAPE (10-year P/E). P/E sources failed; CAPE runs higher.',
            value: `CAPE ~${pe.toFixed(1)}`,
            status: pe > 30 ? 'restrictive' : 'neutral',
            benchmark: 'CAPE, not P/E · avg ~17',
            metric: { value: pe, asOf: fred.peRatioAsOf, stale: false },
            markKey: null, raw: undefined, fmtPrev: (v) => `CAPE ~${v.toFixed(1)}`,
        };
    }
    return {
        icon: '💎', label: 'Market Valuation', tooltip: 'S&P 500 price ÷ earnings. Higher = pricier.',
        value: pe ? `P/E ~${pe.toFixed(1)}` : 'P/E N/A', status: pe > 25 ? 'restrictive' : 'neutral',
        // computed tier (lib/peRatio.js): S&P ÷ EPS by us, not multpl's own figure — say so
        benchmark: fred?.peSource === 'computed' ? 'Fair at ~20 · computed' : 'Fair at ~20',
        metric: { value: pe ?? null, asOf: fred?.peRatioAsOf, stale: false }, markKey: 'peRatio', raw: pe, fmtPrev: (v) => `P/E ~${v.toFixed(1)}`,
    };
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
                <Delta mark={note.tone === 'fresh' ? mark : null} format={ind.fmtPrev} chartKey={ind.markKey} raw={ind.raw}
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
                            { icon: fred.indicators.sahmRule?.status === 'safe' ? '✅' : '🔴', label: 'Sahm Rule', tooltip: "Fires when 3-month jobless rate is 0.5 over its 12-month low.", value: fred.indicators.sahmRule?.value?.toFixed(2) ?? 'N/A', status: fred.indicators.sahmRule?.status, benchmark: 'Safe under 0.50', metric: fred.indicators.sahmRule, markKey: 'sahmRule', raw: fred.indicators.sahmRule?.value, fmtPrev: (v) => v.toFixed(2) },
                            { icon: '🛒', label: 'Consumer Sentiment', tooltip: "UMich consumer confidence survey.", value: fred.indicators.sentiment?.value?.toFixed(1) ?? 'N/A', status: fred.indicators.sentiment?.status, benchmark: 'Strong above 80', metric: fred.indicators.sentiment, markKey: 'sentiment', raw: fred.indicators.sentiment?.value, fmtPrev: (v) => v.toFixed(1) },
                            { icon: '📋', label: 'Initial Claims (4wk)', tooltip: "New jobless claims, 4-week average.", value: fred.indicators.claims?.value ? `${fred.indicators.claims.value.toFixed(0)}K` : 'N/A', status: fred.indicators.claims?.status, benchmark: 'Healthy under 250K', metric: fred.indicators.claims, markKey: 'claims', raw: fred.indicators.claims?.value, fmtPrev: (v) => `${v.toFixed(0)}K` },
                            { icon: '🏦', label: 'BBB Credit Spread', tooltip: "Extra yield companies pay over Treasuries. Wider = stress.", value: fred.indicators.creditSpread?.value ? `${fred.indicators.creditSpread.value.toFixed(2)}%` : 'N/A', status: fred.indicators.creditSpread?.status, benchmark: 'Calm under 1.5%', metric: fred.indicators.creditSpread, markKey: 'creditSpread', raw: fred.indicators.creditSpread?.value, fmtPrev: (v) => `${v.toFixed(2)}%` },
                            { icon: '💵', label: 'Real Yields (10Y)', tooltip: "10-year TIPS yield: borrowing cost after inflation.", value: fred.indicators.realYields?.value ? `${fred.indicators.realYields.value.toFixed(2)}%` : 'N/A', status: fred.indicators.realYields?.status, benchmark: 'Easy at 0% or below', metric: fred.indicators.realYields, markKey: 'realYields', raw: fred.indicators.realYields?.value, fmtPrev: (v) => `${v.toFixed(2)}%` },
                            { icon: '🟠', label: 'Copper/Gold Ratio', tooltip: copperGoldTooltip("Copper vs gold. Rising = growth mood; falling = risk-off.", fred.indicators.copperGold), value: fred.indicators.copperGold?.value != null ? fred.indicators.copperGold.value.toFixed(2) : 'N/A', status: fred.indicators.copperGold?.status === 'rising' ? 'strong' : (fred.indicators.copperGold?.status === 'falling' ? 'weak' : 'neutral'), benchmark: copperGoldBenchmark(fred.indicators.copperGold), metric: fred.indicators.copperGold, markKey: 'copperGold', raw: fred.indicators.copperGold?.value, fmtPrev: (v) => v.toFixed(2) },
                            valuationRow(fred),
                        ].map(ind => <IndicatorRow key={ind.label} ind={ind} statusColor={statusColor} />)}
                    </>
                )}
            </ErrorBoundary>
        </div>
    );
}
