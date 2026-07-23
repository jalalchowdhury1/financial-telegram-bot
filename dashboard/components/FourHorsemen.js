'use client';
import ErrorBoundary from './ErrorBoundary';
import MiniChart from './MiniChart';
import Skeleton from './Skeleton';
import { freshnessNote, formatAsOf } from '../lib/freshness';

/**
 * 🐎 Four Horsemen — Recession Watch. Four classic recession tells on one
 * full-width card, each with its full history + NBER recession shading:
 *   1. Initial Jobless Claims (FRED ICSA, weekly)
 *   2. Unemployment Rate      (FRED UNRATE, monthly)
 *   3. 10Y−2Y Yield Spread    (FRED T10Y2Y, daily — reuses fred.yieldCurve)
 *   4. US Bankruptcies        (AOUSC F-2, quarterly 12-mo business filings)
 * Data all rides in on the /api/fred payload (`horsemen` block + yieldCurve).
 */

const kFmt = (v) => {
    if (v == null || !Number.isFinite(v)) return 'N/A';
    const a = Math.abs(v);
    if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
    if (a >= 1e3) return `${Math.round(v / 1e3)}K`;
    return `${Math.round(v)}`;
};
const pctFmt = (v) => (v == null || !Number.isFinite(v) ? 'N/A' : `${v.toFixed(2)}%`);

// Latest-vs-≈1-year-ago percent change from an ascending history array.
function yoyPct(history, pointsPerYear) {
    if (!history || history.length <= pointsPerYear) return null;
    const now = history[history.length - 1]?.value;
    const ago = history[history.length - 1 - pointsPerYear]?.value;
    if (now == null || ago == null || ago === 0) return null;
    return ((now - ago) / Math.abs(ago)) * 100;
}

function Panel({ title, tooltip, metric, color, gradientId, chart, headline, chip, warn }) {
    const note = freshnessNote(metric);
    const unavailable = metric?.value == null && metric?.current == null;
    return (
        <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '8px', marginBottom: '2px' }}>
                <span className="tooltip-trigger" data-tooltip={`${tooltip}${note.suffix}`}
                    style={{ fontSize: '0.72rem', fontWeight: 700, color, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    {title}
                </span>
                {warn != null && (
                    <span className={`badge ${warn.bad ? 'badge-red' : 'badge-green'}`} style={{ fontSize: '0.58rem', whiteSpace: 'nowrap' }}>{warn.label}</span>
                )}
            </div>
            {unavailable ? (
                <div style={{ padding: '24px 0', color: 'var(--yellow)', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.9rem' }}>
                    N/A <span style={{ color: 'var(--text-muted)', fontSize: '0.68rem' }}>— source busy, try again shortly</span>
                </div>
            ) : (
                <>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px', marginBottom: '4px' }}>
                        <span style={{ fontSize: '1.35rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: metric?.stale ? 'var(--orange)' : 'var(--text)' }}>
                            {metric?.stale ? '🕐 ' : ''}{headline}
                        </span>
                        {chip && (
                            <span style={{ fontSize: '0.66rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: chip.bad ? 'var(--red)' : 'var(--green)' }}>
                                {chip.text}
                            </span>
                        )}
                    </div>
                    {metric?.stale && (
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.66rem', marginBottom: '4px' }}>
                            Last data {formatAsOf(metric.asOf)} (stale)
                        </div>
                    )}
                    {chart}
                </>
            )}
        </div>
    );
}

export default function FourHorsemen({ fred, loading }) {
    const recessions = fred?.recessions || [];
    const claims = fred?.horsemen?.claims;
    const unemployment = fred?.horsemen?.unemployment;
    const spread = fred?.yieldCurve;
    const bk = fred?.horsemen?.bankruptcies;
    const sahm = fred?.indicators?.sahmRule?.value;

    const claimsYoy = yoyPct(claims?.history, 52);
    const unempYoy = unemployment?.history?.length > 13
        ? unemployment.history[unemployment.history.length - 1].value - unemployment.history[unemployment.history.length - 13].value
        : null;

    const panels = !loading && fred && !fred.error ? [
        {
            key: 'claims', title: 'Initial Jobless Claims', color: '#ef4444', gradientId: 'horseClaims',
            tooltip: 'Weekly first-time unemployment filings. A sustained climb is the earliest of the four warnings.',
            metric: claims,
            headline: kFmt(claims?.current),
            chip: claimsYoy != null ? { text: `${claimsYoy >= 0 ? '▲' : '▼'} ${Math.abs(claimsYoy).toFixed(1)}% vs 1y`, bad: claimsYoy > 0 } : null,
            warn: claimsYoy != null ? (claimsYoy > 10 ? { bad: true, label: 'Rising' } : { bad: false, label: 'Contained' }) : null,
            chart: <MiniChart history={claims?.history} color="#ef4444" gradientId="horseClaims" recessions={recessions} cadence="weekly" defaultTimeframe="ALL" fmt={kFmt} />,
        },
        {
            key: 'unemployment', title: 'Unemployment Rate', color: '#22c55e', gradientId: 'horseUnemp',
            tooltip: 'U-3 unemployment rate. The Sahm rule triggers when its 3-month average rises 0.5pp off the 12-month low.',
            metric: unemployment,
            headline: pctFmt(unemployment?.current),
            chip: unempYoy != null ? { text: `${unempYoy >= 0 ? '▲' : '▼'} ${Math.abs(unempYoy).toFixed(1)}pp vs 1y`, bad: unempYoy > 0 } : null,
            warn: sahm != null ? (sahm >= 0.5 ? { bad: true, label: `Sahm ${sahm.toFixed(2)}` } : { bad: false, label: `Sahm ${sahm.toFixed(2)}` }) : null,
            chart: <MiniChart history={unemployment?.history} color="#22c55e" gradientId="horseUnemp" recessions={recessions} cadence="monthly" defaultTimeframe="ALL" />,
        },
        {
            key: 'spread', title: '10Y − 2Y Yield Spread', color: '#3b82f6', gradientId: 'horseSpread',
            tooltip: 'The yield curve. Inversion (below zero) precedes recessions; the re-steepening back through zero is often the late-cycle tell.',
            metric: { value: spread?.current, asOf: spread?.asOf, stale: spread?.stale, unavailable: spread?.current == null, current: spread?.current },
            headline: spread?.current != null ? `${spread.current >= 0 ? '+' : ''}${spread.current.toFixed(2)}%` : 'N/A',
            chip: spread?.current != null ? { text: spread.current >= 0 ? 'Positive' : 'Inverted', bad: spread.current < 0 } : null,
            warn: spread?.current != null ? (spread.current < 0 ? { bad: true, label: 'Inverted' } : { bad: false, label: 'Normal' }) : null,
            chart: <MiniChart history={spread?.history} color="#3b82f6" gradientId="horseSpread" showZero={true} recessions={recessions} defaultTimeframe="ALL" />,
        },
        {
            key: 'bankruptcies', title: 'US Bankruptcies', color: '#e2e8f0', gradientId: 'horseBk',
            tooltip: 'Business bankruptcy filings, 12-month total ending each quarter (Administrative Office of the U.S. Courts, Table F-2).',
            metric: { value: bk?.current, asOf: bk?.asOf, stale: bk?.stale, unavailable: bk?.unavailable, current: bk?.current },
            headline: kFmt(bk?.current),
            chip: bk?.changePct != null ? { text: `${bk.changePct >= 0 ? '▲' : '▼'} ${Math.abs(bk.changePct).toFixed(1)}% YoY`, bad: bk.changePct > 0 } : null,
            warn: bk?.changePct != null ? (bk.changePct > 10 ? { bad: true, label: 'Rising' } : { bad: false, label: 'Contained' }) : null,
            chart: <MiniChart history={bk?.history} color="#e2e8f0" gradientId="horseBk" recessions={recessions} fmt={kFmt} />,
        },
    ] : [];

    const riding = panels.filter((p) => p.warn?.bad).length;

    return (
        <div className="card" style={{ gridColumn: '1 / -1', animationDelay: '0.5s' }}>
            <div className="card-header">
                <h2><span className="tooltip-trigger" data-tooltip="Four classic recession tells on one chart wall: jobless claims, unemployment, the yield curve, and bankruptcies. Shaded bands are NBER recessions — note how all four turn together going into them.">🐎 Four Horsemen — Recession Watch</span></h2>
                {!loading && panels.length > 0 && (
                    <span className={`badge ${riding >= 3 ? 'badge-red' : riding >= 1 ? 'badge-yellow' : 'badge-green'}`}>
                        {riding} of 4 riding
                    </span>
                )}
            </div>
            <ErrorBoundary>
                {loading || !fred ? <Skeleton count={4} /> : panels.length === 0 ? (
                    <div className="hero-price-section">
                        <div className="hero-price" style={{ fontSize: '2.2rem', color: 'var(--yellow)' }}>N/A</div>
                        <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                            Unavailable — source busy, try again shortly
                        </div>
                    </div>
                ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '18px 28px' }}>
                        {panels.map((p) => <Panel key={p.key} {...p} />)}
                    </div>
                )}
            </ErrorBoundary>
        </div>
    );
}
