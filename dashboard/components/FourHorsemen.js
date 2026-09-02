'use client';
import { useEffect, useState } from 'react';
import ErrorBoundary from './ErrorBoundary';
import RunupBars from './HorsemenRunup';
import Skeleton from './Skeleton';
import Delta from './Delta';
import { useMark } from './MarkProvider';
import { freshnessNote, formatAsOf } from '../lib/freshness';

/**
 * 🐎 Four Horsemen — Recession Watch. ONE overlay chart (like the classic
 * "Four Horsemen of the Apocalypse" chart) with all four recession tells:
 *   1. Initial Jobless Claims (FRED ICSA, weekly)      — red, top band
 *   2. Unemployment Rate      (FRED UNRATE, monthly)   — green, upper-middle
 *   3. 10Y−2Y Yield Spread    (FRED T10Y2Y, daily)     — blue, lower-middle
 *   4. US Bankruptcies        (AOUSC F-2, quarterly)   — light gray, bottom
 * The units are incomparable, so each series is min-max normalized into its
 * own (slightly overlapping) vertical band — exactly how the original chart
 * juxtaposes them — with NBER recession shading behind and an inline label
 * pinned to each line. Data all rides on /api/fred (`horsemen` + `yieldCurve`).
 */

// log scale for the strictly-positive count/rate series (the classic chart is
// log — otherwise the 2020 claims spike flattens 40 years of structure);
// linear for the spread, which crosses zero.
const SERIES_STYLE = {
    claims: { label: 'Initial Jobless Claims', shortLabel: 'Jobless Claims', color: '#ef4444', band: [0.02, 0.34], scale: 'log' },
    unemployment: { label: 'Unemployment Rate', shortLabel: 'Unemployment', color: '#22c55e', band: [0.24, 0.58], scale: 'log' },
    spread: { label: '10Y − 2Y Yield Spread', shortLabel: '10Y−2Y Spread', color: '#3b82f6', band: [0.46, 0.80], scale: 'linear' },
    bankruptcies: { label: 'US Bankruptcies', shortLabel: 'Bankruptcies', color: '#e2e8f0', band: [0.68, 0.99], scale: 'log' },
};
// Stagger the inline labels horizontally so they don't stack (fraction of plot
// width). Compact pulls them left, away from the right-edge direction notes.
const LABEL_AT = { claims: 0.58, unemployment: 0.68, spread: 0.78, bankruptcies: 0.48 };
const LABEL_AT_COMPACT = { claims: 0.42, unemployment: 0.55, spread: 0.6, bankruptcies: 0.32 };

const kFmt = (v) => {
    if (v == null || !Number.isFinite(v)) return 'N/A';
    const a = Math.abs(v);
    if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
    if (a >= 1e3) return `${Math.round(v / 1e3)}K`;
    return `${Math.round(v)}`;
};
const pctFmt = (v) => (v == null || !Number.isFinite(v) ? 'N/A' : `${v.toFixed(2)}%`);

// Latest-vs-≈1-year-ago change from an ascending history array.
function yoyPct(history, pointsPerYear) {
    if (!history || history.length <= pointsPerYear) return null;
    const now = history[history.length - 1]?.value;
    const ago = history[history.length - 1 - pointsPerYear]?.value;
    if (now == null || ago == null || ago === 0) return null;
    return ((now - ago) / Math.abs(ago)) * 100;
}


/** True below 640px. Defaults to false (desktop) so SSR/jsdom render wide. */
function useIsNarrow() {
    const [narrow, setNarrow] = useState(false);
    useEffect(() => {
        if (typeof window === 'undefined' || !window.matchMedia) return undefined;
        const mq = window.matchMedia('(max-width: 640px)');
        const update = () => setNarrow(mq.matches);
        update();
        if (mq.addEventListener) { mq.addEventListener('change', update); return () => mq.removeEventListener('change', update); }
        mq.addListener(update); return () => mq.removeListener(update);
    }, []);
    return narrow;
}

function StatChip({ color, label, value, chip, warn, metric, markKey, markRaw, fmtPrev }) {
    // No horseman is markable today: all four are backed by FRED observation series
    // rather than daily snapshots, so there is no honest "what it was yesterday" to
    // show. See the note in lib/marks.js. markKey is left unset and useMark returns
    // null, so this renders exactly as it did before.
    // markRaw is passed explicitly: horsemen metrics key their number as `current`,
    // the spread's synthesised metric uses `value`. Guessing the shape would have
    // silently produced no mark on claims and unemployment.
    const mark = useMark(markKey, markRaw);
    const note = freshnessNote(metric);
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 }}>
            <span className="tooltip-trigger" data-tooltip={`${label}${note.suffix}`}
                style={{ fontSize: '0.64rem', fontWeight: 700, color, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                {label}
            </span>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }}>
                <Delta mark={metric?.stale ? null : mark} format={fmtPrev}>
                    <span style={{ fontSize: '1.05rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: metric?.stale ? 'var(--orange)' : 'var(--text)' }}>
                        {metric?.stale ? '🕐 ' : ''}{value}
                    </span>
                </Delta>
                {chip && (
                    <span style={{ fontSize: '0.62rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: chip.bad ? 'var(--red)' : 'var(--green)', whiteSpace: 'nowrap' }}>
                        {chip.text}
                    </span>
                )}
                {warn && (
                    <span className={`badge ${warn.bad ? 'badge-red' : 'badge-green'}`} style={{ fontSize: '0.56rem', whiteSpace: 'nowrap' }}>{warn.label}</span>
                )}
            </div>
            {metric?.stale && (
                <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>Last data {formatAsOf(metric.asOf)} (stale)</span>
            )}
        </div>
    );
}

export default function FourHorsemen({ fred, loading }) {
    const isNarrow = useIsNarrow();
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

    const stats = !loading && fred && !fred.error ? [
        {
            key: 'claims', color: SERIES_STYLE.claims.color, label: SERIES_STYLE.claims.label,
            value: kFmt(claims?.current),
            metric: claims,
            chip: claimsYoy != null ? { text: `${claimsYoy >= 0 ? '▲' : '▼'} ${Math.abs(claimsYoy).toFixed(1)}% vs 1y`, bad: claimsYoy > 0 } : null,
            warn: claimsYoy != null ? (claimsYoy > 10 ? { bad: true, label: 'Rising' } : { bad: false, label: 'Contained' }) : null,
        },
        {
            key: 'unemployment', color: SERIES_STYLE.unemployment.color, label: SERIES_STYLE.unemployment.label,
            value: pctFmt(unemployment?.current),
            metric: unemployment,
            chip: unempYoy != null ? { text: `${unempYoy >= 0 ? '▲' : '▼'} ${Math.abs(unempYoy).toFixed(1)}pp vs 1y`, bad: unempYoy > 0 } : null,
            warn: sahm != null ? (sahm >= 0.5 ? { bad: true, label: `Sahm ${sahm.toFixed(2)}` } : { bad: false, label: `Sahm ${sahm.toFixed(2)}` }) : null,
        },
        {
            key: 'spread', color: SERIES_STYLE.spread.color, label: SERIES_STYLE.spread.label,
            value: spread?.current != null ? `${spread.current >= 0 ? '+' : ''}${spread.current.toFixed(2)}%` : 'N/A',
            metric: { value: spread?.current, asOf: spread?.asOf, stale: spread?.stale, unavailable: spread?.current == null },
            chip: null,
            warn: spread?.current != null ? (spread.current < 0 ? { bad: true, label: 'Inverted' } : { bad: false, label: 'Normal' }) : null,
        },
        {
            key: 'bankruptcies', color: SERIES_STYLE.bankruptcies.color, label: SERIES_STYLE.bankruptcies.label,
            value: kFmt(bk?.current),
            metric: { value: bk?.current, asOf: bk?.asOf, stale: bk?.stale, unavailable: bk?.unavailable },
            chip: bk?.changePct != null ? { text: `${bk.changePct >= 0 ? '▲' : '▼'} ${Math.abs(bk.changePct).toFixed(1)}% YoY`, bad: bk.changePct > 0 } : null,
            warn: bk?.changePct != null ? (bk.changePct > 10 ? { bad: true, label: 'Rising' } : { bad: false, label: 'Contained' }) : null,
        },
    ] : [];

    const riding = stats.filter((s) => s.warn?.bad).length;
    const histories = [claims?.history, unemployment?.history, spread?.history, bk?.history];
    const hasAnySeries = histories.some((h) => h?.length >= 2);

    return (
        <div className="card" style={{ gridColumn: '1 / -1', animationDelay: '0.5s' }}>
            <div className="card-header">
                <h2><span className="tooltip-trigger" data-tooltip="Four classic recession tells — jobless claims, unemployment, the yield curve and bankruptcies — each shown as how far it has moved in 12 months against how far it had typically moved by the start of past recessions. The level is not the tell: in March 2020 claims sat near their calmest ever.">🐎 Four Horsemen — Recession Watch</span></h2>
                {!loading && stats.length > 0 && (
                    <span className={`badge ${riding >= 3 ? 'badge-red' : riding >= 1 ? 'badge-yellow' : 'badge-green'}`}>
                        {riding} of 4 riding
                    </span>
                )}
            </div>
            <ErrorBoundary>
                {loading || !fred ? <Skeleton count={4} /> : stats.length === 0 || !hasAnySeries ? (
                    <div className="hero-price-section">
                        <div className="hero-price" style={{ fontSize: '2.2rem', color: 'var(--yellow)' }}>N/A</div>
                        <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                            Unavailable — source busy, try again shortly
                        </div>
                    </div>
                ) : (
                    <>
                        {/* Current values + status, doubling as the chart legend.
                            Explicit shrinkable tracks (minmax(0,1fr)) — auto-fit's intrinsic
                            sizing let long chip content widen the whole card on phones. */}
                        <div style={{ display: 'grid', gridTemplateColumns: isNarrow ? 'repeat(2, minmax(0, 1fr))' : 'repeat(4, minmax(0, 1fr))', gap: '10px 20px', marginBottom: '10px' }}>
                            {stats.map((s) => <StatChip key={s.key} {...s} />)}
                        </div>
                        <RunupBars fred={fred} />
                    </>
                )}
            </ErrorBoundary>
        </div>
    );
}
