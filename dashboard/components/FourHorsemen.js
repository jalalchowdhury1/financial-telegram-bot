'use client';
import { useEffect, useState } from 'react';
import ErrorBoundary from './ErrorBoundary';
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

const ms = (dateStr) => new Date(`${dateStr}T00:00:00Z`).getTime();

/**
 * Direction of a series: the sign of a least-squares trendline fitted to the
 * LAST 12 MONTHS of the RAW history (anchored to the series' own newest point,
 * so a stale series is judged on its own data). A fitted slope uses every
 * point in the year — robust to endpoint noise, works identically for daily,
 * weekly, monthly and quarterly cadences, and (unlike the old
 * two-window-means version, which read the THINNED chart data) never changes
 * with the zoom tab or screen size. "Flat" when the fitted change over a year
 * is small: < 2% of the series mean for count-like series (claims,
 * bankruptcies), < 0.08 points for rate-like ones (unemployment, spread).
 * Returns 'up' | 'down' | 'flat' | null.
 */
export function trendOf(pts) {
    if (!pts || pts.length < 2) return null;
    const lastT = ms(pts[pts.length - 1].date);
    const YEAR_MS = 365.25 * 86400000;
    const w = pts.filter((p) => ms(p.date) >= lastT - YEAR_MS && p.value != null);
    if (w.length < 2) return null;

    // OLS slope, x in years so the slope reads as change-per-year.
    const xs = w.map((p) => (ms(p.date) - lastT) / YEAR_MS);
    const ys = w.map((p) => p.value);
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const my = ys.reduce((a, b) => a + b, 0) / ys.length;
    let num = 0, den = 0;
    for (let i = 0; i < xs.length; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
    if (den === 0) return null;
    const slopePerYear = num / den;

    const flat = Math.abs(my) > 10
        ? Math.abs(slopePerYear) / Math.abs(my) < 0.02
        : Math.abs(slopePerYear) < 0.08;
    if (flat) return 'flat';
    return slopePerYear > 0 ? 'up' : 'down';
}

/** Slice to the window, then thin to ≤ maxPoints for a light polyline. */
function windowed(history, cutoffMs, maxPoints = 1500) {
    if (!history?.length) return [];
    const inWin = history.filter((p) => ms(p.date) >= cutoffMs && p.value != null);
    if (inWin.length <= maxPoints) return inWin;
    const step = Math.ceil(inWin.length / maxPoints);
    const out = inWin.filter((_, i) => i % step === 0);
    if (out[out.length - 1] !== inWin[inWin.length - 1]) out.push(inWin[inWin.length - 1]);
    return out;
}

// Phones scale the SVG down ~3×, so the compact variant uses a narrower/taller
// canvas with proportionally larger type and thicker strokes — otherwise the
// chart renders as an unreadable 135px-tall sliver.
const OVERLAY_DIMS = {
    wide: { W: 1200, H: 430, padL: 14, padR: 14, padT: 12, padB: 26, fYear: 12, fLabel: 12.5, labelH: 22, fNote: 12, fZero: 10, stroke: 1.6, strokeBk: 2.2, maxPoints: 1500, maxYears: 10, noteUp: -12, noteDown: 26, noteDownSpread: 42 },
    compact: { W: 720, H: 800, padL: 10, padR: 10, padT: 16, padB: 42, fYear: 20, fLabel: 20, labelH: 32, fNote: 19, fZero: 15, stroke: 2.6, strokeBk: 3.4, maxPoints: 700, maxYears: 5, noteUp: -20, noteDown: 40, noteDownSpread: 64 },
};

function HorsemenOverlay({ series, recessions, timeframe, trends, compact = false }) {
    const D = compact ? OVERLAY_DIMS.compact : OVERLAY_DIMS.wide;
    const { W, H, padL, padR, padT, padB } = D;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    const nowMs = Date.now();
    const CUTOFFS = {
        ALL: ms('1979-01-01'), '20Y': nowMs - 20 * 365.25 * 86400000,
        '10Y': nowMs - 10 * 365.25 * 86400000, '5Y': nowMs - 5 * 365.25 * 86400000,
        '1Y': nowMs - 365.25 * 86400000,
    };
    const cutoff = CUTOFFS[timeframe] ?? CUTOFFS.ALL;

    // Window + normalize each series into its own band.
    const prepared = {};
    let minX = Infinity, maxX = -Infinity;
    for (const [key, s] of Object.entries(series)) {
        const pts = windowed(s.history, cutoff, D.maxPoints);
        if (pts.length >= 2) {
            prepared[key] = pts;
            minX = Math.min(minX, ms(pts[0].date));
            maxX = Math.max(maxX, ms(pts[pts.length - 1].date));
        }
    }
    if (!Number.isFinite(minX) || maxX <= minX) return null;

    const toX = (t) => padL + ((t - minX) / (maxX - minX)) * plotW;
    const lines = {};
    const bandScale = {};
    for (const [key, pts] of Object.entries(prepared)) {
        const log = SERIES_STYLE[key].scale === 'log';
        const tf = log ? (v) => Math.log10(Math.max(v, 1e-9)) : (v) => v;
        const values = pts.map((p) => tf(p.value));
        const lo = Math.min(...values), hi = Math.max(...values);
        const range = hi - lo || 1;
        const [bTop, bBot] = SERIES_STYLE[key].band;
        const yTop = padT + bTop * plotH, yBot = padT + bBot * plotH;
        const toY = (v) => yBot - ((tf(v) - lo) / range) * (yBot - yTop);
        bandScale[key] = { toY, lo: pts.reduce((m, p) => Math.min(m, p.value), Infinity), hi: pts.reduce((m, p) => Math.max(m, p.value), -Infinity) };
        lines[key] = pts.map((p) => `${toX(ms(p.date)).toFixed(1)},${toY(p.value).toFixed(1)}`).join(' ');
    }

    // Inline label anchored to its line at a staggered x position.
    const labelAt = compact ? LABEL_AT_COMPACT : LABEL_AT;
    const labels = Object.entries(prepared).map(([key, pts]) => {
        const targetT = minX + (maxX - minX) * labelAt[key];
        let nearest = pts[0];
        for (const p of pts) { if (Math.abs(ms(p.date) - targetT) < Math.abs(ms(nearest.date) - targetT)) nearest = p; }
        const x = toX(ms(nearest.date));
        const y = bandScale[key].toY(nearest.value);
        const text = compact ? SERIES_STYLE[key].shortLabel : SERIES_STYLE[key].label;
        const w = text.length * D.fLabel * 0.6 + 16;
        // Keep the box inside the plot; nudge above or below the line.
        const bx = Math.min(Math.max(x - w / 2, padL + 4), W - padR - w - 4);
        const by = Math.max(padT + 4, Math.min(y - D.labelH - 8, H - padB - D.labelH));
        return { key, bx, by, w, text, color: SERIES_STYLE[key].color, lineX: x, lineY: y };
    });

    // Year gridlines/labels (at most ~10).
    const years = [];
    const y0 = new Date(minX).getUTCFullYear() + 1, y1 = new Date(maxX).getUTCFullYear();
    const stepYears = Math.max(1, Math.ceil((y1 - y0) / D.maxYears));
    for (let y = y0 + ((stepYears - (y0 % stepYears)) % stepYears); y <= y1; y += stepYears) {
        years.push({ x: toX(Date.UTC(y, 0, 1)), label: String(y) });
    }

    const visibleRecessions = (recessions || []).filter((r) => ms(r.end) >= minX && ms(r.start) <= maxX);
    const spreadZeroY = bandScale.spread && bandScale.spread.lo < 0 && bandScale.spread.hi > 0
        ? bandScale.spread.toY(0) : null;

    // Hand-annotated direction notes at each line's right end, like the original
    // chart ("trending up" / "watch this line"). Verdicts arrive via the
    // `trends` prop — computed ONCE from the raw histories, so they can't drift
    // with the zoom tab or the thinned chart data; only the text position is
    // derived from the drawn line. Claims is the earliest warning of the four,
    // so a flat claims line still gets the "watch this line" nudge.
    const trendNotes = Object.entries(prepared).map(([key, pts]) => {
        const t = trends?.[key];
        if (!t) return null;
        const text = t === 'up' ? '↗ trending up · 1y' : t === 'down' ? '↘ trending down · 1y'
            : key === 'claims' ? '→ watch this line' : '→ flat · 1y';
        const last = pts[pts.length - 1];
        const x = toX(ms(last.date)) - 8;
        // A rising line leaves free space ABOVE its endpoint; a falling one, BELOW.
        // The spread oscillates tightly at its right end, so its note drops further.
        const downOff = key === 'spread' ? D.noteDownSpread : D.noteDown;
        const yRaw = bandScale[key].toY(last.value) + (t === 'up' ? D.noteUp : t === 'down' ? downOff : D.noteUp + 2);
        const y = Math.max(padT + D.fNote, Math.min(yRaw, H - padB - 6));
        return { key, x, y, text, color: SERIES_STYLE[key].color };
    }).filter(Boolean);

    return (
        <div style={{ width: '100%' }}>
            <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
                {/* Year gridlines */}
                {years.map((yr, i) => (
                    <g key={`yr-${i}`}>
                        <line x1={yr.x} x2={yr.x} y1={padT} y2={H - padB} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
                        <text x={yr.x} y={H - 10} fill="rgba(255,255,255,0.28)" fontSize={D.fYear} fontFamily="JetBrains Mono, monospace" textAnchor="middle">{yr.label}</text>
                    </g>
                ))}
                {/* NBER recession bands */}
                {visibleRecessions.map((rec, i) => {
                    const x1 = Math.max(toX(ms(rec.start)), padL);
                    const x2 = Math.min(toX(ms(rec.end)), W - padR);
                    if (x2 <= x1) return null;
                    return <rect key={`rec-${i}`} x={x1} y={padT} width={x2 - x1} height={plotH} fill="rgba(148,163,184,0.13)" rx="2" />;
                })}
                {/* Yield-spread zero (inversion) reference */}
                {spreadZeroY != null && (
                    <g>
                        <line x1={padL} x2={W - padR} y1={spreadZeroY} y2={spreadZeroY} stroke="rgba(59,130,246,0.35)" strokeDasharray="5,4" strokeWidth="1" />
                        <text x={padL + 4} y={spreadZeroY - 4} fill="rgba(59,130,246,0.55)" fontSize={D.fZero} fontFamily="JetBrains Mono, monospace" textAnchor="start">10Y−2Y = 0 (inversion)</text>
                    </g>
                )}
                {/* The four lines */}
                {Object.entries(lines).map(([key, pts]) => (
                    <polyline key={key} points={pts} fill="none" stroke={SERIES_STYLE[key].color}
                        strokeWidth={key === 'bankruptcies' ? D.strokeBk : D.stroke} strokeLinejoin="round" strokeLinecap="round"
                        opacity={key === 'spread' ? 0.9 : 0.95} />
                ))}
                {/* Direction notes at each line's right end */}
                {trendNotes.map((n) => (
                    <text key={`tn-${n.key}`} x={n.x} y={n.y} fill={n.color} fontSize={D.fNote} fontStyle="italic" fontWeight="600"
                        fontFamily="Inter, sans-serif" textAnchor="end" opacity="0.9"
                        transform={`rotate(-6 ${n.x} ${n.y})`}>{n.text}</text>
                ))}
                {/* Inline labels pinned to their lines */}
                {labels.map((l) => (
                    <g key={`lbl-${l.key}`}>
                        <line x1={l.bx + l.w / 2} y1={l.by + D.labelH} x2={l.lineX} y2={l.lineY} stroke={l.color} strokeWidth="1" opacity="0.5" />
                        <rect x={l.bx} y={l.by} width={l.w} height={D.labelH} rx="4" fill="rgba(10,14,23,0.92)" stroke={l.color} strokeWidth="1.2" />
                        <text x={l.bx + l.w / 2} y={l.by + D.labelH * 0.68} fill={l.color} fontSize={D.fLabel} fontWeight="700" fontFamily="Inter, sans-serif" textAnchor="middle">{l.text}</text>
                    </g>
                ))}
            </svg>
        </div>
    );
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
    const [timeframe, setTimeframe] = useState('ALL');
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
    const overlaySeries = {
        claims: { history: claims?.history },
        unemployment: { history: unemployment?.history },
        spread: { history: spread?.history },
        bankruptcies: { history: bk?.history },
    };
    // Direction verdicts from the RAW histories (12-month fitted trend), never
    // from the thinned/zoomed chart data — see trendOf.
    const trends = {
        claims: trendOf(claims?.history),
        unemployment: trendOf(unemployment?.history),
        spread: trendOf(spread?.history),
        bankruptcies: trendOf(bk?.history),
    };
    const hasAnySeries = Object.values(overlaySeries).some((s) => s.history?.length >= 2);
    const TFS = ['ALL', '20Y', '10Y', '5Y', '1Y'];

    return (
        <div className="card" style={{ gridColumn: '1 / -1', animationDelay: '0.5s' }}>
            <div className="card-header">
                <h2><span className="tooltip-trigger" data-tooltip="Four classic recession tells overlaid on one chart, each on its own scale: jobless claims, unemployment, the yield curve, and bankruptcies. Shaded bands are NBER recessions — note how all four turn together going into them.">🐎 Four Horsemen — Recession Watch</span></h2>
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
                        {/* Shared timeframe tabs */}
                        <div style={{ display: 'flex', gap: '3px', marginBottom: '4px' }}>
                            {TFS.map((tf) => (
                                <button key={tf} onClick={() => setTimeframe(tf)}
                                    style={{
                                        padding: '2px 9px', borderRadius: '5px', border: 'none', cursor: 'pointer',
                                        fontSize: '0.62rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
                                        background: tf === timeframe ? 'rgba(148,163,184,0.25)' : 'rgba(255,255,255,0.05)',
                                        color: tf === timeframe ? 'var(--text)' : 'var(--text-muted)',
                                        transition: 'all 0.2s ease',
                                    }}>{tf}</button>
                            ))}
                        </div>
                        <HorsemenOverlay series={overlaySeries} recessions={recessions} timeframe={timeframe} trends={trends} compact={isNarrow} />
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.62rem', marginTop: '6px' }}>
                            Each line on its own scale (normalized) — read the shape, not the height. Shaded bands = NBER recessions. Direction notes = 12-month fitted trend. Bankruptcies = 12-month business filings (AOUSC), data from 2001.
                        </div>
                    </>
                )}
            </ErrorBoundary>
        </div>
    );
}
