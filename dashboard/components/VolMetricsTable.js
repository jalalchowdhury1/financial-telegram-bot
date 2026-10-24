'use client';
import { useRef } from 'react';
import ErrorBoundary from './ErrorBoundary';
import Skeleton from './Skeleton';

const hasRows = (j) => !!(j && Array.isArray(j.tickers) && j.tickers.length);

/**
 * 🌡️ Volatility card. Data: /api/vol.
 * Top half: IV (index proxy), IV rank / percentile (1y) and VRP (IV − 21d realized) for
 * SPY and QQQ — the only rows the owner reads (the route still computes TQQQ / SQQQ / UVXY
 * for the Jev pills). Percentile coloring follows the owner's hedgelab thresholds: ≤10 =
 * options historically cheap (green), ≥90 = panic-rich (red).
 * Bottom half (2026-09-26, lib/volRegime.js): the VIX curve with a Calm / Watch / Stress
 * call, TQQQ's volatility decay at today's QQQ vol, and SPY / QQQ ±1σ moves for the next
 * 5 trading days.
 */
export const SHOWN = ['SPY', 'QQQ'];

const fmt = (v, digits = 1) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits));

const pctColor = (v) => {
    if (v == null) return 'var(--text-muted)';
    if (v >= 90) return 'var(--red)';
    if (v >= 70) return 'var(--orange)';
    if (v <= 10) return 'var(--green)';
    return 'inherit';
};

const vrpColor = (v) => {
    if (v == null) return 'var(--text-muted)';
    if (v < 0) return 'var(--orange)'; // realized above implied = stress regime
    return 'var(--green)';
};

/**
 * '2026-07-15T13:42:31.000-0400' → '2026-07-15, 1:42 PM ET'. Normalizes the
 * narrow no-break space newer ICU builds put before AM/PM. Returns null on
 * anything unparseable so the caller falls back to the plain date.
 */
const formatLiveAt = (iso) => {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return null;
    const date = d.toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
    const time = d.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' }).replace(/[\u202f\u00a0]/g, ' ');
    return `${date}, ${time} ET`;
};

/** Footnote timestamp: live rows get a green dot + ET time + 'intraday'. */
const asOfNote = (data, rows) => {
    const anyLive = rows.some((t) => t.live);
    if (anyLive) {
        const when = (data.live_at && formatLiveAt(data.live_at)) || data.updated_at;
        if (when) return <> <span style={{ color: 'var(--green)' }}>●</span> As of {when} · intraday.</>;
    }
    return data.updated_at ? ` As of ${data.updated_at}.` : '';
};

const COLS = '1.3fr 1.2fr 1fr 1fr 1fr 1fr';
const cellRight = { textAlign: 'right', fontVariantNumeric: 'tabular-nums' };

export default function VolMetricsTable({ data: incoming = null, saved, waiting = false }) {
    // The page fetches /api/vol once per refresh and hands it down (it used to be fetched
    // twice per load — Vercel Active CPU, 21 Oct 2026). `saved` = the page's 🕐 label for a
    // saved copy. A failed or empty answer keeps what the card already shows.
    const last = useRef(null);
    if (hasRows(incoming)) last.current = incoming;
    const data = last.current;
    const status = data ? 'ready' : waiting ? 'loading' : 'error';

    const rows = data ? SHOWN.map((t) => data.tickers.find((x) => x && x.ticker === t)).filter(Boolean) : [];

    return (
        <div className="card" style={{ animationDelay: '0.6s' }} data-cached={saved}>
            <div className="card-header">
                <h2>🌡️ Volatility</h2>
                <span className="badge badge-blue">IV · VIX curve · decay</span>
            </div>
            <ErrorBoundary>
                {status === 'loading' ? <Skeleton count={6} /> : status === 'error' || !rows.length ? (
                    <div className="error-message" style={{ color: 'var(--text-muted)' }}>
                        ⚠️ Volatility data unavailable. Try refreshing the page.
                    </div>
                ) : (
                    <>
                        <div style={{ display: 'grid', gridTemplateColumns: COLS, gap: '4px 8px', fontSize: '0.8rem', padding: '4px 0', whiteSpace: 'nowrap', overflowX: 'auto' }}>
                            <span style={{ color: 'var(--text-muted)' }}>Ticker</span>
                            <span style={{ color: 'var(--text-muted)', ...cellRight }}>IV</span>
                            <span className="tooltip-trigger" data-tooltip="Today's IV between its 1-year low (0) and high (100)." style={{ color: 'var(--text-muted)', ...cellRight }}>Rank 1y</span>
                            <span className="tooltip-trigger" data-tooltip="% of past-year days with IV at or below today. ≤10 cheap, ≥90 panic." style={{ color: 'var(--text-muted)', ...cellRight }}>%ile 1y</span>
                            <span className="tooltip-trigger" data-tooltip="Actual swings, last 21 days, annualized." style={{ color: 'var(--text-muted)', ...cellRight }}>RV 21d</span>
                            <span className="tooltip-trigger" data-tooltip="Implied minus realized. Above 0 = normal; below 0 = stress." style={{ color: 'var(--text-muted)', ...cellRight }}>VRP</span>
                            {rows.map((t) => (
                                <VolRow key={t.ticker} t={t} />
                            ))}
                        </div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.65rem', marginTop: '8px', opacity: 0.7 }}>
                            IV from VIX and VXN.
                            {asOfNote(data, rows)}
                        </div>
                        <ErrorBoundary>
                            <Regime regime={data.regime} />
                        </ErrorBoundary>
                    </>
                )}
            </ErrorBoundary>
        </div>
    );
}

function VolRow({ t }) {
    return (
        <>
            <span style={{ fontWeight: 600 }}>
                {t.ticker} <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: '0.7rem' }}>{t.proxy}</span>
            </span>
            <span style={cellRight}>{fmt(t.iv)}</span>
            <span style={{ ...cellRight, color: pctColor(t.ivRank1y) }}>{fmt(t.ivRank1y, 0)}</span>
            <span style={{ ...cellRight, color: pctColor(t.ivPctile1y) }}>{fmt(t.ivPctile1y, 0)}</span>
            <span style={cellRight}>{fmt(t.rv21)}</span>
            <span style={{ ...cellRight, color: vrpColor(t.vrp) }}>{t.vrp != null && t.vrp > 0 ? '+' : ''}{fmt(t.vrp)}</span>
        </>
    );
}

// ── Bottom half: VIX curve, TQQQ decay, expected moves ───────────────────────────────

export const STATE_COPY = {
    calm: { pill: '🟢 Calm', color: 'var(--green)', line: 'Short-term fear under 3-month: normal.' },
    watch: { pill: '🟡 Watch', color: 'var(--orange)', line: 'Short-term fear catching up to 3-month.' },
    stress: { pill: '🔴 Stress', color: 'var(--red)', line: 'Short-term fear over 3-month: selloff shape.' },
};

const SOURCE_NAMES = { cboe: 'CBOE', cnbc: 'CNBC', fred: 'FRED', yahoo: 'Yahoo', 'cnbc-quote': 'CNBC quote' };
/** 'cboe+live', 'cnbc', … → 'CBOE' or 'CBOE · CNBC' when a backup tier served a point. */
export const curveSources = (points) => [...new Set((points || []).map((p) => {
    const base = String(p.source || '').replace(/\+live$/, '');
    return SOURCE_NAMES[base] || base;
}))].join(' · ');

/** 7.24 → '7.2%', 12.3 → '12%' (a decimal is noise above 10). */
export const fmtDecay = (v) => (v == null || !Number.isFinite(v) ? '—' : `${v >= 10 ? v.toFixed(0) : v.toFixed(1)}%`);

function Regime({ regime }) {
    if (!regime) return null;
    const { curve, decay, moves } = regime;
    return (
        <div className="vol-regime">
            <Curve curve={curve} />
            <div className="vol-fact">
                <div className="vol-fact-head">
                    <span className="vol-fact-label">TQQQ decay</span>
                    <span className="vol-fact-value">
                        ≈{fmtDecay(decay?.realized)}/yr
                        {decay?.implied != null && <span className="vol-fact-sub"> · ≈{fmtDecay(decay.implied)} at VXN</span>}
                    </span>
                </div>
                <div className="vol-note">Loss if QQQ ends flat, at today&apos;s swings.</div>
            </div>
            <div className="vol-fact">
                <div className="vol-fact-head">
                    <span className="vol-fact-label">Next 5 days <span style={{ textTransform: 'none' }}>±1σ</span></span>
                    <span className="vol-fact-value">SPY ±{fmt(moves?.SPY)}% · QQQ ±{fmt(moves?.QQQ)}%</span>
                </div>
                <div className="vol-note">2 weeks in 3 stay inside this range.</div>
            </div>
        </div>
    );
}

function Curve({ curve }) {
    if (!curve || !curve.state || !Array.isArray(curve.points) || !curve.points.length) {
        return (
            <div className="vol-curve">
                <div className="vol-curve-head"><span className="vol-fact-label">VIX curve</span></div>
                <div className="vol-note">VIX curve unavailable right now. Table above is fine.</div>
            </div>
        );
    }
    const copy = STATE_COPY[curve.state];
    const max = Math.max(...curve.points.map((p) => p.value));
    return (
        <div className="vol-curve">
            <div className="vol-curve-head">
                <span className="vol-fact-label">VIX curve</span>
                <span className="vol-pill" style={{ color: copy.color, borderColor: copy.color }}>{copy.pill}</span>
            </div>
            <div className="vol-bars" role="list" aria-label="VIX term structure">
                {curve.points.map((p) => (
                    <div key={p.tenor} className="vol-bar-row" role="listitem" title={`${p.index} ${p.value} · ${p.source} · ${p.asOf}`}>
                        <span className="vol-bar-tenor">{p.tenor}</span>
                        <span className="vol-bar-track">
                            <span className="vol-bar-fill" style={{ width: `${Math.max(4, (p.value / max) * 100)}%`, background: copy.color }} />
                        </span>
                        <span className="vol-bar-value">{fmt(p.value)}</span>
                    </div>
                ))}
            </div>
            <div className="vol-note">
                <strong style={{ color: copy.color }}>VIX ÷ VIX3M {curve.ratio.toFixed(2)}</strong> · calm under 0.90 · stress 1.00+
            </div>
            <div className="vol-note">{copy.line}</div>
            {curve.frontInverted && (
                <div className="vol-note">9-day over 1-month: an event is priced in.</div>
            )}
            {curve.stale && (
                <div className="vol-note" style={{ color: 'var(--orange)' }}>
                    🕐 Saved copy from {curve.asOf}.
                </div>
            )}
            {!curve.stale && curve.asOf && (
                <div className="vol-note vol-asof">{curve.live ? 'Intraday' : `${curve.asOf} close`} · {curveSources(curve.points)}</div>
            )}
        </div>
    );
}
