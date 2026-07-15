'use client';
import { useEffect, useState } from 'react';
import ErrorBoundary from './ErrorBoundary';
import Skeleton from './Skeleton';

/**
 * Volatility metrics table: IV (index proxy), IV rank / percentile (1y) and
 * VRP (IV − 21d realized) for SPY, QQQ, TQQQ, SQQQ, UVXY. Data: /api/vol.
 * Percentile coloring follows the owner's hedgelab thresholds: ≤10 = options
 * historically cheap (green), ≥90 = panic-rich (red).
 */
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
const asOfNote = (data) => {
    const anyLive = Array.isArray(data.tickers) && data.tickers.some((t) => t.live);
    if (anyLive) {
        const when = (data.live_at && formatLiveAt(data.live_at)) || data.updated_at;
        if (when) return <> <span style={{ color: 'var(--green)' }}>●</span> As of {when} · intraday.</>;
    }
    return data.updated_at ? ` As of ${data.updated_at}.` : '';
};

const COLS = '1.3fr 1.2fr 1fr 1fr 1fr 1fr';
const cellRight = { textAlign: 'right', fontVariantNumeric: 'tabular-nums' };

export default function VolMetricsTable() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchVol = async () => {
            try {
                const res = await fetch('/api/vol');
                const json = await res.json();
                if (!json || !Array.isArray(json.tickers) || !json.tickers.length) {
                    setError('no data');
                } else {
                    setData(json);
                }
            } catch {
                setError('fetch failed');
            } finally {
                setLoading(false);
            }
        };
        fetchVol();
    }, []);

    return (
        <div className="card" style={{ animationDelay: '0.6s' }}>
            <div className="card-header">
                <h2>🌡️ Volatility Metrics</h2>
                <span className="badge badge-blue">IV Rank · %ile · VRP</span>
            </div>
            <ErrorBoundary>
                {loading ? <Skeleton count={6} /> : error || !data ? (
                    <div className="error-message" style={{ color: 'var(--text-muted)' }}>
                        ⚠️ Volatility data unavailable. Try refreshing the page.
                    </div>
                ) : (
                    <>
                        <div style={{ display: 'grid', gridTemplateColumns: COLS, gap: '4px 8px', fontSize: '0.8rem', padding: '4px 0' }}>
                            <span style={{ color: 'var(--text-muted)' }}>Ticker</span>
                            <span style={{ color: 'var(--text-muted)', ...cellRight }}>IV</span>
                            <span className="tooltip-trigger" data-tooltip="Where today's IV sits between its 1-year low (0) and high (100)." style={{ color: 'var(--text-muted)', ...cellRight }}>Rank 1y</span>
                            <span className="tooltip-trigger" data-tooltip="Share of the last year's days with IV at or below today's. ≤10 = historically cheap, ≥90 = panic-rich." style={{ color: 'var(--text-muted)', ...cellRight }}>%ile 1y</span>
                            <span className="tooltip-trigger" data-tooltip="Realized volatility of the ETF itself over the last 21 trading days, annualized." style={{ color: 'var(--text-muted)', ...cellRight }}>RV 21d</span>
                            <span className="tooltip-trigger" data-tooltip="Volatility risk premium: implied minus realized. Positive = options priced above delivered vol (normal); negative = market moving more than options imply (stress)." style={{ color: 'var(--text-muted)', ...cellRight }}>VRP</span>
                            {data.tickers.map((t) => (
                                <VolRow key={t.ticker} t={t} />
                            ))}
                        </div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.65rem', marginTop: '8px', opacity: 0.7 }}>
                            IV via index proxies (SPY→VIX · QQQ→VXN · TQQQ/SQQQ→3×VXN · UVXY→VVIX) — approximation, not chain-derived.
                            {asOfNote(data)}
                        </div>
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
