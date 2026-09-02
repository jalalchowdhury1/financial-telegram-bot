'use client';
import { useState, useEffect, useRef } from 'react';
import ErrorBoundary from '../components/ErrorBoundary';
import { freshnessNote, formatAsOf } from '../lib/freshness';

import Gauge from '../components/Gauge';
import Skeleton from '../components/Skeleton';
import SpyChart from '../components/SpyChart';
import MiniChart from '../components/MiniChart';
import MarketPulse from '../components/MarketPulse';
import CustomIndicatorBar from '../components/CustomIndicatorBar';
import EconomicIndicatorGrid from '../components/EconomicIndicatorGrid';
import FourHorsemen from '../components/FourHorsemen';
import BullChecklist from '../components/BullChecklist';
import ExtraMarketsGrid from '../components/ExtraMarketsGrid';
import PolymarketTable from '../components/PolymarketTable';
import VolMetricsTable from '../components/VolMetricsTable';
import RubberBandRadar from '../components/RubberBandRadar';
import Delta from '../components/Delta';
import MarkChip from '../components/MarkChip';
import { MarkProvider, useMark, collectLiveValues } from '../components/MarkProvider';

/**
 * A hero number that can carry a fresh-print mark. Lives here rather than in the JSX
 * because `useMark` is a hook and the cards are rendered inline in Dashboard.
 * A stale value is never marked — an old number reappearing is not a new print.
 */
function HeroValue({ markKey, raw, stale, format, style, children }) {
    const mark = useMark(markKey, raw);
    return (
        <div className="hero-price" style={style}>
            <Delta mark={stale ? null : mark} format={format}>{children}</Delta>
        </div>
    );
}

// ============ MAIN DASHBOARD ============
export default function Dashboard() {
    const [sheets, setSheets] = useState(null);
    const [spyDailyMove, setSpyDailyMove] = useState(null);
    const [spy, setSpy] = useState(null);
    const [fg, setFg] = useState(null);
    const [fred, setFred] = useState(null);
    const [extraMarkets, setExtraMarkets] = useState(null);
    const [loading, setLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState(null);
    const [systemStatus, setSystemStatus] = useState(null);
    const [apiErrors, setApiErrors] = useState([]);
    const [refreshing, setRefreshing] = useState(false);
    const [history, setHistory] = useState(null);
    // Refresh behaviour: `loading` (skeletons) is for the FIRST load only. Every
    // later fetch is a background refresh — the page keeps showing what it has,
    // and only the header spinner moves. Before this, the 5-minute auto-refresh
    // collapsed all 12 cards to skeletons for ~10s while you were reading.
    const hasLoadedRef = useRef(false);
    const lastFetchRef = useRef(0);

    async function fetchAll() {
        if (!hasLoadedRef.current) setLoading(true);
        setRefreshing(true);
        setApiErrors([]);
        lastFetchRef.current = Date.now();
        try {
            const timestamp = Date.now();
            const [sheetsRes, spyRes, spyDailyMoveRes, fgRes, fredRes, extraRes, historyRes] = await Promise.all([
                fetch(`/api/sheets?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch(`/api/spy?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch(`/api/spy-daily-move?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch(`/api/fear-greed?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch(`/api/fred?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch(`/api/market-extra?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                // Baselines for the fresh-print marks. Deliberately last and deliberately
                // swallowed: if it fails the digest is null, no marks render, and every
                // number reads exactly as it does today.
                fetch(`/api/history?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
            ]);

            // A null here means the fetch itself failed (the routes never 500).
            // Keep the previous payload rather than blanking a card that had data.
            setSheets(prev => sheetsRes ?? prev);
            setSpy(prev => spyRes ?? prev);
            setSpyDailyMove(prev => spyDailyMoveRes ?? prev);
            setFg(prev => fgRes ?? prev);
            setFred(prev => fredRes ?? prev);
            setExtraMarkets(prev => extraRes ?? prev);
            setHistory(prev => historyRes ?? prev);
            hasLoadedRef.current = true;

            setSystemStatus({
                spy: spyRes?._meta,
                fred: fredRes?._meta,
                fg: fgRes?._meta,
                sheets: sheetsRes?._meta,
                extra: extraRes?._meta
            });

            const errors = [];
            if (sheetsRes?.error) errors.push(`[SHEETS] ${sheetsRes.error}`);
            if (spyRes?.error) errors.push(`[SPY] ${spyRes.error}`);
            if (fgRes?.error) errors.push(`[F&G] ${fgRes.error}`);
            if (fredRes?.error) errors.push(`[FRED] ${fredRes.error}`);
            if (extraRes?.error) errors.push(`[EXTRA] ${extraRes.error}`);
            setApiErrors(errors);

            const now = new Date();
            const year = now.getFullYear();
            const month = String(now.getMonth() + 1).padStart(2, '0');
            const day = String(now.getDate()).padStart(2, '0');
            const hours = String(now.getHours()).padStart(2, '0');
            const minutes = String(now.getMinutes()).padStart(2, '0');
            setLastUpdated(`${year}-${month}-${day} ${hours}:${minutes}`);
        } catch (e) {
            console.error('Dashboard fetch error:', e);
            setApiErrors(prev => [...prev, `[NETWORK] ${e.toString()}`]);
        }
        setLoading(false);
        setRefreshing(false);
    }

    // The explanatory tooltips are pure CSS :hover, which does not exist on touch —
    // so on a phone every as-of date and metric explanation was unreachable, while the
    // header cheerfully said "hover any number for its date". Tapping a trigger now
    // toggles the same tooltip; tapping elsewhere, or Esc, dismisses it.
    useEffect(() => {
        const closeAll = (except) => document.querySelectorAll('.tooltip-trigger.tooltip-open')
            .forEach((el) => { if (el !== except) el.classList.remove('tooltip-open'); });
        const onClick = (e) => {
            const trigger = e.target.closest?.('.tooltip-trigger');
            closeAll(trigger);
            if (trigger) trigger.classList.toggle('tooltip-open');
        };
        const onKey = (e) => { if (e.key === 'Escape') closeAll(null); };
        document.addEventListener('click', onClick);
        document.addEventListener('keydown', onKey);
        return () => {
            document.removeEventListener('click', onClick);
            document.removeEventListener('keydown', onKey);
        };
    }, []);

    useEffect(() => {
        const REFRESH_MS = 5 * 60 * 1000;
        fetchAll();
        // Auto-refresh every 5 minutes — but not while the tab is hidden. /api/fred
        // alone is ~650KB, so an idle background tab was pulling ~8MB/hour. On
        // returning to the tab, refresh immediately if the data has gone stale.
        const interval = setInterval(() => {
            if (!document.hidden) fetchAll();
        }, REFRESH_MS);
        const onVisible = () => {
            if (!document.hidden && Date.now() - lastFetchRef.current > REFRESH_MS) fetchAll();
        };
        document.addEventListener('visibilitychange', onVisible);
        return () => {
            clearInterval(interval);
            document.removeEventListener('visibilitychange', onVisible);
        };
    }, []);

    const fgSegments = [
        { start: 0, end: 25, color: '#dc2626' },
        { start: 25, end: 45, color: '#f97316' },
        { start: 45, end: 55, color: '#525252' },
        { start: 55, end: 75, color: '#22c55e' },
        { start: 75, end: 100, color: '#15803d' },
    ];

    const rsiSegments = [
        { start: 0, end: 30, color: '#22c55e' },
        { start: 30, end: 70, color: '#3f3f46' },
        { start: 70, end: 100, color: '#dc2626' },
    ];

    const statusColor = (s) => s === 'safe' || s === 'healthy' || s === 'strong' || s === 'tight' || s === 'easy' || s === 'rising' ? 'stat-positive' : s === 'danger' || s === 'weak' || s === 'stressed' || s === 'restrictive' || s === 'falling' ? 'stat-negative' : 'stat-neutral';

    const fgColor = (score) => score < 25 ? 'var(--red)' : score < 45 ? '#f97316' : score < 55 ? 'var(--text-muted)' : score < 75 ? 'var(--green)' : '#15803d';

    return (
        <MarkProvider history={history}>
        <div className="dashboard">
            {/* Auto-Refresh Visualizer */}
            {lastUpdated && <div key={lastUpdated} className="auto-refresh-bar" style={{ animation: 'progress-fill 300s linear forwards' }}></div>}

            {/* HEADER */}
            <header className="dashboard-header">
                <h1>Jalal's Financial Dashboard</h1>
                <p className="subtitle">Live market data, economic indicators & AI-powered assessment</p>
                <div className="header-status">
                    <div className="live-badge">
                        <span className="live-dot" />
                        {loading ? 'Loading live data...' : (
                            <>
                                Updated{' '}
                                {/* the date is hidden on phones — it is always today, and the
                                    full string pushed this badge into the refresh button */}
                                <span className="upd-date">{lastUpdated?.slice(0, 10)} </span>
                                {lastUpdated?.slice(11)}
                            </>
                        )}
                    </div>
                    <MarkChip values={collectLiveValues(fred, extraMarkets, sheets)} />
                    <button className="refresh-btn" onClick={fetchAll} disabled={refreshing} title="Refresh all data">
                        <svg className={refreshing ? 'spinning' : ''} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="23 4 23 10 17 10" />
                            <polyline points="1 20 1 14 7 14" />
                            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                        </svg>
                    </button>
                </div>
                {fred?._meta?.fetchedAt && (
                    <p className="subtitle" style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: '6px' }}>
                        Economic data as of {new Date(fred._meta.fetchedAt).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}<span className="hide-sm"> · refreshes every 30 min</span> · tap any number for its date
                    </p>
                )}
            </header>

            {/* CUSTOM INDICATOR BAR */}
            <CustomIndicatorBar sheets={sheets} loading={loading} />

            {/* MARKET PULSE - Quick summary at top */}
            <MarketPulse spy={spy} spyDailyMove={spyDailyMove} fg={fg} fred={fred} loading={loading} fgColor={fgColor} />

            {/* MAIN GRID */}
            <div className="dashboard-grid">

                {/* ========== REDESIGNED SPY CARD ========== */}
                <div className={`card${spy && !spy.error && (spy.rsi < 30 || spy.rsi > 70) ? ' card-alert' : ''}`} style={{ animationDelay: '0.2s' }}>
                    <div className="card-header">
                        <h2>📊 SPY Market Overview</h2>
                        {spy && !spy.error && <span className={`badge ${spy.rsi > 70 ? 'badge-red' : spy.rsi < 30 ? 'badge-green' : 'badge-blue'}`}>{spy.rsi > 70 ? 'Overbought' : spy.rsi < 30 ? 'Oversold' : 'Neutral'}</span>}
                    </div>
                    <ErrorBoundary>
                        {loading || !spy || spy.error ? <Skeleton count={5} /> : (
                            <>
                                {/* Hero price */}
                                <div className="hero-price-section">
                                    <div className="hero-price">${spy.current.toFixed(2)}</div>
                                    {spyDailyMove?.value ? (
                                        <div className={`daily-change-badge ${parseFloat(spyDailyMove.value) >= 0 ? 'daily-up' : 'daily-down'}`}>
                                            {parseFloat(spyDailyMove.value) >= 0 ? '▲' : '▼'} {spyDailyMove.value} today
                                        </div>
                                    ) : spy.dailyChange && (
                                        <div className={`daily-change-badge ${spy.dailyChange.pct >= 0 ? 'daily-up' : 'daily-down'}`}>
                                            {spy.dailyChange.pct >= 0 ? '▲' : '▼'} ${Math.abs(spy.dailyChange.value).toFixed(2)} ({spy.dailyChange.pct >= 0 ? '+' : ''}{spy.dailyChange.pct.toFixed(2)}%) today
                                        </div>
                                    )}
                                    <div className={`hero-change ${spy.ma200.pct >= 0 ? 'stat-positive' : 'stat-negative'}`} style={{ marginTop: '6px' }}>
                                        {spy.ma200.pct >= 0 ? '▲' : '▼'} {Math.abs(spy.ma200.pct).toFixed(2)}% {spy.ma200.pct >= 0 ? 'above' : 'below'} 200d MA
                                    </div>
                                    <div className={`hero-change`} style={{ color: spy.week52High.pct >= -1 ? 'var(--green)' : 'var(--yellow)', fontSize: '0.78rem', marginTop: '2px' }}>
                                        {spy.week52High.pct >= 0 ? '🔥 At 52-week high' : `${spy.week52High.pct.toFixed(2)}% from 52wk high ($${spy.week52High.value.toFixed(2)})`}
                                    </div>
                                </div>

                                {/* Stats grid */}
                                <div className="stats-mini-grid">
                                    <div className="stat-mini">
                                        <span className="stat-mini-label">200d MA</span>
                                        <span className="stat-mini-value">${spy.ma200.value.toFixed(2)}</span>
                                    </div>
                                    <div className="stat-mini">
                                        <span className="stat-mini-label">52w High</span>
                                        <span className={`stat-mini-value ${spy.week52High.pct >= 0 ? 'stat-positive' : 'stat-negative'}`}>${spy.week52High.value.toFixed(2)}</span>
                                    </div>
                                    <div className="stat-mini">
                                        <span className="stat-mini-label">3Y Return</span>
                                        <span className={`stat-mini-value ${spy.return3y == null ? '' : spy.return3y >= 0 ? 'stat-positive' : 'stat-negative'}`}>{spy.return3y == null ? 'N/A' : `${spy.return3y >= 0 ? '+' : ''}${spy.return3y.toFixed(2)}%`}</span>
                                    </div>
                                    <div className="stat-mini">
                                        <span className="stat-mini-label">9d RSI</span>
                                        <span className={`stat-mini-value ${spy.rsi > 70 ? 'stat-negative' : spy.rsi < 30 ? 'stat-positive' : ''}`}>{spy.rsi.toFixed(2)}</span>
                                    </div>
                                </div>

                                {/* RSI Gauge */}
                                <div className="gauge-section">
                                    <Gauge score={spy.rsi} segments={rsiSegments} labels={[0, 30, 50, 70, 100]} />
                                    <div className="gauge-inline-label">
                                        RSI: <strong>{spy.rsi.toFixed(2)}</strong>
                                        <span style={{ marginLeft: '8px', color: spy.rsi > 70 ? 'var(--red)' : spy.rsi < 30 ? 'var(--green)' : 'var(--text-muted)', fontSize: '0.7rem' }}>
                                            {spy.rsi > 70 ? 'OVERBOUGHT' : spy.rsi < 30 ? 'OVERSOLD' : 'NEUTRAL'}
                                        </span>
                                    </div>
                                </div>
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* ========== REDESIGNED FEAR & GREED CARD ========== */}
                <div className="card" style={{ animationDelay: '0.3s' }}>
                    <div className="card-header">
                        <h2>😨 Fear & Greed Index</h2>
                        {fg && !fg.error && <span className={`badge ${fg.score < 45 ? 'badge-red' : fg.score > 55 ? 'badge-green' : 'badge-yellow'}`}>{fg.rating}</span>}
                    </div>
                    <ErrorBoundary>
                        {loading || !fg || fg.error ? <Skeleton type="gauge" /> : (
                            <>
                                {/* Hero score */}
                                <div className="hero-price-section">
                                    <div className="hero-price" style={{ color: fgColor(fg.score) }}>{Math.round(fg.score)}</div>
                                    <div className="hero-change" style={{ color: fgColor(fg.score) }}>{fg.rating}</div>
                                </div>

                                {/* Gauge */}
                                <div className="gauge-section">
                                    <Gauge score={fg.score} segments={fgSegments} labels={[0, 25, 50, 75, 100]} />
                                </div>

                                {/* Historical */}
                                <div className="fg-history">
                                    {[
                                        { label: 'Prev Close', val: Math.round(fg.previousClose) },
                                        { label: '1 Week', val: Math.round(fg.previousWeek) },
                                        { label: '1 Month', val: Math.round(fg.previousMonth) },
                                        { label: '1 Year', val: Math.round(fg.previousYear) }
                                    ].map(h => {
                                        const current = Math.round(fg.score);
                                        const diff = current - h.val;
                                        const arrow = diff > 0 ? '▲' : diff < 0 ? '▼' : '—';
                                        const arrowColor = diff > 0 ? 'var(--green)' : diff < 0 ? 'var(--red)' : 'var(--text-muted)';
                                        return (
                                            <div key={h.label} className="fg-history-item">
                                                <div className="fg-history-label">{h.label}</div>
                                                <div className="fg-history-value">
                                                    {h.val}
                                                    <span style={{ marginLeft: '6px', fontSize: '0.7rem', color: arrowColor, fontWeight: 600 }}>
                                                        {arrow}{Math.abs(diff)}
                                                    </span>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* YIELD CURVE */}
                <div className="card" style={{ animationDelay: '0.4s' }}>
                    <div className="card-header">
                        <h2><span className="tooltip-trigger" data-tooltip={`When the 2-year yield is higher than the 10-year, it is a classic recession warning.${freshnessNote({ value: fred?.yieldCurve?.current, asOf: fred?.yieldCurve?.asOf, stale: fred?.yieldCurve?.stale }).suffix}`}>📈 Yield Curve (10Y-2Y)</span></h2>
                        {fred?.yieldCurve?.current != null && <span className={`badge ${fred.yieldCurve.current >= 0 ? 'badge-green' : 'badge-red'}`}>{fred.yieldCurve.current >= 0 ? 'Positive' : 'Inverted'}</span>}
                    </div>
                    <ErrorBoundary>
                        {loading || !fred ? <Skeleton count={2} /> : (fred.error || fred.yieldCurve?.current == null) ? (
                            <div className="hero-price-section">
                                <div className="hero-price" style={{ fontSize: '2.2rem', color: 'var(--yellow)' }}>N/A</div>
                                <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                    Unavailable — source busy, try again shortly
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="hero-price-section">
                                    <HeroValue markKey="yieldCurve" raw={fred.yieldCurve.current}
                                        stale={fred.yieldCurve.stale}
                                        format={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(3)}%`}
                                        style={{ fontSize: '2.2rem', color: fred.yieldCurve.stale ? 'var(--orange)' : fred.yieldCurve.current >= 0 ? 'var(--green)' : 'var(--red)' }}>
                                        {fred.yieldCurve.stale ? '🕐 ' : ''}{fred.yieldCurve.current >= 0 ? '+' : ''}{fred.yieldCurve.current.toFixed(3)}%
                                    </HeroValue>
                                    {fred.yieldCurve.stale && (
                                        <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                            Last data {formatAsOf(fred.yieldCurve.asOf)} (stale)
                                        </div>
                                    )}
                                </div>
                                <MiniChart history={fred.yieldCurve.history} color="#818cf8" gradientId="yieldGrad" showZero={true} recessions={fred.recessions || []} />
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* PROFIT MARGIN */}
                <div className="card" style={{ animationDelay: '0.45s' }}>
                    <div className="card-header">
                        <h2><span className="tooltip-trigger" data-tooltip={`Corporate Profits / GDP: High margins indicate strong corporate pricing power.${freshnessNote({ value: fred?.profitMargin?.current, asOf: fred?.profitMargin?.asOf, stale: fred?.profitMargin?.stale }).suffix}`}>💰 Profit Margin</span></h2>
                        {fred?.profitMargin?.current != null && <span className="badge badge-blue">Corp Profits / GDP</span>}
                    </div>
                    <ErrorBoundary>
                        {loading || !fred ? <Skeleton count={2} /> : (fred.error || fred.profitMargin?.current == null) ? (
                            <div className="hero-price-section">
                                <div className="hero-price" style={{ fontSize: '2.2rem', color: 'var(--yellow)' }}>N/A</div>
                                <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                    Unavailable — source busy, try again shortly
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="hero-price-section">
                                    <HeroValue markKey="profitMargin" raw={fred.profitMargin.current}
                                        stale={fred.profitMargin.stale} format={(v) => `${v.toFixed(2)}%`}
                                        style={{ fontSize: '2.2rem', color: fred.profitMargin.stale ? 'var(--orange)' : 'var(--green)' }}>
                                        {fred.profitMargin.stale ? '🕐 ' : ''}{fred.profitMargin.current.toFixed(2)}%
                                    </HeroValue>
                                    {fred.profitMargin.stale && (
                                        <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                            Last data {formatAsOf(fred.profitMargin.asOf)} (stale)
                                        </div>
                                    )}
                                </div>
                                <MiniChart history={fred.profitMargin.history} color="#22c55e" gradientId="profitGrad" recessions={fred.recessions || []} />
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* S&P 500 EPS */}
                <div className="card" style={{ animationDelay: '0.5s' }}>
                    <div className="card-header">
                        <h2><span className="tooltip-trigger" data-tooltip={`S&P 500 earnings per share, trailing 12 months (as-reported) — the E in P/E. Rising EPS means corporate America is earning more. History is inflation-adjusted (today's dollars).${freshnessNote({ value: fred?.spEps?.current, asOf: fred?.spEps?.asOf, stale: fred?.spEps?.stale }).suffix}`}>🧾 S&P 500 EPS</span></h2>
                        {fred?.spEps?.current != null && <span className="badge badge-blue">Trailing 12M</span>}
                    </div>
                    <ErrorBoundary>
                        {loading || !fred ? <Skeleton count={2} /> : (fred.error || fred.spEps?.current == null) ? (
                            <div className="hero-price-section">
                                <div className="hero-price" style={{ fontSize: '2.2rem', color: 'var(--yellow)' }}>N/A</div>
                                <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                    Unavailable — source busy, try again shortly
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="hero-price-section">
                                    {/* spEps has no daily snapshot, so it carries no mark — see lib/marks.js */}
                                    <HeroValue markKey={undefined} raw={fred.spEps.current}
                                        stale={fred.spEps.stale} format={(v) => `$${v.toFixed(2)}`}
                                        style={{ fontSize: '2.2rem', color: fred.spEps.stale ? 'var(--orange)' : 'var(--green)' }}>
                                        {fred.spEps.stale ? '🕐 ' : ''}${fred.spEps.current.toFixed(2)}
                                    </HeroValue>
                                    {fred.spEps.stale && (
                                        <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                            Last data {formatAsOf(fred.spEps.asOf)} (stale)
                                        </div>
                                    )}
                                </div>
                                <MiniChart history={fred.spEps.history} color="#38bdf8" gradientId="epsGrad" recessions={fred.recessions || []} cadence="monthly" />
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* ECONOMIC INDICATORS */}
                <EconomicIndicatorGrid fred={fred} loading={loading} statusColor={statusColor} />

                {/* FOUR HORSEMEN — RECESSION WATCH (full width) */}
                <FourHorsemen fred={fred} loading={loading} />

                {/* RUBBER BAND RADAR — is the dip-buying regime alive? (full width, nightly from the Mac mini) */}
                <RubberBandRadar />

                {/* SPY HISTORICAL CHART */}
                <div className="card" style={{ animationDelay: '0.55s' }}>
                    <div className="card-header">
                        <h2>📈 SPY Historical</h2>
                        <span className="badge badge-blue">Price + 200d MA</span>
                    </div>
                    <ErrorBoundary>
                        {loading || !spy || spy.error || !spy.chartHistory ? <Skeleton count={4} /> : (
                            <SpyChart chartHistory={spy.chartHistory} recessions={fred?.recessions || []} />
                        )}
                    </ErrorBoundary>
                </div>

                {/* VOLATILITY METRICS (IV rank / percentile / VRP) */}
                <VolMetricsTable />

                {/* BULL MARKET CHECKLIST */}
                <BullChecklist fred={fred} loading={loading} />

                {/* EXTRA MARKETS GRID */}
                <ExtraMarketsGrid data={extraMarkets} loading={loading} />

                {/* POLYMARKET TABLE - Integrated in grid naturally */}
                <div style={{ gridColumn: '1 / -1' }}>
                    <PolymarketTable />
                </div>

                {/* FINANCIAL DASHBOARD HISTORY LINK */}
                <div style={{ gridColumn: '1 / -1', textAlign: 'center', marginTop: '1rem' }}>
                    <a
                        href="https://docs.google.com/spreadsheets/d/1lA-_yjLMc3qDTt9sogSPQrCohNULIk5wwJYfb5wIHfc/edit?gid=0#gid=0"
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textDecoration: 'none', opacity: 0.5 }}
                    >
                        Financial Dashboard History
                    </a>
                </div>
            </div>

            {/* FOOTER */}
            <footer className="dashboard-footer">
                <p>Jalal's Financial Dashboard v7.0 — Data from FRED, CNN, Polygon, ExchangeRate-API, Yahoo Finance &amp; Google Sheets</p>
                {process.env.NEXT_PUBLIC_BUILD_TIME && (
                    <p style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: '4px' }}>
                        Deployed: {new Date(process.env.NEXT_PUBLIC_BUILD_TIME).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                    </p>
                )}
            </footer>

            {/* SYSTEM ERROR LOGS */}
            {apiErrors.length > 0 && (
                <div style={{
                    margin: '0 auto 24px',
                    maxWidth: '1200px',
                    width: 'calc(100% - 48px)',
                    padding: '16px',
                    backgroundColor: 'rgba(239, 68, 68, 0.05)',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    borderRadius: '8px',
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: '0.8rem',
                    color: 'rgba(255, 255, 255, 0.8)'
                }}>
                    <div style={{ fontWeight: 600, color: '#ef4444', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        ⚠️ System Diagnostic Logs
                    </div>
                    {apiErrors.map((err, i) => (
                        <div key={i} style={{ marginBottom: '4px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#fca5a5' }}>
                            {err}
                        </div>
                    ))}
                </div>
            )}

            {/* SYSTEM STATUS BAR */}
            {systemStatus && (
                <div className="system-status-bar">
                    <div className="status-items">
                        <span className={`status-item ${systemStatus.spy?.hasErrors ? 'status-error' :
                            systemStatus.spy?.source?.includes('FRED') ? 'status-warn' : ''
                            }`}>
                            [SPY: {
                                systemStatus.spy?.source?.includes('yfinance') ? 'yfinance' :
                                    systemStatus.spy?.source?.includes('Polygon') ? 'Polygon' :
                                        systemStatus.spy?.source?.includes('Google Sheet') ? 'GSheet' :
                                            systemStatus.spy?.source?.includes('FRED') ? 'FRED Fallback' :
                                                systemStatus.spy?.source || 'OK'
                            }]
                        </span>
                        <span className={`status-item ${systemStatus.fred?.hasErrors ? 'status-error' : ''}`}>
                            [FRED: {systemStatus.fred?.messages?.[0]?.replace('Loaded ', '').replace(' series', '') || '18/18'}]
                        </span>
                        <span className={`status-item ${systemStatus.fg?.source?.includes('Stale') || systemStatus.fg?.source?.includes('Failed') ? 'status-error' :
                            (systemStatus.fg?.source?.includes('VIX') || systemStatus.fg?.source?.includes('Proxy')) ? 'status-warn' :
                                systemStatus.fg?.hasErrors ? 'status-warn' : ''
                            }`}>
                            [F&G: {
                                systemStatus.fg?.source?.includes('CNN') ? 'CNN' :
                                    systemStatus.fg?.source?.includes('RapidAPI') ? 'RapidAPI' :
                                        systemStatus.fg?.source?.includes('VIXCLS') ? 'FRED VIX' :
                                            systemStatus.fg?.source?.includes('VIX') ? 'VIX Proxy' :
                                                systemStatus.fg?.source?.includes('Stale') ? 'STALE' :
                                                    systemStatus.fg?.source?.includes('Failed') ? 'FAILED' :
                                                        systemStatus.fg?.hasErrors ? 'PARTIAL' : 'LIVE OK'
                            }]
                        </span>
                        <span className={`status-item ${(systemStatus.sheets?.source?.includes('Failed') || systemStatus.sheets?.source?.includes('Static')) ? 'status-error' :
                            systemStatus.sheets?.source?.includes('Stale') ? 'status-error' :
                                (systemStatus.sheets?.source?.includes('Cached') || systemStatus.sheets?.source?.includes('Alt') || systemStatus.sheets?.source?.includes('Proxy') || systemStatus.sheets?.source?.includes('FRED')) ? 'status-warn' :
                                    ''
                            }`}>
                            [SHEETS: {
                                (systemStatus.sheets?.source?.includes('Failed') || systemStatus.sheets?.source?.includes('Static')) ? 'FAILED' :
                                    systemStatus.sheets?.source?.includes('Stale') ? 'STALE' :
                                        systemStatus.sheets?.source?.includes('Cached') ? 'CACHE' :
                                            systemStatus.sheets?.source?.includes('Alt') ? 'ALT OK' :
                                                (systemStatus.sheets?.source?.includes('Proxy') || systemStatus.sheets?.source?.includes('FRED')) ? 'FRED Proxy' :
                                                    'LIVE OK'
                            }]
                        </span>
                        {systemStatus.extra && (
                            <span className={`status-item ${systemStatus.extra?.messages?.some(m => m.includes('unavailable'))
                                ? 'status-error'
                                : systemStatus.extra?.hasErrors
                                    ? 'status-warn'
                                    : ''
                                }`}>
                                [MKTS: {systemStatus.extra?.messages?.join(' | ') || 'LIVE OK'}]
                            </span>
                        )}
                    </div>
                </div>
            )}
        </div>
        </MarkProvider>
    );
}
