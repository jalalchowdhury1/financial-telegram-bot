'use client';
import { useState, useEffect } from 'react';
import ErrorBoundary from '../components/ErrorBoundary';

import Gauge from '../components/Gauge';
import Skeleton from '../components/Skeleton';
import SpyChart from '../components/SpyChart';
import MiniChart from '../components/MiniChart';
import MarketPulse from '../components/MarketPulse';
import CustomIndicatorBar from '../components/CustomIndicatorBar';
import EconomicIndicatorGrid from '../components/EconomicIndicatorGrid';
import BullChecklist from '../components/BullChecklist';

// ============ MAIN DASHBOARD ============
export default function Dashboard() {
    const [sheets, setSheets] = useState(null);
    const [spy, setSpy] = useState(null);
    const [fg, setFg] = useState(null);
    const [fred, setFred] = useState(null);
    const [assessment, setAssessment] = useState(null);
    const [loading, setLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState(null);
    const [systemStatus, setSystemStatus] = useState(null);
    const [apiErrors, setApiErrors] = useState([]);
    const [refreshing, setRefreshing] = useState(false);

    async function fetchAll() {
        setLoading(true);
        setRefreshing(true);
        setApiErrors([]);
        try {
            const timestamp = Date.now();
            const [sheetsRes, spyRes, fgRes, fredRes] = await Promise.all([
                fetch(`/api/sheets?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch(`/api/spy?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch(`/api/fear-greed?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch(`/api/fred?_t=${timestamp}`, { cache: 'no-store' }).then(r => r.json()).catch(() => null),
            ]);

            setSheets(sheetsRes);
            setSpy(spyRes);
            setFg(fgRes);
            setFred(fredRes);

            setSystemStatus({
                spy: spyRes?._meta,
                fred: fredRes?._meta,
                fg: fgRes?._meta,
                sheets: sheetsRes?._meta
            });

            const errors = [];
            if (sheetsRes?.error) errors.push(`[SHEETS] ${sheetsRes.error}`);
            if (spyRes?.error) errors.push(`[SPY] ${spyRes.error}`);
            if (fgRes?.error) errors.push(`[F&G] ${fgRes.error}`);
            if (fredRes?.error) errors.push(`[FRED] ${fredRes.error}`);
            setApiErrors(errors);

            const now = new Date();
            const year = now.getFullYear();
            const month = String(now.getMonth() + 1).padStart(2, '0');
            const day = String(now.getDate()).padStart(2, '0');
            const hours = String(now.getHours()).padStart(2, '0');
            const minutes = String(now.getMinutes()).padStart(2, '0');
            setLastUpdated(`${year}-${month}-${day} ${hours}:${minutes}`);

            if (fredRes && fgRes && !fredRes.error && !fgRes.error) {
                const assessData = {
                    yieldCurve: fredRes.yieldCurve?.current,
                    sahmRule: fredRes.indicators?.sahmRule?.value,
                    sentiment: fredRes.indicators?.sentiment?.value,
                    claims: fredRes.indicators?.claims?.value,
                    creditSpread: fredRes.indicators?.creditSpread?.value,
                    realYields: fredRes.indicators?.realYields?.value,
                    leiChange: fredRes.indicators?.lei?.change,
                    fearGreed: fgRes.score
                };
                try {
                    const assessRes = await fetch('/api/assessment', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(assessData)
                    });
                    if (!assessRes.ok) {
                        const errText = await assessRes.text();
                        console.error('Assessment API returned non-OK:', errText);
                        setAssessment(`⚠️ Network Error: Could not reach Assessment API (${assessRes.status})`);
                        setApiErrors(prev => [...prev, `[ASSESSMENT] ${assessRes.status} Error: ${errText.substring(0, 50)}`]);
                    } else {
                        const json = await assessRes.json();
                        setAssessment(json.assessment);
                    }
                } catch (assessFetchErr) {
                    console.error('Failed to parse or fetch assessment:', assessFetchErr);
                    setAssessment('⚠️ Fetch Error: Failed to contact Assessment API.');
                    setApiErrors(prev => [...prev, `[ASSESSMENT] Fetch failed: ${assessFetchErr.message}`]);
                }
            }
        } catch (e) {
            console.error('Dashboard fetch error:', e);
            setApiErrors(prev => [...prev, `[NETWORK] ${e.toString()}`]);
        }
        setLoading(false);
        setRefreshing(false);
    }

    useEffect(() => {
        fetchAll();
        const interval = setInterval(fetchAll, 5 * 60 * 1000); // Auto-refresh every 5 minutes
        return () => clearInterval(interval);
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
        <div className="dashboard">
            {/* Auto-Refresh Visualizer */}
            {lastUpdated && <div key={lastUpdated} className="auto-refresh-bar" style={{ animation: 'progress-fill 300s linear forwards' }}></div>}

            {/* HEADER */}
            <header className="dashboard-header">
                <h1>Jalal's Financial Dashboard</h1>
                <p className="subtitle">Live market data, economic indicators & AI-powered assessment</p>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', marginTop: '12px' }}>
                    <div className="live-badge">
                        <span className="live-dot" />
                        {loading ? 'Loading live data...' : `Updated ${lastUpdated}`}
                    </div>
                    <button className="refresh-btn" onClick={fetchAll} disabled={refreshing} title="Refresh all data">
                        <svg className={refreshing ? 'spinning' : ''} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="23 4 23 10 17 10" />
                            <polyline points="1 20 1 14 7 14" />
                            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                        </svg>
                    </button>
                </div>
            </header>

            {/* CUSTOM INDICATOR BAR */}
            <CustomIndicatorBar sheets={sheets} loading={loading} />

            {/* MARKET PULSE - Quick summary at top */}
            <MarketPulse spy={spy} fg={fg} fred={fred} loading={loading} fgColor={fgColor} />

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
                                    {spy.dailyChange && (
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
                                        <span className={`stat-mini-value ${spy.return3y >= 0 ? 'stat-positive' : 'stat-negative'}`}>{spy.return3y >= 0 ? '+' : ''}{spy.return3y.toFixed(2)}%</span>
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
                        <h2><span className="tooltip-trigger" data-tooltip="When the 2-year yield is higher than the 10-year, it is a classic recession warning.">📈 Yield Curve (10Y-2Y)</span></h2>
                        {fred?.yieldCurve?.current !== undefined && <span className={`badge ${fred.yieldCurve.current >= 0 ? 'badge-green' : 'badge-red'}`}>{fred.yieldCurve.current >= 0 ? 'Positive' : 'Inverted'}</span>}
                    </div>
                    <ErrorBoundary>
                        {loading || !fred || fred.error || fred.yieldCurve?.current === undefined ? <Skeleton count={2} /> : (
                            <>
                                <div className="hero-price-section">
                                    <div className="hero-price" style={{ fontSize: '2.2rem', color: fred.yieldCurve.current >= 0 ? 'var(--green)' : 'var(--red)' }}>
                                        {fred.yieldCurve.current >= 0 ? '+' : ''}{fred.yieldCurve.current.toFixed(3)}%
                                    </div>
                                </div>
                                <MiniChart history={fred.yieldCurve.history} color="#818cf8" gradientId="yieldGrad" showZero={true} recessions={fred.recessions || []} />
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* PROFIT MARGIN */}
                <div className="card" style={{ animationDelay: '0.45s' }}>
                    <div className="card-header">
                        <h2><span className="tooltip-trigger" data-tooltip="Corporate Profits / GDP: High margins indicate strong corporate pricing power.">💰 Profit Margin</span></h2>
                        {fred?.profitMargin && <span className="badge badge-blue">Corp Profits / GDP</span>}
                    </div>
                    <ErrorBoundary>
                        {loading || !fred || fred.error || !fred.profitMargin ? <Skeleton count={2} /> : (
                            <>
                                <div className="hero-price-section">
                                    <div className="hero-price" style={{ fontSize: '2.2rem', color: 'var(--green)' }}>
                                        {fred.profitMargin.current.toFixed(2)}%
                                    </div>
                                </div>
                                <MiniChart history={fred.profitMargin.history} color="#22c55e" gradientId="profitGrad" recessions={fred.recessions || []} />
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* ECONOMIC INDICATORS */}
                <EconomicIndicatorGrid fred={fred} loading={loading} statusColor={statusColor} />

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

                {/* BULL MARKET CHECKLIST */}
                <BullChecklist fred={fred} loading={loading} />

                {/* AI ASSESSMENT */}
                <div className="card full-width" style={{ animationDelay: '0.7s' }}>
                    <div className="card-header">
                        <h2>🤖 AI Market Assessment</h2>
                        <span className="badge badge-purple">AI-Powered</span>
                    </div>
                    <ErrorBoundary>
                        {loading || !assessment ? (
                            loading ? <Skeleton count={4} /> : <div className="error-message">Assessment unavailable</div>
                        ) : (
                            <div className="ai-assessment">{assessment}</div>
                        )}
                    </ErrorBoundary>
                </div>
            </div>

            {/* FOOTER */}
            <footer className="dashboard-footer">
                <p>Jalal's Financial Dashboard v7.0 — Data from FRED, CNN, Stooq & Google Sheets</p>
                <p style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: '4px' }}>
                    Deployed: {new Date(process.env.NEXT_PUBLIC_VERCEL_GIT_COMMIT_MESSAGE ? Date.now() : Date.now()).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                </p>
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
                        <span className={`status-item ${systemStatus.spy?.hasErrors ? 'status-error' : systemStatus.spy?.source?.includes('Yahoo') ? 'status-warn' : ''}`}>
                            [SPY: {systemStatus.spy?.source?.includes('Yahoo') ? 'Yahoo Fallback' : systemStatus.spy?.source || 'OK'}]
                        </span>
                        <span className={`status-item ${systemStatus.fred?.hasErrors ? 'status-error' : ''}`}>
                            [FRED: {systemStatus.fred?.messages?.[0]?.replace('Loaded ', '').replace(' series', '') || '18/18'}]
                        </span>
                        <span className={`status-item ${systemStatus.fg?.hasErrors ? 'status-error' : ''}`}>
                            [F&G: {systemStatus.fg?.hasErrors ? 'PARTIAL' : 'LIVE OK'}]
                        </span>
                        <span className={`status-item ${systemStatus.sheets?.source === 'Failed' ? 'status-error' : systemStatus.sheets?.hasErrors ? 'status-warn' : ''}`}>
                            [SHEETS: {systemStatus.sheets?.source === 'Failed' ? 'FAILED' : systemStatus.sheets?.hasErrors ? 'CACHE' : 'LIVE OK'}]
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
}
