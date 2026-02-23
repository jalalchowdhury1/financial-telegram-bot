'use client';
import { useState, useEffect } from 'react';
import ErrorBoundary from './ErrorBoundary';

// ============ GAUGE COMPONENT (SEMICIRCLE) ============
function Gauge({ score, segments, size = 240, labels }) {
    const cx = size / 2;
    const cy = size / 2 + 4;
    const outerR = size / 2 - 18;
    const thickness = outerR * 0.22;
    const midR = outerR - thickness / 2;     // arc centerline
    const clamp = Math.min(Math.max(score, 0), 100);
    const needleAngle = Math.PI - (clamp / 100) * Math.PI;
    const viewH = size / 2 + 16;

    // Helper: arc path from angle a1 to a2 at radius r
    function arcPath(a1Deg, a2Deg, r) {
        const a1 = (Math.PI / 180) * a1Deg;
        const a2 = (Math.PI / 180) * a2Deg;
        const x1 = cx + r * Math.cos(a1);
        const y1 = cy - r * Math.sin(a1);
        const x2 = cx + r * Math.cos(a2);
        const y2 = cy - r * Math.sin(a2);
        const sweep = a2Deg - a1Deg > 180 ? 1 : 0;
        return `M${x1},${y1} A${r},${r} 0 ${sweep} 0 ${x2},${y2}`;
    }

    return (
        <svg viewBox={`0 0 ${size} ${viewH}`} style={{ width: '100%', maxWidth: `${size}px`, display: 'block', margin: '0 auto' }}>
            {/* Background track */}
            <path
                d={arcPath(0, 180, midR)}
                fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={thickness}
                strokeLinecap="butt"
            />
            {/* Colored segments */}
            {segments.map(({ start, end, color }, i) => {
                // Convert 0-100 score range to 0-180 degrees
                const startAngle = (start / 100) * 180;
                const endAngle = (end / 100) * 180;
                return (
                    <path
                        key={i}
                        d={arcPath(startAngle, endAngle, midR)}
                        fill="none" stroke={color} strokeWidth={thickness}
                        strokeLinecap="butt"
                    />
                );
            })}
            {/* Tick labels */}
            {labels && labels.map((val, i) => {
                const theta = Math.PI - (val / 100) * Math.PI;
                return (
                    <text
                        key={i}
                        x={cx + (outerR + 10) * Math.cos(theta)}
                        y={cy - (outerR + 10) * Math.sin(theta)}
                        fill="rgba(255,255,255,0.35)" fontSize="9" fontWeight="600"
                        textAnchor="middle" dominantBaseline="middle"
                    >{val}</text>
                );
            })}
            {/* Needle */}
            <line
                x1={cx} y1={cy}
                x2={cx + (outerR - 4) * Math.cos(needleAngle)}
                y2={cy - (outerR - 4) * Math.sin(needleAngle)}
                stroke="#f1f5f9" strokeWidth="2.5" strokeLinecap="round"
            />
            <circle cx={cx} cy={cy} r="5" fill="#f1f5f9" />
            <circle cx={cx} cy={cy} r="2" fill="var(--bg-primary, #0a0e17)" />
        </svg>
    );
}

// ============ LOADING SKELETON ============
function Skeleton({ type = 'text', count = 3 }) {
    if (type === 'gauge') return <div className="skeleton skeleton-gauge" />;
    return (
        <div>
            {Array.from({ length: count }).map((_, i) => (
                <div key={i} className={`skeleton skeleton-text ${i === count - 1 ? 'short' : i % 2 ? 'medium' : ''}`} />
            ))}
        </div>
    );
}

// ============ SPY CHART (enhanced with timeframes, MAs, crosses) ============
function SpyChart({ chartHistory, recessions = [] }) {
    const [timeframe, setTimeframe] = useState('5Y');
    if (!chartHistory || chartHistory.length < 2) return null;

    const tfDays = { '1Y': 252, '5Y': 1260, '10Y': 2520, 'ALL': chartHistory.length };
    const sliceLen = Math.min(tfDays[timeframe] || chartHistory.length, chartHistory.length);
    const data = chartHistory.slice(-sliceLen);

    const w = 540, h = 220, padL = 45, padR = 8, padT = 12, padB = 24;
    const prices = data.map(d => d.price);
    const ma50s = data.map(d => d.ma50);
    const ma200s = data.map(d => d.ma200);
    const allVals = [...prices, ...ma50s, ...ma200s];
    const min = Math.min(...allVals), max = Math.max(...allVals);
    const range = max - min || 1;
    const dates = data.map(d => d.date);

    const toX = (i) => padL + (i / (data.length - 1)) * (w - padL - padR);
    const toY = (v) => h - padB - ((v - min) / range) * (h - padT - padB);

    const priceLine = prices.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
    const ma50Line = ma50s.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
    const ma200Line = ma200s.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');

    // Date to X for recessions
    const dateToX = (dateStr) => {
        const firstDate = new Date(dates[0]);
        const lastDate = new Date(dates[dates.length - 1]);
        const totalMs = lastDate - firstDate || 1;
        const d = new Date(dateStr);
        const frac = (d - firstDate) / totalMs;
        return padL + frac * (w - padL - padR);
    };
    const visibleRecessions = recessions.filter(r => r.start <= dates[dates.length - 1] && r.end >= dates[0]);

    // Year labels on x-axis
    const yearLabels = [];
    let lastYear = '';
    for (let i = 0; i < dates.length; i++) {
        const yr = dates[i].substring(0, 4);
        if (yr !== lastYear) { yearLabels.push({ x: toX(i), label: yr }); lastYear = yr; }
    }

    // Golden / Death Cross detection
    let lastCross = null;
    for (let i = 1; i < data.length; i++) {
        const prev50above = data[i - 1].ma50 > data[i - 1].ma200;
        const curr50above = data[i].ma50 > data[i].ma200;
        if (!prev50above && curr50above) lastCross = { type: 'golden', date: data[i].date, idx: i };
        else if (prev50above && !curr50above) lastCross = { type: 'death', date: data[i].date, idx: i };
    }

    // Price range labels
    const maxPrice = Math.max(...prices);
    const minPrice = Math.min(...prices);
    const returnPct = ((prices[prices.length - 1] - prices[0]) / prices[0]) * 100;

    // Y-axis price ticks (5 levels)
    const yTicks = [];
    for (let i = 0; i <= 4; i++) {
        const val = min + (range * i) / 4;
        yTicks.push({ y: toY(val), label: val >= 100 ? `$${Math.round(val)}` : `$${val.toFixed(1)}` });
    }

    return (
        <div>
            {/* Timeframe selector */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '8px', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: '4px' }}>
                    {['1Y', '5Y', '10Y', 'ALL'].map(tf => (
                        <button key={tf} onClick={() => setTimeframe(tf)}
                            style={{
                                padding: '3px 10px', borderRadius: '6px', border: 'none', cursor: 'pointer',
                                fontSize: '0.65rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
                                background: tf === timeframe ? 'rgba(56,189,248,0.2)' : 'rgba(255,255,255,0.05)',
                                color: tf === timeframe ? '#38bdf8' : 'var(--text-muted)',
                                transition: 'all 0.2s ease'
                            }}>{tf}</button>
                    ))}
                </div>
                <span style={{
                    fontSize: '0.65rem', fontFamily: "'JetBrains Mono', monospace",
                    color: returnPct >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 700
                }}>
                    {returnPct >= 0 ? '+' : ''}{returnPct.toFixed(1)}% return
                </span>
            </div>
            {/* Cross signal */}
            {lastCross && (
                <div style={{
                    fontSize: '0.68rem', fontWeight: 700, marginBottom: '6px', padding: '4px 10px',
                    borderRadius: '6px', display: 'inline-block',
                    background: lastCross.type === 'golden' ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                    color: lastCross.type === 'golden' ? 'var(--green)' : 'var(--red)'
                }}>
                    {lastCross.type === 'golden' ? '✨ Golden Cross' : '💀 Death Cross'} — {lastCross.date}
                </div>
            )}
            <div className="mini-chart" style={{ height: '220px' }}>
                <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
                    <defs>
                        <linearGradient id="spyGrad2" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.12" />
                            <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
                        </linearGradient>
                    </defs>
                    {/* Y-axis grid + labels */}
                    {yTicks.map((t, i) => (
                        <g key={i}>
                            <line x1={padL} x2={w - padR} y1={t.y} y2={t.y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                            <text x={padL - 4} y={t.y + 3} fill="rgba(255,255,255,0.25)" fontSize="7" fontFamily="JetBrains Mono, monospace" textAnchor="end">{t.label}</text>
                        </g>
                    ))}
                    {/* Year labels on x-axis */}
                    {yearLabels.map((yl, i) => (
                        <text key={i} x={yl.x} y={h - 4} fill="rgba(255,255,255,0.2)" fontSize="7" fontFamily="JetBrains Mono, monospace" textAnchor="middle">{yl.label}</text>
                    ))}
                    {/* Recession bands */}
                    {visibleRecessions.map((rec, i) => {
                        const x1 = Math.max(dateToX(rec.start), padL);
                        const x2 = Math.min(dateToX(rec.end), w - padR);
                        if (x2 <= x1) return null;
                        return <rect key={`rec-${i}`} x={x1} y={padT} width={x2 - x1} height={h - padT - padB} fill="rgba(239,68,68,0.08)" rx="2" />;
                    })}
                    {/* Price area fill */}
                    <polygon points={priceLine + ` ${w - padR},${h - padB} ${padL},${h - padB}`} fill="url(#spyGrad2)" />
                    {/* 200d MA */}
                    <polyline points={ma200Line} fill="none" stroke="#fb923c" strokeWidth="1.3" strokeLinejoin="round" opacity="0.7" />
                    {/* 50d MA */}
                    <polyline points={ma50Line} fill="none" stroke="#4ade80" strokeWidth="1.2" strokeLinejoin="round" opacity="0.7" />
                    {/* Price line */}
                    <polyline points={priceLine} fill="none" stroke="#38bdf8" strokeWidth="1.8" strokeLinejoin="round" />
                    {/* Cross marker */}
                    {lastCross && (
                        <circle cx={toX(lastCross.idx)} cy={toY(data[lastCross.idx].price)} r="3.5"
                            fill={lastCross.type === 'golden' ? '#4ade80' : '#ef4444'} stroke="#0a0e17" strokeWidth="1.5" />
                    )}
                </svg>
            </div>
            {/* Legend */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '14px', marginTop: '6px', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '0.6rem', color: '#38bdf8', fontWeight: 600 }}>━ Price</span>
                <span style={{ fontSize: '0.6rem', color: '#4ade80', fontWeight: 600 }}>━ 50d MA</span>
                <span style={{ fontSize: '0.6rem', color: '#fb923c', fontWeight: 600 }}>━ 200d MA</span>
                <span style={{ fontSize: '0.6rem', color: 'rgba(239,68,68,0.4)', fontWeight: 600 }}>█ Recessions</span>
            </div>
        </div>
    );
}
// ============ ENHANCED CHART (with timeframes, axes, recessions) ============
function MiniChart({ history, color = '#818cf8', gradientId = 'chartGrad', showZero = false, recessions = [], label = '' }) {
    const [timeframe, setTimeframe] = useState('5Y');
    if (!history || history.length < 2) return null;

    // Detect if data is quarterly (~4 points/yr) or daily (~252 points/yr)
    const isQuarterly = history.length < 500;
    const tfMap = isQuarterly
        ? { '10Y': 40, '20Y': 80, '30Y': 120, 'ALL': history.length }
        : { '1Y': 252, '5Y': 1260, '10Y': 2520, 'ALL': history.length };
    const tfKeys = isQuarterly ? ['10Y', '20Y', '30Y', 'ALL'] : ['1Y', '5Y', '10Y', 'ALL'];
    const defaultTf = isQuarterly ? 'ALL' : '5Y';

    // Use default if current timeframe isn't valid for this chart
    const activeTf = tfKeys.includes(timeframe) ? timeframe : defaultTf;
    const sliceLen = Math.min(tfMap[activeTf] || history.length, history.length);
    const data = history.slice(-sliceLen);

    const w = 480, h = 180, padL = 42, padR = 8, padT = 10, padB = 22;
    const values = data.map(d => d.value);
    const dates = data.map(d => d.date);
    const min = Math.min(...values), max = Math.max(...values);
    const range = max - min || 1;

    const toX = (i) => padL + (i / (data.length - 1)) * (w - padL - padR);
    const toY = (v) => h - padB - ((v - min) / range) * (h - padT - padB);

    const line = values.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
    const area = line + ` ${w - padR},${h - padB} ${padL},${h - padB}`;

    // Date to X for recessions
    const dateToX = (dateStr) => {
        const firstDate = new Date(dates[0]);
        const lastDate = new Date(dates[dates.length - 1]);
        const totalMs = lastDate - firstDate || 1;
        const d = new Date(dateStr);
        return padL + ((d - firstDate) / totalMs) * (w - padL - padR);
    };
    const visibleRecessions = recessions.filter(r => r.start <= dates[dates.length - 1] && r.end >= dates[0]);

    // Year labels
    const yearLabels = [];
    let lastYear = '';
    for (let i = 0; i < dates.length; i++) {
        const yr = dates[i].substring(0, 4);
        if (yr !== lastYear) { yearLabels.push({ x: toX(i), label: yr }); lastYear = yr; }
    }

    // Y-axis ticks
    const yTicks = [];
    for (let i = 0; i <= 4; i++) {
        const val = min + (range * i) / 4;
        yTicks.push({ y: toY(val), label: val >= 10 ? val.toFixed(1) : val.toFixed(2) });
    }

    // Change over period
    const change = values[values.length - 1] - values[0];
    const changePct = (change / Math.abs(values[0] || 1)) * 100;

    return (
        <div>
            {/* Timeframe selector + change */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '6px', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: '3px' }}>
                    {tfKeys.map(tf => (
                        <button key={tf} onClick={() => setTimeframe(tf)}
                            style={{
                                padding: '2px 8px', borderRadius: '5px', border: 'none', cursor: 'pointer',
                                fontSize: '0.6rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
                                background: tf === activeTf ? `${color}33` : 'rgba(255,255,255,0.05)',
                                color: tf === activeTf ? color : 'var(--text-muted)',
                                transition: 'all 0.2s ease'
                            }}>{tf}</button>
                    ))}
                </div>
                <span style={{
                    fontSize: '0.6rem', fontFamily: "'JetBrains Mono', monospace",
                    color: change >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 700
                }}>
                    {change >= 0 ? '▲' : '▼'} {Math.abs(change).toFixed(2)}
                </span>
            </div>
            <div className="mini-chart" style={{ height: '180px' }}>
                <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
                    <defs>
                        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={color} stopOpacity="0.2" />
                            <stop offset="100%" stopColor={color} stopOpacity="0" />
                        </linearGradient>
                    </defs>
                    {/* Y-axis grid + labels */}
                    {yTicks.map((t, i) => (
                        <g key={i}>
                            <line x1={padL} x2={w - padR} y1={t.y} y2={t.y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                            <text x={padL - 4} y={t.y + 3} fill="rgba(255,255,255,0.25)" fontSize="7" fontFamily="JetBrains Mono, monospace" textAnchor="end">{t.label}</text>
                        </g>
                    ))}
                    {/* Year labels */}
                    {yearLabels.map((yl, i) => (
                        <text key={i} x={yl.x} y={h - 4} fill="rgba(255,255,255,0.2)" fontSize="7" fontFamily="JetBrains Mono, monospace" textAnchor="middle">{yl.label}</text>
                    ))}
                    {/* Recession bands */}
                    {visibleRecessions.map((rec, i) => {
                        const x1 = Math.max(dateToX(rec.start), padL);
                        const x2 = Math.min(dateToX(rec.end), w - padR);
                        if (x2 <= x1) return null;
                        return <rect key={`rec-${i}`} x={x1} y={padT} width={x2 - x1} height={h - padT - padB} fill="rgba(239,68,68,0.08)" rx="2" />;
                    })}
                    {/* Zero line */}
                    {showZero && min < 0 && max > 0 && (
                        <line
                            x1={padL} x2={w - padR}
                            y1={toY(0)} y2={toY(0)}
                            stroke="rgba(239,68,68,0.35)" strokeDasharray="4,3" strokeWidth="1"
                        />
                    )}
                    <polygon points={area} fill={`url(#${gradientId})`} />
                    <polyline points={line} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
                </svg>
            </div>
        </div>
    );
}

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
            const [sheetsRes, spyRes, fgRes, fredRes] = await Promise.all([
                fetch('/api/sheets', { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch('/api/spy', { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch('/api/fear-greed', { cache: 'no-store' }).then(r => r.json()).catch(() => null),
                fetch('/api/fred', { cache: 'no-store' }).then(r => r.json()).catch(() => null),
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
                const assessRes = await fetch('/api/assessment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(assessData)
                }).then(r => r.json()).catch(() => null);
                setAssessment(assessRes?.assessment);
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
            <div className="indicator-bar">
                <div className={`indicator-pill${sheets?.NotSoBoring && sheets.NotSoBoring !== 'ON' ? ' pill-alert' : ''}`}>
                    <div className="label"><span className="emoji">🛡️</span>NotSoBoring</div>
                    <div className="value">{sheets?.NotSoBoring || (loading ? '...' : 'N/A')}</div>
                </div>
                <div className={`indicator-pill${sheets?.FrontRunner && !sheets.FrontRunner.startsWith('BIL') ? ' pill-alert' : ''}`}>
                    <div className="label"><span className="emoji">🔑</span>FrontRunner</div>
                    <div className="value">{sheets?.FrontRunner || (loading ? '...' : 'N/A')}</div>
                </div>
                <div className={`indicator-pill${sheets?.AAIIDiff && parseFloat(sheets.AAIIDiff) > 20 ? ' pill-alert' : ''}`}>
                    <div className="label"><span className="emoji">🔸</span>AAII Diff</div>
                    {sheets?.AAIIDiff ? (() => {
                        const val = parseFloat(sheets.AAIIDiff);
                        const isBullish = val > 20;
                        const targetDate = new Date();
                        targetDate.setMonth(targetDate.getMonth() + 6);
                        const dateStr = targetDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
                        return (
                            <>
                                <div className="value" style={{ color: isBullish ? 'var(--green)' : 'var(--text-primary)' }}>
                                    {sheets.AAIIDiff}
                                </div>
                                <div className="pill-detail" style={{ fontSize: '0.68rem', marginTop: '4px', color: isBullish ? 'var(--green)' : 'var(--text-muted)', fontWeight: 600 }}>
                                    {isBullish ? '🟢' : '⚪'} {isBullish ? 'Bullish' : 'Neutral'} outlook → {dateStr}
                                </div>
                                <div className="pill-detail" style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                                    Threshold: &gt;20% = bullish 6mo forward
                                </div>
                            </>
                        );
                    })() : <div className="value">{loading ? '...' : 'N/A'}</div>}
                </div>
                <div className={`indicator-pill${sheets?.VIX?.current && parseFloat(sheets.VIX.current) > parseFloat(sheets.VIX.threeMonth) ? ' pill-alert' : ''}`}>
                    <div className="label"><span className="emoji">🎢</span>VIX (Current | 3M)</div>
                    <div className="value">
                        {sheets?.VIX?.current
                            ? `${sheets.VIX.current} | ${sheets.VIX.threeMonth} | ${sheets.VIX.fearGreed}`
                            : (loading ? '...' : 'N/A')}
                    </div>
                </div>
            </div>

            {/* MARKET PULSE - Quick summary at top */}
            {!loading && spy && fg && !spy.error && !fg.error && (
                <div className="market-pulse">
                    <span className="pulse-label">📡 Market Pulse</span>
                    <span className="pulse-items">
                        <span className={spy.dailyChange?.pct >= 0 ? 'stat-positive' : 'stat-negative'}>
                            SPY {spy.dailyChange?.pct >= 0 ? '▲' : '▼'}{Math.abs(spy.dailyChange?.pct || 0).toFixed(2)}%
                        </span>
                        <span className="pulse-sep">·</span>
                        <span style={{ color: fgColor(fg.score) }}>
                            F&G {Math.round(fg.score)} {fg.rating}
                        </span>
                        <span className="pulse-sep">·</span>
                        <span className={spy.rsi > 70 ? 'stat-negative' : spy.rsi < 30 ? 'stat-positive' : ''}>
                            RSI {spy.rsi.toFixed(0)}
                        </span>
                        {fred?.checklist && (() => {
                            const bullish = Object.values(fred.checklist).filter(i => i.bullish).length;
                            const total = Object.values(fred.checklist).length;
                            return <><span className="pulse-sep">·</span><span style={{ color: 'var(--green)' }}>Bull {bullish}/{total}</span></>;
                        })()}
                    </span>
                </div>
            )}

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
                <div className="card" style={{ animationDelay: '0.5s' }}>
                    <div className="card-header">
                        <h2>📊 Economic Indicators</h2>
                    </div>
                    <ErrorBoundary>
                        {loading || !fred || fred.error ? <Skeleton count={6} /> : (
                            <>
                                {[
                                    { icon: fred.indicators.sahmRule?.status === 'safe' ? '✅' : '🔴', label: 'Sahm Rule', tooltip: "Recession indicator: triggers if 3mo average unemployment rises 0.5% above its 12mo low.", value: fred.indicators.sahmRule?.value?.toFixed(2) ?? 'N/A', status: fred.indicators.sahmRule?.status, benchmark: '< 0.50' },
                                    { icon: '🛒', label: 'Consumer Sentiment', tooltip: "University of Michigan survey assessing consumer confidence.", value: fred.indicators.sentiment?.value?.toFixed(1) ?? 'N/A', status: fred.indicators.sentiment?.status, benchmark: '> 80 strong' },
                                    { icon: '📋', label: 'Initial Claims (4wk)', tooltip: "4-week moving average of initial jobless claims. A critical real-time labor market gauge.", value: fred.indicators.claims?.value ? `${fred.indicators.claims.value.toFixed(0)}K` : 'N/A', status: fred.indicators.claims?.status, benchmark: '< 250K healthy' },
                                    { icon: '🏦', label: 'BBB Credit Spread', tooltip: "The premium corporations pay over Treasuries to borrow. Widening indicates market stress.", value: fred.indicators.creditSpread?.value ? `${fred.indicators.creditSpread.value.toFixed(2)}%` : 'N/A', status: fred.indicators.creditSpread?.status, benchmark: '< 1.5% tight' },
                                    { icon: '💵', label: 'Real Yields (10Y TIPS)', tooltip: "10-Year Treasury Inflation-Indexed Security. Shows the true inflation-adjusted cost of capital.", value: fred.indicators.realYields?.value ? `${fred.indicators.realYields.value.toFixed(2)}%` : 'N/A', status: fred.indicators.realYields?.status, benchmark: '< 0% easy' },
                                    { icon: '📊', label: 'Leading Economic Index', tooltip: "The Conference Board LEI tracks 10 forward-looking economic components.", value: fred.indicators.lei?.change ? `${fred.indicators.lei.change >= 0 ? '+' : ''}${fred.indicators.lei.change.toFixed(2)}%` : 'N/A', status: fred.indicators.lei?.status, benchmark: '> 0% rising' },
                                    { icon: '💎', label: 'Market Valuation', tooltip: "Current S&P 500 P/E Ratio. A measure of how expensive the market is historically.", value: fred.peRatio ? `P/E ~${fred.peRatio.toFixed(1)}` : 'P/E N/A', status: fred.peRatio > 25 ? 'restrictive' : 'neutral', benchmark: 'Fair at ~20' },
                                ].map(ind => (
                                    <div className="stat-row" key={ind.label}>
                                        <span className="stat-label">
                                            {ind.icon} <span className="tooltip-trigger" data-tooltip={ind.tooltip}>{ind.label}</span>
                                        </span>
                                        <span className="stat-right">
                                            <span className={`stat-value ${statusColor(ind.status)}`}>{ind.value}</span>
                                            <span className="stat-benchmark">{ind.benchmark}</span>
                                        </span>
                                    </div>
                                ))}
                            </>
                        )}
                    </ErrorBoundary>
                </div>

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
                <div className="card full-width" style={{ animationDelay: '0.6s' }}>
                    <div className="card-header">
                        <h2>📋 Bull Market Checklist</h2>
                        {fred?.checklist && (() => {
                            const items = Object.values(fred.checklist);
                            const bullish = items.filter(i => i.bullish).length;
                            const pct = (bullish / items.length) * 100;
                            return <span className={`badge ${pct >= 75 ? 'badge-green' : pct >= 50 ? 'badge-yellow' : 'badge-red'}`}>{bullish}/{items.length} ({pct.toFixed(0)}%)</span>;
                        })()}
                    </div>
                    <ErrorBoundary>
                        {loading || !fred || fred.error ? <Skeleton count={8} /> : (
                            <>
                                <div className="checklist-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
                                    {Object.entries(fred.checklist).map(([key, item]) => (
                                        <div className="checklist-item" key={key} style={{ padding: '12px 14px', alignItems: 'center' }}>
                                            <span className="checklist-icon">{item.bullish ? '✅' : '🔴'}</span>
                                            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '3px' }}>
                                                <span className="checklist-text" style={{ flex: 'none', color: 'var(--text-primary)' }}>{item.label}</span>
                                                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', lineHeight: 1.2 }}>
                                                    {{
                                                        nfci: 'System tightness (<0 = easy, >0 = tight)',
                                                        m2: 'YoY liquidity growth (>2% = expanding)',
                                                        retail: 'Consumer spending strength',
                                                        housing: 'Housing market health (>1,400K = strong, >1,300K = OK)',
                                                        indpro: '6-month manufacturing trend',
                                                        jolts: 'Labor demand (>7,000K = strong, >6,000K = OK)',
                                                        durable: 'Business investment (3mo trend)',
                                                        savings: 'Consumer cushion (>5% = healthy, ≥ 3.5% = OK)'
                                                    }[key]}
                                                </span>
                                            </div>
                                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '3px' }}>
                                                <span className={`checklist-value ${item.bullish ? 'stat-positive' : 'stat-negative'}`} style={{ fontSize: '0.95rem' }}>
                                                    {typeof item.value === 'number' ? (
                                                        key === 'housing' || key === 'jolts' ? `${item.value.toFixed(0)}K` :
                                                            key === 'nfci' ? `${item.value.toFixed(2)}` :
                                                                `${item.value >= 0 && key !== 'nfci' ? '+' : ''}${item.value.toFixed(1)}%`
                                                    ) : item.value}
                                                </span>
                                                <span className="checklist-benchmark" style={{ opacity: 0.9 }}>
                                                    {{
                                                        nfci: item.status === 'strong' ? '← Easy' : item.status === 'good' ? '← Easy' : '← Tight',
                                                        m2: item.status === 'strong' ? '← Growing' : item.status === 'good' ? '← Growing' : '← Contracting',
                                                        retail: item.status === 'strong' ? '← Growing' : item.status === 'good' ? '← Growing' : '← Declining',
                                                        housing: item.status === 'strong' ? '← Strong' : item.status === 'good' ? '← OK' : '← Weak',
                                                        indpro: item.status === 'strong' ? '← Expanding' : item.status === 'good' ? '← Expanding' : '← Contracting',
                                                        jolts: item.status === 'strong' ? '← Strong' : item.status === 'good' ? '← OK' : '← Weak',
                                                        durable: item.status === 'strong' ? '← Rising' : item.status === 'good' ? '← Rising' : '← Falling',
                                                        savings: item.status === 'strong' ? '← Healthy' : item.status === 'good' ? '← OK' : '← Low'
                                                    }[key]}
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                {(() => {
                                    const items = Object.values(fred.checklist);
                                    const bullish = items.filter(i => i.bullish).length;
                                    const pct = (bullish / items.length) * 100;
                                    const regime = pct >= 75 ? '🟢 CONFIRMED BULL MARKET' : pct >= 50 ? '🟡 CAUTIOUS / MIXED' : '🔴 BEAR MARKET WARNING';
                                    const bg = pct >= 75 ? 'var(--green-bg)' : pct >= 50 ? 'var(--yellow-bg)' : 'var(--red-bg)';
                                    const color = pct >= 75 ? 'var(--green)' : pct >= 50 ? 'var(--yellow)' : 'var(--red)';
                                    return (
                                        <div className="checklist-score" style={{ background: bg, color }}>
                                            {regime} — Score: {bullish}/{items.length} ({pct.toFixed(0)}% Bullish)
                                        </div>
                                    );
                                })()}
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* AI ASSESSMENT */}
                <div className="card full-width" style={{ animationDelay: '0.7s' }}>
                    <div className="card-header">
                        <h2>🤖 AI Market Assessment</h2>
                        <span className="badge badge-purple">AI-Powered</span>
                    </div>
                    <ErrorBoundary>
                        {!assessment ? (
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
