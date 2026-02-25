'use client';
import { useState } from 'react';

export default function SpyChart({ chartHistory, recessions = [] }) {
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
