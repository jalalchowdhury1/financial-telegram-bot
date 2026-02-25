'use client';
import { useState } from 'react';

export default function MiniChart({ history, color = '#818cf8', gradientId = 'chartGrad', showZero = false, recessions = [], label = '' }) {
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
