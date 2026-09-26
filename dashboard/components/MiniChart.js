'use client';
import { useState, useEffect, useRef } from 'react';
import { yearTicks, indexFromPointer, tfAvailable, fmtDay, readChoice, saveChoice } from '../lib/chartAxis';

export default function MiniChart({ history, color = '#818cf8', gradientId = 'chartGrad', showZero = false, recessions = [], label = '', cadence = 'auto', defaultTimeframe = null, fmt = null }) {
    const [timeframe, setTimeframe] = useState(defaultTimeframe || '5Y');
    const [hover, setHover] = useState(null);
    const svgRef = useRef(null);
    // Each chart remembers its timeframe per device (gradientId is unique per card).
    const storeKey = `ftb:tf:${gradientId}`;
    useEffect(() => { const v = readChoice(storeKey); if (v) setTimeframe(v); }, [storeKey]);
    if (!history || history.length < 2) return null;

    // Explicit cadence='monthly' (12 points/yr) or 'weekly' (52 points/yr)
    // unlocks the right tab row; otherwise auto-detect quarterly (~4 points/yr)
    // vs daily (~252 points/yr) from the point count, as before.
    let tfMap, tfKeys, defaultTf;
    if (cadence === 'monthly') {
        tfMap = { '1Y': 12, '3Y': 36, '5Y': 60, '10Y': 120, '20Y': 240, '30Y': 360, 'ALL': history.length };
        tfKeys = ['1Y', '3Y', '5Y', '10Y', '20Y', '30Y', 'ALL'];
        defaultTf = 'ALL';
    } else if (cadence === 'weekly') {
        tfMap = { '1Y': 52, '3Y': 156, '5Y': 260, '10Y': 520, '20Y': 1040, 'ALL': history.length };
        tfKeys = ['1Y', '3Y', '5Y', '10Y', '20Y', 'ALL'];
        defaultTf = 'ALL';
    } else if (history.length < 500) {
        tfMap = { '10Y': 40, '20Y': 80, '30Y': 120, 'ALL': history.length };
        tfKeys = ['10Y', '20Y', '30Y', 'ALL'];
        defaultTf = 'ALL';
    } else {
        tfMap = { '1Y': 252, '5Y': 1260, '10Y': 2520, 'ALL': history.length };
        tfKeys = ['1Y', '5Y', '10Y', 'ALL'];
        defaultTf = '5Y';
    }

    // Only offer tabs the history really covers; fall back to the default, then ALL.
    const usable = tfKeys.filter((tf) => tfAvailable(tf === 'ALL' ? null : tfMap[tf], history.length));
    const activeTf = usable.includes(timeframe) ? timeframe : usable.includes(defaultTf) ? defaultTf : 'ALL';
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

    // Year labels, thinned so a 1947→today axis stays readable
    const yearLabels = yearTicks(dates, toX);

    // Y-axis ticks (fmt lets big-number series render compact labels like 350K)
    const fmtTick = fmt || ((v) => (v >= 10 ? v.toFixed(1) : v.toFixed(2)));
    const yTicks = [];
    for (let i = 0; i <= 4; i++) {
        const val = min + (range * i) / 4;
        yTicks.push({ y: toY(val), label: fmtTick(val) });
    }

    // Change over period
    const change = values[values.length - 1] - values[0];
    const changePct = (change / Math.abs(values[0] || 1)) * 100;

    // Tap / hover readout: the point under the pointer replaces the change label.
    const fmtVal = fmt || ((v) => v.toFixed(2));
    const hi = hover != null && hover < data.length ? hover : null;
    const pick = (tf) => { setTimeframe(tf); saveChoice(storeKey, tf); setHover(null); };
    const onPoint = (e) => {
        const i = indexFromPointer(e.clientX, svgRef.current?.getBoundingClientRect(), data.length, { w, padL, padR });
        if (i != null) setHover(i);
    };
    const onLeave = (e) => { if (e.pointerType === 'mouse') setHover(null); };

    return (
        <div>
            {/* Timeframe selector + change */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '6px', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: '3px' }}>
                    {tfKeys.map(tf => (
                        <button key={tf} onClick={() => pick(tf)} disabled={!usable.includes(tf)}
                            title={usable.includes(tf) ? undefined : 'Not enough history yet'}
                            style={{
                                padding: '2px 8px', borderRadius: '5px', border: 'none',
                                cursor: usable.includes(tf) ? 'pointer' : 'default', opacity: usable.includes(tf) ? 1 : 0.35,
                                fontSize: '0.6rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
                                background: tf === activeTf ? `${color}33` : 'rgba(255,255,255,0.05)',
                                color: tf === activeTf ? color : 'var(--text-muted)',
                                transition: 'all 0.2s ease'
                            }}>{tf}</button>
                    ))}
                </div>
                {hi != null ? (
                    <span className="chart-readout" style={{
                        fontSize: '0.6rem', fontFamily: "'JetBrains Mono', monospace",
                        color: 'var(--text-primary)', fontWeight: 700, whiteSpace: 'nowrap'
                    }}>
                        {fmtDay(dates[hi])} · {fmtVal(values[hi])}
                    </span>
                ) : (
                    <span style={{
                        fontSize: '0.6rem', fontFamily: "'JetBrains Mono', monospace",
                        color: change >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 700
                    }}>
                        {change >= 0 ? '▲' : '▼'} {fmt ? fmt(Math.abs(change)) : Math.abs(change).toFixed(2)}
                    </span>
                )}
            </div>
            <div className="mini-chart" style={{ height: '180px' }}>
                <svg ref={svgRef} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
                    onPointerMove={onPoint} onPointerDown={onPoint} onPointerLeave={onLeave}
                    style={{ touchAction: 'pan-y' }}>
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
                    {hi != null && (
                        <g className="chart-cursor" pointerEvents="none">
                            <line x1={toX(hi)} x2={toX(hi)} y1={padT} y2={h - padB} stroke="rgba(255,255,255,0.35)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
                            <circle cx={toX(hi)} cy={toY(values[hi])} r="3" fill={color} stroke="#0a0e17" strokeWidth="1.5" />
                        </g>
                    )}
                </svg>
            </div>
        </div>
    );
}
