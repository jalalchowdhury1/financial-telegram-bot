'use client';
import { useState, useEffect, useRef } from 'react';
import { yearTicks, indexFromPointer, tfAvailable, fmtDay, readChoice, saveChoice, unitsForPx, gutterFor, clampUnits, LABEL_CHAR_PX } from '../lib/chartAxis';
import useElementWidth from './useElementWidth';
import AxisLabels from './AxisLabels';

const TF_KEYS = ['1Y', '5Y', '10Y', 'ALL'];
const STORE_KEY = 'ftb:tf:spy';

export default function SpyChart({ chartHistory, recessions = [], current = null }) {
    const [timeframe, setTimeframe] = useState('5Y');
    const [hover, setHover] = useState(null);
    const svgRef = useRef(null);
    const [plotRef, plotPx] = useElementWidth(); // real px width → 10px labels spaced honestly
    useEffect(() => { const v = readChoice(STORE_KEY); if (v) setTimeframe(v); }, []);
    if (!chartHistory || chartHistory.length < 2) return null;

    const tfDays = { '1Y': 252, '5Y': 1260, '10Y': 2520, 'ALL': chartHistory.length };
    // Only offer tabs the history covers (the Polygon path carries ~14 months, so
    // "5Y" there was 14 months under a 5Y label). Fall back to 5Y, then ALL.
    const usable = TF_KEYS.filter((tf) => tfAvailable(tf === 'ALL' ? null : tfDays[tf], chartHistory.length));
    const activeTf = usable.includes(timeframe) ? timeframe : usable.includes('5Y') ? '5Y' : 'ALL';
    const sliceLen = Math.min(tfDays[activeTf] || chartHistory.length, chartHistory.length);
    const data = chartHistory.slice(-sliceLen);

    const w = 540, h = 220, padR = 8, padT = 12, padB = 24;
    const prices = data.map(d => d.price);
    const ma50s = data.map(d => d.ma50);
    const ma200s = data.map(d => d.ma200);
    const allVals = [...prices, ...ma50s, ...ma200s];
    const min = Math.min(...allVals), max = Math.max(...allVals);
    const range = max - min || 1;
    const dates = data.map(d => d.date);
    const toY = (v) => h - padB - ((v - min) / range) * (h - padT - padB);

    // The Google Sheet path charts FRED's S&P 500 INDEX (~10x SPY), not SPY dollars.
    // Label it as index points instead of "$7,799" next to a $771 SPY price.
    const isIndex = current > 0 && prices[prices.length - 1] > 3 * current;

    // Y-axis price ticks (5 levels)
    const yTicks = [];
    for (let i = 0; i <= 4; i++) {
        const val = min + (range * i) / 4;
        yTicks.push({ y: toY(val), label: isIndex ? Math.round(val).toLocaleString('en-US') : val >= 100 ? `$${Math.round(val)}` : `$${val.toFixed(1)}` });
    }
    // The left gutter fits the longest tick at a real 10px (the old 45 units until measured).
    const padL = gutterFor(yTicks.map((t) => t.label), w, plotPx, 45);

    const toX = (i) => padL + (i / (data.length - 1)) * (w - padL - padR);

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

    // Year labels on x-axis, thinned so they never overlap (at a real 10px once measured)
    const yearLabels = yearTicks(dates, toX, { minGap: Math.max(26, unitsForPx(LABEL_CHAR_PX * 4 + 10, w, plotPx) ?? 26) });
    const lblGap = unitsForPx(4, w, plotPx) ?? 4;
    const axisLabels = [
        ...yTicks.map((t) => ({ x: padL - lblGap, y: t.y, text: t.label, ax: 'end', ay: 'middle' })),
        ...yearLabels.map((yl) => ({ x: clampUnits(yl.x, (yl.label.length * LABEL_CHAR_PX) / 2, w, plotPx), y: h, text: yl.label, ax: 'middle', ay: 'bottom' })),
    ];

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

    // Tap / hover readout: the point under the pointer replaces the return label.
    const hi = hover != null && hover < data.length ? hover : null;
    const pick = (tf) => { setTimeframe(tf); saveChoice(STORE_KEY, tf); setHover(null); };
    const onPoint = (e) => {
        const i = indexFromPointer(e.clientX, svgRef.current?.getBoundingClientRect(), data.length, { w, padL, padR });
        if (i != null) setHover(i);
    };
    const onLeave = (e) => { if (e.pointerType === 'mouse') setHover(null); };
    const fmtPx = (v) => (v == null ? '—' : isIndex ? Math.round(v).toLocaleString('en-US') : `$${v.toFixed(2)}`);

    return (
        <div>
            {/* Timeframe selector */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '8px', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: '4px' }}>
                    {TF_KEYS.map(tf => (
                        <button key={tf} className="tf-btn tf-btn-spy" onClick={() => pick(tf)} disabled={!usable.includes(tf)}
                            title={usable.includes(tf) ? undefined : 'Not enough history from this source'}
                            style={{
                                padding: '3px 10px', borderRadius: '6px', border: 'none',
                                cursor: usable.includes(tf) ? 'pointer' : 'default', opacity: usable.includes(tf) ? 1 : 0.35,
                                fontSize: '0.65rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
                                background: tf === activeTf ? 'rgba(56,189,248,0.2)' : 'rgba(255,255,255,0.05)',
                                color: tf === activeTf ? '#38bdf8' : 'var(--text-muted)',
                                transition: 'all 0.2s ease'
                            }}>{tf}</button>
                    ))}
                </div>
                {hi != null ? (
                    <span className="chart-readout" style={{
                        fontSize: '0.65rem', fontFamily: "'JetBrains Mono', monospace",
                        color: 'var(--text-primary)', fontWeight: 700, whiteSpace: 'nowrap'
                    }}>
                        {fmtDay(dates[hi])} · {fmtPx(prices[hi])}
                    </span>
                ) : (
                    <span style={{
                        fontSize: '0.65rem', fontFamily: "'JetBrains Mono', monospace",
                        color: returnPct >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 700
                    }}>
                        {returnPct >= 0 ? '+' : ''}{returnPct.toFixed(1)}% return
                    </span>
                )}
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
                <div className="chart-plot" ref={plotRef}>
                    <svg ref={svgRef} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
                        onPointerMove={onPoint} onPointerDown={onPoint} onPointerLeave={onLeave}
                        style={{ touchAction: 'pan-y' }}>
                        <defs>
                            <linearGradient id="spyGrad2" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.12" />
                                <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
                            </linearGradient>
                        </defs>
                        {/* Y-axis grid (its labels are HTML, below) */}
                        {yTicks.map((t, i) => (
                            <line key={i} x1={padL} x2={w - padR} y1={t.y} y2={t.y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
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
                        {hi != null && (
                            <g className="chart-cursor" pointerEvents="none">
                                <line x1={toX(hi)} x2={toX(hi)} y1={padT} y2={h - padB} stroke="rgba(255,255,255,0.35)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
                                <circle cx={toX(hi)} cy={toY(prices[hi])} r="3" fill="#38bdf8" stroke="#0a0e17" strokeWidth="1.5" />
                            </g>
                        )}
                    </svg>
                    <AxisLabels w={w} h={h} labels={axisLabels} />
                </div>
            </div>
            {/* Legend */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '14px', marginTop: '6px', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '0.6rem', color: '#38bdf8', fontWeight: 600 }}>━ Price</span>
                <span style={{ fontSize: '0.6rem', color: '#4ade80', fontWeight: 600 }}>━ 50d MA</span>
                <span style={{ fontSize: '0.6rem', color: '#fb923c', fontWeight: 600 }}>━ 200d MA</span>
                <span style={{ fontSize: '0.6rem', color: 'rgba(239,68,68,0.4)', fontWeight: 600 }}>█ Recessions</span>
            </div>
            {isIndex && (
                <div className="chart-note" style={{ fontSize: '0.58rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '4px' }}>
                    Chart in S&amp;P 500 index points (backup source) · shape matches SPY
                </div>
            )}
        </div>
    );
}
