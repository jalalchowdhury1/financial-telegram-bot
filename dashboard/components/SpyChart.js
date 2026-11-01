'use client';
import { useState, useEffect, useRef } from 'react';
import { yearTicks, shortTicks, indexFromPointer, tfAvailable, fmtDay, readChoice, saveChoice, unitsForPx, gutterFor, clampUnits, LABEL_CHAR_PX } from '../lib/chartAxis';
import useElementWidth from './useElementWidth';
import AxisLabels from './AxisLabels';
import { decodeLong, loadLong, joinLong, longInfo } from '../lib/longHistory';
import ValuationLens, { loadFwd } from './ValuationLens';
import { useChart } from './MarkProvider';
import { fwdDaily, attention } from '../lib/valuation';

const TF_KEYS = ['1M', '3M', '6M', '1Y', '5Y', '10Y', '15Y', '20Y', '30Y', 'ALL'];
const STORE_KEY = 'ftb:tf:spy';
const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;
const LENSES = ['Price', 'Fwd P/E', 'P/E', 'Earnings'];
const BLUE = '#38bdf8', TEAL = '#2dd4bf', AMBER = '#fbbf24', VIOLET = '#a78bfa';

/** A view button's last year, `[{date, value}]`: one line, no axes, spaced by date (the P/E's older points are monthly). */
function Spark({ pts, color }) {
    if (!(pts?.length > 1)) return null;
    const vs = pts.map((p) => p.value), lo = Math.min(...vs), hi = Math.max(...vs);
    const t0 = Date.parse(pts[0].date), span = Date.parse(pts[pts.length - 1].date) - t0 || 1;
    const d = pts.map((p) => `${(((Date.parse(p.date) - t0) / span) * 60).toFixed(1)},${(13 - ((p.value - lo) / (hi - lo || 1)) * 12).toFixed(1)}`).join(' ');
    return (
        <svg viewBox="0 0 60 14" preserveAspectRatio="none" aria-hidden="true" style={{ width: '24px', flex: '1 1 24px', maxWidth: '80px', height: '14px' }}>
            <polyline points={d} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
        </svg>
    );
}

/** Baked closes strictly before the live window, then the live bars; 50/200d MAs over the whole run.
 *  The live bars alone when the two disagree (the backup path charts the S&P INDEX, ~10x SPY). */
export function joinBaked(baked, live) {
    const cut = live[0].date;
    const before = (baked || []).filter((p) => p.date < cut);
    const at = (baked || []).find((p) => p.date >= cut);
    if (!before.length || !at || Math.abs(at.value / live[0].price - 1) > 0.05) return live;
    const prices = [...before.map((p) => p.value), ...live.map((d) => d.price)];
    const dates = [...before.map((p) => p.date), ...live.map((d) => d.date)];
    let s50 = 0, s200 = 0;
    return prices.map((p, i) => {
        s50 += p - (i >= 50 ? prices[i - 50] : 0);
        s200 += p - (i >= 200 ? prices[i - 200] : 0);
        const r = (v) => Math.round(v * 100) / 100;
        return { date: dates[i], price: p, ma50: r(s50 / Math.min(i + 1, 50)), ma200: r(s200 / Math.min(i + 1, 200)) };
    });
}

/** `lenses` (v2 only, `{peNow}`): a Price · Fwd P/E · P/E · Earnings row above the chart (components/ValuationLens.js). */
export default function SpyChart({ chartHistory, recessions = [], current = null, lenses = null }) {
    const [timeframe, setTimeframe] = useState('5Y');
    const [lens, setLens] = useState('Price');
    const [hover, setHover] = useState(null);
    const svgRef = useRef(null);
    const [plotRef, plotPx] = useElementWidth(); // real px width → 10px labels spaced honestly
    const [baked, setBaked] = useState(null);
    const [dateRange, setDateRange] = useState(null); // 📅 {from, to} ISO dates, or null = a preset tab
    const [fwd, setFwd] = useState(null); // FactSet's weekly rows (false = failed): the view buttons' numbers + the attention pick
    const [peLong, setPeLong] = useState(null);
    const [attn, setAttn] = useState(null);
    const tapped = useRef(false);
    const peChart = useChart('peRatio', lenses?.peNow);
    const v2 = !!lenses;
    useEffect(() => { const v = readChoice(STORE_KEY); if (v) setTimeframe(v); }, []);
    useEffect(() => {
        if (!v2) return undefined;
        let on = true;
        loadFwd().then((rows) => { if (on) setFwd(rows || false); });
        loadLong('peRatio').then((p) => { if (on) setPeLong(p); });
        return () => { on = false; };
    }, [v2]);
    // /api/spy carries ~14 months; the baked daily file (scripts/bake_spy_daily.py) fills 5Y…30Y/ALL.
    useEffect(() => {
        fetch('/history/spy-daily.json').then((r) => (r.ok ? r.json() : null)).then(decodeLong).then(setBaked).catch(() => {});
    }, []);
    // The view that needs a look opens first (his 1 Nov 2026 "default view should be the one that should get my
    // immediate attention"); a tap wins for the rest of the visit. Needs the baked years: 14 months can't say "since 2025".
    // FactSet down (fwd false): SPY's own low still counts.
    useEffect(() => {
        const spy = fwd !== null && baked && chartHistory?.length > 1 ? joinBaked(baked, chartHistory) : null;
        if (!spy || spy === chartHistory) return;
        const a = attention(fwd ? fwdDaily(fwd, spy) : [], spy);
        setAttn(a);
        if (a && !tapped.current) setLens(a.lens);
    }, [fwd, baked, chartHistory]);
    if (!chartHistory || chartHistory.length < 2) return null;

    const full = joinBaked(baked, chartHistory);
    const tfDays = { '1M': 21, '3M': 63, '6M': 126, '1Y': 252, '5Y': 1260, '10Y': 2520, '15Y': 3780, '20Y': 5040, '30Y': 7560, 'ALL': full.length };
    // Only offer tabs the history covers (until the baked file lands, or on the index
    // backup path, that is ~14 months). Fall back to 5Y, then ALL.
    const usable = TF_KEYS.filter((tf) => tfAvailable(tf === 'ALL' ? null : tfDays[tf], full.length));
    const activeTf = usable.includes(timeframe) ? timeframe : usable.includes('5Y') ? '5Y' : 'ALL';
    const sliceLen = Math.min(tfDays[activeTf] || full.length, full.length);
    // ~1,200 points is plenty for a 540-wide chart; 33 years of days is 8,500.
    // 📅 dates: the bars between them; fewer than 2 trading days keeps the preset tab, with a note.
    const picked = dateRange ? full.filter((d) => d.date >= dateRange.from && d.date <= dateRange.to) : null;
    const tooFew = picked != null && picked.length < 2;
    const win = picked && !tooFew ? picked : full.slice(-sliceLen);
    const step = Math.ceil(win.length / 1200);
    const data = win.filter((_, i) => (win.length - 1 - i) % step === 0);

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

    // Year labels on x-axis, thinned so they never overlap (at a real 10px once measured);
    // under ~18 months, months or weeks instead.
    const spanDays = (Date.parse(dates[dates.length - 1]) - Date.parse(dates[0])) / 864e5;
    const yearLabels = spanDays > 550
        ? yearTicks(dates, toX, { minGap: Math.max(26, unitsForPx(LABEL_CHAR_PX * 4 + 10, w, plotPx) ?? 26) })
        : shortTicks(dates, toX, { minGap: unitsForPx(LABEL_CHAR_PX * 6 + 10, w, plotPx) ?? 40 });
    const lblGap = unitsForPx(4, w, plotPx) ?? 4;
    const axisLabels = [
        ...yTicks.map((t) => ({ x: padL - lblGap, y: t.y, text: t.label, ax: 'end', ay: 'middle' })),
        ...yearLabels.map((yl) => ({ x: clampUnits(yl.x, (yl.label.length * LABEL_CHAR_PX) / 2, w, plotPx), y: h, text: yl.label, ax: 'middle', ay: 'bottom' })),
    ];

    // Golden / Death Cross detection, on every day (thinning would move the date); idx = its drawn point
    let lastCross = null;
    for (let i = 1; i < win.length; i++) {
        const prev50above = win[i - 1].ma50 > win[i - 1].ma200;
        const curr50above = win[i].ma50 > win[i].ma200;
        const idx = data.length - 1 - Math.floor((win.length - 1 - i) / step);
        if (!prev50above && curr50above) lastCross = { type: 'golden', date: win[i].date, idx };
        else if (prev50above && !curr50above) lastCross = { type: 'death', date: win[i].date, idx };
    }

    // Price range labels
    const maxPrice = Math.max(...prices);
    const minPrice = Math.min(...prices);
    const returnPct = ((prices[prices.length - 1] - prices[0]) / prices[0]) * 100;

    // Tap / hover readout: the point under the pointer replaces the return label.
    const hi = hover != null && hover < data.length ? hover : null;
    const pick = (tf) => { setTimeframe(tf); saveChoice(STORE_KEY, tf); setDateRange(null); setHover(null); };
    const custom = dateRange != null && !tooFew;
    const toggleRange = () => { setDateRange(dateRange ? null : { from: win[0].date, to: full[full.length - 1].date }); setHover(null); };
    const setEnd = (k) => (e) => { if (ISO_DAY.test(e.target.value)) { setDateRange({ ...dateRange, [k]: e.target.value }); setHover(null); } };
    const chip = (on, enabled = true) => ({
        padding: '3px 7px', borderRadius: '6px', border: 'none', whiteSpace: 'nowrap', flexShrink: 0,
        cursor: enabled ? 'pointer' : 'default', opacity: enabled ? 1 : 0.35,
        fontSize: '0.65rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
        background: on ? 'rgba(56,189,248,0.2)' : 'rgba(255,255,255,0.05)',
        color: on ? '#38bdf8' : 'var(--text-muted)',
        transition: 'all 0.2s ease'
    });
    const dateBox = { background: 'rgba(255,255,255,0.05)', color: 'var(--text-primary)', border: 'none', borderRadius: '6px', padding: '2px 4px', font: 'inherit', colorScheme: 'dark' };
    const onPoint = (e) => {
        const i = indexFromPointer(e.clientX, svgRef.current?.getBoundingClientRect(), data.length, { w, padL, padR });
        if (i != null) setHover(i);
    };
    const onLeave = (e) => { if (e.pointerType === 'mouse') setHover(null); };
    const fmtPx = (v) => (v == null ? '—' : isIndex ? Math.round(v).toLocaleString('en-US') : `$${v.toFixed(2)}`);
    const pickLens = (l) => { tapped.current = true; setLens(l); setHover(null); };
    // overflowY hidden on the chip rows: each chip's tap band (.tf-btn::before) sits 12px below it, so a sideways row was
    // 12px taller inside than out and one scroll over it slid the chips out of sight (1 Nov 2026, "i can't see the buttons").
    // Each view's button shows its number now and its last year (his 1 Nov "display all of these more creatively");
    // a dot marks the one that needs a look.
    const daily = fwd ? fwdDaily(fwd, full) : [];
    const now = daily[daily.length - 1], yr = full[Math.max(0, full.length - 252)].date;
    const epsYr = daily.find((d) => d.date >= yr);
    const px = current ?? full[full.length - 1].price;
    const peNow = lenses?.peNow == null || lenses.peNow === '' ? NaN : Number(lenses.peNow);
    const tiles = lenses ? {
        Price: { value: isIndex ? Math.round(px).toLocaleString('en-US') : `$${Math.round(px)}`, color: BLUE, pts: full.slice(-252).map((p) => ({ date: p.date, value: p.price })) },
        'Fwd P/E': now && { value: `${now.pe.toFixed(1)}x`, color: now.a5 != null && now.pe > now.a5 ? AMBER : TEAL, pts: daily.slice(-252).map((d) => ({ date: d.date, value: d.pe })) },
        'P/E': Number.isFinite(peNow) && { value: `${peNow.toFixed(1)}x`, color: 'var(--text-primary)', pts: joinLong(peChart?.points, peLong, longInfo('peRatio')).filter((p) => p.date >= yr) },
        Earnings: now && epsYr && { value: `${now.eps >= epsYr.eps ? '+' : '−'}${Math.abs((now.eps / epsYr.eps - 1) * 100).toFixed(0)}%`, color: VIOLET, pts: daily.slice(-252).map((d) => ({ date: d.date, value: d.eps })) },
    } : {};
    const hotColor = attn && (attn.lens === 'Fwd P/E' && attn.low ? TEAL : AMBER);
    const sinceText = attn && fmtDay(attn.since || (attn.lens === 'Price' ? full : daily)[0].date).replace(/ \d+,/, ''); // 'Apr 2025'
    const lensRow = lenses && (
        <div role="tablist" aria-label="Chart" style={{ display: 'flex', flexWrap: 'nowrap', gap: '4px', overflowX: 'auto', overflowY: 'hidden', marginBottom: '8px' }}>
            {LENSES.map((l) => {
                const t = tiles[l], hot = attn?.lens === l;
                return (
                    <button key={l} role="tab" aria-selected={l === lens} aria-label={l} className="tf-btn tf-btn-spy" onClick={() => pickLens(l)}
                        title={hot ? `${attn.low ? 'Lowest' : 'Highest'} since ${sinceText}: worth a look` : undefined}
                        style={{ ...chip(l === lens), display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '2px', padding: '5px 7px', flex: '1 0 auto', maxWidth: '160px' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            {l}{hot && <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: hotColor }} />}
                        </span>
                        {t && (
                            <span style={{ display: 'flex', alignItems: 'center', gap: '6px', width: '100%' }}>
                                <span style={{ color: t.color }}>{t.value}</span><Spark pts={t.pts} color={t.color} />
                            </span>
                        )}
                    </button>
                );
            })}
        </div>
    );
    // why it opened here: the Fwd P/E view says it with its own chip
    const why = attn && lens === attn.lens && attn.lens !== 'Fwd P/E' && (
        <div style={{ marginBottom: '6px' }}>
            <span style={{ fontSize: '0.68rem', fontWeight: 700, padding: '4px 10px', borderRadius: '6px', whiteSpace: 'nowrap', background: 'rgba(251,191,36,0.12)', color: AMBER }}>
                📉 {attn.lens === 'Price' ? 'SPY' : 'Earnings'} lowest since {sinceText}
            </span>
        </div>
    );
    // The index backup path has no SPY dollars to anchor forward P/E on: price only.
    if (lensRow && lens !== 'Price' && !isIndex) {
        return <div>{lensRow}{why}<ValuationLens lens={lens} spy={full} tf={timeframe} onTf={pick} peNow={lenses.peNow} chip={chip} /></div>;
    }

    return (
        <div>
            {lensRow}
            {why}
            {/* Timeframe selector */}
            {/* The chips stay one row (sideways scroll if ever too narrow); the return label drops below on a phone. */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 8px', marginBottom: '8px', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', flexWrap: 'nowrap', gap: '4px', overflowX: 'auto', overflowY: 'hidden', maxWidth: '100%' }}>
                    {TF_KEYS.map(tf => (
                        <button key={tf} className="tf-btn tf-btn-spy" onClick={() => pick(tf)} disabled={!usable.includes(tf)}
                            title={usable.includes(tf) ? undefined : 'Not enough history from this source'}
                            style={chip(!custom && tf === activeTf, usable.includes(tf))}>{tf}</button>
                    ))}
                    <button className="tf-btn tf-btn-spy" onClick={toggleRange} title="Pick dates" aria-label="Pick dates"
                        aria-pressed={dateRange != null} style={chip(dateRange != null)}>📅</button>
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
                        color: returnPct >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 700, whiteSpace: 'nowrap'
                    }}>
                        {returnPct >= 0 ? '+' : ''}{returnPct.toFixed(1)}% return
                    </span>
                )}
            </div>
            {dateRange && (
                <div className="spy-range" style={{
                    display: 'flex', flexWrap: 'nowrap', gap: '8px', alignItems: 'center', marginBottom: '8px', overflowX: 'auto',
                    fontSize: '0.65rem', fontFamily: "'JetBrains Mono', monospace", color: 'var(--text-muted)', whiteSpace: 'nowrap'
                }}>
                    <label>From <input type="date" aria-label="From" value={dateRange.from} min={full[0].date} max={dateRange.to} onChange={setEnd('from')} style={dateBox} /></label>
                    <label>To <input type="date" aria-label="To" value={dateRange.to} min={dateRange.from} max={full[full.length - 1].date} onChange={setEnd('to')} style={dateBox} /></label>
                </div>
            )}
            {tooFew && (
                <div className="chart-note" style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginBottom: '6px', whiteSpace: 'nowrap' }}>
                    Under 2 trading days in those dates · showing {activeTf}
                </div>
            )}
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
