'use client';
/**
 * The v2 SPY card's valuation lenses (SpyChart `lenses`), drawn in place of the price chart:
 *   Fwd P/E   FactSet's forward P/E every day (lib/valuation.js fwdDaily), the 5- and 10-yr averages FactSet
 *             printed at the time, teal where it sits under that 5-yr average, amber over it. "Then what?" =
 *             SPY after each first week under it (SignalStudy), so the study's weeks are the chart's teal.
 *   P/E       the trailing P/E since 1871 (the P/E number's own history: baked file + sheet + live).
 *   Earnings  SPY against forward earnings × the 10-yr average P/E: the gap is the P/E, and the window
 *             splits SPY's move into earnings and P/E.
 * Same timeframe chips as the price chart (shared pick); chips a lens cannot fill are dimmed.
 */
import { useState, useEffect, useMemo, useRef, useId } from 'react';
import { yearTicks, shortTicks, fmtDay, unitsForPx, gutterFor, clampUnits, LABEL_CHAR_PX } from '../lib/chartAxis';
import useElementWidth from './useElementWidth';
import AxisLabels from './AxisLabels';
import SignalStudy from './SignalStudy';
import { useChart } from './MarkProvider';
import { loadLong, joinLong, longInfo } from '../lib/longHistory';
import { decodeFwd, joinFwd } from '../lib/fwdPe';
import { getJson } from '../lib/loadJson';
import { fwdDaily, extremeSince, timeAvg, decompose, underAvg } from '../lib/valuation';

const TF_KEYS = ['1M', '3M', '6M', '1Y', '5Y', '10Y', '15Y', '20Y', '30Y', 'ALL'];
const TF_DAYS = { '1M': 31, '3M': 92, '6M': 183, '1Y': 366, '5Y': 1827, '10Y': 3653, '15Y': 5479, '20Y': 7305, '30Y': 10958 };
const DAY = 864e5;
const ms = (iso) => Date.parse(`${iso}T00:00:00Z`);
const back = (iso, days) => new Date(ms(iso) - days * DAY).toISOString().slice(0, 10);
const signed = (x, dp = 1) => `${x > 0 ? '+' : x < 0 ? '−' : ''}${Math.abs(x).toFixed(dp)}`;
const pct = (r) => `${signed(r * 100, 0)}%`;
const monthYear = (iso) => fmtDay(iso).replace(/ \d+,/, ''); // 'Apr 7, 2025' → 'Apr 2025'
const BLUE = '#38bdf8', TEAL = '#2dd4bf', AMBER = '#fbbf24', VIOLET = '#a78bfa';
const MONO = "'JetBrains Mono', monospace";

let fwdLoad = null;
const bake = () => fetch('/history/fwdPE.json').then((r) => (r.ok ? r.json() : null)).then(decodeFwd).catch(() => null);
/**
 * FactSet's weekly rows: the baked file (a second try if the first fails), then the reports since (/api/fwd-pe, given
 * 8 s: the bake alone draws every view, so a route busy at FactSet never holds the card). null when the file fails.
 * LeadCombos uses it too.
 */
export function loadFwd() {
    if (!fwdLoad) {
        fwdLoad = Promise.all([
            bake().then((b) => b || bake()),
            getJson('/api/fwd-pe', { timeoutMs: 8000, retries: 0 }),
        ]).then(([baked, live]) => (baked ? joinFwd(baked, live?.rows) : null));
        fwdLoad.then((rows) => { if (!rows) fwdLoad = null; }); // the next open tries again
    }
    return fwdLoad;
}

export default function ValuationLens({ lens, spy, tf, onTf, peNow, chip }) {
    const [fwd, setFwd] = useState(null); // weekly rows | false when unavailable
    const [long, setLong] = useState(null);
    const [hover, setHover] = useState(null);
    const [study, setStudy] = useState(false);
    const svgRef = useRef(null);
    const [plotRef, plotPx] = useElementWidth();
    const uid = useId().replace(/[^\w-]/g, '');
    const peChart = useChart('peRatio', peNow);
    useEffect(() => {
        let on = true;
        loadFwd().then((rows) => { if (on) setFwd(rows || false); });
        return () => { on = false; };
    }, []);
    useEffect(() => {
        if (lens !== 'P/E') return undefined;
        let on = true;
        loadLong('peRatio').then((p) => { if (on) setLong(p); });
        return () => { on = false; };
    }, [lens]);

    const daily = useMemo(() => (fwd && spy?.length ? fwdDaily(fwd, spy) : []), [fwd, spy]);
    const newest = useMemo(() => (fwd ? [...fwd].reverse().find((r) => r.a5 != null && r.a10 != null) || null : null), [fwd]);
    const series = useMemo(() => {
        // no P/E on the page (N/A, or CAPE standing in): no chart, not the history up to the sheet's first day as if current
        if (lens === 'P/E') return peChart ? joinLong(peChart.points, long, longInfo('peRatio')).map((p) => ({ date: p.date, value: p.value })) : [];
        if (lens === 'Earnings') return newest ? daily.map((d) => ({ date: d.date, value: d.price, line: d.eps * newest.a10, d })) : [];
        return daily.map((d) => ({ date: d.date, value: d.pe, line: d.a5, line2: d.a10 }));
    }, [lens, peChart, long, daily, newest]);

    if (series.length < 2) {
        return (
            <div className="mini-chart" style={{ height: '220px', display: 'grid', placeItems: 'center' }}>
                <span className="chart-note" style={{ fontSize: '0.65rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {lens === 'P/E' ? (peNow == null || peNow === '' ? 'P/E unavailable right now' : 'Loading…') : fwd === false ? 'Forward P/E unavailable right now' : 'Loading…'}
                </span>
            </div>
        );
    }

    const last = series[series.length - 1].date;
    const usable = TF_KEYS.filter((k) => k === 'ALL' || ms(series[0].date) <= ms(last) - TF_DAYS[k] * DAY * 0.9);
    const active = usable.includes(tf) ? tf : 'ALL';
    const win = active === 'ALL' ? series : series.filter((p) => p.date >= back(last, TF_DAYS[active]));
    const step = Math.ceil(win.length / 1200);
    const isPe = lens !== 'Earnings';
    // line / line2 = the dashed 5- / 10-yr averages (the trailing P/E's: the last 5 / 10 years, flat); Earnings' line = its violet one
    const peAvg = lens === 'P/E' && { line: timeAvg(series, back(last, 1827)), line2: timeAvg(series, back(last, 3653)) };
    const data = win.filter((_, i) => (win.length - 1 - i) % step === 0).map((p) => (peAvg ? { ...p, ...peAvg } : p));

    // y: every line in view plus the averages; a spike past 3× the median (the trailing P/E in 2009,
    // when earnings vanished) runs off the top rather than flattening the rest.
    const vals = data.flatMap((p) => [p.value, p.line, p.line2].filter(Number.isFinite)).sort((a, b) => a - b);
    const lo = vals[0], hi = Math.min(vals[vals.length - 1], vals[Math.floor(vals.length / 2)] * 3);
    const pad = ((hi - lo) || 1) * 0.06, min = lo - pad, max = hi + pad;
    const w = 540, h = 220, padT = 12, padB = 24;
    const toY = (v) => h - padB - ((Math.min(v, max) - min) / (max - min)) * (h - padT - padB);
    // y ticks on round numbers (1, 2 or 5 × 10ⁿ apart, at most 6), so the 18x line sits at 18
    const base = 10 ** Math.floor(Math.log10(max - min)) / 10;
    const tStep = [1, 2, 5, 10, 20, 50].map((m) => m * base).find((s) => Math.floor(max / s) - Math.ceil(min / s) < 6);
    const tickFmt = (v) => (isPe ? `${v.toFixed(Math.max(0, -Math.floor(Math.log10(tStep) + 1e-9)))}x` : `$${Math.round(v)}`);
    const yTicks = [];
    for (let k = Math.ceil(min / tStep); k <= Math.floor(max / tStep); k++) yTicks.push({ y: toY(k * tStep), label: tickFmt(k * tStep) });
    const padL = gutterFor(yTicks.map((t) => t.label), w, plotPx, 34);
    const padR = isPe ? (unitsForPx(46, w, plotPx) ?? 44) : 8; // room for the end pill
    const t0 = ms(data[0].date), t1 = ms(data[data.length - 1].date);
    const toX = (i) => padL + ((ms(data[i].date) - t0) / (t1 - t0 || 1)) * (w - padL - padR);
    const line = (key) => data.flatMap((p, i) => (Number.isFinite(p[key]) ? [`${toX(i).toFixed(1)},${toY(p[key]).toFixed(1)}`] : [])).join(' ');
    const main = line('value');
    const xL = padL, xR = toX(data.length - 1);
    const a = data.findIndex((p) => Number.isFinite(p.line)), xa = a < 0 ? null : toX(a); // the line starts here (FactSet's 5-yr avg: 2013)

    const dates = data.map((p) => p.date);
    const spanDays = (t1 - t0) / DAY;
    const yrGap = Math.max(26, unitsForPx(LABEL_CHAR_PX * 4 + 10, w, plotPx) ?? 26);
    const xLabels = spanDays > 550 // phone: fewer years, else 2011–2026 at a 2-year step crowds every label out
        ? yearTicks(dates, toX, { minGap: yrGap, maxLabels: Math.min(8, Math.floor((w - padL - padR) / yrGap)) })
        : shortTicks(dates, toX, { minGap: unitsForPx(LABEL_CHAR_PX * 6 + 10, w, plotPx) ?? 40 });
    const lblGap = unitsForPx(4, w, plotPx) ?? 4;
    const axisLabels = [
        ...yTicks.map((t) => ({ x: padL - lblGap, y: t.y, text: t.label, ax: 'end', ay: 'middle' })),
        ...xLabels.map((l) => ({ x: clampUnits(l.x, (l.label.length * LABEL_CHAR_PX) / 2, w, plotPx), y: h, text: l.label, ax: 'middle', ay: 'bottom' })),
    ];

    const first = data[0], end = data[data.length - 1];
    const turns = end.value - first.value;
    const parts = !isPe && decompose(first.d, end.d);
    const extreme = isPe ? extremeSince(series) : null;
    const si = extreme?.since && extreme.since >= first.date ? data.findIndex((p) => p.date >= extreme.since) : -1; // where "since" starts on screen
    const hi2 = hover != null && hover < data.length ? data[hover] : null;
    // the time machine: what SPY did in the 6 months after a hovered day (or since, for the last 6 months)
    let then = null;
    if (isPe && hi2 && spy?.length && spy[0].date <= hi2.date) {
        const a = spy.find((p) => p.date >= hi2.date), b = spy.find((p) => p.date >= back(hi2.date, -182));
        const z = b || spy[spy.length - 1];
        if (a && z !== a) then = `SPY ${pct(z.price / a.price - 1)} ${b ? 'in 6M' : 'since'}`;
    }
    // FactSet skipped a week or more: the line still moves with SPY, on the newest report's earnings
    const asOf = lens !== 'P/E' && fwd ? fwd[fwd.length - 1].date : null;
    const est = asOf && ms(last) - ms(asOf) > 9 * DAY ? ` · est. since ${fmtDay(asOf).replace(/, \d{4}$/, '')}` : '';
    const onPoint = (e) => {
        const r = svgRef.current?.getBoundingClientRect();
        if (!r?.width) return;
        const t = t0 + ((((e.clientX - r.left) / r.width) * w - padL) / (w - padL - padR)) * (t1 - t0);
        let i = data.findIndex((p) => ms(p.date) >= t);
        if (i < 0) i = data.length - 1;
        else if (i > 0 && t - ms(data[i - 1].date) < ms(data[i].date) - t) i -= 1;
        setHover(i);
    };
    const label = { fontSize: '0.65rem', fontFamily: MONO, fontWeight: 700, whiteSpace: 'nowrap' };
    const legend = { fontSize: '0.6rem', fontWeight: 600, whiteSpace: 'nowrap' };

    return (
        <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 8px', marginBottom: '8px', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', flexWrap: 'nowrap', gap: '4px', overflowX: 'auto', overflowY: 'hidden', maxWidth: '100%' }}>
                    {TF_KEYS.map((k) => (
                        <button key={k} className="tf-btn tf-btn-spy" onClick={() => { onTf(k); setHover(null); }} disabled={!usable.includes(k)}
                            title={usable.includes(k) ? undefined : 'Not enough history for this view'}
                            style={chip(k === active, usable.includes(k))}>{k}</button>
                    ))}
                </div>
                {hi2 ? (
                    <span className="chart-readout" style={{ ...label, color: 'var(--text-primary)' }}>
                        {fmtDay(hi2.date)} · {isPe ? `${hi2.value.toFixed(1)}x${then ? ` · ${then}` : ''}` : `SPY $${hi2.value.toFixed(0)} · line $${hi2.line.toFixed(0)}`}
                    </span>
                ) : isPe ? (
                    <span style={{ ...label, color: turns <= 0 ? TEAL : AMBER }}>{first.value.toFixed(1)}x → {end.value.toFixed(1)}x</span>
                ) : (
                    <span style={label}>
                        <span style={{ color: VIOLET }}>EPS {pct(parts.eps)}</span>
                        <span style={{ color: 'var(--text-muted)' }}> · </span>
                        <span style={{ color: parts.pe <= 0 ? TEAL : AMBER }}>P/E {pct(parts.pe)}</span>
                        <span style={{ color: 'var(--text-muted)' }}> → </span>
                        <span style={{ color: BLUE }}>SPY {pct(parts.price)}</span>
                    </span>
                )}
            </div>
            {isPe && (extreme || lens === 'Fwd P/E') && (
                <div style={{ display: 'flex', flexWrap: 'nowrap', gap: '8px', alignItems: 'center', marginBottom: '6px' }}>
                    {extreme && (
                        <span style={{
                            fontSize: '0.68rem', fontWeight: 700, padding: '4px 10px', borderRadius: '6px', whiteSpace: 'nowrap',
                            background: extreme.low ? 'rgba(45,212,191,0.12)' : 'rgba(251,191,36,0.12)', color: extreme.low ? TEAL : AMBER,
                        }}>
                            {extreme.low ? '📉 Lowest' : '📈 Highest'} since {monthYear(extreme.since || series[0].date)}
                        </span>
                    )}
                    {lens === 'Fwd P/E' && (
                        <button type="button" onClick={() => setStudy(true)} aria-haspopup="dialog" style={{ ...chip(false), marginLeft: 'auto', color: TEAL }}>
                            Then what? ›
                        </button>
                    )}
                </div>
            )}
            <div className="mini-chart" style={{ height: '220px' }}>
                <div className="chart-plot" ref={plotRef}>
                    <svg ref={svgRef} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
                        onPointerMove={onPoint} onPointerDown={onPoint} onPointerLeave={(e) => { if (e.pointerType === 'mouse') setHover(null); }}
                        style={{ touchAction: 'pan-y' }}>
                        <defs>
                            {xa != null && <>
                                <clipPath id={`${uid}-under`}><polygon points={`${line('line')} ${xR},${h} ${xa},${h}`} /></clipPath>
                                <clipPath id={`${uid}-over`}><polygon points={`${line('line')} ${xR},0 ${xa},0`} /></clipPath>
                            </>}
                            <clipPath id={`${uid}-plot`}><rect x="0" y={padT - 2} width={w} height={h - padT - padB + 4} /></clipPath>
                        </defs>
                        {yTicks.map((t, i) => <line key={i} x1={padL} x2={w - padR} y1={t.y} y2={t.y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />)}
                        <g clipPath={`url(#${uid}-plot)`}>
                            {/* teal = cheaper than usual (P/E under its 5-yr average / SPY under its earnings line), amber = pricier */}
                            {xa != null && <>
                                <polygon points={`${main} ${xR},0 ${xL},0`} fill="rgba(45,212,191,0.14)" clipPath={`url(#${uid}-under)`} />
                                <polygon points={`${main} ${xR},${h} ${xL},${h}`} fill="rgba(251,191,36,0.07)" clipPath={`url(#${uid}-over)`} />
                            </>}
                            {/* "Lowest since …": today's level, back to the last day it was past it */}
                            {si >= 0 && <line x1={toX(si)} x2={xR} y1={toY(end.value)} y2={toY(end.value)} stroke={extreme.low ? TEAL : AMBER}
                                strokeOpacity="0.7" strokeWidth="1" strokeDasharray="2 3" vectorEffect="non-scaling-stroke" />}
                            {isPe ? ['line', 'line2'].map((k, i) => (
                                <polyline key={k} points={line(k)} fill="none" stroke={i ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.6)'}
                                    strokeWidth="1" strokeDasharray="4 3" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
                            )) : <polyline points={line('line')} fill="none" stroke={VIOLET} strokeWidth="1.5" strokeLinejoin="round" />}
                            <polyline points={main} fill="none" stroke={BLUE} strokeWidth="1.8" strokeLinejoin="round" />
                        </g>
                        <circle cx={xR} cy={toY(end.value)} r="3" fill={BLUE} stroke="#0a0e17" strokeWidth="1.5" />
                        {hi2 && (
                            <g className="chart-cursor" pointerEvents="none">
                                <line x1={toX(hover)} x2={toX(hover)} y1={padT} y2={h - padB} stroke="rgba(255,255,255,0.35)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
                                <circle cx={toX(hover)} cy={toY(hi2.value)} r="3" fill={BLUE} stroke="#0a0e17" strokeWidth="1.5" />
                            </g>
                        )}
                    </svg>
                    <AxisLabels w={w} h={h} labels={axisLabels} />
                    {isPe && (
                        <span aria-hidden="true" style={{
                            position: 'absolute', left: `${((xR + 3) / w) * 100}%`, top: `${(toY(end.value) / h) * 100}%`, transform: 'translateY(-50%)',
                            pointerEvents: 'none', padding: '1px 5px', borderRadius: '5px', background: BLUE, color: '#0a0e17',
                            fontSize: '0.62rem', fontWeight: 800, fontFamily: MONO, whiteSpace: 'nowrap',
                        }}>{end.value.toFixed(1)}x</span>
                    )}
                </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '14px', marginTop: '6px', flexWrap: 'nowrap', overflowX: 'auto' }}>
                {isPe ? <>
                    <span style={{ ...legend, color: est ? AMBER : BLUE }} title={lens === 'P/E' ? 'S&P 500 price ÷ the last 12 months of earnings (multpl, monthly since 1871)' : `S&P 500 price ÷ the next 12 months of expected earnings (FactSet, weekly since 2011; moves with SPY between reports). Newest report: ${fmtDay(asOf)}.`}>━ {lens === 'P/E' ? 'P/E (trailing)' : 'Fwd P/E'}{est}</span>
                    {Number.isFinite(end.line) && <span style={{ ...legend, color: 'rgba(255,255,255,0.75)' }}>┄ 5-yr avg {end.line.toFixed(1)}</span>}
                    {Number.isFinite(end.line2) && <span style={{ ...legend, color: 'rgba(255,255,255,0.45)' }}>┄ 10-yr avg {end.line2.toFixed(1)}</span>}
                </> : <>
                    <span style={{ ...legend, color: BLUE }}>━ SPY</span>
                    <span style={{ ...legend, color: est ? AMBER : VIOLET }} title={`Forward 12-month earnings × the 10-yr average P/E (${newest.a10.toFixed(1)}): where SPY would trade at a normal valuation. Newest FactSet report: ${fmtDay(asOf)}.`}>━ Earnings × {newest.a10.toFixed(1)}{est}</span>
                </>}
            </div>
            {study && (
                <SignalStudy id="fwdpe" src="/history/fwdPE.json" decode={(body) => underAvg(decodeFwd(body))} line={0} horizons={[1, 3, 6]} start={3}
                    title="Fwd P/E under 5-yr avg" sub={newest ? `Now ${newest.pe.toFixed(1)} · 5-yr avg ${newest.a5.toFixed(1)} · ${fmtDay(newest.date)}` : ''}
                    label={(v, p) => `P/E ${p.pe.toFixed(1)} · avg ${p.a5.toFixed(1)}`} foot="first week under the 5-yr avg after 3 quiet months"
                    sheet={underAvg(fwd)} spyLive={spy} onClose={() => setStudy(false)} />
            )}
        </div>
    );
}
