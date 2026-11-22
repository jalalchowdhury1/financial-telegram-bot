'use client';
/**
 * The v2 SPY card's Bands view (SpyChart `lenses`; his 22 Nov 2026 "S&P 500 Forward P/E Valuation Bands" chart).
 * SPY on a ladder: each rung = forward earnings × a whole P/E (lib/valuation.js bandRungs: the ones the window
 * touches), so the rungs SPY sits between are its forward P/E. Quiet on purpose (his "minimalistic"): the rungs are
 * grey hairlines, brighter near SPY; one band is lit, the one SPY is in today, teal under FactSet's 5-yr average
 * P/E and amber over it (the Fwd P/E view's colours); SPY is the card's blue. Right of today the rungs run on at the earnings growth analysts expect (`g` on the newest
 * FactSet report, lib/fwdPe.js): what SPY would cost at each P/E by then. No `g` in the last 45 days: no projection.
 * 🔭 (his 22 Nov "i would like to see till 2030. but not on the default screen"): a tap runs the rungs out to the end
 * of the 4th year ahead, a second tap comes back; the chart glides between the two. Analysts name no year past the
 * next, so after the first 12 months the rungs grow at forward earnings' own pace over the last 10 years (measured
 * from this chart's data, needs 5+ years of it) and are drawn fainter.
 * Log scale: rungs keep their distance and steady growth is a straight line. The labels on the right give each
 * rung's price at the right edge; a tap reads a day (ahead of today: SPY at today's P/E and at the 5-yr average).
 */
import { useState, useRef, useId, useEffect } from 'react';
import { yearTicks, shortTicks, fmtDay, unitsForPx, gutterFor, clampUnits, LABEL_CHAR_PX } from '../lib/chartAxis';
import useElementWidth from './useElementWidth';
import AxisLabels from './AxisLabels';
import { TF_KEYS, TF_DAYS } from './ValuationLens';
import { bandRungs } from '../lib/valuation';

const DAY = 864e5;
const ms = (iso) => Date.parse(`${iso}T00:00:00Z`);
const back = (iso, days) => new Date(ms(iso) - days * DAY).toISOString().slice(0, 10);
const monthYear = (iso) => fmtDay(iso).replace(/ \d+,/, ''); // 'Apr 7, 2025' → 'Apr 2025'
const signedPct = (x) => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(1)}%`;
const usd = (v) => `$${Math.round(v).toLocaleString('en-US')}`;
const MONO = "'JetBrains Mono', monospace";
const BLUE = '#38bdf8', TEAL = '#2dd4bf', AMBER = '#fbbf24';

export default function ValuationBands({ daily, fwd, tf, onTf, chip }) {
    const [hover, setHover] = useState(null);
    const [far, setFar] = useState(false); // 🔭 on
    const [glide, setGlide] = useState(null); // days ahead on screen while the chart moves between the two views
    const svgRef = useRef(null);
    const frame = useRef(0);
    useEffect(() => () => window.cancelAnimationFrame?.(frame.current), []);
    const [plotRef, plotPx] = useElementWidth();
    const uid = useId().replace(/[^\w-]/g, '');

    if (!(daily?.length > 1)) {
        return (
            <div className="mini-chart" style={{ height: '260px', display: 'grid', placeItems: 'center' }}>
                <span className="chart-note" style={{ fontSize: '0.65rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {fwd === false ? 'Forward P/E unavailable right now' : 'Loading…'}
                </span>
            </div>
        );
    }

    const last = daily[daily.length - 1];
    const usable = TF_KEYS.filter((k) => k === 'ALL' || ms(daily[0].date) <= ms(last.date) - TF_DAYS[k] * DAY * 0.9);
    const active = usable.includes(tf) ? tf : 'ALL';
    const win = active === 'ALL' ? daily : daily.filter((p) => p.date >= back(last.date, TF_DAYS[active]));
    const step = Math.ceil(win.length / 1200);
    const data = win.filter((_, i) => (win.length - 1 - i) % step === 0);

    // right of today: forward earnings growing at the analysts' rate, a point a week
    // the newest growth number of the last 45 days, for the furthest year any of those reports names (a blog-read
    // week names only this year: its bigger number must not replace the PDF's next-year one)
    const recent = fwd.filter((r) => r.g != null && ms(last.date) - ms(r.date) <= 45 * DAY);
    const gy = Math.max(...recent.map((r) => r.gy ?? 0));
    const grow = recent.reverse().find((r) => (r.gy ?? 0) === gy);
    const g = grow ? grow.g / 100 : null;
    const whose = grow?.gy ? `analysts' ${grow.gy}` : 'analysts';
    const near = g == null ? 0 : Math.min(365, Math.round(((ms(last.date) - ms(data[0].date)) / DAY) * 0.6));
    // past the analysts' year: forward earnings' own yearly pace over the last 10 years (5+ on file, else no 🔭)
    const old = daily.find((p) => p.date >= back(last.date, 3653)), oldYears = (ms(last.date) - ms(old.date)) / DAY / 365.25;
    const pace = oldYears >= 5 && old.eps > 0 ? (last.eps / old.eps) ** (1 / oldYears) - 1 : null;
    const farYear = Number(last.date.slice(0, 4)) + 4;
    const farDays = near && pace != null ? Math.round((ms(`${farYear}-12-31`) - ms(last.date)) / DAY) : 0;
    const ahead = glide ?? (far && farDays ? farDays : near);
    const epsIn = (days) => last.eps * (1 + g) ** (Math.min(days, 365) / 365) * (1 + (pace ?? 0)) ** (Math.max(0, days - 365) / 365);
    const future = [];
    for (let d = 7; d < ahead + 7 && ahead; d += 7) {
        const days = Math.min(d, ahead);
        future.push({ date: back(last.date, -days), eps: epsIn(days), days });
    }
    const year1 = future.filter((p) => p.days <= 365), later = future.filter((p) => p.days >= 365 || p === year1[year1.length - 1]);
    const look = () => {
        const from = ahead, to = far ? near : farDays, t0 = Date.now();
        setFar(!far); setHover(null);
        if (typeof window === 'undefined' || !window.requestAnimationFrame || window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return;
        const tick = () => {
            const k = Math.min(1, (Date.now() - t0) / 700), e = k < 0.5 ? 2 * k * k : 1 - (2 - 2 * k) ** 2 / 2;
            setGlide(k < 1 ? Math.round(from + (to - from) * e) : null);
            if (k < 1) frame.current = window.requestAnimationFrame(tick);
        };
        tick();
    };
    const pts = [...data, ...future], end = pts[pts.length - 1];

    const pes = data.map((p) => p.pe), eps = pts.map((p) => p.eps);
    const rungs = bandRungs(Math.min(...pes), Math.max(...pes)), top = rungs.length - 1;
    const rung = (p) => Math.max(0, Math.min(top - 1, rungs.findLastIndex((m) => p.pe >= m))); // the ribbon a day sits in
    const cur = rung(last);
    const lit = last.a5 != null && last.pe > last.a5 ? AMBER : TEAL;
    // the lit band is always one P/E wide, whatever the rungs' step (15Y steps by 2): the same band the tile names
    const lo = Math.min(Math.floor(last.pe), rungs[top] - 1);
    const mine = (i) => rungs[i] === lo || rungs[i] === lo + 1;
    const ink = (i) => `rgba(255,255,255,${Math.max(0.06, 0.24 - Math.min(Math.abs(i - cur), Math.abs(i - cur - 1)) * 0.05).toFixed(2)})`; // dimmer further from SPY

    const w = 540, h = 260, padT = 12, padB = 24;
    const lnLo = Math.log(Math.min(...eps) * rungs[0]), lnHi = Math.log(Math.max(...eps) * rungs[top]);
    const pad = (lnHi - lnLo) * 0.04, min = lnLo - pad, max = lnHi + pad;
    const toY = (v) => h - padB - ((Math.log(v) - min) / (max - min)) * (h - padT - padB);
    // y ticks on round dollars; the log scale packs the top ones, so any closer than 16px to the one below is dropped
    const lo$ = Math.exp(min), hi$ = Math.exp(max), base = 10 ** Math.floor(Math.log10(hi$ - lo$)) / 10;
    const tStep = [1, 2, 5, 10, 20, 50].map((m) => m * base).find((s) => Math.floor(hi$ / s) - Math.ceil(lo$ / s) < 6);
    const yTicks = [];
    for (let k = Math.ceil(lo$ / tStep); k <= Math.floor(hi$ / tStep); k++) {
        const y = toY(k * tStep);
        if (!yTicks.length || yTicks[yTicks.length - 1].y - y >= 16) yTicks.push({ y, label: usd(k * tStep) });
    }
    const padL = gutterFor(yTicks.map((t) => t.label), w, plotPx, 34);
    const padR = unitsForPx(66, w, plotPx) ?? 62; // the rung labels
    const t0 = ms(pts[0].date), t1 = ms(end.date);
    const xAt = (iso) => padL + ((ms(iso) - t0) / (t1 - t0 || 1)) * (w - padL - padR);
    const xy = (p, v) => `${xAt(p.date).toFixed(1)},${toY(v).toFixed(1)}`;
    const xNow = xAt(last.date), xEnd = xAt(end.date);
    const lineOf = (m, from = pts) => from.map((p) => xy(p, p.eps * m));

    const dates = pts.map((p) => p.date), toX = (i) => xAt(dates[i]);
    const yrGap = Math.max(26, unitsForPx(LABEL_CHAR_PX * 4 + 10, w, plotPx) ?? 26);
    const xLabels = (t1 - t0) / DAY > 550
        ? yearTicks(dates, toX, { minGap: yrGap, maxLabels: Math.min(8, Math.floor((w - padL - padR) / yrGap)) })
        : shortTicks(dates, toX, { minGap: unitsForPx(LABEL_CHAR_PX * 6 + 10, w, plotPx) ?? 40 });
    const lblGap = unitsForPx(4, w, plotPx) ?? 4;
    const axisLabels = [
        ...yTicks.map((t) => ({ x: padL - lblGap, y: t.y, text: t.label, ax: 'end', ay: 'middle' })),
        ...xLabels.map((l) => ({ x: clampUnits(l.x, (l.label.length * LABEL_CHAR_PX) / 2, w, plotPx), y: h, text: l.label, ax: 'middle', ay: 'bottom' })),
    ];

    const hov = hover != null && hover < pts.length ? pts[hover] : null;
    // rung labels at the right edge: SPY's own two rungs first, then top down, none within 11px of another
    const shown = [];
    for (const i of [cur, cur + 1, ...rungs.map((_, k) => top - k)]) {
        const y = toY(end.eps * rungs[i]);
        if (!shown.some((s) => Math.abs(s.y - y) < 11)) shown.push({ i, y });
    }
    const onPoint = (e) => {
        const r = svgRef.current?.getBoundingClientRect();
        if (!r?.width) return;
        const t = t0 + ((((e.clientX - r.left) / r.width) * w - padL) / (w - padL - padR)) * (t1 - t0);
        let i = pts.findIndex((p) => ms(p.date) >= t);
        if (i < 0) i = pts.length - 1;
        else if (i > 0 && t - ms(pts[i - 1].date) < ms(pts[i].date) - t) i -= 1;
        setHover(i);
    };
    const asOf = fwd[fwd.length - 1].date;
    const est = ms(last.date) - ms(asOf) > 9 * DAY ? ` · est. since ${fmtDay(asOf).replace(/, \d{4}$/, '')}` : '';
    const label = { fontSize: '0.65rem', fontFamily: MONO, fontWeight: 700, whiteSpace: 'nowrap' };
    const legend = { fontSize: '0.6rem', fontWeight: 600, whiteSpace: 'nowrap' };
    const dot = (x, y, color, size = 5) => <path d={`M${x.toFixed(1)} ${y.toFixed(1)}h0`} stroke={color} strokeWidth={size} strokeLinecap="round" vectorEffect="non-scaling-stroke" />;
    const pctOf = (x, y) => ({ left: `${(x / w) * 100}%`, top: `${(y / h) * 100}%` });

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
                {!hov ? (
                    <span style={{ ...label, color: lit }}>{usd(last.price)} = {last.pe.toFixed(1)}x earnings</span>
                ) : hover < data.length ? (
                    <span className="chart-readout" style={{ ...label, color: 'var(--text-primary)' }}>{fmtDay(hov.date)} · {usd(hov.price)} · {hov.pe.toFixed(1)}x</span>
                ) : (
                    <span className="chart-readout" style={{ ...label, color: 'var(--text-primary)' }}>
                        {monthYear(hov.date)} · at {last.pe.toFixed(1)}x {usd(hov.eps * last.pe)}{last.a5 != null ? ` · at ${last.a5.toFixed(1)}x ${usd(hov.eps * last.a5)}` : ''}
                    </span>
                )}
            </div>
            <div style={{ display: 'flex', flexWrap: 'nowrap', gap: '8px', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontSize: '0.68rem', fontWeight: 700, padding: '4px 10px', borderRadius: '6px', whiteSpace: 'nowrap', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', background: lit === TEAL ? 'rgba(45,212,191,0.12)' : 'rgba(251,191,36,0.12)', color: lit }}>
                    Between {lo}x and {lo + 1}x{last.a5 != null ? ` · 5-yr avg ${last.a5.toFixed(1)}x = ${usd(last.eps * last.a5)}` : ''}
                </span>
                {farDays > 0 && (
                    <button type="button" onClick={look} aria-pressed={far} aria-label={`Look ahead to ${farYear}`} title={far ? 'Back to the year ahead' : `Run the bands out to the end of ${farYear}`}
                        style={{ ...chip(far), marginLeft: 'auto', flexShrink: 0 }}>🔭 {farYear}</button>
                )}
            </div>
            <div className="mini-chart" style={{ height: '260px' }}>
                <div className="chart-plot" ref={plotRef}>
                    <svg ref={svgRef} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img"
                        aria-label={`SPY at ${last.pe.toFixed(1)} times forward earnings, between the ${lo}x and ${lo + 1}x bands`}
                        onPointerMove={onPoint} onPointerDown={onPoint} onPointerLeave={(e) => { if (e.pointerType === 'mouse') setHover(null); }}
                        style={{ touchAction: 'pan-y' }}>
                        <defs>
                            <clipPath id={`${uid}-plot`}><rect x={padL} y="0" width={xEnd - padL} height={h - padB + 2} /></clipPath>
                            <filter id={`${uid}-glow`} x="-5%" y="-30%" width="110%" height="160%"><feGaussianBlur stdDeviation="3" /></filter>
                        </defs>
                        {yTicks.map((t, i) => <line key={i} x1={padL} x2={xEnd} y1={t.y} y2={t.y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />)}
                        <g clipPath={`url(#${uid}-plot)`} key={active}>
                            <polygon className="vb-lit" points={[...lineOf(lo), ...lineOf(lo + 1).reverse()].join(' ')} fill={lit} fillOpacity="0.13" />
                            {rungs.map((m, i) => (
                                <polyline key={m} points={lineOf(m, data).join(' ')} fill="none" stroke={ink(i)} strokeWidth="1" vectorEffect="non-scaling-stroke" />
                            ))}
                            {ahead > 0 && (
                                <g className="vb-ahead">
                                    <line x1={xNow} x2={xNow} y1={padT - 2} y2={h - padB} stroke="rgba(255,255,255,0.16)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
                                    {rungs.map((m, i) => (
                                        <g key={m} fill="none" stroke={ink(i)} strokeWidth="1">
                                            <polyline points={lineOf(m, [last, ...year1]).join(' ')} strokeDasharray="2 4" vectorEffect="non-scaling-stroke" />
                                            {later.length > 1 && <polyline points={lineOf(m, later).join(' ')} strokeOpacity="0.55" strokeDasharray="1 5" vectorEffect="non-scaling-stroke" />}
                                        </g>
                                    ))}
                                </g>
                            )}
                            <polyline className="vb-glow" points={data.map((p) => xy(p, p.price)).join(' ')} fill="none" stroke={BLUE} strokeOpacity="0.3" strokeWidth="4" filter={`url(#${uid}-glow)`} />
                            <polyline className="vb-spy" pathLength="1" points={data.map((p) => xy(p, p.price)).join(' ')} fill="none" stroke={BLUE} strokeWidth="1.8" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
                        </g>
                        {hov && (
                            <g className="chart-cursor" pointerEvents="none">
                                <line x1={xAt(hov.date)} x2={xAt(hov.date)} y1={padT} y2={h - padB} stroke="rgba(255,255,255,0.35)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
                                {rungs.map((m, i) => <g key={m}>{dot(xAt(hov.date), toY(hov.eps * m), mine(i) ? lit : 'rgba(255,255,255,0.45)', 4)}</g>)}
                                {hov.price != null && dot(xAt(hov.date), toY(hov.price), BLUE, 7)}
                            </g>
                        )}
                    </svg>
                    <AxisLabels w={w} h={h} labels={axisLabels} />
                    <span className="vb-now" aria-hidden="true" style={{ ...pctOf(xNow, toY(last.price)), color: BLUE }} />
                    {shown.map(({ i, y }) => (
                        <span key={rungs[i]} aria-hidden="true" style={{
                            position: 'absolute', ...pctOf(xEnd + 4, y), transform: 'translateY(-50%)', pointerEvents: 'none', whiteSpace: 'nowrap',
                            fontSize: '0.6rem', fontWeight: 700, fontFamily: MONO, color: mine(i) ? lit : 'var(--text-muted)',
                        }}>{rungs[i]}x <span style={{ color: mine(i) ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{usd(end.eps * rungs[i])}</span></span>
                    ))}
                </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'safe center', gap: '14px', marginTop: '6px', flexWrap: 'nowrap', overflowX: 'auto' }}>
                <span style={{ ...legend, color: est ? AMBER : BLUE }} title={`Each line = the next 12 months of expected earnings (FactSet, weekly) × a P/E. The lit band is the one SPY sits in. Newest report: ${fmtDay(asOf)}.`}>━ SPY{est}</span>
                {ahead > 0 && (
                    <span style={{ ...legend, color: 'var(--text-muted)' }} title={`The bands carried forward as if earnings grow ${grow.g.toFixed(1)}% a year, the growth analysts expect for ${grow.gy || 'the furthest year'} in FactSet's ${fmtDay(grow.date)} report. An assumption, not a forecast. ${later.length > 1 ? ` After 12 months: ${signedPct(pace * 100)} a year, the pace forward earnings grew over the last ${Math.round(oldYears)} years (no analyst number that far out).` : ''} Prices on the right = each P/E at ${monthYear(end.date)}.`}>
                        {later.length > 1
                            ? `┄ if ${signedPct(grow.g)} (${whose}), then ${signedPct(pace * 100)}/yr (10-yr pace)`
                            : `┄ if earnings ${signedPct(grow.g)} (${whose}) → ${monthYear(end.date)}`}
                    </span>
                )}
            </div>
        </div>
    );
}
