'use client';
/**
 * v2 — the same live feeds as v1 (useFeeds.js copies v1's loading rules), laid out
 * top-down by importance: the market now (SPY, Fear & Greed, the read), his trade
 * signals, then Market → Risk → Economy → World → Crowd. v1 at / is untouched.
 */
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import ErrorBoundary from '../../components/ErrorBoundary';
import Skeleton from '../../components/Skeleton';
import SpyChart from '../../components/SpyChart';
import MiniChart from '../../components/MiniChart';
import { SEVERITY, FRIENDLY, PILL_ENTRIES } from '../../components/JevPills';
import JevPillModal from '../../components/JevPillModal';
import { pillWhy } from '../../lib/jevWhy';
import { FR_OVER, FR_UNDER } from '../../lib/frTriggers';
import { valuationRow } from '../../components/EconomicIndicatorGrid';
import FourHorsemen from '../../components/FourHorsemen';
import PolymarketTable from '../../components/PolymarketTable';
import VolMetricsTable from '../../components/VolMetricsTable';
import RubberBandRadar from '../../components/RubberBandRadar';
import FactorRow from '../../components/FactorRow';
import Delta from '../../components/Delta';
import SignalStudy from '../../components/SignalStudy';
import MarkChip from '../../components/MarkChip';
import MarksSheet from '../../components/MarksSheet';
import { MarkProvider, useMark, useChart, collectLiveValues } from '../../components/MarkProvider';
import BackPill from '../../components/BackPill';
import { jumpToCard } from '../../components/WhatMoved';
import StatusLine from '../../components/StatusLine';
import SinceLastVisit from '../../components/SinceLastVisit';
import MarketClock from '../../components/MarketClock';
import { UpdatedAgo, OfflineBanner, PullToRefresh } from '../../components/PhonePolish';
import { staleNote } from '../../components/CustomIndicatorBar';
import { isLiveAnswer } from '../../lib/snapshot';
import { freshnessNote, formatAsOf } from '../../lib/freshness';
import { fgHistoryCells } from '../../lib/fgHistory';
import { staleOf } from '../../lib/glance';
import { collectMoves, fmtLevel, movedWhen, sigmaOf, BIG_Z } from '../../lib/whatMoved';
import { marketWindow } from '../../lib/marketWindow';
import { ivRank, ivPercentile } from '../../lib/vol';
import { pulseVerdicts, horsemenRiding } from '../../lib/pulseVerdicts';
import { decodeLong, joinLong } from '../../lib/longHistory';
import useFeeds from './useFeeds';

const SECTIONS = [
    ['now', 'Now'], ['signals', 'Signals'], ['market', 'Market'], ['risk', 'Risk'],
    ['economy', 'Economy'], ['world', 'World'], ['crowd', 'Crowd'],
];

const FG_ZONES = [
    { to: 25, tone: 'bad' }, { to: 45, tone: 'caution' }, { to: 55, tone: 'flat' }, { to: 75, tone: 'good' }, { to: 100, tone: 'good2' },
];
const RSI_ZONES = [{ to: 30, tone: 'good' }, { to: 70, tone: 'flat' }, { to: 100, tone: 'bad' }];
const fgTone = (s) => (s < 25 ? 'bad' : s < 45 ? 'caution' : s < 55 ? 'flat' : 'good');
const signed = (v, d = 2) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`;
const usd = (v) => `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
// Same bands as the Volatility table: 90+ panic, 70+ rich, 10 or under cheap.
const ivTone = (v) => (v >= 90 ? 'bad' : v >= 70 ? 'caution' : v <= 10 ? 'good' : 'flat');

/** A card body as its own component, so the card's ErrorBoundary catches a throw in it. */
function Body({ children }) { return children(); }

function Unavailable({ what, onRetry }) {
    return <button type="button" className="card-unavailable" onClick={onRetry}>{what} unavailable · tap ↻</button>;
}

const DAY = 86400000;

/**
 * F&G's year as one line on a zoom-in timeline: 1Y · 1M · 1W · 1D · Now sit evenly spaced (each
 * stretch drawn in its own time), CNN's readings as dots, the history sheet's daily record between
 * them, the zones behind. A stretch with no daily record is a dashed straight line, not a guess.
 */
function FgTrail({ fg }) {
    const sheet = useChart('cnnFearGreed', fg._meta?.proxy ? null : fg.score);
    const now = Date.parse(String(fg.asOf || new Date().toISOString()).slice(0, 10));
    const cells = fgHistoryCells(fg); // Prev Close, 1 Week, 1 Month, 1 Year
    const anchors = [[365, '1Y', 3], [30, '1M', 2], [7, '1W', 1], [1, '1D', 0]]
        .map(([d, k, i]) => ({ k, t: now - d * DAY, ...cells[i], v: cells[i].val }))
        .filter((a) => a.v != null)
        .concat({ k: 'Now', t: now, v: Math.round(fg.score) });
    const last = anchors.length - 1;
    if (!last) return null;
    const X = (t) => {
        let j = 0;
        while (j < last - 1 && t > anchors[j + 1].t) j++;
        return ((j + Math.min(1, Math.max(0, (t - anchors[j].t) / (anchors[j + 1].t - anchors[j].t)))) / last) * 100;
    };
    const daily = (sheet?.points || []).map((p) => ({ t: Date.parse(p.date), v: p.value }))
        .filter((p) => Number.isFinite(p.v) && p.t > anchors[0].t && p.t < now && anchors.every((a) => Math.abs(a.t - p.t) >= DAY / 2));
    const pts = [...anchors, ...daily].sort((a, b) => a.t - b.t);
    let solid = '', dashed = '';
    for (let i = 1; i < pts.length; i++) {
        const seg = `M${X(pts[i - 1].t)} ${100 - pts[i - 1].v}L${X(pts[i].t)} ${100 - pts[i].v}`;
        if (pts[i].t - pts[i - 1].t > 10 * DAY) dashed += seg; else solid += seg;
    }
    const arrow = (d) => (d > 0 ? `▲${d}` : d < 0 ? `▼${-d}` : '');
    return (
        <div className="v2-fgt" role="img" aria-label={anchors.map((a) => `${a.k} ${a.v}`).join(', ')}>
            <div className="v2-fgt-plot">
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                    {FG_ZONES.map((z, i) => <rect key={z.to} className={`v2-fgt-band is-${z.tone}`} x="0" width="100" y={100 - z.to} height={z.to - (i ? FG_ZONES[i - 1].to : 0)} />)}
                    <path className="v2-fgt-gap" d={dashed} />
                    <path className="v2-fgt-line" d={solid} />
                </svg>
                {anchors.map((a) => (
                    <span key={a.k} className={`v2-fgt-pt tone-${fgTone(a.v)}${a.k === 'Now' ? ' is-now' : ''}`} style={{ left: `${X(a.t)}%`, top: `${100 - a.v}%` }}
                        title={a.k === 'Now' ? `Now ${a.v}` : `${a.label} ${a.v} · ${a.diff ? `${arrow(a.diff)} to now` : 'same as now'}`}>
                        {a.k !== 'Now' && <b>{a.v}</b>}
                    </span>
                ))}
            </div>
            <div className="v2-fgt-axis">
                {anchors.map((a) => (
                    <span key={a.k} style={{ left: `${X(a.t)}%` }}>{a.k}{a.diff ? <small className={a.diff > 0 ? 'tone-good' : 'tone-bad'}>{arrow(a.diff)}</small> : null}</span>
                ))}
            </div>
        </div>
    );
}

/** 0–100 strip with tinted zones and one marker: F&G and RSI read the same way. */
function Scale({ value, zones, ticks }) {
    let from = 0;
    return (
        <div className="v2-scale" role="img" aria-label={`${Math.round(value)} of 100`}>
            <div className="v2-scale-track">
                {zones.map((z) => {
                    const el = <span key={z.to} className={`v2-zone tone-${z.tone}`} style={{ left: `${from}%`, width: `${z.to - from}%` }} />;
                    from = z.to;
                    return el;
                })}
                <span className="v2-scale-mark" style={{ left: `${Math.max(0, Math.min(100, value))}%` }} />
            </div>
            <div className="v2-scale-ticks">{ticks.map((t) => <span key={t} style={{ left: `${t}%` }}>{t}</span>)}</div>
        </div>
    );
}

/** Last ~3 months of SPY with its 200-day average and 52-week high drawn in, each tagged at the right edge. */
function Spark({ spy }) {
    const rows = (spy.chartHistory || []).slice(-63).filter((p) => Number.isFinite(p.price));
    if (rows.length < 2) return null;
    const pts = rows.map((p) => p.price);
    const ma = rows.every((p) => Number.isFinite(p.ma200)) && Number.isFinite(spy.ma200?.value) ? rows.map((p) => p.ma200) : null;
    const high = Number.isFinite(spy.week52High?.value) ? spy.week52High.value : null;
    const all = [...pts, ...(ma || []), ...(high != null ? [high] : [])];
    const lo = Math.min(...all), hi = Math.max(...all), w = 320, h = 100, pad = 16;
    const y = (v) => pad + (1 - (v - lo) / (hi - lo || 1)) * (h - 2 * pad);
    const path = (vs) => vs.map((v, i) => `${i ? 'L' : 'M'}${((i / (vs.length - 1)) * w).toFixed(1)},${y(v).toFixed(1)}`).join('');
    const line = path(pts);
    const up = pts[pts.length - 1] >= pts[0];
    const highPct = spy.week52High?.pct;
    return (
        <div className={`v2-spark ${up ? 'is-up' : 'is-down'}`}>
            <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
                <defs>
                    <linearGradient id="v2SparkFill" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stopColor="currentColor" stopOpacity="0.16" />
                        <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
                    </linearGradient>
                </defs>
                <path d={`${line}L${w},${h}L0,${h}Z`} fill="url(#v2SparkFill)" />
                {high != null && <path className="v2-spark-high" d={`M0,${y(high).toFixed(1)}H${w}`} vectorEffect="non-scaling-stroke" />}
                {ma && <path className="v2-spark-ma" d={path(ma)} vectorEffect="non-scaling-stroke" />}
                <path d={line} fill="none" stroke="currentColor" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
            </svg>
            <span className="v2-spark-tag is-span" style={{ top: `${high != null ? y(high) : pad}%` }}>3M</span>
            {high != null && (
                <span className="v2-spark-tag is-high" style={{ top: `${y(high)}%` }}>
                    {highPct >= 0 ? <span className="tone-good">At the 52W high</span> : <>52W high <b>{usd(high)}</b> <em className={highPct >= -1 ? 'tone-good' : 'tone-caution'}>{signed(highPct)}%</em></>}
                </span>
            )}
            {ma && (
                // a rising line runs below its tag's left end, so the tag sits above it (and below a falling one)
                <span className={`v2-spark-tag is-ma${ma[ma.length - 1] >= ma[Math.max(0, ma.length - 16)] ? ' is-above' : ''}`} style={{ top: `${y(ma[ma.length - 1])}%` }}>
                    200D <b>{usd(spy.ma200.value)}</b> <em className={spy.ma200.pct >= 0 ? 'tone-good' : 'tone-bad'}>{signed(spy.ma200.pct)}%</em>
                </span>
            )}
        </div>
    );
}

const MOVE_CAP = 4; // bars run out to 4× a normal day; anything bigger pins at the end

/** What moved since the last close, each move drawn against that series' normal day (the shaded ±1× band). */
function Moves({ spy, fg, extra, vol, history, saved, waiting }) {
    const moves = useMemo(() => collectMoves({ spy, fg, extra, vol, history }), [spy, fg, extra, vol, history]);
    const when = useMemo(() => { try { return movedWhen({ spy, vol }); } catch { return null; } }, [spy, vol]);
    if (!moves.length) return waiting ? <div className="v2-moves is-waiting" aria-busy="true"><div className="v2-moves-head">What moved</div></div> : null;
    return (
        <div className="v2-moves" data-cached={saved || undefined}>
            <div className="v2-moves-head">
                <span>What moved{when ? ` · ${when}` : ''}{saved ? ` · ${saved}` : ''}</span>
                <span>× a normal day</span>
            </div>
            <div className="v2-moves-grid">
                {moves.map((m) => <MoveTile key={m.key} m={m} feeds={{ spy, extra, vol }} />)}
            </div>
        </div>
    );
}

const MOVE_DAYS = 22; // about a month of sessions in each tile's line
// green = good for stocks: up, except VIX (fear rising); the 10Y, oil and gold just read up = green
const prices = (h) => (Array.isArray(h) ? h.map((p) => p?.price) : []);

/** One move: its last month, with the shaded band = a normal day around yesterday's level. A big move leaps out of it. */
function MoveTile({ m, feeds: { spy, extra, vol } }) {
    const fgLine = useChart('cnnFearGreed', m.key === 'fg' ? m.value : null);
    const geo = useMemo(() => {
        const raw = m.key === 'spy' ? prices(spy?.chartHistory) : m.key === 'tnx' ? prices(extra?.rates?.tnx?.history)
            : ['cl', 'gc', 'btc'].includes(m.key) ? prices(extra?.commodities?.[m.key]?.history)
            : m.key === 'vix' ? vol?.vixYear || [] : (fgLine?.points || []).map((p) => p.value);
        const s = raw.map(Number).filter(Number.isFinite);
        while (s.length && Math.abs(s[s.length - 1] - m.value) < 1e-9) s.pop(); // today's own close, if the series has it
        const pts = [...s.slice(-(MOVE_DAYS - 2)), m.prev, m.value];
        if (Math.abs(pts[pts.length - 3] - m.prev) < 1e-9) pts.splice(-2, 1); // yesterday's close already in the series
        if (pts.length < 6 || !(m.z > 0)) return null;
        const hw = Math.abs(m.delta) / m.z; // a normal day, in the move's own unit
        const band = m.unit === 'pct' ? (m.prev * hw) / 100 : m.unit === 'bp' ? hw / 100 : hw;
        const lo = Math.min(...pts, m.prev - band), hi = Math.max(...pts, m.prev + band);
        const pad = (hi - lo) * 0.12 || 1;
        const y = (v) => 100 - ((v - lo + pad) / (hi - lo + 2 * pad)) * 100;
        const x = (i) => (i / (pts.length - 1)) * 100;
        const n = pts.length;
        return {
            past: pts.slice(0, -1).map((v, i) => `${i ? 'L' : 'M'}${x(i)} ${y(v)}`).join(''),
            today: `M${x(n - 2)} ${y(pts[n - 2])}L100 ${y(m.value)}`,
            band: [y(m.prev + band), y(m.prev - band)],
            end: y(m.value),
        };
    }, [m, spy, extra, vol, fgLine]);
    return (
        <button type="button" className={`v2-mv${m.big ? ' is-big' : ''} is-${(m.delta > 0) === (m.key !== 'vix') ? 'good' : 'bad'}`} onClick={() => jumpToCard(m.jump)}
            style={{ '--heat': Math.min(m.z, MOVE_CAP) / MOVE_CAP }}
            title={`${m.label} ${fmtLevel(m.prev)} → ${fmtLevel(m.value)}${m.day ? ` on ${m.day}` : ''} · ${m.z.toFixed(1)}× a normal day · tap for the card`}>
            <span className="v2-mv-label">{m.label}{m.day && <small>{m.day}</small>}</span>
            <span className="v2-mv-val">{m.text}</span>
            {geo ? (
                <span className="v2-mv-spark" style={{ '--end': `${geo.end}%` }}>
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                        <rect className="v2-mv-band" x="0" width="100" y={geo.band[0]} height={geo.band[1] - geo.band[0]} />
                        <path className="v2-mv-past" d={geo.past} />
                        <path className="v2-mv-today" d={geo.today} />
                    </svg>
                    <i />
                </span>
            ) : (
                <span className="v2-mv-bar"><i style={{ [m.delta < 0 ? 'right' : 'left']: '50%', width: `${(Math.min(m.z, MOVE_CAP) / MOVE_CAP) * 50}%` }} /></span>
            )}
            <span className="v2-mv-x">{m.z.toFixed(1)}×</span>
        </button>
    );
}

const pctS = (v) => `${signed(v, 1)}%`;
const pct2 = (v) => `${v.toFixed(2)}%`;
const kfmt = (v) => `${Math.round(v).toLocaleString('en-US')}K`;
const fixed = (d) => (v) => v.toFixed(d);

/**
 * Economy: each signal drawn as its distance from the line where the backend turns it green
 * (app/api/fred/route.js). `band` is that route's next tier step (OK → Strong, or green → the
 * next colour), so a dot one band out is one tier past the line; the track shows two a side.
 * `low`: lower is better. Rows run worst first, so what needs a look is on top.
 */
const BULL = [
    { key: 'nfci', label: 'Financial conditions', line: 0, band: 0.5, low: true, fmt: fixed(2), tip: 'Chicago Fed money tightness. Below 0 = easy.' },
    { key: 'm2', label: 'M2 money supply', line: 2, band: 2, fmt: pctS, tip: 'Cash + deposits, 1-year growth.' },
    { key: 'retail', label: 'Retail sales', line: 0, band: 1, fmt: pctS, tip: 'Retail sales, 3-month growth.' },
    { key: 'housing', label: 'Housing starts', line: 1300, band: 100, fmt: kfmt, tip: 'New homes started. Green also needs a rising trend.' },
    { key: 'indpro', label: 'Industrial production', line: 0, band: 1, fmt: pctS, tip: 'Factory output, 6-month trend.' },
    { key: 'jolts', label: 'Job openings', line: 6000, band: 1000, fmt: kfmt, tip: 'JOLTS job openings.' },
    { key: 'durable', label: 'Durable goods', line: 0, band: 2, fmt: pctS, tip: 'Big-ticket orders, 3-month trend.' },
    { key: 'savings', label: 'Savings rate', line: 3.5, band: 1.5, fmt: (v) => `${v.toFixed(1)}%`, tip: 'Share of income saved.' },
].map((d) => ({ ...d, ticks: [[d.line, 'OK'], [d.line + (d.low ? -d.band : d.band), 'Strong']],
    rule: `OK ${d.low ? '<' : '>'} ${d.fmt(d.line)} · Strong ${d.low ? '<' : '>'} ${d.fmt(d.line + (d.low ? -d.band : d.band))}` }));

// Copper/gold has no tiers: green while its 1-month change is up, so that change is its distance (one band = a 5% month).
const INDICATORS = [
    { key: 'sahmRule', ticks: [[0.5, 'Fires']], label: 'Sahm rule', line: 0.5, band: 0.5, low: true, fmt: fixed(2), rule: 'Safe < 0.50', tip: 'Fires when the 3-month jobless rate is 0.5 over its 12-month low.' },
    { key: 'sentiment', ticks: [[80, 'Strong'], [60, 'Weak']], label: 'Consumer sentiment', line: 80, band: 20, fmt: fixed(1), rule: 'Strong > 80 · Weak < 60', tip: 'UMich consumer confidence survey.' },
    { key: 'claims', ticks: [[250, 'Healthy'], [350, 'Weak']], label: 'Jobless claims', line: 250, band: 100, low: true, fmt: kfmt, rule: 'Healthy < 250K', tip: 'New jobless claims, 4-week average.' },
    { key: 'creditSpread', ticks: [[1.5, 'Calm'], [2.5, 'Stressed']], label: 'Credit spread', line: 1.5, band: 1, low: true, fmt: pct2, rule: 'Calm < 1.50%', tip: 'BBB yield over Treasuries. Wider = stress.' },
    { key: 'realYields', ticks: [[0, 'Easy'], [2, 'Tight']], label: 'Real yields', line: 0, band: 2, low: true, fmt: pct2, rule: 'Easy ≤ 0% · Tight > 2%', tip: '10-year TIPS yield: borrowing cost after inflation.' },
    { key: 'copperGold', ticks: [[0, 'Flat']], tfmt: (v) => `${v}%`, label: 'Copper/gold', line: 0, band: 5, fmt: fixed(2), pos: (m) => m.changePct ?? m.changePct3mo, tip: 'Copper vs gold. Rising = growth mood.' },
];

function gauge(def, m, tone) {
    const pos = def.pos ? def.pos(m || {}) : m?.value;
    const s = Number.isFinite(m?.value) && Number.isFinite(pos) ? ((pos - def.line) / def.band) * (def.low ? -1 : 1) : NaN;
    const rule = def.key === 'copperGold' && Number.isFinite(pos) ? `Green while rising · 1mo ${pctS(pos)}` : def.rule;
    return { markKey: def.key, ...def, m, rule, s, tone };
}

function bullRows(fred) {
    const c = fred?.checklist;
    return c ? BULL.map((d) => gauge(d, c[d.key], c[d.key]?.bullish ? 'good' : 'bad')) : null;
}

function indicatorRows(fred) {
    const ind = fred?.indicators;
    if (!ind) return null;
    const pe = valuationRow(fred);
    const cape = !!fred.peIsCape;
    return [
        ...INDICATORS.map((d) => gauge(d, ind[d.key], statusTone(ind[d.key]?.status))),
        gauge({ key: 'peRatio', markKey: pe.markKey, label: cape ? 'CAPE' : 'P/E', line: cape ? 17 : 20, band: cape ? 13 : 5, low: true, fmt: fixed(1),
            ticks: cape ? [[17, 'Avg'], [30, 'Rich']] : [[20, 'Fair'], [25, 'Rich']],
            rule: cape ? 'Avg ~17 · Tight > 30' : 'Fair ~20 · Tight > 25', tip: pe.tooltip }, pe.metric, statusTone(pe.status)),
    ];
}

const NEAR = 0.25; // a green dot this close to its line is about to flip: drawn as a ring

// Where a level sits on the track: the line in the middle, one band a quarter of the width, two bands the end.
const trackX = (g, at) => 50 + Math.max(-2, Math.min(2, ((at - g.line) / g.band) * (g.low ? -1 : 1))) * 25;

/** The track's zones on hover/open: red below the line, grey for a middle tier a band down, green past it, brighter past Strong. */
function zones(g) {
    const xs = g.ticks.map(([at]) => Math.round(trackX(g, at)));
    const weak = xs.includes(25) ? 25 : 50;
    const strong = xs.includes(75) ? 75 : 100;
    return `linear-gradient(90deg, rgb(239 107 115 / .3) 0 ${weak}%, rgb(163 163 173 / .16) ${weak}% 50%, rgb(76 195 138 / .24) 50% ${strong}%, rgb(76 195 138 / .5) ${strong}%)`;
}

/** A row opened: its months since the history sheet began, split at the line, then how far it is from flipping. */
function GaugePanel({ g, note }) {
    const v = g.m?.value;
    const chart = useChart(g.markKey, v);
    // The sheet once logged housing starts in units, not thousands: days 500× off the median are dropped, not drawn.
    const raw = (chart?.points || []).filter((p) => Number.isFinite(p.value));
    const mid = raw.map((p) => p.value).sort((a, b) => a - b)[raw.length >> 1];
    const pts = raw.filter((p) => !(Math.abs(mid) > 0 && Math.abs(p.value / mid) >= 500));
    const vals = pts.map((p) => p.value);
    const name = g.ticks.find(([at]) => at === g.line)?.[1];
    const tfmt = g.tfmt || g.fmt;
    const amt = Number.isFinite(g.s) && !g.pos ? tfmt(Math.abs(v - g.line)).replace(/^[+−-]/, '') : null;
    let plot = null;
    if (vals.length > 1) {
        const line = g.pos ? null : g.line;
        const lo = Math.min(...vals, line ?? Infinity);
        const hi = Math.max(...vals, line ?? -Infinity);
        const pad = (hi - lo) * 0.18 || 1;
        const y = (val) => 100 - ((val - lo + pad) / (hi - lo + 2 * pad)) * 100;
        plot = { d: vals.map((val, i) => `${i ? 'L' : 'M'}${(i / (vals.length - 1)) * 100} ${y(val)}`).join(''), ty: line == null ? null : y(line), end: y(vals[vals.length - 1]) };
    }
    const since = pts[0] && formatAsOf(pts[0].date)?.replace(/, \d{4}$/, '');
    return (
        <div className="v2-gp">
            {plot && (
                <div className="v2-gp-chart" style={{ '--ty': `${plot.ty ?? 0}%`, '--end': `${plot.end}%` }}>
                    {[g.low ? 'is-bad' : 'is-good', g.low ? 'is-good' : 'is-bad'].map((cls, i) => (
                        <svg key={cls} className={`${plot.ty == null ? 'is-flat' : cls} ${i ? 'is-below' : 'is-above'}`} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d={plot.d} /></svg>
                    ))}
                    {plot.ty != null && <span className="v2-gp-line"><em>{name} {tfmt(g.line)}</em></span>}
                    <i className="v2-gp-end" />
                    {since && <span className="v2-gp-since">{since}</span>}
                </div>
            )}
            <div className="v2-gp-info">
                <b>{amt && (g.s < 0 ? `${amt} to ${name}` : `${amt} cushion`)}</b>
                <span className="v2-gp-rule">{g.rule}</span>
                <span className="v2-gp-asof">{note.tone === 'stale' ? '🕐 ' : ''}{formatAsOf(g.m?.asOf)}</span>
            </div>
            <div className="v2-gp-tip">{g.tip}</div>
        </div>
    );
}

function Gauge({ g, open, onToggle }) {
    const note = freshnessNote(g.m);
    const v = g.m?.value;
    const mark = useMark(g.markKey, v);
    const x = Number.isFinite(g.s) ? Math.max(-2, Math.min(2, g.s)) * 25 : null; // two bands a side fill the half track
    const tfmt = g.tfmt || g.fmt;
    return (
        <div className={`v2-g${x != null && g.s >= 0 && g.s < NEAR ? ' is-near' : ''}${open ? ' is-open' : ''}`} data-tone={note.tone === 'stale' ? 'caution' : Number.isFinite(v) ? g.tone : 'flat'}
            onClick={(e) => { if (!e.target.closest('.chartable, .mark-pop')) onToggle(); }}>
            <span className="v2-g-label"><button type="button" aria-expanded={open}><span className="tooltip-trigger" data-tooltip={`${g.tip}${note.suffix}`}>{g.label}</span></button></span>
            <span className="v2-g-track" style={{ '--zones': zones(g) }}>
                {g.ticks.map(([at, name]) => <em key={name} style={{ left: `${trackX(g, at)}%` }}><span>{tfmt(at)}</span></em>)}
                {x != null && <><i style={{ [x < 0 ? 'right' : 'left']: '50%', width: `${Math.abs(x)}%` }} /><b style={{ left: `${50 + x}%` }} /></>}
            </span>
            <Delta mark={note.tone === 'fresh' ? mark : null} format={g.fmt} chartKey={g.markKey} raw={v} className="v2-g-val">
                {Number.isFinite(v) ? g.fmt(v) : 'N/A'}
            </Delta>
            <span className="v2-g-rule">{g.rule}</span>
            {open && <GaugePanel g={g} note={note} />}
        </div>
    );
}

function Board({ rows }) {
    const [openKey, setOpenKey] = useState(null);
    if (!rows) return <div className="card"><Skeleton count={6} /></div>;
    const live = rows.filter((r) => Number.isFinite(r.m?.value));
    const good = live.filter((r) => r.tone === 'good');
    const sorted = [...rows].sort((a, b) => (Number.isFinite(a.s) ? a.s : Infinity) - (Number.isFinite(b.s) ? b.s : Infinity));
    return (
        <div className="card">
            <div className="card-header">
                <span className="v2-score" title={`${good.length} of ${live.length} green`}>
                    <span className="v2-score-bar">{[...sorted].reverse().filter((r) => live.includes(r)).map((r) => <i key={r.key} data-tone={r.tone} title={r.label} />)}</span>
                    {good.length}/{live.length}
                </span>
            </div>
            <div className="v2-gauges">{sorted.map((g) => <Gauge key={g.key} g={g} open={openKey === g.key} onToggle={() => setOpenKey(openKey === g.key ? null : g.key)} />)}</div>
        </div>
    );
}

/** Global markets: the same rows as ExtraMarketsGrid. `chart` is the history-sheet key; `mark` rows also get fresh-print marks. */
const MARKETS = [
    { t: 'GOLD', label: 'Gold', pick: (e) => e.commodities?.gc, fmt: usd, chart: 'gold' },
    { t: 'BTC', label: 'Bitcoin', pick: (e) => e.commodities?.btc, fmt: (v) => `$${Math.round(v).toLocaleString('en-US')}`, chart: 'btc' },
    { t: 'CL', label: 'Oil', pick: (e) => e.commodities?.cl, fmt: usd, chart: 'cl' },
    { t: 'TNX', label: '10Y yield', pick: (e) => e.rates?.tnx, fmt: pct2, rate: true, chart: 'tnx' },
    { t: 'T2Y', label: '2Y yield', pick: (e) => e.rates?.t2y, fmt: pct2, rate: true, chart: 't2y' },
    { t: 'DXY', label: 'Dollar index', pick: (e) => e.fx?.dxy, fmt: fixed(2), chart: 'dxy' },
    { t: 'USD/CAD', label: 'USD/CAD', pick: (e) => e.fx?.usdcad, fmt: fixed(4), chart: 'usdcad' },
    { t: 'USD/INR', label: 'USD/INR', pick: (e) => e.fx?.usdinr, fmt: fixed(2), chart: 'usdinr' },
    { t: 'USD/BDT', label: 'USD/BDT', pick: (e) => e.fx?.usdbdt, fmt: fixed(2), chart: 'usdbdt' },
    { t: 'CAD/BDT', label: 'CAD/BDT', pick: (e) => e.fx?.cadbdt, fmt: fixed(2), chart: 'cadbdt' },
    { t: 'INR/BDT', label: 'INR/BDT', pick: (e) => e.fx?.inrbdt, fmt: fixed(4), chart: 'inrbdt' },
    { t: 'CAD/INR', label: 'CAD/INR', pick: (e) => e.fx?.cadinr, fmt: fixed(2), chart: 'cadinr' },
    { t: 'MORT30', label: '30Y mortgage', pick: (e) => e.rates?.mortgageRate, fmt: pct2, rate: true, chart: 'mortgageRate', mark: true, home: true },
    { t: 'MTGPMT', label: 'Mortgage payment', pick: (e) => e.realEstate?.mortgagePayment, fmt: (v) => `$${Math.round(v).toLocaleString('en-US')}`, chart: 'mortgagePayment', mark: true, home: true },
    { t: 'ZRI', label: 'Rent', pick: (e) => e.realEstate?.rentIndex, fmt: (v) => `$${Math.round(v).toLocaleString('en-US')}`, chart: 'rentIndex', mark: true, home: true },
    { t: 'ATNHPI', label: 'House prices', pick: (e) => e.realEstate?.atnhpi, fmt: fixed(2), chart: 'atnhpi', mark: true, home: true },
];

function MarketSpark({ history }) {
    const v = (history || []).slice(-63).map((p) => p?.price ?? p?.value).filter(Number.isFinite);
    if (v.length < 2) return <span className="v2-mk-spark" />;
    const lo = Math.min(...v), k = Math.max(...v) - lo || 1;
    const pts = v.map((y, i) => `${((i / (v.length - 1)) * 80).toFixed(1)},${(22 - ((y - lo) / k) * 20).toFixed(1)}`).join(' ');
    return (
        <svg className={`v2-mk-spark ${v[v.length - 1] >= v[0] ? 'is-up' : 'is-down'}`} viewBox="0 0 80 24" preserveAspectRatio="none" aria-hidden="true">
            <polyline points={pts} />
        </svg>
    );
}

function MarketRow({ r }) {
    const mark = useMark(r.mark ? r.chart : undefined, r.d.current);
    const heat = r.z == null ? 0 : Math.min(r.z, MOVE_CAP) / MOVE_CAP;
    return (
        <div className={`v2-mk${r.z == null ? ' is-quiet' : ''}${r.z >= BIG_Z ? ' is-big' : ''}`} data-dir={r.win?.dir ?? 0} style={{ '--heat': heat }}
            title={r.d.stale ? 'Live sources failed: last good value' : undefined}>
            <span className="v2-mk-label">{r.label}</span>
            {r.z != null && <MarketSpark history={r.d.history} />}
            <Delta mark={mark} format={r.fmt} chartKey={r.chart} raw={r.d.current} className={`v2-mk-val${r.d.stale ? ' tone-caution' : ''}`}>
                {Number.isFinite(r.d.current) ? r.fmt(r.d.current) : 'N/A'}
            </Delta>
            {r.win && <span className="v2-mk-chg">{r.win.text}{r.home && r.win.tag && <small>{r.win.tag}</small>}</span>}
            {r.z != null && <span className="v2-mv-bar"><i style={{ [r.win.dir < 0 ? 'right' : 'left']: '50%', width: `${heat * 50}%` }} /></span>}
            {r.z != null && <span className="v2-mk-x">{r.z.toFixed(1)}×</span>}
        </div>
    );
}

const TAKA_RANGES = [['1M', 31], ['1Y', 366], ['5Y', 1827], ['10Y', 3653], ['15Y', 5479], ['20Y', 7305], ['30Y', 10958], ['All', Infinity]];
const TAKA_KEY = 'v2-taka-range';
const isoMinus = (iso, days) => new Date(Date.parse(`${iso}T00:00:00Z`) - days * 86400000).toISOString().slice(0, 10);
const taka = (v) => `৳${Math.round(v).toLocaleString('en-US')}`;
const longDate = (iso) => new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });

/**
 * USD/BDT, all of it: the baked history (public/v2/usdbdt.json: IMF official month-ends from 1972, Wise daily from Oct 2021), then the
 * history sheet's daily record (lib/longHistory join), then today's live rate. Up = each dollar buys more taka.
 */
function Taka({ extra, loading }) {
    const live = extra?.fx?.usdbdt?.current;
    const sheet = useChart('usdbdt', live);
    const [baked, setBaked] = useState(null);
    const [range, setRange] = useState('1Y');
    const [at, setAt] = useState(null);
    useEffect(() => {
        try { const r = window.localStorage.getItem(TAKA_KEY); if (TAKA_RANGES.some(([id]) => id === r)) setRange(r); } catch { /* private mode */ }
        fetch('/v2/usdbdt.json').then((r) => (r.ok ? r.json() : null)).then((b) => setBaked(decodeLong(b))).catch(() => {});
    }, []);
    const all = useMemo(() => {
        const pts = sheet?.points?.length ? [...joinLong(sheet.points, baked, null)] : [...(baked || [])];
        if (Number.isFinite(live) && pts.length) {
            const today = new Date().toLocaleDateString('en-CA'); // the local calendar day, as the sheet dates it
            const last = pts[pts.length - 1];
            if (last.date >= today) pts[pts.length - 1] = { ...last, value: live }; else pts.push({ date: today, value: live });
        }
        return pts;
    }, [sheet, baked, live]);
    if (all.length < 2) return loading || !baked ? <Skeleton count={3} /> : <div className="v2-fred-na">Unavailable · source busy, try again shortly</div>;

    const last = all[all.length - 1], cur = last.value;
    const span = (Date.parse(last.date) - Date.parse(all[0].date)) / 86400000;
    const chips = TAKA_RANGES.filter(([, d]) => d === Infinity || d < span);
    const [id, days] = chips.find(([r]) => r === range) || chips[chips.length - 1];
    const from = days === Infinity ? all[0].date : isoMinus(last.date, days - 1);
    const view = all.filter((p) => p.date >= from);
    const start = view[0].value;
    const ath = all.reduce((m, p) => (p.value > m.value ? p : m), all[0]);
    const step = 5, next = Math.floor(cur / step) * step + step;
    // 5Y and longer: the compound yearly rise, and the steady path at that rate (dashed) to race the real one
    const t0 = Date.parse(view[0].date), T = Date.parse(last.date) - t0;
    const cagr = days >= 1827 && T > 0 ? Math.pow(cur / start, 31557600000 / T) - 1 : null;

    // the chart: ≤ 600 points, milestone lines at round levels (≤ 4 in view)
    const stride = Math.ceil(view.length / 600);
    const pts = view.filter((p, i) => i % stride === 0 || i === view.length - 1);
    let lo = Math.min(...pts.map((p) => p.value)), hi = Math.max(...pts.map((p) => p.value));
    const pad = (hi - lo) * 0.08 || 0.5; lo -= pad; hi += pad;
    const y = (v) => 100 - ((v - lo) / (hi - lo)) * 100;
    const x = (i) => ((Date.parse(pts[i].date) - t0) / T) * 100; // by date: the IMF years are monthly, the Wise years daily
    const line = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i)} ${y(p.value)}`).join('');
    const pace = cagr != null && pts.map((p, i) => `${i ? 'L' : 'M'}${x(i)} ${y(start * Math.pow(cur / start, (Date.parse(p.date) - t0) / T))}`).join('');
    const grid = [0.5, 1, 2, 5, 10, 20, 50].find((g) => (hi - lo) / g <= 4) || 50;
    const levels = [];
    for (let v = Math.ceil(lo / grid) * grid; v < hi; v += grid) levels.push(v);
    const up = cur >= start;
    const pick = (r) => { setRange(r); try { window.localStorage.setItem(TAKA_KEY, r); } catch { /* still works this visit */ } };
    const read = (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const t = t0 + Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * T;
        const i = Math.max(0, pts.findIndex((p) => Date.parse(p.date) >= t));
        setAt({ i, p: pts[i] });
    };
    return (
        <div className={`card v2-taka is-${up ? 'up' : 'down'}`}>
            <div className="card-header">
                <div className="v2-taka-tfs" role="group" aria-label="Chart range">
                    {chips.map(([r]) => <button key={r} type="button" className={r === id ? 'is-on' : ''} aria-pressed={r === id} onClick={() => pick(r)}>{r}</button>)}
                </div>
            </div>
            <div className="v2-taka-top">
                <span className="v2-taka-now">{cur.toFixed(2)}</span>
                <span className={`v2-taka-chg tone-${up ? 'good' : 'bad'}`}>{up ? '▲' : '▼'} {cur / start >= 2 ? `${(cur / start).toFixed(1)}×` : `${signed((cur / start - 1) * 100, 1)}%`} <small>{id === 'All' ? `since ${view[0].date.slice(0, 4)}` : id}</small></span>
                {cagr != null && <span className="v2-taka-cagr" title={`The steady yearly rise from ${start.toFixed(2)} to ${cur.toFixed(2)} over ${(T / 31557600000).toFixed(1)} years (dashed line)`}>{signed(cagr * 100, 1)}%/yr</span>}
                {cur.toFixed(2) === ath.value.toFixed(2) || cur > ath.value
                    ? <span className="v2-taka-ath">All-time high</span>
                    : <span className="v2-taka-off">High {ath.value.toFixed(2)} · {longDate(ath.date)}</span>}
            </div>
            <div className="v2-taka-plot" onPointerMove={read} onPointerDown={read} onPointerLeave={() => setAt(null)}
                style={{ '--end': `${y(cur)}%` }}>
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                    <path className="v2-taka-area" d={`${line}L100 100L0 100Z`} />
                    {pace && <path className="v2-taka-pace" d={pace} />}
                    <path className="v2-taka-line" d={line} />
                </svg>
                {levels.map((v) => <span key={v} className="v2-taka-lvl" style={{ top: `${y(v)}%` }}><b>{Number.isInteger(v) ? v : v.toFixed(1)}</b></span>)}
                <i className="v2-taka-dot" />
                {at && (
                    <>
                        <i className="v2-taka-at" style={{ left: `${x(at.i)}%`, top: `${y(at.p.value)}%` }} />
                        <span className="v2-taka-read" style={{ '--x': `${x(at.i)}%` }}>{longDate(at.p.date)} · {at.p.value.toFixed(2)} · $1k = {taka(at.p.value * 1000)}</span>
                    </>
                )}
            </div>
            <div className="v2-taka-foot">
                <span>$1k = <b>{taka(cur * 1000)}</b> <em className={`tone-${up ? 'good' : 'bad'}`}>{up ? '+' : '−'}{taka(Math.abs(cur - start) * 1000)} {id === 'All' ? `since ${view[0].date.slice(0, 4)}` : `vs ${id} ago`}</em></span>
                <span className="v2-taka-next">Next {next} <i style={{ '--p': `${((cur - (next - step)) / step) * 100}%` }} /> {((next / cur - 1) * 100).toFixed(1)}% to go</span>
            </div>
        </div>
    );
}

/** Daily markets ranked by their last move against their own normal day (σ, as What moved); slow prints and once-a-day FX after, quiet. */
function Markets({ extra, loading }) {
    if (loading) return <div className="card"><Skeleton count={7} /></div>;
    if (!extra || extra.error) return null;
    const rows = MARKETS.map((r) => {
        const d = r.pick(extra);
        if (!d) return null;
        const win = marketWindow(d, { rate: r.rate, ticker: r.t });
        const sigma = win && !r.home ? sigmaOf((d.history || []).map((p) => p?.price ?? p?.value), r.rate ? 'bp' : 'pct') : null;
        // z from the shown change, so the bar always agrees with the number beside it
        return { ...r, d, win, z: sigma ? Math.abs(parseFloat(win.text.replace('−', '-'))) / sigma : null };
    }).filter(Boolean);
    const movers = rows.filter((r) => r.z != null).sort((a, b) => b.z - a.z);
    const home = rows.filter((r) => r.home);
    const rates = rows.filter((r) => !r.home && r.z == null);
    return (
        <div className="card">
            <div className="card-header" />
            <div className="v2-mkt">
                <div>
                    <div className="v2-mkt-head"><span>Biggest move first</span><span>× a normal day</span></div>
                    <div className="v2-mk-list">{movers.map((r) => <MarketRow key={r.t} r={r} />)}</div>
                </div>
                <div className="v2-mkt-side">
                    {rates.length > 0 && <div><div className="v2-mkt-head">Daily rates</div><div className="v2-mk-list">{rates.map((r) => <MarketRow key={r.t} r={r} />)}</div></div>}
                    {home.length > 0 && <div className="v2-mkt-home"><div className="v2-mkt-head">Home</div><div className="v2-mk-list">{home.map((r) => <MarketRow key={r.t} r={r} />)}</div></div>}
                </div>
            </div>
        </div>
    );
}

function HeroValue({ markKey, raw, stale, format, className, children }) {
    const mark = useMark(markKey, raw);
    return <Delta mark={stale ? null : mark} format={format} chartKey={markKey} raw={raw} className={className}>{children}</Delta>;
}

/** Section: plain title on a hairline, then its content. `id` is the sticky nav's anchor. */
function Section({ id, title, note, children }) {
    return (
        <section id={id} className="v2-section" aria-labelledby={`${id}-h`}>
            <header className="v2-section-head">
                <h2 id={`${id}-h`}>{title}</h2>
                {note && <span className="v2-section-note">{note}</span>}
            </header>
            {children}
        </section>
    );
}

/**
 * A reused v1 card in a v2 slot: v2 hides the card's own emoji title (CSS) and draws
 * its own, so every panel reads the same. `jump` keeps v1's data-jump label (chips jump here).
 * `more`: the card opens short (CSS hides the detail under .is-calm); "More" opens it, remembered per card.
 */
/** Open/closed for a "More" card, remembered per card on this device. */
function useOpen(key) {
    const [open, setOpen] = useState(false);
    useEffect(() => { try { setOpen(localStorage.getItem(`v2-open:${key}`) === '1'); } catch { /* private mode */ } }, [key]);
    const toggle = () => {
        try { localStorage.setItem(`v2-open:${key}`, open ? '0' : '1'); } catch { /* private mode */ }
        setOpen(!open);
    };
    return [open, toggle];
}

function MoreButton({ open, onClick, className = 'v2-more' }) {
    return (
        <button type="button" className={className} onClick={onClick} aria-expanded={open}>
            {open ? 'Less' : 'More'}
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </button>
    );
}

/**
 * SPY's 1-year IV rank and percentile as one picture: the 252 VIX closes they use, sorted calmest → wildest.
 * Today's dot sits at their coordinates: across = %ile (share of days at or below it), up = Rank (year low → high).
 * The hollow dot is the last close; pointing anywhere on the curve reads that level.
 */
function ivYearGeo(year, t) {
    const vals = (year || []).filter(Number.isFinite);
    const days = vals.slice(-252);
    const sorted = [...days].sort((a, b) => a - b);
    const n = sorted.length;
    if (n < 20 || !(sorted[n - 1] > sorted[0]) || !Number.isFinite(t.iv)) return null;
    const y = (v) => 100 - ivRank(days, v);
    const steps = (to) => sorted.slice(0, to).map((v, i) => `L${(i / n) * 100} ${y(v)}H${((i + 1) / n) * 100}`).join('');
    const k = sorted.filter((v) => v <= t.iv).length;
    // The last close against the 252 days up to it (live: today's window; after the close: one day back).
    const prev = t.live ? vals[vals.length - 1] : vals[vals.length - 2];
    const prevDays = t.live ? days : vals.slice(-253, -1);
    const was = Number.isFinite(prev) && prevDays.length === n
        ? { rank: ivRank(prevDays, prev), pct: ivPercentile(prevDays, prev), x: ivPercentile(days, prev), y: ivRank(days, prev) } : null;
    return { days, sorted, n, area: `M0 100${steps(n)}V100Z`, lit: `M0 100${steps(k)}V100Z`, edge: `M0 ${y(sorted[0])}${steps(n)}`, litEdge: `M0 ${y(sorted[0])}${steps(k)}`, was };
}

function IvYear({ vol, waiting }) {
    const t = vol?.tickers?.find((x) => x.ticker === 'SPY' && Number.isFinite(x.ivRank1y) && Number.isFinite(x.ivPctile1y));
    const geo = useMemo(() => (t ? ivYearGeo(vol.vixYear, t) : null), [vol, t]);
    const [at, setAt] = useState(null);
    if (!t) return waiting ? <Skeleton count={3} /> : <div className="v2-fred-na">Unavailable · source busy, try again shortly</div>;
    const delta = (now, was) => {
        const d = Math.round(now) - Math.round(was);
        return d ? <small>{d > 0 ? '▲' : '▼'}{Math.abs(d)}</small> : null;
    };
    const pill = (cls, name, v, was) => (
        <span className={`v2-ivy-pill ${cls}`}>{name} <b className={`tone-${ivTone(v)}`}>{Math.round(v)}</b>{was != null && delta(v, was)}</span>
    );
    const rank = pill('is-rank', 'Rank', t.ivRank1y, geo?.was?.rank);
    const pct = pill('is-pct', '%ile', t.ivPctile1y, geo?.was?.pct);
    if (!geo) return <div className="v2-ivy-plain">{rank}{pct}</div>;
    const read = (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const v = geo.sorted[Math.min(geo.n - 1, Math.floor(Math.max(0, (e.clientX - r.left) / r.width) * geo.n))];
        setAt({ v, x: ivPercentile(geo.days, v), y: ivRank(geo.days, v) });
    };
    return (
        <div className="v2-ivy" style={{ '--x': `${t.ivPctile1y}%`, '--y': `${t.ivRank1y}%` }}>
            <div className={`v2-ivy-plot tone-${ivTone(t.ivPctile1y)}`} onPointerMove={read} onPointerDown={read} onPointerLeave={() => setAt(null)}>
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                    <path className="v2-ivy-all" d={geo.area} />
                    <path className="v2-ivy-lit" d={geo.lit} />
                    <path className="v2-ivy-edge" d={geo.edge} />
                    <path className="v2-ivy-edge is-lit" d={geo.litEdge} />
                </svg>
                <i className="v2-ivy-h" />
                <i className="v2-ivy-v" />
                {geo.was && <i className="v2-ivy-was" style={{ left: `${geo.was.x}%`, bottom: `${geo.was.y}%` }} />}
                <i className="v2-ivy-dot" />
                {at && (
                    <>
                        <i className="v2-ivy-at" style={{ left: `${at.x}%`, bottom: `${at.y}%` }} />
                        <span className="v2-ivy-read" style={{ '--x': `${at.x}%` }}>VIX {at.v.toFixed(1)} · Rank {Math.round(at.y)} · %ile {Math.round(at.x)}</span>
                    </>
                )}
                {rank}
            </div>
            <div className="v2-ivy-axis">{pct}</div>
        </div>
    );
}

function Slot({ jump, title, tip, span = 12, saved, more, children }) {
    const [open, toggle] = useOpen(jump);
    const mode = more ? (open ? ' has-more is-open' : ' has-more is-calm') : '';
    return (
        <div className={`v2-slot span-${span}${mode}`} data-jump={jump} data-cached={saved}>
            <div className="v2-slot-title">{tip ? <span className="tooltip-trigger" data-tooltip={tip}>{title}</span> : title}</div>
            {children}
            {more && <MoreButton open={open} onClick={toggle} />}
        </div>
    );
}

const monYear = (iso) => new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });

/**
 * What stands out about today against the WHOLE history (not the chart's range): a badge, lit
 * when it matters, and a ruler from the low to the high since the first year (tick = the
 * long-run average, or 0 for the yield curve; dot = today).
 */
function fredStory(metric, hist) {
    if (!Array.isArray(hist) || hist.length < 24) return null;
    const v = hist.map((p) => p.value);
    const cur = v[v.length - 1], last = hist[hist.length - 1].date, from = hist[0].date.slice(0, 4);
    const lo = Math.min(...v), hi = Math.max(...v), avg = v.reduce((a, b) => a + b, 0) / v.length;
    const share = v.filter((x) => x < cur).length / v.length;
    const short = metric === 'spEps' ? (x) => `$${Math.round(x)}` : (x) => `${x < 0 ? '−' : ''}${Math.abs(x).toFixed(1)}`;
    const mark = metric === 'yieldCurve' ? 0 : avg;
    const ruler = { lo, hi, cur, mark, short, tip: `Since ${from}: low ${short(lo)}, high ${short(hi)}, ${metric === 'yieldCurve' ? 'tick at 0' : `avg ${short(avg)} (tick)`}` };
    let text, sub, tone = 'quiet';
    if (metric === 'yieldCurve') {
        let i = v.length - 1;
        while (i > 0 && (v[i - 1] < 0) === (cur < 0)) i--;
        const months = (Date.parse(last) - Date.parse(hist[i].date)) / 2.63e9;
        text = cur < 0 ? 'Inverted' : 'Positive';
        sub = ` since ${monYear(hist[i].date)}`;
        tone = cur < 0 ? 'bad' : months < 12 ? 'caution' : 'good'; // a fresh un-inversion is the classic late-cycle tell
    } else if (metric === 'spEps') {
        const ago = hist.findLast((p) => p.date <= plusMonths(last, -12));
        const yoy = ago ? (cur / ago.value - 1) * 100 : null;
        sub = yoy == null ? '' : ` · ${yoy >= 0 ? '+' : '−'}${Math.abs(yoy).toFixed(0)}% y/y`;
        text = cur >= hi ? 'Record high' : `${((1 - cur / hi) * 100).toFixed(0)}% off high`;
        tone = cur >= hi ? 'good' : yoy < 0 ? 'bad' : 'quiet';
    } else {
        sub = ` since ${from}`;
        if (cur >= hi) { text = 'Record high'; tone = 'good'; }
        else if (share >= 0.9) { text = `Top ${Math.max(1, Math.round((1 - share) * 100))}%`; tone = 'good'; }
        else if (share <= 0.1) { text = `Bottom ${Math.max(1, Math.round(share * 100))}%`; tone = 'bad'; }
        else text = `%ile ${Math.round(share * 100)}`;
    }
    return { text, sub, tone, lit: tone !== 'quiet' && !(metric === 'yieldCurve' && tone === 'good'), ruler };
}

function HistRuler({ lo, hi, cur, mark, short, tip }) {
    const at = (x) => `${((x - lo) / (hi - lo || 1)) * 100}%`;
    return (
        <span className="v2-hr" title={tip}>
            <em>{short(lo)}</em>
            <span className="v2-hr-rail" style={{ '--at': at(cur) }}><b style={{ left: at(mark) }} /><i /></span>
            <em>{short(hi)}</em>
        </span>
    );
}

/** Yield curve / profit margin / EPS: one number, one long chart, and what stands out about today. */
function FredTile({ fred, metric, title, jump, tip, markKey, fmt, tone, chart, refreshTick }) {
    const m = fred?.[metric];
    const story = useMemo(() => fredStory(metric, m?.history), [metric, m?.history]);
    return (
        <div className={`card v2-fred${story ? ` lit-${story.tone}` : ''}${story?.lit ? ' is-lit' : ''}`} data-jump={jump}>
            <div className="v2-card-head">
                <span className="tooltip-trigger" data-tooltip={`${tip}${freshnessNote({ value: m?.current, asOf: m?.asOf, stale: m?.stale }).suffix}`}>{title}</span>
                {story && <span className="v2-fred-badge"><i />{story.text}<small>{story.sub}</small></span>}
            </div>
            <ErrorBoundary resetKey={refreshTick}>
                <Body>{() => !fred ? <Skeleton count={2} /> : (fred.error || m?.current == null) ? (
                    <div className="v2-fred-na">Unavailable · source busy, try again shortly</div>
                ) : (
                    <>
                        <div className="v2-fred-value">
                            <HeroValue markKey={markKey} raw={m.current} stale={m.stale} format={fmt}
                                className={`v2-num-lg tone-${m.stale ? 'caution' : tone(m.current)}`}>
                                {m.stale ? '🕐 ' : ''}{fmt(m.current)}
                            </HeroValue>
                            {m.stale && <span className="v2-muted">Last data {formatAsOf(m.asOf)}</span>}
                            {story && <HistRuler {...story.ruler} />}
                        </div>
                        <MiniChart history={m.history} recessions={fred.recessions || []} {...chart} />
                    </>
                )}</Body>
            </ErrorBoundary>
        </div>
    );
}

// ---------- signals: each card also draws how close it is to firing ----------
const NSB_LINE = 8.5;     // NotSoBoring trips past an 8.5% TMF drop from its 10-day high (lib/signals.js)
const AAII_LINE = 20;     // bears beat bulls by 20+ pts ...
const AAII_MONTHS = 6;    // ... and the 6 months after have tended to be good ones
const shortDate = (iso) => new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
const plusMonths = (iso, n) => { const d = new Date(`${iso}T12:00:00Z`); d.setUTCMonth(d.getUTCMonth() + n); return d.toISOString().slice(0, 10); };
const FR_MODE = (v) => (/^BIL/.test(v) ? 'Cash' : /VIX/.test(v) ? 'VIX hedge' : v === 'LABD' ? 'Biotech short' : 'Dip buy');

/** NotSoBoring: TMF's drop from its 10-day high, filling toward the 8.5% line. */
function DropMeter({ pct }) {
    const max = Math.max(10, pct + 1);
    const tone = pct > NSB_LINE ? 'bad' : pct >= NSB_LINE * 0.9 ? 'caution' : 'quiet';
    return (
        <span className={`v2-meter tone-${tone}`} style={{ '--f': `${(pct / max) * 100}%`, '--l': `${(NSB_LINE / max) * 100}%` }}
            title={`TMF ${pct.toFixed(2)}% under its 10-day high · trips past ${NSB_LINE}%`}>
            <span className="v2-meter-tag">{pct.toFixed(1)} / {NSB_LINE}%</span>
            <span className="v2-meter-bar"><i /></span>
        </span>
    );
}

/** FrontRunner: the 15 RSIs it watches. Top row hedges past its amber zone, bottom row buys dips under its green one. */
function RsiStrip({ rsi, pick }) {
    const row = (map, side) => Object.entries(map).filter(([t]) => Number.isFinite(rsi?.[t])).map(([t, n]) => {
        const v = rsi[t];
        const hit = side === 'over' ? v > n : v < n;
        return <i key={t} className={`${hit ? 'is-hit' : ''}${t === pick ? ' is-pick' : ''}`} style={{ left: `${v}%` }}
            title={`${t} RSI ${v.toFixed(1)} · ${side === 'over' ? 'hedges over' : 'buys under'} ${n}`} />;
    });
    const over = row(FR_OVER, 'over'), under = row(FR_UNDER, 'under');
    if (over.length + under.length < 8) return null;
    return (
        <span className="v2-rsi">
            <span className="v2-rsi-row is-over"><b style={{ left: `${Math.min(...Object.values(FR_OVER))}%` }} />{over}</span>
            <span className="v2-rsi-row is-under"><b style={{ width: `${Math.max(...Object.values(FR_UNDER))}%` }} />{under}</span>
        </span>
    );
}

/** A signal's path since spring against its trigger line; the stretch past the line is lit. */
function TriggerLine({ points, line, tag, tone }) {
    const id = useId();
    const geo = useMemo(() => {
        const vals = (points || []).map((p) => p.value).filter(Number.isFinite);
        if (vals.length < 6) return null;
        let lo = Math.min(...vals, line), hi = Math.max(...vals, line);
        const pad = (hi - lo) * 0.14 || 1;
        lo -= pad; hi += pad;
        const y = (v) => 36 - ((v - lo) / (hi - lo)) * 36;
        const d = vals.map((v, i) => `${i ? 'L' : 'M'}${((i / (vals.length - 1)) * 120).toFixed(1)},${y(v).toFixed(1)}`).join('');
        return { d, ly: y(line), ey: y(vals[vals.length - 1]) };
    }, [points, line]);
    if (!geo) return null;
    return (
        <span className={`v2-trig tone-${tone}`} style={{ '--ly': `${(geo.ly / 36) * 100}%`, '--ey': `${(geo.ey / 36) * 100}%` }}>
            <svg viewBox="0 0 120 36" preserveAspectRatio="none" aria-hidden="true">
                <clipPath id={id}><rect x="0" y="-4" width="120" height={geo.ly + 4} /></clipPath>
                <rect className="v2-trig-zone" x="0" y="0" width="120" height={geo.ly} />
                <line className="v2-trig-line" x1="0" x2="120" y1={geo.ly} y2={geo.ly} />
                <path className="v2-trig-path" d={geo.d} />
                <path className="v2-trig-path is-lit" d={geo.d} clipPath={`url(#${id})`} />
            </svg>
            <span className="v2-trig-tag">{tag}</span>
            <i className="v2-trig-dot" />
        </span>
    );
}

function Signals({ sheets, spy, loading }) {
    const na = loading ? '…' : 'N/A';
    const [study, setStudy] = useState(false);
    const aaii = parseFloat(sheets?.AAIIDiff);
    const vixCur = parseFloat(sheets?.VIX?.current);
    const vix3m = parseFloat(sheets?.VIX?.threeMonth);
    const aaiiMark = useMark('aaiiDiff', aaii);
    const aaiiHist = useChart('aaiiDiff', aaii);
    const vixHist = useChart('vixCurrent', vixCur);
    const v3Hist = useChart('vix3m', vix3m);
    const nsb = sheets?._meta?.fields?.NotSoBoring?.detail;
    const frRsi = sheets?._meta?.fields?.FrontRunner?.detail;
    const fr = String(sheets?.FrontRunner || '');

    const aaiiPts = useMemo(() => (Number.isFinite(aaii) ? [...(aaiiHist?.points || []), { value: aaii }] : []), [aaiiHist, aaii]);
    // The latest 20+ reading (daily rows repeat each weekly survey): it fired on the run's first day, and speaks for 6 months after its last survey.
    const fired = useMemo(() => {
        const p = aaiiHist?.points || [];
        let j = p.length - 1;
        while (j >= 0 && !(p[j].value >= AAII_LINE)) j--;
        if (j < 0) return null;
        let i = j;
        while (i > 0 && p[i - 1].value >= AAII_LINE) i--;
        while (j > i && p[j - 1].value === p[j].value) j--;
        return { on: p[i].date, until: plusMonths(p[j].date, AAII_MONTHS) };
    }, [aaiiHist]);
    const today = new Date().toISOString().slice(0, 10);
    const bullUntil = Number.isFinite(aaii) && aaii >= AAII_LINE ? plusMonths(today, AAII_MONTHS) : fired && fired.until >= today ? fired.until : null;

    const ratioPts = useMemo(() => {
        const v3 = new Map((v3Hist?.points || []).map((p) => [p.date, p.value]));
        const pts = (vixHist?.points || []).filter((p) => v3.get(p.date) > 0).map((p) => ({ date: p.date, value: p.value / v3.get(p.date) }));
        return Number.isFinite(vixCur) && vix3m > 0 ? [...pts, { value: vixCur / vix3m }] : pts;
    }, [vixHist, v3Hist, vixCur, vix3m]);

    const tiles = [
        {
            key: 'nsb', label: 'NotSoBoring', tip: `Goes defensive when TMF drops ${NSB_LINE}%+ from its 10-day high.`,
            alert: !loading && sheets?.NotSoBoring && sheets.NotSoBoring !== 'ON',
            value: sheets?.NotSoBoring || na,
            sub: staleNote(sheets, ['NotSoBoring'])
                || (Number.isFinite(nsb?.watchOutAt) ? <span className="v2-muted">Watch out at ${nsb.watchOutAt.toFixed(2)}</span> : null),
            viz: Number.isFinite(nsb?.drawdownPct) ? <DropMeter pct={nsb.drawdownPct} /> : null,
        },
        {
            key: 'fr', label: 'FrontRunner', tip: 'Hedges with VIX when RSIs run hot; buys deep tech dips. Tap for the NUTS radar (a different algo).',
            href: 'https://nuts-radar.vercel.app',
            alert: !loading && fr && !fr.startsWith('BIL'),
            value: fr || na,
            sub: staleNote(sheets, ['FrontRunner']) || <span className="v2-muted">{fr ? `${FR_MODE(fr)} · ` : ''}NUTS radar ↗</span>,
            viz: frRsi ? <RsiStrip rsi={frRsi} pick={fr} /> : null,
        },
        {
            key: 'aaii', label: 'AAII', tip: `Bears minus bulls. ${AAII_LINE}%+ has meant a good next 6 months. Tap for the proof.`,
            alert: !!bullUntil, good: true, open: Number.isFinite(aaii) ? () => setStudy('aaii') : null,
            value: Number.isFinite(aaii) ? (
                <Delta mark={aaiiMark} chartKey="aaiiDiff" raw={aaii} format={(v) => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`}>
                    <span className={aaii >= AAII_LINE ? 'tone-good' : ''}>{sheets.AAIIDiff}</span>
                </Delta>
            ) : na,
            sub: !Number.isFinite(aaii) ? null : bullUntil
                ? <span className="tone-good" title={fired ? `Fired ${shortDate(fired.on)}` : undefined}>Bullish to {shortDate(bullUntil)}</span>
                : <span className="v2-muted">{(AAII_LINE - aaii).toFixed(1)} pts to {AAII_LINE}%</span>,
            viz: aaiiPts.length > 6 && <TriggerLine points={aaiiPts} line={AAII_LINE} tag={`${AAII_LINE}%`} tone="good" />,
        },
        {
            key: 'vix', label: 'VIX / 3M', tip: 'VIX above 3-month VIX: panic. Often before a rebound. Tap for the proof.',
            alert: Number.isFinite(vixCur) && vixCur > vix3m, open: Number.isFinite(vixCur) && vix3m > 0 ? () => setStudy('vix') : null,
            value: Number.isFinite(vixCur) ? (
                <span className="v2-pair">
                    <Delta chartKey="vixCurrent" raw={vixCur}>{sheets.VIX.current}</Delta>
                    <span className="v2-pair-sep">/</span>
                    <Delta chartKey="vix3m" raw={vix3m}>{sheets.VIX.threeMonth}</Delta>
                </span>
            ) : na,
            sub: staleNote(sheets, ['vixCurrent', 'vixThreeMonth', 'vixFearGreed'])
                || (Number.isFinite(vixCur) && vix3m > 0
                    ? <span className="v2-muted">{(vixCur / vix3m).toFixed(2)}{sheets?.VIX?.fearGreed ? ` · ${sheets.VIX.fearGreed}` : ''}</span> : null),
            viz: ratioPts.length > 6 && <TriggerLine points={ratioPts} line={1} tag="1.0" tone="bad" />,
        },
    ];
    return (
        <div className="v2-signals">
            {tiles.map((t) => {
                const inner = (
                    <>
                        <span className="v2-signal-label"><span className="tooltip-trigger" data-tooltip={t.tip}>{t.label}</span></span>
                        <span className="v2-signal-value">{t.value}</span>
                        <span className="v2-signal-sub">{t.sub}</span>
                        {t.viz && <span className="v2-signal-viz">{t.viz}</span>}
                    </>
                );
                const cls = `v2-signal${t.alert ? ` is-alert${t.good ? ' is-good' : ''}` : ''}`;
                if (t.href) return <a key={t.key} className={`${cls} is-link`} href={t.href} target="_blank" rel="noopener noreferrer">{inner}</a>;
                // a tap on the number still opens its own chart (Delta is a role=button inside)
                return t.open ? (
                    <div key={t.key} className={`${cls} is-link`} role="button" tabIndex={0} aria-haspopup="dialog"
                        onClick={(e) => { if (e.target.closest('[role="button"]') === e.currentTarget) t.open(); }}
                        onKeyDown={(e) => { if (e.target === e.currentTarget && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); t.open(); } }}>{inner}</div>
                ) : <div key={t.key} className={cls}>{inner}</div>;
            })}
            {study === 'aaii' && (
                <SignalStudy id="aaii" src="/history/aaiiDiff.json" line={AAII_LINE} horizons={[3, 6, 12]} start={6} title={`AAII fear at ${AAII_LINE}+`}
                    sub={`Now ${aaii.toFixed(1)} · ${bullUntil ? `bullish to ${shortDate(bullUntil)}` : `${(AAII_LINE - aaii).toFixed(1)} pts to go`}`}
                    label={(v) => `AAII ${v.toFixed(1)}`} foot={`first ${AAII_LINE}+ week after 3 quiet months`}
                    now={aaii} sheet={aaiiHist?.points} spyLive={spy} onClose={() => setStudy(false)} />
            )}
            {study === 'vix' && (
                <SignalStudy id="vix" src="/history/vix-ratio-daily.json" line={1} horizons={[1, 3, 6, 12]} start={1} title="VIX over VIX 3M"
                    sub={`Now ${sheets.VIX.current} / ${sheets.VIX.threeMonth} · ${(vixCur / vix3m).toFixed(2)}`}
                    label={(v) => `VIX/3M ${v.toFixed(3)}`} foot="VIX > VIX 3M after 3 quiet months"
                    now={vixCur / vix3m} sheet={ratioPts.filter((p) => p.date)} spyLive={spy} onClose={() => setStudy(false)} />
            )}
        </div>
    );
}

const RG_TONE = ['good', 'caution', 'bad'];
const RG_TREND = { hardening: ['Hardening ↑', 'bad'], softening: ['Softening ↓', 'good'], mixed: ['Mixed', 'caution'] };

/** The five Jev regime calls in one row. Each has a best → worst scale: today's step lit, yesterday's outlined when it moved. Tap → the full read. */
function RegimeRow({ data }) {
    const [open, setOpen] = useState(null);
    const since = data?.since;
    const was = Object.fromEntries((since?.changed || []).map((c) => [c.pill, SEVERITY[c.from]?.order]));
    const trend = since?.changed?.length && !since.noBaseline ? RG_TREND[since.direction] || RG_TREND.mixed : null;
    const label = Object.fromEntries(PILL_ENTRIES.map((e) => [e.key, e.label]));
    return (
        <div className="card v2-rg-row">
            {PILL_ENTRIES.map(({ key }) => {
                const p = data?.pills?.[key];
                if (!p) return null;
                const at = SEVERITY[p.verdict]?.order ?? -1;
                return (
                    <button key={key} type="button" className={`v2-rg tone-${RG_TONE[at] || 'quiet'}`} onClick={() => setOpen(key)} title={pillWhy(data, key) || p.reason || undefined}>
                        <span className="v2-rg-k">{label[key]}</span>
                        <span className="v2-rg-v">{FRIENDLY[p.verdict] || p.verdict}</span>
                        <span className="v2-rg-scale" aria-hidden="true">
                            {RG_TONE.map((_, i) => <i key={i} className={i === at ? 'is-on' : i === was[key] ? 'is-was' : undefined} />)}
                        </span>
                    </button>
                );
            })}
            {trend && (
                <span className={`v2-rg-trend tone-${trend[1]}`}
                    title={since.changed.map((c) => `${label[c.pill]}: ${FRIENDLY[c.from] || c.from} → ${FRIENDLY[c.to] || c.to}`).join(' · ')}>
                    {trend[0]}
                </span>
            )}
            {open && <JevPillModal pillKey={open} data={data} onClose={() => setOpen(null)} />}
        </div>
    );
}

/** The five verdicts v1 shows as Market Pulse chips, as a readable row: verdict + why. */
function Readout({ fred, vol, rubberBand, saved, waiting }) {
    const chips = useMemo(() => pulseVerdicts({ fred, vol, rubberBand, saved }), [fred, vol, rubberBand, saved?.fred, saved?.vol]); // eslint-disable-line react-hooks/exhaustive-deps
    if (!chips.length) return waiting ? <div className="v2-readout is-waiting"><Skeleton count={1} /></div> : null;
    return (
        <div className="v2-readout" aria-label="The read: verdicts from the cards below">
            {chips.map((c) => (
                <button key={c.key} type="button" className={`v2-read tone-${c.tone}`} onClick={() => jumpToCard(c.jump)}
                    title={`${c.why}${c.old ? ` · ${c.old.note}` : ''} · tap for the card`}>
                    <span className="v2-read-text">{c.old?.kind === 'stale' ? '🕐 ' : ''}{c.text.replace(/\s*✓$/, '')}</span>
                    <ReadGlyph k={c.key} fred={fred} vol={vol} rubberBand={rubberBand} />
                </button>
            ))}
        </div>
    );
}

const RB_TONE = { green: 'good', amber: 'caution', red: 'bad' };
/** A w×h line through `vals`; `also` widens the scale (e.g. 0, so a zero line always fits). */
const spark = (vals, w, h, also = []) => {
    const all = [...vals, ...also];
    const lo = Math.min(...all), span = Math.max(...all) - lo || 1;
    const y = (v) => +(h - 2 - ((v - lo) / span) * (h - 4)).toFixed(1);
    const x = (i) => +((i / (vals.length - 1)) * w).toFixed(1);
    return { x, y, d: vals.map((v, i) => `${i ? 'L' : 'M'}${x(i)} ${y(v)}`).join('') };
};

/** Each verdict's own little picture: the VIX curve, the 4 horsemen, the 10Y−2Y line, the 8 bull tests, the 5 tripwires. */
function ReadGlyph({ k, fred, vol, rubberBand }) {
    const yc = useChart('yieldCurve', k === 'curve' ? fred?.yieldCurve?.current : null);
    if (k === 'vol') {
        const v = (vol?.regime?.curve?.points || []).map((p) => p.value).filter(Number.isFinite);
        if (v.length < 2) return null;
        const s = spark(v, 40, 14);
        return <svg className="v2-glyph" width="40" height="14" viewBox="0 0 40 14" aria-hidden="true"><path d={s.d} />{v.map((x, i) => <circle key={i} cx={s.x(i)} cy={s.y(x)} r="1.6" />)}</svg>;
    }
    if (k === 'curve') {
        const v = (yc?.points || []).slice(-90).map((p) => p.value).filter(Number.isFinite);
        if (v.length < 2) return null;
        const s = spark(v, 44, 14, [0]);
        return <svg className="v2-glyph" width="44" height="14" viewBox="0 0 44 14" aria-hidden="true"><line className="is-zero" x1="0" x2="44" y1={s.y(0)} y2={s.y(0)} /><path d={s.d} /><circle cx="44" cy={s.y(v[v.length - 1])} r="2" /></svg>;
    }
    let pips = null;
    if (k === 'horsemen') {
        const r = horsemenRiding(fred);
        if (r) pips = [0, 1, 2, 3].map((i) => (i < r.riding ? 'bad' : i < r.known ? 'off' : 'na'));
    } else if (k === 'bull') {
        pips = Object.values(fred?.checklist || {}).filter((i) => i && typeof i === 'object').map((i) => (i.bullish ? 'good' : 'off'));
    } else if (k === 'dips') {
        pips = Object.values(rubberBand?.dials || {}).map((d) => RB_TONE[d?.colour] || 'na');
    }
    if (!pips?.length) return null;
    return <span className={`v2-pips is-${k}`} aria-hidden="true">{pips.map((p, i) => <i key={i} className={`is-${p}`} />)}</span>;
}

/** Sticky section tabs; the one in view is lit (IntersectionObserver, no scroll listener). */
function SectionNav({ glance }) {
    const [active, setActive] = useState('now');
    useEffect(() => {
        if (typeof IntersectionObserver === 'undefined') return undefined;
        const els = SECTIONS.map(([id]) => document.getElementById(id)).filter(Boolean);
        const io = new IntersectionObserver((entries) => {
            const hit = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
            if (hit) setActive(hit.target.id);
        }, { rootMargin: '-120px 0px -60% 0px' });
        els.forEach((el) => io.observe(el));
        return () => io.disconnect();
    }, []);
    const go = (id) => {
        const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
        document.getElementById(id)?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
    };
    return (
        <nav className="v2-tabs" aria-label="Sections">
            <div className="v2-tabs-row">
                {SECTIONS.map(([id, label]) => (
                    <button key={id} type="button" className={`v2-tab${active === id ? ' is-active' : ''}`} onClick={() => go(id)}>{label}</button>
                ))}
            </div>
            {glance}
        </nav>
    );
}

export default function DashboardV2() {
    const f = useFeeds();
    const { spy, spyDailyMove, fg, fred, extra, sheets, history, jev, vol } = f.data;
    const { pending, saved, refreshTick, refreshNow } = f;
    const [rubberBand, setRubberBand] = useState(undefined);
    const [trioOpen, toggleTrio] = useOpen('fred-trio');
    const heroRef = useRef(null);
    const [heroGone, setHeroGone] = useState(false);
    useEffect(() => {
        if (!heroRef.current || typeof IntersectionObserver === 'undefined') return undefined;
        const io = new IntersectionObserver(([e]) => setHeroGone(!e.isIntersecting && e.boundingClientRect.top < 0));
        io.observe(heroRef.current);
        return () => io.disconnect();
    }, []);

    const spyOk = isLiveAnswer('spy', spy);
    const fgOk = isLiveAnswer('fg', fg);
    const ivSpy = vol?.tickers?.find((t) => t.ticker === 'SPY' && Number.isFinite(t.ivRank1y) && Number.isFinite(t.ivPctile1y));
    const moveOk = isLiveAnswer('spyDailyMove', spyDailyMove);
    const spyStale = spyOk ? staleOf(spy)?.label : undefined;
    const spyWhen = useMemo(() => { try { return movedWhen({ spy, vol }) || 'today'; } catch { return 'today'; } }, [spy, vol]);
    const dayPct = moveOk ? parseFloat(spyDailyMove.value) : spyOk && spy.dailyChange ? spy.dailyChange.pct : null;
    const dayText = moveOk ? spyDailyMove.value.replace(/^-/, '−') : dayPct != null ? `${signed(dayPct)}%` : null;
    const marks = collectLiveValues(fred, extra, sheets);

    const glance = heroGone && spyOk ? (
        <button type="button" className="v2-glance" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} aria-label="Back to top">
            <span>SPY <b>{usd(spy.current)}</b></span>
            {dayPct != null && <span className={dayPct >= 0 ? 'tone-good' : 'tone-bad'}>{dayText}</span>}
            {fgOk && <span>F&G <b className={`tone-${fgTone(fg.score)}`}>{Math.round(fg.score)}</b></span>}
        </button>
    ) : null;

    return (
        <MarkProvider history={f.marksOff ? null : history} series={history?.series || null}>
            <OfflineBanner onBack={refreshNow} since={f.anySaved} />
            <PullToRefresh onRefresh={refreshNow} busy={f.refreshing} />
            <div className="v2">
                <header className="v2-top">
                    <div className="v2-top-row">
                        <a className="v2-brand" href="/v2">Markets</a>
                        <div className="v2-top-clock"><ErrorBoundary><MarketClock /></ErrorBoundary></div>
                        <div className="v2-top-right">
                            <span className={`v2-updated${f.anySaved && (f.loading || !f.updatedAt) ? ' is-saved' : ''}`}>
                                {f.loading ? (f.anySaved ? `Saved ${f.anySaved} · loading` : 'Loading') : !f.updatedAt ? 'No live data yet' : <UpdatedAgo at={f.updatedAt} />}
                            </span>
                            <ErrorBoundary resetKey={refreshTick}><MarkChip values={marks} /></ErrorBoundary>
                            <button className="v2-refresh" onClick={refreshNow} disabled={f.refreshing} title="Refresh (R)" aria-label="Refresh all data">
                                <svg className={f.refreshing ? 'spinning' : ''} width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" />
                                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                                </svg>
                            </button>
                        </div>
                    </div>
                    <SectionNav glance={glance} />
                </header>

                <main className="v2-main">
                    {/* NOW — the price, the mood, the read */}
                    <section id="now" className="v2-now" aria-label="The market now">
                        <div className="v2-phone-clock"><ErrorBoundary><MarketClock /></ErrorBoundary></div>
                        <div className="v2-hero" ref={heroRef}>
                            <div className="v2-hero-spy" data-jump="SPY overview" data-cached={saved('spy', 'spyDailyMove')}>
                                <ErrorBoundary resetKey={refreshTick}>
                                    <Body>{() => !spyOk ? (pending.spy ? <Skeleton count={4} /> : <Unavailable what="SPY" onRetry={refreshNow} />) : (
                                        <>
                                            <div className="v2-hero-label">
                                                <span>S&P 500 · SPY</span>
                                                {spy.return3y != null && <span className="v2-hero-aside">3Y <b className={spy.return3y >= 0 ? 'tone-good' : 'tone-bad'}>{signed(spy.return3y, 1)}%</b></span>}
                                                {spyStale && <span className="v2-flag" data-testid="spy-provenance">Stale · {/ close$/.test(spyStale) ? spyStale : `cached ${spyStale}`}</span>}
                                            </div>
                                            <div className="v2-hero-price-row">
                                                <span className="v2-hero-price">{usd(spy.current)}</span>
                                                {dayPct != null && (
                                                    <span className={`v2-move ${dayPct >= 0 ? 'is-up' : 'is-down'}`}>
                                                        {dayPct >= 0 ? '▲' : '▼'} {dayText.replace(/^[+−-]/, '')} <span className="v2-move-when">{spyWhen}</span>
                                                    </span>
                                                )}
                                            </div>
                                            <Spark spy={spy} />
                                        </>
                                    )}</Body>
                                </ErrorBoundary>
                            </div>

                            <div className="v2-hero-side">
                                <div className="v2-gauge-block" data-jump="Fear & Greed" data-cached={saved('fg')}>
                                    <ErrorBoundary resetKey={refreshTick}>
                                        <Body>{() => !fgOk ? (pending.fg ? <Skeleton type="gauge" /> : <Unavailable what="F&G" onRetry={refreshNow} />) : (
                                            <>
                                                <div className="v2-hero-label">
                                                    <span>Fear & Greed</span>
                                                    {(fg._meta?.proxy || fg._meta?.stale) && (
                                                        <span className="v2-flag" data-testid="fg-provenance">{fg._meta?.proxy ? 'VIX proxy, not CNN' : `Stale · ${String(fg._meta?.lastGoodAt || '').slice(0, 10)}`}</span>
                                                    )}
                                                </div>
                                                <div className="v2-gauge-head">
                                                    <HeroValue markKey="cnnFearGreed" raw={fg._meta?.proxy ? null : fg.score} className={`v2-num-xl tone-${fgTone(fg.score)}`}>{Math.round(fg.score)}</HeroValue>
                                                    <span className={`v2-gauge-word tone-${fgTone(fg.score)}`}>{fg.rating}</span>
                                                </div>
                                                <FgTrail fg={fg} />
                                            </>
                                        )}</Body>
                                    </ErrorBoundary>
                                </div>
                                {spyOk && (
                                    <div className="v2-gauge-block is-rsi">
                                        <div className="v2-hero-label">
                                            <span className="tooltip-trigger" data-tooltip="9-day RSI of SPY. Under 30 oversold, over 70 overbought.">RSI 9-day</span>
                                            <span className={`v2-rsi-word tone-${spy.rsi > 70 ? 'bad' : spy.rsi < 30 ? 'good' : 'flat'}`}>{spy.rsi > 70 ? 'Overbought' : spy.rsi < 30 ? 'Oversold' : 'Neutral'}</span>
                                            <span className="v2-rsi-num">{spy.rsi.toFixed(1)}</span>
                                        </div>
                                        <Scale value={spy.rsi} zones={RSI_ZONES} ticks={[0, 30, 70, 100]} />
                                    </div>
                                )}
                                {ivSpy && (
                                    <button type="button" className="v2-gauge-block v2-iv" data-cached={saved('vol')} onClick={() => jumpToCard('Volatility')}
                                        title="SPY implied volatility (VIX) vs its past year · tap for the full table">
                                        <span>SPY IV · 1y</span>
                                        <span>Rank <b className={`tone-${ivTone(ivSpy.ivRank1y)}`}>{Math.round(ivSpy.ivRank1y)}</b></span>
                                        <span>%ile <b className={`tone-${ivTone(ivSpy.ivPctile1y)}`}>{Math.round(ivSpy.ivPctile1y)}</b></span>
                                    </button>
                                )}
                            </div>
                        </div>

                        <div className="saved-wrap" data-cached={saved('fred', 'vol')} style={{ display: 'contents', '--saved': saved('fred', 'vol') ? JSON.stringify(saved('fred', 'vol')) : undefined }}>
                            <ErrorBoundary resetKey={refreshTick}>
                                <Readout fred={fred} vol={vol} rubberBand={rubberBand} saved={{ fred: saved('fred'), vol: saved('vol') }} waiting={pending.fred || pending.vol} />
                            </ErrorBoundary>
                        </div>
                        <div className="v2-strips">
                            <ErrorBoundary resetKey={refreshTick}>
                                <Moves spy={spy} fg={fg} extra={extra} vol={vol} history={history}
                                    saved={saved('spy', 'fg', 'extra', 'vol', 'history')}
                                    waiting={['spy', 'fg', 'extra', 'vol', 'history'].some((k) => pending[k])} />
                            </ErrorBoundary>
                            <ErrorBoundary resetKey={refreshTick}><SinceLastVisit base={f.seenBase.rec} live={f.seenNow} /></ErrorBoundary>
                        </div>
                    </section>

                    <Section id="signals" title="Signals">
                        <div data-jump="Indicators" data-cached={saved('sheets')}>
                            <ErrorBoundary resetKey={refreshTick}><Signals sheets={sheets} spy={spyOk ? spy.chartHistory : null} loading={pending.sheets && !sheets} /></ErrorBoundary>
                        </div>
                        <div className="v2-grid">
                            <ErrorBoundary resetKey={refreshTick}>
                                {jev && <Slot title="Regime" span={12} saved={saved('jev')}><RegimeRow data={jev} /></Slot>}
                            </ErrorBoundary>
                        </div>
                    </Section>

                    <Section id="market" title="Market">
                        <div className="v2-grid">
                            <Slot jump="SPY chart" title="SPY" span={7} saved={saved('spy')}>
                                <div className="card">
                                    <div className="card-header" />
                                    <ErrorBoundary resetKey={refreshTick}>
                                        <Body>{() => !spyOk || !spy.chartHistory ? (pending.spy ? <Skeleton count={4} /> : <Unavailable what="SPY chart" onRetry={refreshNow} />) : (
                                            <SpyChart chartHistory={spy.chartHistory} recessions={fred?.recessions || []} current={spy.current} />
                                        )}</Body>
                                    </ErrorBoundary>
                                </div>
                            </Slot>
                            <Slot jump="Volatility" title="Volatility" tip="SPY implied vol (VIX): the last 252 days, calmest → wildest. Dot = today: across = %ile, up = Rank." span={5} more>
                                <div className="card v2-ivcard" data-cached={saved('vol')}>
                                    <div className="card-header" />
                                    <ErrorBoundary resetKey={refreshTick}><IvYear vol={vol} waiting={pending.vol} /></ErrorBoundary>
                                    <div className="v2-ivcard-full">
                                        <ErrorBoundary resetKey={refreshTick}><VolMetricsTable data={vol} saved={saved('vol')} waiting={pending.vol} /></ErrorBoundary>
                                    </div>
                                </div>
                            </Slot>
                            <div className="v2-slot span-12 is-bare">
                                <ErrorBoundary resetKey={refreshTick}><FactorRow refreshKey={refreshTick} bust={f.bustKey} race /></ErrorBoundary>
                            </div>
                        </div>
                    </Section>

                    <Section id="risk" title="Risk">
                        <div className="v2-grid">
                            <Slot jump="Rubber band" title="Rubber band" tip="Is dip-buying still paying? Nightly check of three tripwires." span={12} more>
                                <ErrorBoundary resetKey={refreshTick}><RubberBandRadar onVerdict={setRubberBand} /></ErrorBoundary>
                            </Slot>
                            <Slot jump="Recession watch" title="Recession watch" tip="Four recession tells: each one's 1-year move vs its usual move before past recessions." span={12} saved={saved('fred')} more>
                                <ErrorBoundary resetKey={refreshTick}><FourHorsemen fred={fred} loading={pending.fred && !fred} /></ErrorBoundary>
                            </Slot>
                        </div>
                    </Section>

                    <Section id="economy" title="Economy">
                        <div className="v2-grid">
                            <Slot jump="Bull checklist" title="Bull checklist" span={6} saved={saved('fred')} more>
                                <ErrorBoundary resetKey={refreshTick}><Board rows={bullRows(fred)} /></ErrorBoundary>
                            </Slot>
                            <Slot jump="Economy" title="Indicators" span={6} saved={saved('fred')} more>
                                <ErrorBoundary resetKey={refreshTick}><Board rows={indicatorRows(fred)} /></ErrorBoundary>
                            </Slot>
                            <div className={`v2-trio span-12 ${trioOpen ? 'is-open' : 'is-calm'}`} data-cached={saved('fred')}>
                                <FredTile fred={fred} metric="yieldCurve" title="Yield curve" jump="Yield curve" markKey="yieldCurve" refreshTick={refreshTick}
                                    tip="10-year minus 2-year. Below 0 = classic recession warning."
                                    fmt={(v) => `${signed(v)}%`} tone={(v) => (v >= 0 ? 'good' : 'bad')}
                                    chart={{ color: '#9fb4c7', gradientId: 'v2Yield', showZero: true }} />
                                <FredTile fred={fred} metric="profitMargin" title="Profit margin" jump="Profit margin" markKey="profitMargin" refreshTick={refreshTick}
                                    tip="Corporate profits ÷ GDP. High = pricing power."
                                    fmt={(v) => `${v.toFixed(2)}%`} tone={() => 'flat'}
                                    chart={{ color: '#9fb4c7', gradientId: 'v2Profit' }} />
                                <FredTile fred={fred} metric="spEps" title="S&P 500 EPS" jump="S&P 500 EPS" markKey={undefined} refreshTick={refreshTick}
                                    tip="S&P 500 earnings per share, last 12 months. History in today's dollars."
                                    fmt={(v) => `$${v.toFixed(2)}`} tone={() => 'flat'}
                                    chart={{ color: '#9fb4c7', gradientId: 'v2Eps', cadence: 'monthly' }} />
                                <MoreButton open={trioOpen} onClick={toggleTrio} className="v2-trio-more" />
                            </div>
                        </div>
                    </Section>

                    <Section id="world" title="World">
                        <div className="v2-grid">
                            <Slot jump="USD/BDT" title="USD/BDT" span={12} saved={saved('extra')}
                                tip="Taka per dollar. Up = each dollar buys more taka. IMF official month-ends from 1972, Wise daily from Oct 2021, then the history sheet + today's live rate.">
                                <ErrorBoundary resetKey={refreshTick}><Taka extra={extra} loading={pending.extra && !extra} /></ErrorBoundary>
                            </Slot>
                            <Slot jump="Markets" title="Global markets" span={12} saved={saved('extra')} more>
                                <ErrorBoundary resetKey={refreshTick}><Markets extra={extra} loading={pending.extra && !extra} /></ErrorBoundary>
                            </Slot>
                        </div>
                    </Section>

                    <Section id="crowd" title="Crowd">
                        <div className="v2-grid">
                            <Slot jump="Polymarket" title="Polymarket" span={12} more>
                                <ErrorBoundary resetKey={refreshTick}><PolymarketTable refreshKey={refreshTick} bust={f.bustKey} /></ErrorBoundary>
                            </Slot>
                        </div>
                    </Section>
                </main>

                <footer className="v2-foot">
                    {f.systemStatus && (
                        <StatusLine bodies={f.systemStatus}
                            saved={{ spy: saved('spy'), fred: saved('fred'), fg: saved('fg'), sheets: saved('sheets'), extra: saved('extra') }} />
                    )}
                    {f.apiErrors.length > 0 && (
                        <div className="v2-errors" role="status">{f.apiErrors.map((e, i) => <div key={i}>{e}</div>)}</div>
                    )}
                    <div className="v2-foot-links">
                        <a href="/">Classic view</a>
                        <a href="https://docs.google.com/spreadsheets/d/1lA-_yjLMc3qDTt9sogSPQrCohNULIk5wwJYfb5wIHfc/edit?gid=0#gid=0" target="_blank" rel="noopener noreferrer">History sheet</a>
                    </div>
                </footer>

                <ErrorBoundary resetKey={refreshTick}><MarksSheet values={marks} /></ErrorBoundary>
                <ErrorBoundary resetKey={refreshTick}><BackPill /></ErrorBoundary>
            </div>
        </MarkProvider>
    );
}

// The same word → colour map v1 hands the economic indicator grid.
function statusTone(s) {
    return ['safe', 'healthy', 'strong', 'tight', 'easy', 'rising'].includes(s) ? 'good'
        : ['danger', 'weak', 'stressed', 'restrictive', 'falling'].includes(s) ? 'bad' : 'flat';
}
