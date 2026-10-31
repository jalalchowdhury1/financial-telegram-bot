'use client';
import { useEffect, useId, useMemo, useState } from 'react';
import Skeleton from './Skeleton';
import {
    COLOUR_WORD, BADGE, PLAIN, EXPLAIN, dayName, nextRun, signed, pct0, minus,
    contextOf, legColour, legUsed, tripFuse,
} from './RubberBandRadar';
import { kFmt } from './FourHorsemen';
import { latestYoY, preRecessionRunups, runupMedian, horsemanStatus, lastInversion } from '../lib/horsemenRunup';
import { horsemenRiding } from '../lib/pulseVerdicts';
import { formatAsOf } from '../lib/freshness';

/**
 * v2's Risk section, drawn natively. v1 keeps RubberBandRadar + FourHorsemen; the rules and the
 * numbers here are theirs (imported), only the picture is new. Closed: one answer per card and a
 * row of small pictures. `open` (the slot's More): the proof, each series against its own alarm
 * line, hover or drag to read any day.
 */

const TC = { green: 'var(--up)', amber: 'var(--warn)', red: 'var(--down)', blue: 'var(--accent)', grey: 'var(--ink-3)' };
const tc = (c) => ({ '--tc': TC[c] || TC.grey });
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const W = 1000, H = 100;
const when = (d) => new Date(`${d.length === 7 ? `${d}-15` : d}T12:00:00Z`).toLocaleDateString('en-US',
    d.length === 7 ? { month: 'short', year: 'numeric', timeZone: 'UTC' } : { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });

/**
 * One series (`pts` [{d, v, g?}], oldest first) against its alarm `line`: the side past it (`bad`
 * 'above' | 'below') is washed red and the line turns red there, grey bars are `bands`
 * (recessions), the dot is the latest point. `clip` trims the scale to the 1st–98.5th percentile
 * (claims' 2020 spike would flatten 50 years); the readout still says the real number.
 */
function Plot({ pts, ghost, line, bad, bands, clip, fmt, safe = 'var(--up)', tag, size = 'lane', mini }) {
    const gid = `rk${useId().replace(/[^a-zA-Z0-9]/g, '')}`;
    const [at, setAt] = useState(null);
    const g = useMemo(() => {
        if (!pts || pts.length < 12) return null;
        const vals = pts.map((p) => p.v).concat(ghost ? pts.map((p) => p.g).filter(Number.isFinite) : []);
        let lo = Math.min(line, ...vals), hi = Math.max(line, ...vals);
        if (clip) {
            const s = [...vals].sort((a, b) => a - b), q = (f) => s[Math.floor(f * (s.length - 1))];
            lo = Math.min(line, 0, Math.max(lo, q(0.01)));
            hi = Math.max(line, Math.min(hi, q(0.985)));
        }
        const pad = (hi - lo) * 0.1 || 1;
        lo -= pad; hi += pad;
        const y = (v) => (1 - (clamp(v, lo, hi) - lo) / (hi - lo)) * H;
        const x = (i) => (i / (pts.length - 1)) * W;
        const path = (k) => {
            let d = '', pen = false;
            pts.forEach((p, i) => {
                if (!Number.isFinite(p[k])) { pen = false; return; }
                d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(p[k]).toFixed(1)}`;
                pen = true;
            });
            return d;
        };
        const last = pts[pts.length - 1];
        const idx = (d) => { const i = pts.findIndex((p) => p.d >= d); return i < 0 ? pts.length - 1 : i; };
        const rects = (bands || []).filter(([s, e]) => e >= pts[0].d && s <= last.d).map(([s, e]) => [x(idx(s)), x(idx(e))]);
        const span = Number(last.d.slice(0, 4)) - Number(pts[0].d.slice(0, 4));
        const step = span <= 6 ? 1 : span <= 14 ? 2 : span <= 30 ? 5 : 10;
        const years = [];
        pts.forEach((p, i) => {
            const yr = Number(p.d.slice(0, 4));
            if (i && yr % step === 0 && yr !== Number(pts[i - 1].d.slice(0, 4))) years.push({ yr, at: (x(i) / W) * 100 });
        });
        return {
            ly: y(line), end: y(last.v), main: path('v'), ghost: ghost ? path('g') : null, rects,
            zero: line !== 0 && lo < 0 && hi > 0 ? y(0) : null, years: years.filter((t) => t.at > 4 && t.at < 93),
        };
    }, [pts, ghost, line, clip, bands]);
    if (!g) return null;
    const off = (g.ly / H).toFixed(4);
    const [top, bottom] = bad === 'above' ? ['var(--down)', safe] : [safe, 'var(--down)'];
    const read = (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        if (r.width) setAt(clamp(Math.round(((e.clientX - r.left) / r.width) * (pts.length - 1)), 0, pts.length - 1));
    };
    const ax = at == null ? null : (at / (pts.length - 1)) * 100;
    return (
        <div className={`rk-plot is-${size}`} {...(mini ? {} : { onPointerMove: read, onPointerDown: read, onPointerLeave: () => setAt(null) })}>
            <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
                <defs>
                    <linearGradient id={gid} gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2={H}>
                        <stop offset="0" stopColor={top} /><stop offset={off} stopColor={top} />
                        <stop offset={off} stopColor={bottom} /><stop offset="1" stopColor={bottom} />
                    </linearGradient>
                </defs>
                {g.rects.map(([a, b]) => <rect key={a} className="rk-band" x={a} y="0" width={Math.max(b - a, 3)} height={H} />)}
                <rect className="rk-wash" x="0" width={W} y={bad === 'above' ? 0 : g.ly} height={bad === 'above' ? g.ly : H - g.ly} />
                {g.zero != null && <line className="rk-zero" x1="0" x2={W} y1={g.zero} y2={g.zero} />}
                <line className="rk-at" x1="0" x2={W} y1={g.ly} y2={g.ly} />
                {g.ghost && <path className="rk-ghost" d={g.ghost} />}
                <path className="rk-path" d={g.main} stroke={`url(#${gid})`} />
            </svg>
            <i className="rk-end" style={{ top: `${g.end}%` }} />
            {tag && <span className={`rk-tag${g.ly < 22 ? ' is-below' : ''}`} style={{ top: `${g.ly}%` }}>{tag}</span>}
            {!mini && g.years.map((t) => <span key={t.yr} className="rk-yr" style={{ left: `${t.at}%` }}>{t.yr}</span>)}
            {ax != null && (
                <>
                    <b className="rk-hair" style={{ left: `${ax}%` }} />
                    <span className={`rk-read${ax > 60 ? ' is-left' : ''}`} style={{ left: `${ax}%` }}>{when(pts[at].d)} · {fmt(pts[at].v)}</span>
                </>
            )}
        </div>
    );
}

/* ---------------- Rubber band ---------------- */

const rb = (v) => minus(signed(v));
const series = (history, k, ghost) => (history || []).filter((h) => h?.d && Number.isFinite(h[k])).map((h) => ({ d: h.d, v: h[k], g: ghost ? h[ghost] : undefined }));

/** The call, as the v1 hero makes it, in one line. */
function verdictOf(data) {
    const mode = data.defensive?.mode, v = data.verdict.colour;
    if (mode === 'PENDING_DEFENSIVE') return { c: 'red', t: 'Go defensive', s: 'Sell half, reply "done"' };
    if (mode === 'DEFENSIVE') return { c: 'amber', t: 'Defensive', s: 'Half in cash. Wait.' };
    if (mode === 'PENDING_REENTRY') return { c: 'blue', t: 'Go back in', s: 'Buy back, reply "done"' };
    if (v === 'red') return { c: 'red', t: 'A tripwire is red', s: data.verdict.text };
    if (v === 'amber') return { c: 'amber', t: 'Stay invested', s: 'Keep an eye on it' };
    if (v === 'grey') return { c: 'grey', t: 'No reading today', s: '' };
    return { c: 'green', t: 'Stay invested', s: 'Nothing to do' };
}

/** Each machine's drop so far, as a share of the limit that would trip it. */
function LegBars({ legs, spm }) {
    return (
        <span className="rk-legbars" aria-hidden="true">
            {legs.map((l) => {
                const u = legUsed(l, spm);
                return <i key={l.name} style={{ ...tc(legColour(l, spm)), '--w': `${u ? Math.max(u.frac * 100, 4) : 0}%` }} />;
            })}
        </span>
    );
}

function Leg({ l, spm }) {
    const u = legUsed(l, spm);
    const uw = l.months_underwater == null ? '' : `, ${l.months_underwater}${l.longest_underwater_prior_months != null ? ` of ${l.longest_underwater_prior_months}` : ''} months under water`;
    return (
        <div className="rk-leg tooltip-trigger" style={tc(legColour(l, spm))}
            data-tooltip={`${l.name}: ${minus(signed(l.dd_pct, 1))} from its peak${u ? `, trips at ${minus(signed(u.lim.v, 0))} (${u.lim.why})` : ''}${uw}`}>
            <span className="rk-leg-top"><b>{l.name}</b>{l.role === 'hedge' && <em>hedge</em>}<span>{minus(signed(l.dd_pct, 1))}</span></span>
            <span className="rk-gauge"><i style={{ width: `${u ? Math.max(u.frac * 100, l.dd_pct < 0 ? 2 : 0) : 0}%` }} /></span>
            <span className="rk-leg-sub">{u ? `trips at ${minus(signed(u.lim.v, 0))}` : 'no limit yet'}</span>
        </div>
    );
}

/** The alarm's fuse: one segment per day it takes to trip, lit as they burn. */
function Fuse({ count, of, what, tip }) {
    const n = Number.isFinite(of) && of > 0 ? Math.min(of, 60) : 0;
    const known = Number.isFinite(count);
    const lit = known ? clamp(count, 0, n) : 0;
    const c = !known ? 'grey' : lit >= n ? 'red' : lit >= 0.75 * n ? 'amber' : 'green';
    return (
        <div className="rk-fuse tooltip-trigger" data-tooltip={tip} style={tc(c)}>
            <span className="rk-segs">{Array.from({ length: n }, (_, i) => <i key={i} className={i < lit ? 'is-on' : undefined} />)}</span>
            <span className="rk-fuse-n"><b>{known ? count : '—'}</b>/{of} {what}</span>
        </div>
    );
}

const FUSE = { slow: 'bad days', rip: 'hot days', machines: 'red closes' };

function Lane({ k, d, ctx, def, value, cap, children }) {
    const e = EXPLAIN[k](d, ctx);
    const f = tripFuse(k, d, ctx, def);
    return (
        <div className="rk-lane" style={tc(d.colour)}>
            <div className="rk-lane-head">
                <span className="rk-q tooltip-trigger" data-tooltip={`${[].concat(e.what).join('. ')}. ${[].concat(e.record).join('. ')}.`}><i />{PLAIN[k].q}</span>
                <span className="rk-lane-v"><b>{value}</b><em>{cap}</em></span>
            </div>
            <div className="rk-lane-body">{children}</div>
            <Fuse count={f.count} of={f.of} what={FUSE[k]} tip={`Red: ${[].concat(e.red).join('; ')}`} />
        </div>
    );
}

const LOOP = [
    { mode: 'INVESTED', c: 'green', t: 'Invested', s: () => 'Nothing to do' },
    { mode: 'PENDING_DEFENSIVE', c: 'red', t: 'Tripwire red', s: () => 'Sell half' },
    { mode: 'DEFENSIVE', c: 'amber', t: 'Defensive', s: () => 'Half in cash' },
    { mode: 'PENDING_REENTRY', c: 'blue', t: 'Back in', s: (r) => `After ${r.reentry_closes} green days` },
];

/** The trigger's loop with "you are here"; the full rule on hover/tap. */
function Loop({ def, rules }) {
    const here = def ? LOOP.findIndex((s) => s.mode === def.mode) : -1;
    const fa = rules.fire_after;
    const closes = (n) => (n === 1 ? 'a red close' : `${n} red closes in a row`);
    const tip = `Sell half after ${closes(fa.slow)} on dips, ${closes(fa.rip)} on rallies or ${closes(fa.machines)} on machines. Buy back after ${rules.reentry_closes} green closes with dips over +${rules.reentry_edge_pct}%.`;
    return (
        <ol className="rk-loop tooltip-trigger" data-tooltip={tip} style={tc(LOOP[here]?.c)}>
            {LOOP.map((s, i) => (
                <li key={s.mode} className={i === here ? 'is-here' : undefined}>
                    <i /><b>{s.t}</b><span>{i === here && i === 0 ? `${def.green_streak ?? 0} green days` : s.s(rules)}</span>
                </li>
            ))}
        </ol>
    );
}

function RubberBandBody({ data, open }) {
    const ctx = contextOf(data);
    const h = verdictOf(data);
    const dial = (k) => data.dials[k] || { colour: 'grey' };
    const legs = dial('machines').legs || [];
    const judged = legs.filter((l) => l.dd_pct != null);
    const mach = judged.length ? `${judged.filter((l) => legColour(l, ctx.machines) !== 'red').length} of ${judged.length}` : '—';
    const val = (k) => (dial(k).colour === 'grey' ? '—' : k === 'machines' ? mach : rb(dial(k).excess_pct));
    const pts = useMemo(() => ({ slow: series(data.history, 'slow', 'fast'), rip: series(data.history, 'rip') }), [data.history]);
    const bad = { slow: 'below', rip: 'above' };
    return (
        <>
            <div className="rk-verdict" style={tc(h.c)}><b>{h.t}</b>{h.s && <span>{h.s}</span>}</div>
            {!open ? (
                <div className="rk-wires">
                    {['slow', 'rip', 'machines'].map((k) => (
                        <div key={k} className="rk-wire" style={tc(dial(k).colour)}>
                            <span className="rk-q"><i />{PLAIN[k].q}</span>
                            {k === 'machines' ? <LegBars legs={judged} spm={ctx.machines} />
                                : <Plot mini size="spark" pts={pts[k]} line={0} bad={bad[k]} />}
                            <b className="rk-v">{val(k)}</b>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="rk-open">
                    <Loop def={data.defensive} rules={ctx.rules} />
                    <div className="rk-lanes">
                        <Lane k="slow" d={dial('slow')} ctx={ctx} def={data.defensive} value={val('slow')} cap={`per dip vs a normal day · paid ${pct0(dial('slow').hit)}`}>
                            <Plot pts={pts.slow} ghost line={0} bad="below" fmt={rb} tag="normal day" />
                        </Lane>
                        <Lane k="rip" d={dial('rip')} ctx={ctx} def={data.defensive} value={val('rip')} cap="the day after a hot run">
                            <Plot pts={pts.rip} line={0} bad="above" fmt={rb} tag="normal day" />
                        </Lane>
                        <Lane k="machines" d={dial('machines')} ctx={ctx} def={data.defensive} value={mach} cap="inside their limits">
                            <div className="rk-legs">{legs.map((l) => <Leg key={l.name} l={l} spm={ctx.machines} />)}</div>
                            {dial('machines').reasons?.length > 0 && <div className="rk-reasons">{dial('machines').reasons.join(' · ')}</div>}
                        </Lane>
                    </div>
                    <div className="rk-foot">
                        {['fast', 'age'].map((k) => {
                            const d = dial(k), e = EXPLAIN[k](d, ctx);
                            return (
                                <span key={k} className="rk-look tooltip-trigger" style={tc(d.colour)} data-tooltip={`${[].concat(e.what).join('. ')}. Look only, never trips.`}>
                                    <i />{k === 'fast' ? 'Early warning' : 'Evidence age'}
                                    <b>{d.colour === 'grey' ? '—' : k === 'fast' ? rb(d.excess_pct) : `${d.years?.toFixed(1)} yrs`}</b>
                                </span>
                            );
                        })}
                        <span className="rk-meta">Checked {dayName(data.asOf)}{nextRun(data.asOf) ? ` · next ${nextRun(data.asOf)}` : ''}{data._meta?.stale ? ` · ${data._meta.ageDays ?? '?'}d old` : ''}</span>
                    </div>
                </div>
            )}
        </>
    );
}

export function RubberBandCard({ open, onVerdict = null }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // The Readout shows this card's verdict as a chip: hand it the answer (null = failed).
    useEffect(() => {
        if (loading || !onVerdict) return;
        try { onVerdict(data); } catch { /* the readout never breaks this card */ }
    }, [loading, data, onVerdict]);

    useEffect(() => {
        fetch('/api/rubber-band')
            .then((res) => res.json())
            .then((json) => (!json || !json.dials || !json.verdict ? setError(json?._meta?.messages?.[0] || 'no data') : setData(json)))
            .catch(() => setError('fetch failed'))
            .finally(() => setLoading(false));
    }, []);

    const v = data?.verdict?.colour || 'grey';
    return (
        <div className="card rk">
            <div className="card-header">
                {data?._meta?.stale && <span className="badge badge-red">Stale</span>}
                <span className={`badge ${BADGE[v]}`}>{loading ? '…' : COLOUR_WORD[v]}</span>
            </div>
            {loading ? <Skeleton count={3} /> : error || !data
                ? <div className="rk-na">Snapshot unavailable{error ? ` (${error})` : ''}. The Mac mini publishes it after each close.</div>
                : <RubberBandBody data={data} open={open} />}
        </div>
    );
}

/* ---------------- Recession watch ---------------- */

const lvlPct = (v) => `${v.toFixed(2)}%`;
const lvlSpread = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}%`;
const chg = (unit) => (v) => (v == null ? '—' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}${unit}`);
const STATUS = { improving: ['green', 'Improving'], watch: ['amber', 'Rising'], 'recession-like': ['red', 'Recession-like'], unknown: ['grey', 'No data'] };
const HORSES = [
    { key: 'claims', label: 'Jobless claims', mode: 'pct', unit: '%', pick: (f) => f?.horsemen?.claims, level: kFmt },
    { key: 'unemployment', label: 'Unemployment', mode: 'pp', unit: 'pp', pick: (f) => f?.horsemen?.unemployment, level: lvlPct },
    { key: 'spread', label: '10Y − 2Y', pick: (f) => f?.yieldCurve, level: lvlSpread },
    { key: 'bankruptcies', label: 'Bankruptcies', mode: 'pct', unit: '%', pick: (f) => f?.horsemen?.bankruptcies, level: kFmt },
];

/** Last print in each month, [{d: 'YYYY-MM', v}]. */
function monthly(history) {
    const out = [];
    for (const p of history || []) {
        const v = Number(p?.value);
        if (!p?.date || p.value == null || !Number.isFinite(v)) continue;
        const d = String(p.date).slice(0, 7);
        if (out.length && out[out.length - 1].d === d) out[out.length - 1].v = v; else out.push({ d, v });
    }
    return out;
}

/** Each month's change on the same month a year before; the last point is the header's own number. */
function yoyMonthly(history, mode, latest) {
    const m = monthly(history);
    const at = new Map(m.map((p) => [p.d, p.v]));
    const out = [];
    for (const p of m) {
        const prev = at.get(`${Number(p.d.slice(0, 4)) - 1}${p.d.slice(4)}`);
        if (prev == null || (mode === 'pct' && prev === 0)) continue;
        out.push({ d: p.d, v: mode === 'pct' ? 100 * (p.v / prev - 1) : p.v - prev });
    }
    if (latest != null && out.length && out[out.length - 1].d === m[m.length - 1].d) out[out.length - 1].v = latest;
    return out;
}

function horse(r, fred, now) {
    const m = r.pick(fred);
    const cur = Number.isFinite(m?.current) ? m.current : null;
    if (r.key === 'spread') {
        const inv = lastInversion(m?.history, now);
        return {
            ...r, m, cur, move: cur, line: 0, bad: 'below', c: cur == null ? 'grey' : cur < 0 ? 'red' : 'green',
            word: cur == null ? 'No data' : cur < 0 ? 'Inverted' : 'Normal',
            sub: inv ? (inv.currentlyInverted ? 'inverted now' : `inverted ${inv.startYear}–${String(inv.endYear).slice(2)}`) : 'level',
            tip: `Below 0 = inverted, the tell a year or more before past recessions.${inv && !inv.currentlyInverted ? ` Last inverted ${inv.startYear}–${inv.endYear}, ended ${inv.monthsSince} months ago.` : ''}`,
            fmt: lvlSpread, tag: 'inverted below 0',
        };
    }
    const yoy = latestYoY(m?.history, r.mode) ?? (r.key === 'bankruptcies' && Number.isFinite(m?.changePct) ? m.changePct : null);
    const runups = preRecessionRunups(m?.history, fred?.recessions || [], r.mode);
    const median = runupMedian(runups);
    const [c, word] = STATUS[horsemanStatus(yoy, median, true)];
    const f = chg(r.unit);
    const sahm = fred?.indicators?.sahmRule?.value;
    return {
        ...r, m, cur, yoy, median, move: yoy, line: median, bad: 'above', c, word, fmt: f,
        sub: <>{cur != null ? r.level(cur) : '—'} now{r.key === 'unemployment' && Number.isFinite(sahm) && <em> · Sahm {sahm.toFixed(2)}</em>}</>,
        tag: median != null ? `${runups.length} recessions began near ${f(median)}` : null,
        tip: `1-year move ${f(yoy)}.${median != null ? ` ${runups.length} recessions began with a move near ${f(median)} (the red zone).` : ''}${m?.stale ? ` Last data ${formatAsOf(m.asOf)} (stale).` : ''}`,
    };
}

/** Today's move as a dot; the red stretch is where past recessions began. */
function Meter({ v, line, bad }) {
    if (v == null) return <span className="rw-meter" aria-hidden="true" />;
    const half = Math.max(Math.abs(line ?? 0) * 1.6, Math.abs(v) * 1.15, bad === 'below' ? 0.5 : 1e-9);
    const pos = (x) => 50 + 50 * clamp(x / half, -1, 1);
    const zone = line == null ? null : bad === 'above' ? { left: `${pos(line)}%`, right: 0 } : { left: 0, right: `${100 - pos(line)}%` };
    return (
        <span className="rw-meter" aria-hidden="true">
            {zone && <b style={zone} />}
            <u />
            <i style={{ left: `${pos(v)}%` }} />
        </span>
    );
}

export function RecessionCard({ fred, loading, open }) {
    const rows = useMemo(() => (fred && !fred.error ? HORSES.map((r) => horse(r, fred, Date.now())) : []), [fred]);
    const plots = useMemo(() => (open ? Object.fromEntries(rows.map((r) => [r.key, r.key === 'spread' ? monthly(r.m?.history) : yoyMonthly(r.m?.history, r.mode, r.yoy)])) : null), [rows, open]);
    const bands = useMemo(() => (fred?.recessions || []).map((b) => [String(b.start).slice(0, 7), String(b.end).slice(0, 7)]), [fred]);
    const riding = horsemenRiding(fred);
    return (
        <div className="card rw">
            <div className="card-header">
                {!loading && riding && <span className={`badge ${riding.riding >= 3 ? 'badge-red' : riding.riding >= 1 ? 'badge-yellow' : 'badge-green'}`}>{riding.riding} of 4 riding</span>}
            </div>
            {loading || !fred ? <Skeleton count={2} /> : !rows.some((r) => r.cur != null) ? (
                <div className="rk-na">Unavailable — source busy, try again shortly</div>
            ) : !open ? (
                <div className="rw-tiles">
                    {rows.map((r) => (
                        <div key={r.key} className="rw-tile" style={tc(r.c)}>
                            <span className="rw-label tooltip-trigger" data-tooltip={r.tip}>{r.label}</span>
                            <span className={`rw-val${r.m?.stale ? ' is-stale' : ''}`}>
                                <b>{r.cur != null ? r.level(r.cur) : '—'}</b>
                                {r.key === 'spread' ? <em>{r.word}</em>
                                    : r.yoy != null && <em className={r.yoy > 0 ? 'is-bad' : 'is-good'}>{r.yoy >= 0 ? '▲' : '▼'} {Math.abs(r.yoy).toFixed(1)}{r.unit} vs 1y</em>}
                            </span>
                            <Meter v={r.move} line={r.line} bad={r.bad} />
                        </div>
                    ))}
                </div>
            ) : (
                <>
                    <div className="rw-panels">
                        {rows.map((r) => (
                            <div key={r.key} className="rw-panel" style={tc(r.c)}>
                                <div className="rw-head">
                                    <span className="rw-name tooltip-trigger" data-tooltip={r.tip}>{r.label}</span>
                                    <span className="rw-sub">{r.sub}</span>
                                    <span className="rw-now">{r.key === 'spread' ? (r.cur != null ? lvlSpread(r.cur) : '—') : r.fmt(r.yoy)}</span>
                                    <span className="rw-word">{r.word}</span>
                                </div>
                                {plots[r.key].length >= 12
                                    ? <Plot pts={plots[r.key]} line={r.line ?? 0} bad={r.bad} bands={bands} clip={r.key !== 'spread'} fmt={r.fmt} safe="var(--ink-2)" tag={r.tag} size="panel" />
                                    : <div className="rk-na rw-none">Not enough history</div>}
                            </div>
                        ))}
                    </div>
                    <div className="rw-foot">
                        <span><i className="rw-k-band" />recession</span>
                        <span><i className="rw-k-line" />where past ones began</span>
                        <span className="rw-hint">1-year change · drag to read</span>
                    </div>
                </>
            )}
        </div>
    );
}
