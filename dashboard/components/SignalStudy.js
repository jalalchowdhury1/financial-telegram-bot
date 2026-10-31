'use client';
/**
 * A v2 Signals tile's drill-in (AAII 20+, VIX over VIX 3M): every time the signal fired, what SPY did
 * next, in one overlay chart. Thin lines = each signal (green ended up, red ended down), bold = their average,
 * dashed = any day's average, blue = the signal still running. Under it, every signal best → worst.
 * Hover or tap a line or bar to read it; ← → step through them. The two baked files load on open
 * only; lib/signalStudy.js does the math.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import useSheetLock, { useSheetFocus } from './useSheetLock';
import { decodeLong } from '../lib/longHistory';
import { signalStudy } from '../lib/signalStudy';

const pct = (r, dp = 1) => `${r > 0 ? '+' : r < 0 ? '−' : ''}${Math.abs(r * 100).toFixed(dp)}%`;
const day = (iso) => new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
const load = (src) => fetch(src).then((r) => (r.ok ? r.json() : null)).then(decodeLong);
const XS = { 1: ['Hit', '2W', '1M'], 3: ['Hit', '1M', '2M', '3M'], 6: ['Hit', '2M', '4M', '6M'], 12: ['Hit', '3M', '6M', '9M', '12M'] };
const tail = (pts, more) => [...pts, ...(more || []).filter((p) => p.date > pts[pts.length - 1].date)];

/** r on a path at x (0..1), or null past where it ends (a live signal). */
function rAt(path, x) {
    const k = path.findIndex((p) => p.x >= x);
    if (k < 0) return null;
    if (k === 0) return path[0].r;
    const a = path[k - 1], b = path[k];
    return a.r + ((b.r - a.r) * (x - a.x)) / (b.x - a.x || 1);
}

/**
 * `id` keys the remembered horizon; `src` = the signal's baked daily/weekly file, `sheet` = its newer
 * history-sheet points and `now` today's reading (both appended after the file); `line` = where it
 * fires; `label(v)` names a reading; `sub` = the header's one line; `foot` = what counts as a signal;
 * `horizons` = the month chips, `start` = the one it opens on.
 */
export default function SignalStudy({ id, src, line, title, sub, label, foot, horizons, start, now, sheet, spyLive, onClose }) {
    const KEY = `v2-${id}-months`;
    const backRef = useRef(null), sheetRef = useRef(null);
    const [raw, setRaw] = useState(null); // {a, s} | false when a file failed
    const [months, setMonths] = useState(start);
    const [hot, setHot] = useState(null); // {date, x?} under the pointer
    const [pin, setPin] = useState(null); // date tapped
    useSheetLock(true);
    useSheetFocus(true, sheetRef);

    useEffect(() => {
        try { const m = Number(window.localStorage.getItem(KEY)); if (horizons.includes(m)) setMonths(m); } catch { /* private mode */ }
        let live = true;
        Promise.all([load(src), load('/history/spy-daily.json')])
            .then(([a, s]) => { if (live) setRaw(a && s ? { a, s } : false); })
            .catch(() => { if (live) setRaw(false); });
        return () => { live = false; };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const study = useMemo(() => {
        if (!raw) return null;
        const today = new Date().toLocaleDateString('en-CA');
        const pts = tail(raw.a, Number.isFinite(now) ? [...(sheet || []), { date: today, value: now }] : sheet);
        const spy = tail(raw.s.map((p) => ({ date: p.date, price: p.value })), (spyLive || []).map((p) => ({ date: p.date, price: p.price })));
        return signalStudy(pts, spy, months, line);
    }, [raw, sheet, now, spyLive, months, line]);

    const eps = study?.episodes || [];
    const ranked = useMemo(() => [...eps].sort((a, b) => b.ret - a.ret), [eps]);
    const live = eps.findLast((e) => e.live);
    const focus = hot?.date || pin;
    const sel = eps.find((e) => e.date === (focus || live?.date));

    useEffect(() => {
        const k = (e) => {
            if (e.key === 'Escape') onClose();
            if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && eps.length) {
                const i = eps.findIndex((ep) => ep.date === pin);
                const j = i < 0 ? eps.length - 1 : Math.min(eps.length - 1, Math.max(0, i + (e.key === 'ArrowLeft' ? -1 : 1)));
                setPin(eps[j].date);
            }
        };
        document.addEventListener('keydown', k);
        return () => document.removeEventListener('keydown', k);
    }, [onClose, eps, pin]);

    const geo = useMemo(() => {
        if (!study?.stats) return null;
        const rs = eps.flatMap((e) => e.path.map((p) => p.r)).concat(0, study.base.avg);
        let lo = Math.min(...rs), hi = Math.max(...rs);
        const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
        const y = (r) => ((hi - r) / (hi - lo)) * 100;
        const d = (path) => path.map((p, i) => `${i ? 'L' : 'M'}${(p.x * 100).toFixed(2)} ${y(p.r).toFixed(2)}`).join('');
        const step = [0.05, 0.1, 0.2, 0.25, 0.5].find((s) => (hi - lo) / s <= 5) || 1;
        const ticks = [];
        for (let v = Math.ceil(lo / step) * step; v < hi; v += step) ticks.push(Math.round(v * 100) / 100);
        return { lo, hi, y, d, ticks };
    }, [study, eps]);

    const pick = (m) => { setMonths(m); setHot(null); try { window.localStorage.setItem(KEY, String(m)); } catch { /* still works this visit */ } };
    const read = (e) => {
        const b = e.currentTarget.getBoundingClientRect();
        const x = Math.min(1, Math.max(0, (e.clientX - b.left) / b.width));
        const v = geo.hi - ((e.clientY - b.top) / b.height) * (geo.hi - geo.lo);
        let best = null, gap = Infinity;
        for (const ep of eps) {
            const r = rAt(ep.path, x);
            if (r != null && Math.abs(r - v) < gap) { gap = Math.abs(r - v); best = ep; }
        }
        setHot(best && { date: best.date, x });
    };
    const toggle = (date) => setPin((p) => (p === date ? null : date));

    let body;
    if (raw === null) body = <div className="v2-aa-wait" />;
    else if (!study?.stats || !geo) body = <div className="v2-fred-na">Unavailable · try again shortly</div>;
    else {
        const { stats: st, base } = study, { y } = geo;
        const yAvg = y(study.avgPath[study.avgPath.length - 1].r), yBase = y(base.avg);
        const lastLive = live?.path[live.path.length - 1];
        const hotR = hot?.x != null && sel ? rAt(sel.path, hot.x) : null;
        const maxUp = Math.max(0.01, ...eps.map((e) => e.ret)), maxDn = Math.max(0.01, ...eps.map((e) => -e.ret));
        const span = maxUp + maxDn;
        const stat = (label, v, b, show) => (
            <span className="v2-aa-stat">
                <small>{label}</small>
                <b className={`tone-${v > b ? 'good' : 'caution'}`}>{show(v)}</b>
                <em>any day {show(b)}</em>
            </span>
        );
        body = (
            <>
                <div className="v2-aa-stats">
                    {stat(`Avg after ${months}M`, st.avg, base.avg, (v) => pct(v))}
                    {stat('Rose', st.win, base.win, (v) => `${Math.round(v * 100)}%`)}
                    {stat('Median', st.median, base.median, (v) => pct(v))}
                </div>
                <div className="v2-aa-read">
                    {sel ? (
                        <>
                            <b>{day(sel.date)}</b>
                            <span>{label(sel.value)}</span>
                            <span className={`tone-${sel.live ? 'flat' : sel.ret >= 0 ? 'good' : 'bad'}`}>SPY {pct(sel.ret)}{sel.live ? ' so far' : ''}</span>
                            <span className="v2-aa-low">low {pct(sel.low)}</span>
                        </>
                    ) : <span>Avg of {st.n} signals · {pct(st.avg)}</span>}
                </div>
                <div className={`v2-aa-plot${focus ? ' has-focus' : ''}`} onPointerMove={read} onPointerDown={read}
                    onPointerLeave={() => setHot(null)} onClick={() => hot && toggle(hot.date)}>
                    {geo.ticks.map((v) => <span key={v} className={`v2-aa-y${v === 0 ? ' is-zero' : ''}`} style={{ top: `${y(v)}%` }}><b>{v ? pct(v, 0) : '0'}</b></span>)}
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                        <g key={months} className="v2-aa-lines">
                            {[...eps].sort((a, b) => (a.date === focus) - (b.date === focus)).map((ep) => (
                                <path key={ep.date} d={geo.d(ep.path)}
                                    className={`v2-aa-ep ${ep.live ? 'is-live' : ep.ret >= 0 ? 'is-up' : 'is-down'}${ep.date === focus ? ' is-on' : ''}`} />
                            ))}
                            <path className="v2-aa-base" d={`M0 ${y(0)}L100 ${yBase}`} />
                            <path className="v2-aa-avg" d={geo.d(study.avgPath)} />
                        </g>
                    </svg>
                    <span className="v2-aa-end is-avg" style={{ top: `${yAvg}%` }}>avg {pct(st.avg)}</span>
                    <span className="v2-aa-end" style={{ top: `${Math.abs(yAvg - yBase) < 8 ? yAvg + (yBase >= yAvg ? 8 : -8) : yBase}%` }}>any day {pct(base.avg)}</span>
                    {lastLive && (
                        <span className={`v2-aa-now${lastLive.x > 0.5 ? ' is-left' : ''}`} style={{ left: `${lastLive.x * 100}%`, top: `${y(lastLive.r)}%` }}>
                            <i /><b>now {pct(live.ret)}</b>
                        </span>
                    )}
                    {hotR != null && <i className="v2-aa-dot" style={{ left: `${hot.x * 100}%`, top: `${y(hotR)}%` }} />}
                </div>
                <div className="v2-aa-xs">{XS[months].map((t, i, a) => <span key={t} style={{ left: `${(i / (a.length - 1)) * 100}%` }}>{t}</span>)}</div>
                <div className="v2-aa-rank-head"><span>{eps.length} signals since {eps[0].date.slice(0, 4)} · best → worst</span><span className="v2-aa-key">┄ any day</span></div>
                <div className="v2-aa-rank" role="group" aria-label="Every signal, best to worst">
                    <i className="v2-aa-rank-base" style={{ top: `${((maxUp - base.avg) / span) * 100}%` }} />
                    {ranked.map((ep) => (
                        <button key={ep.date} type="button" aria-label={`${day(ep.date)} · SPY ${pct(ep.ret)}`} aria-pressed={pin === ep.date}
                            className={`${ep.live ? 'is-live' : ep.ret >= 0 ? 'is-up' : 'is-down'}${ep.date === focus ? ' is-on' : ''}`}
                            style={{ '--t': `${((maxUp - Math.max(0, ep.ret)) / span) * 100}%`, '--h': `${(Math.abs(ep.ret) / span) * 100}%` }}
                            onPointerEnter={() => setHot({ date: ep.date })} onPointerLeave={() => setHot(null)} onClick={() => toggle(ep.date)} />
                    ))}
                </div>
                <p className="v2-aa-foot">SPY price only · a signal = {foot}</p>
            </>
        );
    }

    if (typeof document === 'undefined') return null;
    return createPortal(
        <div ref={backRef} className="v2-aa-back" role="dialog" aria-modal="true" aria-labelledby="v2-aa-title"
            onClick={(e) => { if (e.target === backRef.current) onClose(); }}>
            <div className="v2-aa" ref={sheetRef} data-sheet-scroll="">
                <div className="v2-aa-head">
                    <div className="v2-aa-title">
                        <h2 id="v2-aa-title">{title}</h2>
                        <p>{sub}</p>
                    </div>
                    <div className="v2-aa-tfs" role="group" aria-label="Months after">
                        {horizons.map((m) => <button key={m} type="button" className={m === months ? 'is-on' : ''} aria-pressed={m === months} onClick={() => pick(m)}>{m}M</button>)}
                    </div>
                    <button type="button" className="v2-aa-close sheet-x" onClick={onClose} aria-label="Close">×</button>
                </div>
                {body}
            </div>
        </div>,
        document.querySelector('.v2') || document.body,
    );
}
