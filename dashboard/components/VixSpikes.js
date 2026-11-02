'use client';
/**
 * The v2 Volatility card's "After spikes": every time VIX first closed at 30+ (after 3 calm months), what SPY
 * did 1, 3, 6 and 12 months later. One row a spike, newest first: its peak (number + bar), then SPY's move per
 * window, tinted by its size (dashed blue = a window still running). On top, the median and how often SPY rose,
 * each against any day. Tap a spike for its year as a line. The baked daily VIX (public/history/vix-daily.json,
 * re-baked quarterly; the card's own year fills in after it) loads on open only; lib/signalStudy.js does the math.
 */
import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import useSheetLock, { useSheetFocus } from './useSheetLock';
import { load, tail, pct, day } from './SignalStudy';
import { spikeStudy, SPIKE_MONTHS } from '../lib/signalStudy';

export const SPIKE_LINE = 30;
const COLS = SPIKE_MONTHS.map((m) => (m === 12 ? '1Y' : `${m}M`));
const monYear = (iso) => new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });
const better = (v, b) => (v == null || b == null ? '' : v > b ? 'tone-good' : 'tone-caution');

function Cell({ c, max }) {
    if (!c) return <span className="v2-vs-cell">—</span>;
    if (c.live) return <span className="v2-vs-cell is-live">{pct(c.ret)}</span>;
    return <span className={`v2-vs-cell ${c.ret >= 0 ? 'is-up' : 'is-down'}`} style={{ '--a': `${Math.round(12 + 48 * Math.min(1, Math.abs(c.ret) / max))}%` }}>{pct(c.ret)}</span>;
}

/** One spike's year: SPY from the day VIX hit 30, with 0, the 1M/3M/6M marks, each finished window's dot and the low. */
function Year({ row }) {
    const rs = row.path.map((p) => p.r);
    const lo = Math.min(-0.05, ...rs), hi = Math.max(0.05, ...rs);
    const y = (r) => 10 + ((hi - r) / (hi - lo)) * 66; // the bottom quarter stays free for the low's label
    const end = row.path[row.path.length - 1], low = row.path.reduce((m, p) => (p.r < m.r ? p : m));
    const tone = row.live ? 'is-live' : end.r >= 0 ? 'is-up' : 'is-down';
    return (
        <div className="v2-vs-year">
            <div className="v2-vs-plot">
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                    {SPIKE_MONTHS.slice(0, -1).map((m) => <path key={m} className="v2-vs-tick" d={`M${((m / 12) * 100).toFixed(2)} 0V100`} />)}
                    <path className="v2-vs-zero" d={`M0 ${y(0).toFixed(2)}H100`} />
                    <path className={`v2-vs-line ${tone}`} d={row.path.map((p, i) => `${i ? 'L' : 'M'}${(p.x * 100).toFixed(2)} ${y(p.r).toFixed(2)}`).join('')} />
                </svg>
                {row.at.map((c, k) => c && !c.live && <i key={k} className="v2-vs-dot" style={{ left: `${(SPIKE_MONTHS[k] / 12) * 100}%`, top: `${y(c.ret)}%` }} />)}
                {low !== end && low.r < -0.01 && (
                    <span className={`v2-vs-low${low.x > 0.8 ? ' is-left' : low.x < 0.15 ? ' is-right' : ''}`} style={{ left: `${low.x * 100}%`, top: `${y(low.r)}%` }}>low {pct(low.r)}</span>
                )}
                <span className={`v2-vs-end ${tone}${end.x > 0.7 ? ' is-left' : ''}`} style={{ left: `${end.x * 100}%`, top: `${y(end.r)}%` }}>
                    {row.live ? 'now' : '1Y'} {pct(end.r)}
                </span>
            </div>
            <div className="v2-aa-xs">{['Hit', ...COLS].map((t, i) => <span key={t} style={{ left: `${(i ? SPIKE_MONTHS[i - 1] / 12 : 0) * 100}%` }}>{t}</span>)}</div>
        </div>
    );
}

export default function VixSpikes({ vol, spy, onClose }) {
    const backRef = useRef(null), sheetRef = useRef(null);
    const [raw, setRaw] = useState(null); // {v, s} | false when a file failed
    const [open, setOpen] = useState(undefined); // the spike tapped open; undefined = the newest, null = none
    useSheetLock(true);
    useSheetFocus(true, sheetRef);

    useEffect(() => {
        let live = true;
        Promise.all([load('/history/vix-daily.json'), load('/history/spy-daily.json')])
            .then(([v, s]) => { if (live) setRaw(v && s ? { v, s } : false); })
            .catch(() => { if (live) setRaw(false); });
        return () => { live = false; };
    }, []);

    useEffect(() => {
        const k = (e) => { if (e.key === 'Escape') onClose(); };
        document.addEventListener('keydown', k);
        return () => document.removeEventListener('keydown', k);
    }, [onClose]);

    const study = useMemo(() => {
        if (!raw) return null;
        const dates = vol?.vixYearDates || [];
        const year = (vol?.vixYear || []).map((value, i) => ({ date: dates[i], value })).filter((p) => p.date && Number.isFinite(p.value));
        const px = (spy?.chartHistory || []).filter((p) => p.date && Number.isFinite(p.price)).map((p) => ({ date: p.date, price: p.price }));
        return spikeStudy(tail(raw.v, year), tail(raw.s.map((p) => ({ date: p.date, price: p.value })), px), SPIKE_LINE);
    }, [raw, vol, spy]);

    const now = vol?.vixYear?.[vol.vixYear.length - 1];
    let body;
    if (raw === null) body = <div className="v2-aa-wait" />;
    else if (!study) body = <div className="v2-fred-na">Unavailable · try again shortly</div>;
    else {
        const { rows, cols } = study;
        const max = SPIKE_MONTHS.map((_, k) => Math.max(0.01, ...rows.map((r) => (r.at[k] && !r.at[k].live ? Math.abs(r.at[k].ret) : 0))));
        const top = Math.max(...rows.map((r) => r.peak));
        const shown = open === undefined ? rows[0].date : open;
        const sum = (label, f, show, base) => (
            <div className={`v2-vs-sum${base ? ' is-base' : ''}`}>
                <span>{label}</span>
                {cols.map((c, k) => {
                    const v = f(base ? c.base : c.stats);
                    return <b key={COLS[k]} className={base ? '' : better(v, f(c.base))}>{v == null ? '—' : show(v)}</b>;
                })}
            </div>
        );
        const median = (s) => s?.median, win = (s) => s?.win, share = (v) => `${Math.round(v * 100)}%`;
        body = (
            <>
                <div className="v2-vs" role="group" aria-label="SPY after each VIX spike">
                    <div className="v2-vs-head"><span>{rows.length} spike{rows.length === 1 ? '' : 's'}</span><span>Peak</span>{COLS.map((c) => <span key={c}>{c}</span>)}</div>
                    {sum('Median', median, pct)}
                    {sum('any day', median, pct, true)}
                    {sum('Rose', win, share)}
                    {sum('any day', win, share, true)}
                    <div className="v2-vs-rule" />
                    {rows.map((r) => (
                        <Fragment key={r.date}>
                            <button type="button" className={`v2-vs-row${r.date === shown ? ' is-on' : ''}`} aria-expanded={r.date === shown}
                                aria-label={`${day(r.date)} · peak ${r.peak.toFixed(1)} · ${COLS.map((c, k) => `${c} ${r.at[k] ? `${pct(r.at[k].ret)}${r.at[k].live ? ' so far' : ''}` : '—'}`).join(', ')}`}
                                onClick={() => setOpen(r.date === shown ? null : r.date)}>
                                <span className="v2-vs-when">{monYear(r.date)}<i><b style={{ width: `${(r.peak / top) * 100}%` }} /></i></span>
                                <span className="v2-vs-peak">{r.peak.toFixed(0)}</span>
                                {r.at.map((c, k) => <Cell key={COLS[k]} c={c} max={max[k]} />)}
                            </button>
                            {r.date === shown && <Year row={r} />}
                        </Fragment>
                    ))}
                </div>
                <p className="v2-aa-foot">Spike = VIX&apos;s first close at 30+ after 3 calm months · SPY price only</p>
                <p className="v2-aa-foot">Peak = its high in the next 3 months · dashed = so far</p>
            </>
        );
    }

    if (typeof document === 'undefined') return null;
    return createPortal(
        <div ref={backRef} className="v2-aa-back" role="dialog" aria-modal="true" aria-labelledby="v2-vs-title"
            onClick={(e) => { if (e.target === backRef.current) onClose(); }}>
            <div className="v2-aa" ref={sheetRef} data-sheet-scroll="">
                <div className="v2-aa-head">
                    <div className="v2-aa-title">
                        <h2 id="v2-vs-title">After VIX spikes</h2>
                        <p>{Number.isFinite(now) ? `VIX ${now.toFixed(1)} now · ${now >= SPIKE_LINE ? 'a spike is on' : `${(SPIKE_LINE - now).toFixed(1)} below 30`}` : 'Every first close at 30+'}</p>
                    </div>
                    <button type="button" className="v2-aa-close sheet-x" onClick={onClose} aria-label="Close">×</button>
                </div>
                {body}
            </div>
        </div>,
        document.querySelector('.v2') || document.body,
    );
}
