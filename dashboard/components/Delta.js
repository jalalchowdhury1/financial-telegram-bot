'use client';
/**
 * Delta — wraps a rendered number and marks it when its change is NEWS.
 *
 * Two marks, one colour: a filled dot for a new print, a chevron for an outsized
 * (2σ) move. Cyan is the only hue the dashboard's semantic palette hasn't claimed,
 * so a mark can never be misread as bullish/bearish/stale.
 *
 * The mark breathes three times as it scrolls into view, then rests. Firing on
 * scroll-into-view is deliberate — scanning is exactly when the owner is looking —
 * and nothing pulses forever, so a page with four marks never strobes.
 *
 * Tap / click the value (or the dot) to see what it was before — and, for any number
 * with history-sheet data (`chartKey`), its chart (2026-09-26; range chips 2026-10-09; decades
 * of baked history before the sheet, lib/longHistory.js, 2026-10-10). A number with a
 * chart but no mark gets a faint dotted underline so it reads as tappable.
 *
 * THE POPOVER IS PORTALLED TO document.body AND POSITIONED FIXED. It must not live
 * inside the card: `.card` sets backdrop-filter, which creates a stacking context, so
 * an absolutely-positioned child is trapped there and the NEXT card paints over it.
 * (globals.css already carries a comment from a previous run-in with this bug.)
 */
import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { useChart } from './MarkProvider';
import { RANGES, getRange, setRange, sliceRange, rangeLabel, rangesFor, pickFor } from '../lib/chartRange';
import { longInfo, loadLong, peekLong, joinLong, cutFor, spanDays } from '../lib/longHistory';

const MAX_SPARK = 8;

function fmtDate(iso, long = false) {
    if (!iso) return null;
    const d = new Date(`${iso}T00:00:00`);
    if (Number.isNaN(d.getTime())) return null;
    // past ~10 months "Oct 9" is ambiguous: month + year instead
    return d.toLocaleDateString('en-US', long ? { month: 'short', year: 'numeric' } : { month: 'short', day: 'numeric' });
}

/** Sparkline of recent values, with the newest point emphasised. */
function Spark({ runs, dir }) {
    if (!Array.isArray(runs) || runs.length < 2) return null;
    const pts = runs.slice(-MAX_SPARK).filter((v) => Number.isFinite(v));
    if (pts.length < 2) return null;
    const w = 218, h = 26;
    const lo = Math.min(...pts), hi = Math.max(...pts), span = (hi - lo) || 1;
    const x = (i) => (i / (pts.length - 1)) * (w - 6) + 3;
    const y = (v) => h - 4 - ((v - lo) / span) * (h - 8);
    const coords = pts.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
    const stroke = dir > 0 ? 'var(--green)' : 'var(--red)';
    return (
        <svg className="mark-spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
            <polyline points={coords.slice(0, -1).join(' ')} fill="none"
                stroke="rgba(148,163,184,.55)" strokeWidth="1.4" strokeLinejoin="round" />
            <polyline points={coords.slice(-2).join(' ')} fill="none"
                stroke={stroke} strokeWidth="1.8" strokeLinecap="round" />
            <circle cx={x(pts.length - 1)} cy={y(pts[pts.length - 1])} r="2.6" fill={stroke} />
            <circle cx={x(pts.length - 2)} cy={y(pts[pts.length - 2])} r="1.8" fill="rgba(148,163,184,.75)" />
        </svg>
    );
}

const fmtNum = (v) => v.toLocaleString('en-US', { maximumFractionDigits: Math.abs(v) >= 100 ? 0 : 2 });

/**
 * Line of the window the range chips picked: low / high, first → last, and the dates. Drawn in a
 * neutral colour: "up" is bad news for VIX, claims or spreads, so green/red would mislead.
 * `joinAt` (the date the sheet's snapshots take over from baked history) gets a faint dotted
 * rule when it falls inside the window.
 */
export function SeriesChart({ chart, format, joinAt }) {
    const pts = chart?.points || [];
    if (pts.length < 2) return null;
    const f = (v) => { try { return format ? format(v) : fmtNum(v); } catch { return fmtNum(v); } };
    const w = 218, h = 64;
    const t0 = Date.parse(`${pts[0].date}T00:00:00Z`);
    const span = Math.max(1, Date.parse(`${pts[pts.length - 1].date}T00:00:00Z`) - t0);
    const vals = pts.map((p) => p.value);
    const lo = Math.min(...vals), hi = Math.max(...vals), vr = (hi - lo) || 1;
    const x = (p) => ((Date.parse(`${p.date}T00:00:00Z`) - t0) / span) * (w - 6) + 3;
    const y = (v) => h - 4 - ((v - lo) / vr) * (h - 8);
    const first = pts[0], last = pts[pts.length - 1];
    const long = span > 300 * 86400000;
    const jx = joinAt && pts[0].long && joinAt > first.date && joinAt <= last.date ? x({ date: joinAt }) : null;
    return (
        <div className="series-chart">
            <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
                {jx != null && <line className="series-join" x1={jx} x2={jx} y1="0" y2={h}
                    stroke="rgba(148,163,184,.4)" strokeWidth="1" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />}
                <polyline points={pts.map((p) => `${x(p).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ')}
                    fill="none" stroke="var(--mark, #22d3ee)" strokeWidth="1.6" strokeLinejoin="round" />
                <circle cx={x(last)} cy={y(last.value)} r="2.6" fill="var(--mark, #22d3ee)" />
            </svg>
            <div className="series-range">
                <span>low {f(lo)}</span><span>high {f(hi)}</span>
            </div>
            <div className="series-dates">
                <span>{fmtDate(first.date, long)}: {f(first.value)}</span>
                <span>{fmtDate(last.date, long)}: {f(last.value)}</span>
            </div>
        </div>
    );
}

/**
 * Position a portalled popover against its trigger: prefer below, flip above when the
 * viewport runs out, clamp horizontally, and aim the arrow at the trigger's centre.
 */
function place(el, btn) {
    if (!el || !btn) return;
    const r = btn.getBoundingClientRect();
    const pw = el.offsetWidth, ph = el.offsetHeight;
    const GAP = 10, PAD = 12;
    let top = r.bottom + GAP, flip = false;
    if (top + ph > window.innerHeight - PAD) {
        if (r.top - GAP - ph > PAD) { top = r.top - GAP - ph; flip = true; }
        else { top = Math.max(PAD, window.innerHeight - ph - PAD); }
    }
    const left = Math.min(Math.max(PAD, r.right - pw), Math.max(PAD, window.innerWidth - pw - PAD));
    el.style.top = `${top}px`;
    el.style.left = `${left}px`;
    el.classList.toggle('flip', flip);
    el.style.setProperty('--ax', `${Math.min(Math.max(10, r.left + r.width / 2 - left - 4), pw - 18)}px`);
}

/** Format the delta between prev and value, matching the precision of the rendered text. */
function fmtDelta(mark) {
    const d = mark.value - mark.prev;
    const mag = Math.abs(d);
    const dp = mag >= 100 ? 0 : mag >= 1 ? 2 : 3;
    return `${d > 0 ? '+' : '−'}${mag.toFixed(dp)}`;
}

/**
 * A stat's baked long history (lib/longHistory.js), fetched when its popover opens.
 * undefined while loading, null when there is none or it failed, else the points.
 */
function useLong(key) {
    const has = !!longInfo(key);
    const [pts, setPts] = useState(() => (has ? peekLong(key) : null));
    useEffect(() => {
        if (!has) { setPts(null); return undefined; }
        let live = true;
        loadLong(key).then((p) => { if (live) setPts(p); });
        return () => { live = false; };
    }, [key, has]);
    return pts;
}

/**
 * The popover's chart section: 1M · 3M · 6M · 1Y · 5Y · MAX chips (styled like the cards'
 * timeframe pills; only the ones this stat's history can fill), the eyebrow, the line. Mounts
 * only while the popover is open, so it reads the remembered pick fresh each time — a chip
 * picked on one number carries to the next.
 */
export function ChartBlock({ chart, format, marked, onRangeChange }) {
    const info = longInfo(chart.key);
    const long = useLong(chart.key);
    const all = useMemo(() => joinLong(chart.points, long, info), [chart.points, long, info]);
    const chips = rangesFor(spanDays(chart.points, info));
    const [range, setLocal] = useState(getRange);
    const shown = pickFor(range, chips);
    const pts = sliceRange(all, shown);
    const cut = info ? cutFor(chart.points, info) : null;
    const pick = (id) => {
        setRange(id);
        setLocal(id);
        if (onRangeChange) window.requestAnimationFrame?.(onRangeChange);
    };
    // Footer: where the line comes from. "loading…" only while the window needs the baked part.
    const r = RANGES.find((x) => x.id === shown);
    const last = chart.points[chart.points.length - 1]?.date;
    const wantsOld = !!(info && cut && last && (r.days === Infinity
        || new Date(Date.parse(`${last}T00:00:00Z`) - (r.days - 1) * 86400000).toISOString().slice(0, 10) < cut));
    let foot = 'daily snapshots · history sheet';
    if (pts[0]?.long) foot = `${info.source} · snapshots from ${fmtDate(cut)}`;
    else if (wantsOld && long === undefined) foot = `loading history since ${info.from.slice(0, 4)}…`;
    return (
        <>
            {marked && <div className="mark-pop-sep" />}
            <div className="series-head">
                <div className="mark-pop-eyebrow">{chart.label} · {rangeLabel(pts, shown)}</div>
                <div className="series-tfs" role="group" aria-label="Chart range">
                    {chips.map((c) => (
                        <button key={c.id} type="button"
                            className={`series-tf${c.id === shown ? ' is-on' : ''}`}
                            aria-pressed={c.id === shown}
                            onClick={(e) => { e.stopPropagation(); pick(c.id); }}>{c.id}</button>
                    ))}
                </div>
            </div>
            <SeriesChart chart={{ ...chart, points: pts }} format={format} joinAt={cut} />
            <div className="mark-pop-foot">{foot}</div>
        </>
    );
}

/**
 * @param {object}   props
 * @param {object=}  props.mark      a `markFor` result, or null/undefined for no mark
 * @param {string=}  props.format    how the PREVIOUS value should be rendered (defaults to toString)
 * @param {node}     props.children  the already-formatted current value
 */
export default function Delta({ mark, format, className, children, chartKey, raw }) {
    const btnRef = useRef(null);
    const popRef = useRef(null);
    const [open, setOpen] = useState(false);
    // No IntersectionObserver (jsdom, very old browsers) means no scroll trigger —
    // settle into the rested state immediately rather than never showing the mark.
    const [seen, setSeen] = useState(() => typeof IntersectionObserver === 'undefined');
    const marked = !!mark;
    const chart = useChart(chartKey, raw);

    // announce once, on entering view
    useEffect(() => {
        const el = btnRef.current;
        if (!marked || !el || seen) return;
        const io = new IntersectionObserver((entries) => {
            entries.forEach((e) => { if (e.isIntersecting) { setSeen(true); io.unobserve(e.target); } });
        }, { threshold: 0.9 });
        io.observe(el);
        return () => io.disconnect();
    }, [marked, seen]);

    const reposition = useCallback(() => {
        if (!popRef.current || !btnRef.current) return;
        const r = btnRef.current.getBoundingClientRect();
        if (r.bottom < 0 || r.top > window.innerHeight) { setOpen(false); return; }
        place(popRef.current, btnRef.current);
    }, []);

    useEffect(() => {
        if (!open) return;
        reposition();
        const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
        const onDown = (e) => {
            if (popRef.current?.contains(e.target) || btnRef.current?.contains(e.target)) return;
            setOpen(false);
        };
        window.addEventListener('scroll', reposition, true);
        window.addEventListener('resize', reposition);
        document.addEventListener('keydown', onKey);
        document.addEventListener('mousedown', onDown);
        return () => {
            window.removeEventListener('scroll', reposition, true);
            window.removeEventListener('resize', reposition);
            document.removeEventListener('keydown', onKey);
            document.removeEventListener('mousedown', onDown);
        };
    }, [open, reposition]);

    if (!marked && !chart) return <span className={className}>{children}</span>;

    // One tap toggles. The 2nd/3rd click of a double/triple click is ignored (e.detail), so a
    // double-click opens once instead of open → close (→ open, replaying the animation).
    const toggle = (e) => { if (e?.detail > 1) return; setOpen((o) => !o); };
    const chartBlock = chart && <ChartBlock chart={chart} format={format} marked={marked} onRangeChange={reposition} />;
    const popover = (body) => (open && typeof document !== 'undefined' ? createPortal(
        <div
            ref={popRef}
            className="mark-pop"
            role="dialog"
            aria-label={chart ? `${chart.label} chart` : 'Previous value'}
            onClick={(e) => e.stopPropagation()}
        >
            {body}
        </div>,
        document.body,
    ) : null);

    if (!marked) {
        return (
            <span
                ref={btnRef}
                role="button"
                tabIndex={0}
                className={`chartable ${className || ''}`}
                data-open={open ? 'true' : undefined}
                aria-haspopup="dialog"
                aria-expanded={open}
                title="Tap for its chart"
                onClick={toggle}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } }}
            >
                {children}
                {popover(chartBlock)}
            </span>
        );
    }

    const isMove = mark.kind === 'move';
    const prevText = format ? format(mark.prev) : String(mark.prev);
    const held = mark.heldDays != null
        ? `held ${mark.heldDays} day${mark.heldDays === 1 ? '' : 's'}${fmtDate(mark.heldFrom) ? ` · since ${fmtDate(mark.heldFrom)}` : ''}`
        : 'Moved more than 2σ of its own daily range';

    return (
        <span
            ref={btnRef}
            role="button"
            tabIndex={0}
            className={`mark ${className || ''}`}
            data-mark={mark.kind}
            data-open={open ? 'true' : undefined}
            aria-label={`${prevText} before this change. Activate to see details${chart ? ' and its chart' : ''}.`}
            aria-expanded={open}
            onClick={toggle}
            onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen((o) => !o); }
            }}
        >
            <span className={`mark-num${seen ? ' seen' : ''}`}>{children}</span>
            <span
                className={`mark-glyph${seen ? ' seen' : ''}`}
                onClick={(e) => { e.stopPropagation(); toggle(e); }}
            >
                {isMove ? (mark.dir > 0 ? '⌃' : '⌄') : <span className="mark-dot" />}
            </span>

            {popover(
                <>
                    <div className="mark-pop-eyebrow">{isMove ? 'Yesterday' : 'Before this print'}</div>
                    <div className="mark-pop-row">
                        <span className="mark-pop-prev">{prevText}</span>
                        <span className="mark-pop-delta" style={{ color: mark.dir > 0 ? 'var(--green)' : 'var(--red)' }}>
                            {mark.dir > 0 ? '▲' : '▼'} {fmtDelta(mark)}
                        </span>
                    </div>
                    <div className="mark-pop-held">{held}</div>
                    <Spark runs={mark.runs} dir={mark.dir} />
                    <div className="mark-pop-foot">
                        {isMove ? 'last sessions' : `last ${Math.min(mark.runs?.length || 0, MAX_SPARK)} prints`}
                    </div>
                    {chartBlock}
                </>,
            )}
        </span>
    );
}
