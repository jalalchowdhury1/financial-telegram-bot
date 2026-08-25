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
 * Double-click the value (or click/tap the dot) to see what it was before.
 *
 * THE POPOVER IS PORTALLED TO document.body AND POSITIONED FIXED. It must not live
 * inside the card: `.card` sets backdrop-filter, which creates a stacking context, so
 * an absolutely-positioned child is trapped there and the NEXT card paints over it.
 * (globals.css already carries a comment from a previous run-in with this bug.)
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';

const MAX_SPARK = 8;

function fmtDate(iso) {
    if (!iso) return null;
    const d = new Date(`${iso}T00:00:00`);
    if (Number.isNaN(d.getTime())) return null;
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
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
 * @param {object}   props
 * @param {object=}  props.mark      a `markFor` result, or null/undefined for no mark
 * @param {string=}  props.format    how the PREVIOUS value should be rendered (defaults to toString)
 * @param {node}     props.children  the already-formatted current value
 */
export default function Delta({ mark, format, className, children }) {
    const btnRef = useRef(null);
    const popRef = useRef(null);
    const [open, setOpen] = useState(false);
    // No IntersectionObserver (jsdom, very old browsers) means no scroll trigger —
    // settle into the rested state immediately rather than never showing the mark.
    const [seen, setSeen] = useState(() => typeof IntersectionObserver === 'undefined');
    const marked = !!mark;

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

    if (!marked) return <span className={className}>{children}</span>;

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
            aria-label={`${prevText} before this change. Activate to see details.`}
            onDoubleClick={(e) => { e.preventDefault(); setOpen((o) => !o); }}
            onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen((o) => !o); }
            }}
        >
            <span className={`mark-num${seen ? ' seen' : ''}`}>{children}</span>
            <span
                className={`mark-glyph${seen ? ' seen' : ''}`}
                onClick={(e) => { e.stopPropagation(); setOpen((o) => !o); }}
            >
                {isMove ? (mark.dir > 0 ? '⌃' : '⌄') : <span className="mark-dot" />}
            </span>

            {open && typeof document !== 'undefined' && createPortal(
                <div
                    ref={popRef}
                    className="mark-pop"
                    role="dialog"
                    onClick={(e) => e.stopPropagation()}
                    onDoubleClick={(e) => e.stopPropagation()}
                >
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
                </div>,
                document.body,
            )}
        </span>
    );
}
