'use client';
/**
 * 🔝 Glance bar. Once Market Pulse scrolls off the top, a slim glass capsule slides in:
 * "SPY 769.64 ▲0.74% · F&G 31 · 3 min ago ↻". From 5,000px down (Volatility, Markets) the
 * live state and the refresh used to be a long thumb-scroll away.
 *  - Numbers come from the page's own state (lib/glance.js) — no fetch of its own. The move
 *    carries the SPY card's session word off the day ("▲0.74% Fri"). F&G down = SPY + ↻ stay.
 *  - The age is the header's: <UpdatedAgo> (amber past 10 min), or "🕐 Saved 10:42" while
 *    the page shows a saved copy and nothing live has landed.
 *  - Tap the numbers → back to the top, noted like ☰ → ↑ Top so the ↩ pill can bring him
 *    back. ↻ = the header's refresh (skips the edge cache).
 *  - ● N (cyan) = the header's "new prints" chip in miniature: each tap walks to the next
 *    marked number, sharing the chip's cursor. Only there when something is lit.
 *  - Hidden near the top; CSS also hides it while a sheet (html.sheet-open), the jump menu
 *    or the offline banner is up. Reduced motion: no slide.
 */
import { useEffect, useState } from 'react';
import { UpdatedAgo } from './PhonePolish';
import { glanceNumbers, glanceAge, shouldShowGlance } from '../lib/glance';
import { noteJumpFrom } from './JumpNav';
import { useMarkCounts } from './MarkProvider';
import { jumpToNextMark, markSummary } from './MarkChip';

// Market Pulse is the anchor; if it is not on the page (a crashed card), the header is.
const ANCHORS = ['.market-pulse', '.dashboard-header'];

export default function GlanceBar({ spy, spyDailyMove, fg, fgColor, when, updatedAt, saved, loading, onRefresh, busy, markValues }) {
    const [on, setOn] = useState(false);
    const marks = useMarkCounts(markValues);

    useEffect(() => {
        // one layout read per frame, however many scroll events fire
        let pending = false;
        let raf = 0;
        const check = () => {
            pending = false;
            const a = ANCHORS.map((s) => document.querySelector(s)).find(Boolean);
            setOn(a ? shouldShowGlance(a.getBoundingClientRect().bottom) : false);
        };
        const onScroll = () => {
            if (pending) return;
            pending = true;
            raf = window.requestAnimationFrame(check);
        };
        check();
        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('resize', onScroll);
        return () => {
            window.removeEventListener('scroll', onScroll);
            window.removeEventListener('resize', onScroll);
            if (pending) window.cancelAnimationFrame?.(raf);
        };
    }, []);

    const n = glanceNumbers({ spy, spyDailyMove, fg, when });
    if (!n) return null;
    const age = glanceAge({ updatedAt, saved, loading });
    const toTop = () => {
        const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
        noteJumpFrom();
        window.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
    };
    const moveSaid = n.move ? ` ${n.move.text}${n.move.when ? ` ${n.move.when}` : ''}` : '';
    const said = `SPY ${n.price}${moveSaid}${n.fg != null ? `, Fear & Greed ${n.fg}` : ''}`;

    return (
        <div className={`glance-bar${on ? ' is-on' : ''}`} role="region" aria-label="Glance bar">
            <button type="button" className="glance-main" onClick={toTop} tabIndex={on ? 0 : -1} aria-label={`${said} — back to top`}>
                <span className="glance-k">SPY</span> <b>{n.price}</b>
                {n.move && <> <span className={n.move.up ? 'stat-positive' : 'stat-negative'}>{n.move.text}</span></>}
                {n.move?.when && <span className="glance-when"> <span className="glance-k">{n.move.when}</span></span>}
                {n.fg != null && (
                    <>
                        <span className="glance-sep" aria-hidden="true"> · </span>
                        <span className="glance-k">F&amp;G</span> <b style={{ color: fgColor?.(n.fgScore) }}>{n.fg}</b>
                    </>
                )}
            </button>
            {age && (
                <>
                    <span className="glance-sep" aria-hidden="true"> · </span>
                    {age.saved ? <span className="glance-saved">🕐 <span className="glance-saved-word">Saved </span>{age.saved}</span> : <UpdatedAgo at={age.at} />}
                </>
            )}
            {marks.total > 0 && (
                <button
                    type="button"
                    className="glance-mark"
                    onClick={jumpToNextMark}
                    tabIndex={on ? 0 : -1}
                    aria-label={`Jump to the next changed number (${markSummary(marks)})`}
                    title={`${markSummary(marks)} — tap for the next one`}
                >
                    <span className="mark-chip-dot" aria-hidden="true" />{marks.total}
                </button>
            )}
            <button
                type="button"
                className="glance-refresh"
                onClick={onRefresh}
                disabled={busy}
                tabIndex={on ? 0 : -1}
                aria-label="Refresh now"
                title="Refresh all data (R)"
            >
                <svg className={busy ? 'spinning' : ''} width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <polyline points="23 4 23 10 17 10" />
                    <polyline points="1 20 1 14 7 14" />
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                </svg>
            </button>
        </div>
    );
}
