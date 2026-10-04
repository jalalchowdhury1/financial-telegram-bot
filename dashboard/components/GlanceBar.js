'use client';
/**
 * 🔝 Glance bar. Once Market Pulse scrolls off the top, a slim glass capsule slides in:
 * "SPY 769.64 ▲0.74% · F&G 31 · 3 min ago ↻". From 5,000px down (Volatility, Markets) the
 * live state and the refresh used to be a long thumb-scroll away.
 *  - Numbers come from the page's own state (lib/glance.js) — no fetch of its own.
 *  - The age is the header's: <UpdatedAgo> (amber past 10 min), or "🕐 Saved 10:42" while
 *    the page shows a saved copy and nothing live has landed.
 *  - Tap the numbers → back to the top. ↻ = the header's refresh (skips the edge cache).
 *  - Hidden near the top; CSS also hides it while a sheet (html.sheet-open), the jump menu
 *    or the offline banner is up. Reduced motion: no slide.
 */
import { useEffect, useState } from 'react';
import { UpdatedAgo } from './PhonePolish';
import { glanceNumbers, glanceAge, shouldShowGlance } from '../lib/glance';

// Market Pulse is the anchor; if it is not on the page (a crashed card), the header is.
const ANCHORS = ['.market-pulse', '.dashboard-header'];

export default function GlanceBar({ spy, spyDailyMove, fg, fgColor, updatedAt, saved, loading, onRefresh, busy }) {
    const [on, setOn] = useState(false);

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

    const n = glanceNumbers({ spy, spyDailyMove, fg });
    if (!n) return null;
    const age = glanceAge({ updatedAt, saved, loading });
    const toTop = () => {
        const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
        window.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
    };
    const said = `SPY ${n.price}${n.move ? ` ${n.move.text}` : ''}${n.fg != null ? `, Fear & Greed ${n.fg}` : ''}`;

    return (
        <div className={`glance-bar${on ? ' is-on' : ''}`} role="region" aria-label="Glance bar">
            <button type="button" className="glance-main" onClick={toTop} tabIndex={on ? 0 : -1} aria-label={`${said} — back to top`}>
                <span className="glance-k">SPY</span> <b>{n.price}</b>
                {n.move && <> <span className={n.move.up ? 'stat-positive' : 'stat-negative'}>{n.move.text}</span></>}
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
                    {age.saved ? <span className="glance-saved">🕐 Saved {age.saved}</span> : <UpdatedAgo at={age.at} />}
                </>
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
