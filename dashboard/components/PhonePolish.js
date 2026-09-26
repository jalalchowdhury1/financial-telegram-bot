'use client';
/**
 * 📱 Phone polish — three small pieces the page mounts once:
 *  - <UpdatedAgo>      "3 min ago" that keeps ticking (turns amber past 10 min)
 *  - <OfflineBanner>   a clear bar while the phone is offline; refreshes on reconnect
 *  - <PullToRefresh>   pull down at the top of the page → same refresh as the ↻ button
 * Each owns its own timer / listeners, so the ticking clock never re-renders the cards.
 */
import { useEffect, useRef, useState } from 'react';

export const STALE_AFTER_MS = 10 * 60 * 1000;
export const TICK_MS = 15 * 1000;

export function agoLabel(at, now = Date.now()) {
    const s = Math.max(0, Math.round((now - at) / 1000));
    if (s < 60) return 'just now';
    const m = Math.floor(s / 60);
    if (m < 60) return `${m} min ago`;
    const h = Math.floor(m / 60);
    return h < 24 ? `${h} h ago` : `${Math.floor(h / 24)} d ago`;
}

export function UpdatedAgo({ at }) {
    const [now, setNow] = useState(() => Date.now());
    useEffect(() => {
        setNow(Date.now());
        const id = setInterval(() => setNow(Date.now()), TICK_MS);
        // A phone tab wakes from sleep with a stale clock — catch up at once.
        const onVisible = () => { if (!document.hidden) setNow(Date.now()); };
        document.addEventListener('visibilitychange', onVisible);
        return () => { clearInterval(id); document.removeEventListener('visibilitychange', onVisible); };
    }, [at]);
    if (!Number.isFinite(at)) return null;
    return <span className={`upd-ago${now - at > STALE_AFTER_MS ? ' is-old' : ''}`}>{agoLabel(at, now)}</span>;
}

export function OfflineBanner({ onBack, since }) {
    const [offline, setOffline] = useState(false);
    const back = useRef(onBack);
    back.current = onBack;
    useEffect(() => {
        setOffline(typeof navigator !== 'undefined' && navigator.onLine === false);
        const off = () => setOffline(true);
        const on = () => { setOffline(false); back.current?.(); };
        window.addEventListener('offline', off);
        window.addEventListener('online', on);
        return () => { window.removeEventListener('offline', off); window.removeEventListener('online', on); };
    }, []);
    if (!offline) return null;
    return (
        <div className="offline-banner" role="status">
            📴 Offline — these numbers are from {since || 'your last visit'}. It refreshes by itself when you are back online.
        </div>
    );
}

export const PULL_TRIGGER_PX = 70;
const DAMP = 0.5;
const MAX_PULL = 110;
const START_PX = 10;

/** A touch that starts in a modal, the jump menu or a scrolled inner box is not a page pull. */
function ignoreTouch(target) {
    let el = target;
    for (let i = 0; el && el !== document.body && i < 15; i++, el = el.parentElement) {
        if (el.scrollTop > 0) return true;
        if (window.getComputedStyle?.(el).position === 'fixed') return true;
    }
    return false;
}

export function PullToRefresh({ onRefresh, busy }) {
    const [pull, setPull] = useState(0);
    const [fired, setFired] = useState(false);
    const refresh = useRef(onRefresh);
    refresh.current = onRefresh;
    const g = useRef({ y0: null });
    const wasBusy = useRef(false);

    useEffect(() => {
        const touch = (e) => (e.touches && e.touches[0]) || null;
        // A second finger (pinch) or a zoomed-in page is zooming/panning, never a pull.
        const zoomed = () => (window.visualViewport?.scale || 1) > 1.01;
        const reset = () => { g.current = { y0: null }; setPull(0); };
        const start = (e) => {
            const t = touch(e);
            if (!t || e.touches.length > 1 || zoomed() || window.scrollY > 0 || ignoreTouch(e.target)) { g.current = { y0: null }; return; }
            g.current = { y0: t.clientY, x0: t.clientX, pull: 0, active: false };
        };
        const move = (e) => {
            const s = g.current;
            const t = touch(e);
            if (s.y0 == null || !t) return;
            if (e.touches.length > 1 || zoomed()) { reset(); return; }
            const dy = t.clientY - s.y0;
            const dx = t.clientX - s.x0;
            if (!s.active) {
                // sideways (the factor timeline) or upward: this gesture is not a pull
                if (Math.abs(dx) > Math.abs(dy) || dy < -START_PX) { g.current = { y0: null }; return; }
                if (dy < START_PX) return;
                s.active = true;
            }
            if (window.scrollY > 0) { reset(); return; }
            s.pull = Math.min(MAX_PULL, Math.max(0, (dy - START_PX) * DAMP));
            setPull(s.pull);
        };
        const end = () => {
            const s = g.current;
            if (s.y0 != null && s.active && s.pull >= PULL_TRIGGER_PX) {
                setFired(true);
                refresh.current?.();
            }
            reset();
        };
        window.addEventListener('touchstart', start, { passive: true });
        window.addEventListener('touchmove', move, { passive: true });
        window.addEventListener('touchend', end);
        window.addEventListener('touchcancel', reset);
        return () => {
            window.removeEventListener('touchstart', start);
            window.removeEventListener('touchmove', move);
            window.removeEventListener('touchend', end);
            window.removeEventListener('touchcancel', reset);
        };
    }, []);

    // "Refreshing…" stays up until the refresh finishes (busy true → false); 15 s cap.
    useEffect(() => {
        if (wasBusy.current && !busy) setFired(false);
        wasBusy.current = busy;
    }, [busy]);
    useEffect(() => {
        if (!fired) return undefined;
        const id = setTimeout(() => setFired(false), 15000);
        return () => clearTimeout(id);
    }, [fired]);

    if (!pull && !fired) return null;
    const ready = pull >= PULL_TRIGGER_PX;
    const y = fired ? 12 : Math.min(12, pull - 40);
    return (
        <div className={`ptr${ready || fired ? ' is-ready' : ''}`} style={{ transform: `translate(-50%, ${y}px)` }} role="status">
            {fired ? '↻ Refreshing…' : ready ? '↑ Release to refresh' : '↓ Pull to refresh'}
        </div>
    );
}
