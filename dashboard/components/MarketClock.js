'use client';
/**
 * 🕰️ Market clock pill — "Open · closes in 2h 10m", "Closed · opens Mon 9:30 ET".
 * lib/marketClock.js does the maths on the device (no network). Renders nothing until
 * mounted: the page is prerendered, and a time baked in at build would be wrong and
 * would not match the client's first render.
 */
import { useEffect, useState } from 'react';
import { marketStatus, clockLabel } from '../lib/marketClock';

export const CLOCK_TICK_MS = 30 * 1000;

export default function MarketClock() {
    const [now, setNow] = useState(null);
    useEffect(() => {
        setNow(Date.now());
        const id = setInterval(() => setNow(Date.now()), CLOCK_TICK_MS);
        const onVisible = () => { if (!document.hidden) setNow(Date.now()); };
        document.addEventListener('visibilitychange', onVisible);
        return () => { clearInterval(id); document.removeEventListener('visibilitychange', onVisible); };
    }, []);
    if (now == null) return null;
    let st = null;
    try { st = marketStatus(now); } catch { return null; }
    return (
        <span className={`mkt-clock is-${st.state}`} title="NYSE regular session: 9:30 am – 4:00 pm ET, holidays from nyse.com">
            <span className="mkt-dot" aria-hidden="true" />
            {clockLabel(st)}
        </span>
    );
}
