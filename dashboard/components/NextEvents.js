'use client';
/**
 * 🔔 Next releases — one quiet line under the market clock pill:
 *   "Next · Jobs Fri 8:30 · CPI Wed 8:30 · FOMC Oct 28 2:00 ET"
 * On release day, until the time, that piece turns amber ("🔔 CPI today 8:30 ET · in 1h 10m");
 * after it, "CPI out 8:30".
 * Dates are hand-copied from federalreserve.gov + bls.gov (lib/econCalendar.js). `now` comes
 * from MarketClock, so it ticks with the clock and shows nothing before the page has mounted.
 */
import { econLine } from '../lib/econCalendar';

export default function NextEvents({ now }) {
    if (now == null) return null;
    let line = null;
    try { line = econLine(now); } catch { return null; }
    if (!line) return null;
    return (
        <div className={`econ-line${line.alert ? ' is-today' : ''}`}
            title="US release times in New York time: jobs + CPI 8:30 am (bls.gov), FOMC statement 2:00 pm (federalreserve.gov)">
            {(line.segments || [{ text: line.text, hot: false }]).map((g, i) => (
                <span key={i} className={g.hot ? 'econ-hot' : undefined}>{i > 0 ? ' · ' : ''}{g.text}</span>
            ))}
        </div>
    );
}
