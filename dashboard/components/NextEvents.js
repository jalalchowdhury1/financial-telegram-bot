'use client';
/**
 * 🔔 Next releases — one quiet line under the market clock pill:
 *   "Next · Jobs Fri 8:30 · CPI Wed 8:30 · FOMC Oct 28 2:00 ET"
 * On release day, until the time, that piece turns amber ("🔔 CPI today 8:30 ET · in 1h 10m");
 * after it, "CPI out 8:30".
 * Dates are hand-copied from federalreserve.gov + bls.gov (lib/econCalendar.js). `now` comes
 * from MarketClock, so it ticks with the clock and shows nothing before the page has mounted.
 */
import { Fragment } from 'react';
import { econLine } from '../lib/econCalendar';

export default function NextEvents({ now }) {
    if (now == null) return null;
    let line = null;
    try { line = econLine(now); } catch { return null; }
    if (!line) return null;
    const segs = line.segments || [{ text: line.text, hot: false }];
    // Each release is one nowrap piece and carries the " ·" after it; the plain space between pieces is
    // the only place the line may break, so it never splits "CPI" from "Nov 10 8:30 ET".
    return (
        <div className={`econ-line${line.alert ? ' is-today' : ''}`}
            title="US release times in New York time: jobs + CPI 8:30 am (bls.gov), FOMC statement 2:00 pm (federalreserve.gov)">
            {segs.map((g, i) => (
                <Fragment key={i}>
                    <span className={`econ-seg${g.hot ? ' econ-hot' : ''}`}>{g.text}</span>
                    {i < segs.length - 1 && <><span className="econ-sep"> ·</span>{' '}</>}
                </Fragment>
            ))}
        </div>
    );
}
