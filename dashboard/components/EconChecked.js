'use client';
import { useState } from 'react';
import { freshnessNote } from '../lib/freshness';

/**
 * Header line "Econ checked 4:58 PM": when /api/fred last ran (New York time), not the
 * data's date — that used to read "Economic data as of …", which it never was. A tap
 * reveals the newest fresh print ("newest print: Initial Claims · Oct 23").
 */
const NAMES = {
    sahmRule: 'Sahm Rule', sentiment: 'Consumer Sentiment', claims: 'Initial Claims',
    creditSpread: 'BBB Credit Spread', realYields: 'Real Yields', copperGold: 'Copper/Gold',
};

/** The newest print among the econ numbers that are neither stale nor missing, or null. */
export function newestPrint(fred) {
    const rows = [
        ...Object.entries(fred?.indicators || {}).map(([k, x]) => [NAMES[k], x]),
        ...Object.values(fred?.checklist || {}).map((x) => [x?.label, x]),
        ['Yield Curve', fred?.yieldCurve && { ...fred.yieldCurve, value: fred.yieldCurve.current }],
    ];
    let best = null;
    for (const [name, x] of rows) {
        const day = /^\d{4}-\d{2}-\d{2}/.exec(x?.asOf || '')?.[0];
        if (!name || !day || freshnessNote(x).tone !== 'fresh') continue;
        if (!best || day > best.day) best = { name, day };
    }
    // a bare date is a calendar day: read at UTC noon so no device timezone moves it
    return best && { name: best.name, date: new Date(`${best.day}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' }) };
}

export default function EconChecked({ fred }) {
    const [open, setOpen] = useState(false);
    const newest = newestPrint(fred);
    const label = `Econ checked ${new Date(fred._meta.fetchedAt).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/New_York' })}`;
    const style = { fontSize: '0.7rem', opacity: 0.6, marginTop: '6px' };
    return (
        <>
            <p className="subtitle data-asof" style={style}>
                {newest ? <button type="button" className="data-asof-btn" onClick={() => setOpen((o) => !o)} aria-expanded={open}>{label}</button> : label}
                <span className="hide-sm"> · every 3 h · tap a number for its chart</span>
            </p>
            {open && newest && <p className="subtitle data-asof data-asof-newest" style={style}>newest print: {newest.name} · {newest.date}</p>}
        </>
    );
}
