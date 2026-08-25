'use client';
/**
 * MarkProvider — the single place that decides which numbers are marked.
 *
 * Two sources feed it, because the dashboard's metrics split cleanly in two:
 *   1. /api/history — baselines for the 22 metrics carried by the history sheet.
 *   2. /api/fred    — already ships full `history[]` arrays for spEps and the Four
 *                     Horsemen, so those need no sheet and no extra request.
 *
 * Consumers call `useMark(key, liveValue)` and get either a mark object or null.
 * They never see where the baseline came from.
 */
import { createContext, useContext, useMemo } from 'react';
import { historyFor, markFor, todayET } from '../lib/marks';

const MarkContext = createContext({ entries: {}, today: null });

/** A FRED `history[]` ([{date,value}]) → the same shape /api/history returns. */
function fromFredHistory(history, today) {
    if (!Array.isArray(history) || history.length < 2) return null;
    const series = history
        .filter((p) => p && p.date && Number.isFinite(Number(p.value)))
        .map((p) => ({ date: String(p.date).slice(0, 10), value: Number(p.value) }))
        .sort((a, b) => (a.date < b.date ? -1 : 1));
    return historyFor(series, today);
}

export function MarkProvider({ history, fred, children }) {
    const value = useMemo(() => {
        const today = history?.today || todayET();
        const entries = { ...(history?.metrics || {}) };

        // Metrics the sheet does not carry, derived from data already on the page.
        const fredSeries = {
            spEps: fred?.spEps?.history,
            unemployment: fred?.horsemen?.unemployment?.history,
            bankruptcies: fred?.horsemen?.bankruptcies?.history,
            hClaims: fred?.horsemen?.claims?.history,
        };
        for (const [key, hist] of Object.entries(fredSeries)) {
            const entry = fromFredHistory(hist, today);
            if (entry) entries[key] = { kind: 'print', ...entry };
        }
        return { entries, today };
    }, [history, fred]);

    return <MarkContext.Provider value={value}>{children}</MarkContext.Provider>;
}

/**
 * @param {string} key    a metric key from lib/marks.js
 * @param {number} live   the value currently being rendered
 * @returns {object|null} a mark, or null when this number's change is not news
 */
export function useMark(key, live) {
    const { entries, today } = useContext(MarkContext);
    return useMemo(
        () => markFor(key, Number(live), entries[key], today),
        [key, live, entries, today],
    );
}

/** Counts for the header chip: `{ print, move, total }`. */
export function useMarkCounts(values) {
    const { entries, today } = useContext(MarkContext);
    return useMemo(() => {
        let print = 0, move = 0;
        for (const [key, live] of Object.entries(values || {})) {
            const m = markFor(key, Number(live), entries[key], today);
            if (!m) continue;
            if (m.kind === 'move') move++; else print++;
        }
        return { print, move, total: print + move };
    }, [values, entries, today]);
}
