'use client';
/**
 * MarkProvider — the single place that decides which numbers are marked.
 *
 * Baselines come from /api/history, which reads the history sheet's daily snapshots.
 * Metrics with no daily snapshot cannot be marked at all — see the note in lib/marks.js
 * about why a FRED `history[]` array cannot stand in for one.
 *
 * Consumers call `useMark(key, liveValue)` and get either a mark object or null.
 * They never see where the baseline came from.
 */
import { createContext, useContext, useMemo } from 'react';
import { markFor, todayET } from '../lib/marks';

const MarkContext = createContext({ entries: {}, today: null });

export function MarkProvider({ history, children }) {
    const value = useMemo(() => ({
        entries: history?.metrics || {},
        today: history?.today || todayET(),
    }), [history]);

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

/**
 * A metric's value ONLY when it is fresh.
 *
 * The cards refuse to mark a stale or unavailable number — an old value resurfacing
 * from a cache is not a new print. The chip has to apply the SAME rule or it reports a
 * count the page cannot show: on a Sheet-last-known-good load every FRED metric is
 * stale, and the chip claimed "4 new prints" above a page with no marks on it at all.
 */
const fresh = (m) => (m && !m.stale && !m.unavailable && Number.isFinite(m.value) ? m.value : undefined);

/**
 * Flatten the dashboard's live payloads into `{ markKey: value }`, so the header chip
 * can count what is lit without every card reporting upward. Keys here MUST match the
 * keys the cards pass to `useMark`, and freshness MUST be filtered the same way, or the
 * count and the page disagree.
 */
export function collectLiveValues(fred, extra, sheets) {
    const i = fred?.indicators || {};
    const c = fred?.checklist || {};
    const re = extra?.realEstate || {};
    const ra = extra?.rates || {};
    const yc = fred?.yieldCurve, pm = fred?.profitMargin;
    return {
        sahmRule: fresh(i.sahmRule),
        sentiment: fresh(i.sentiment),
        claims: fresh(i.claims),
        creditSpread: fresh(i.creditSpread),
        realYields: fresh(i.realYields),
        copperGold: fresh(i.copperGold),
        peRatio: fred?.peRatio,
        nfci: fresh(c.nfci), m2: fresh(c.m2), retail: fresh(c.retail),
        housing: fresh(c.housing), indpro: fresh(c.indpro), jolts: fresh(c.jolts),
        durable: fresh(c.durable), savings: fresh(c.savings),
        // market-extra rows carry no staleness flag of their own
        rentIndex: re.rentIndex?.current,
        mortgagePayment: re.mortgagePayment?.current,
        mortgageRate: ra.mortgageRate?.current,
        atnhpi: re.atnhpi?.current,
        aaiiDiff: parseFloat(sheets?.AAIIDiff),
        yieldCurve: yc?.stale ? undefined : yc?.current,
        profitMargin: pm?.stale ? undefined : pm?.current,
    };
}
