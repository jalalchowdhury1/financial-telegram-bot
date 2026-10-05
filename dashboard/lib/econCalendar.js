/**
 * econCalendar.js — 🔔 the next market-moving US releases (jobs report, CPI, Fed decision),
 * as one quiet line under the market clock. Hand-copied from the official schedules, exactly
 * like lib/marketClock.js does for NYSE holidays: no network call, nothing to fail.
 *
 * Sources, read 2026-10-04:
 *  - FOMC  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
 *          (page "Last Update: September 16, 2026"; lists 2026 and 2027). The date kept is the
 *          meeting's LAST day, when the statement comes out at 2:00 p.m. ET ("For release at
 *          2:00 p.m.", e.g. https://www.federalreserve.gov/newsevents/pressreleases/monetary20260729a.htm).
 *          The Fed: "Each meeting date is tentative until confirmed at the meeting immediately
 *          preceding it."
 *  - CPI   https://www.bls.gov/schedule/news_release/cpi.htm     (8:30 AM ET)
 *  - Jobs  https://www.bls.gov/schedule/news_release/empsit.htm  (Employment Situation, 8:30 AM ET)
 *          Both BLS pages (and the bls.gov/schedule/news_release/bls.ics feed) list release
 *          dates only through December 2026 — BLS has not posted its 2027 schedule yet. 2027 CPI
 *          and jobs dates are LEFT OUT, never guessed; BLS_THROUGH marks where they stop.
 *
 * Reminders (lib/__tests__/econCalendar.test.js): a test fails 6 months before CALENDAR_THROUGH
 * runs out, and another 30 days before BLS_THROUGH does — re-copy the dates from the pages above.
 *
 * All maths is in America/New_York via the marketClock helpers, whatever the device's zone.
 */
import { etParts, etWallToMs, weekdayOf, fmtCountdown } from './marketClock';

const AM_830 = 8 * 60 + 30;
const PM_200 = 14 * 60;

export const SERIES = {
    jobs: {
        name: 'Jobs', min: AM_830,
        dates: ['2026-01-09', '2026-02-11', '2026-03-06', '2026-04-03', '2026-05-08', '2026-06-05',
            '2026-07-02', '2026-08-07', '2026-09-04', '2026-10-02', '2026-11-06', '2026-12-04'],
    },
    cpi: {
        name: 'CPI', min: AM_830,
        dates: ['2026-01-13', '2026-02-13', '2026-03-11', '2026-04-10', '2026-05-12', '2026-06-10',
            '2026-07-14', '2026-08-12', '2026-09-11', '2026-10-14', '2026-11-10', '2026-12-10'],
    },
    fomc: {
        name: 'FOMC', min: PM_200,
        dates: ['2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17', '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09',
            '2027-01-27', '2027-03-17', '2027-04-28', '2027-06-09', '2027-07-28', '2027-09-15', '2027-10-27', '2027-12-08'],
    },
};
/** Last date the calendar covers (the FOMC list). */
export const CALENDAR_THROUGH = '2027-12-08';
/** Last BLS (CPI / jobs) date copied — BLS's own schedule stops in December 2026. */
export const BLS_THROUGH = '2026-12-10';

export const WINDOW_DAYS = 14;
export const MAX_EVENTS = 3;

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const dayMs = (date) => Date.parse(`${date}T12:00:00Z`);
const addDays = (date, n) => new Date(dayMs(date) + n * 864e5).toISOString().slice(0, 10);
/** 510 → "8:30", 840 → "2:00" (the line says ET once). */
const hm = (min) => `${((Math.floor(min / 60) + 11) % 12) + 1}:${String(min % 60).padStart(2, '0')}`;

/**
 * Releases from today (ET) through today + `days`, in time order, at most `max`.
 * @returns {{key:string,name:string,date:string,min:number,at:number,today:boolean,out:boolean}[]}
 */
export function upcomingEvents(now, { days = WINDOW_DAYS, max = MAX_EVENTS } = {}) {
    if (!Number.isFinite(now)) return [];
    const today = etParts(now).date;
    const last = addDays(today, days);
    const out = [];
    for (const [key, s] of Object.entries(SERIES)) {
        for (const date of s.dates) {
            if (date < today || date > last) continue;
            const at = etWallToMs(date, s.min);
            out.push({ key, name: s.name, date, min: s.min, at, today: date === today, out: date === today && now >= at });
        }
    }
    out.sort((a, b) => a.at - b.at || a.name.localeCompare(b.name));
    return out.slice(0, max);
}

/** This week → weekday ("Fri"); further out → "Oct 28". */
function dayLabel(date, today) {
    const ahead = Math.round((dayMs(date) - dayMs(today)) / 864e5);
    if (ahead < 7) return weekdayOf(date);
    return `${MONTHS[Number(date.slice(5, 7)) - 1]} ${Number(date.slice(8, 10))}`;
}

/**
 * The line under the market clock, or null when nothing is due within the window.
 *   "Next · Jobs Fri 8:30 · CPI Wed 8:30 · FOMC Oct 28 2:00 ET"
 *   "🔔 CPI today 8:30 ET · in 1h 10m"   (alert: amber, before the time on the day)
 *   "CPI out 8:30"                       (after the time, the rest of that day)
 * `segments` is the same text in pieces; only the `hot` (release-day countdown) piece is amber.
 */
export function econLine(now) {
    const evs = upcomingEvents(now);
    if (!evs.length) return null;
    const today = etParts(now).date;
    const segments = [];
    const later = [];
    for (const e of evs) {
        if (e.out) segments.push({ text: `${e.name} out ${hm(e.min)}`, hot: false });
        else if (e.today) segments.push({ text: `🔔 ${e.name} today ${hm(e.min)} ET · in ${fmtCountdown(e.at - now)}`, hot: true });
        else later.push(`${e.name} ${dayLabel(e.date, today)} ${hm(e.min)}`);
    }
    if (later.length) segments.push({ text: `Next · ${later.join(' · ')} ET`, hot: false });
    return { text: segments.map((g) => g.text).join(' · '), alert: segments.some((g) => g.hot), segments };
}
