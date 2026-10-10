/**
 * marketClock.js — 🕰️ is the US stock market open, and for how long? Computed on the
 * device from the NYSE calendar: no network call, so nothing to fail or fall back from.
 *
 * Holidays and 1:00 pm early closes are copied from nyse.com/markets/hours-calendars
 * (read 2026-09-26; the page lists 2026–2028). Past CALENDAR_THROUGH the weekday +
 * 9:30–16:00 rule still runs but holidays are unknown, so the pill says "≈" — and
 * lib/__tests__/marketClock.test.js starts failing a year ahead as the reminder to extend.
 *
 * All wall-clock maths is in America/New_York (Intl), whatever the device's time zone,
 * so the countdown stays right from Dhaka too; far-off opens are labelled "ET".
 */

/** date → holiday name. Jan 1 2028 is a Saturday and NYSE does not observe it. */
export const NYSE_HOLIDAYS = {
    '2026-01-01': "New Year's Day", '2026-01-19': 'Martin Luther King Jr. Day', '2026-02-16': "Washington's Birthday",
    '2026-04-03': 'Good Friday', '2026-05-25': 'Memorial Day', '2026-06-19': 'Juneteenth',
    '2026-07-03': 'Independence Day', '2026-09-07': 'Labor Day', '2026-11-26': 'Thanksgiving', '2026-12-25': 'Christmas',
    '2027-01-01': "New Year's Day", '2027-01-18': 'Martin Luther King Jr. Day', '2027-02-15': "Washington's Birthday",
    '2027-03-26': 'Good Friday', '2027-05-31': 'Memorial Day', '2027-06-18': 'Juneteenth',
    '2027-07-05': 'Independence Day', '2027-09-06': 'Labor Day', '2027-11-25': 'Thanksgiving', '2027-12-24': 'Christmas',
    '2028-01-17': 'Martin Luther King Jr. Day', '2028-02-21': "Washington's Birthday", '2028-04-14': 'Good Friday',
    '2028-05-29': 'Memorial Day', '2028-06-19': 'Juneteenth', '2028-07-04': 'Independence Day',
    '2028-09-04': 'Labor Day', '2028-11-23': 'Thanksgiving', '2028-12-25': 'Christmas',
};
/** 1:00 pm ET closes. */
export const EARLY_CLOSES = new Set(['2026-11-27', '2026-12-24', '2027-11-26', '2028-07-03', '2028-11-24']);
export const CALENDAR_THROUGH = '2028-12-31';

export const PRE_MIN = 4 * 60; // pre-market trading starts 4:00 ET; before that it is night
export const OPEN_MIN = 9 * 60 + 30;
export const CLOSE_MIN = 16 * 60;
export const EARLY_CLOSE_MIN = 13 * 60;

let fmt = null;
/** Wall clock in New York: {date:'YYYY-MM-DD', min: minutes since midnight}. */
export function etParts(ms) {
    fmt = fmt || new Intl.DateTimeFormat('en-US', {
        timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
    });
    const p = Object.fromEntries(fmt.formatToParts(new Date(ms)).map((x) => [x.type, x.value]));
    return { date: `${p.year}-${p.month}-${p.day}`, min: (Number(p.hour) % 24) * 60 + Number(p.minute) };
}

/** Epoch ms of a New York wall-clock time (DST-correct: two correction passes). */
export function etWallToMs(date, min) {
    const [y, m, d] = date.split('-').map(Number);
    const want = Date.UTC(y, m - 1, d, Math.floor(min / 60), min % 60);
    let t = want + 5 * 3600e3;
    for (let i = 0; i < 2; i++) {
        const p = etParts(t);
        const [py, pm, pd] = p.date.split('-').map(Number);
        t += want - Date.UTC(py, pm - 1, pd, Math.floor(p.min / 60), p.min % 60);
    }
    return t;
}

const addDays = (date, n) => new Date(Date.parse(`${date}T12:00:00Z`) + n * 864e5).toISOString().slice(0, 10);
export const weekdayOf = (date) => new Date(`${date}T12:00:00Z`).toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' });

/** The day's regular session, or null (weekend / holiday). */
export function sessionOf(date) {
    const wd = new Date(`${date}T12:00:00Z`).getUTCDay();
    if (wd === 0 || wd === 6 || NYSE_HOLIDAYS[date]) return null;
    const early = EARLY_CLOSES.has(date);
    return { open: OPEN_MIN, close: early ? EARLY_CLOSE_MIN : CLOSE_MIN, early };
}

/**
 * @returns {{state:'open'|'pre'|'closed', at:number, ms:number, date:string,
 *   early?:boolean, holiday?:string|null, estimated:boolean}}
 *   `at` = the next close (open) or the next open (pre / closed); `ms` = time until it.
 */
export function marketStatus(now = Date.now()) {
    const { date, min } = etParts(now);
    const today = sessionOf(date);
    const estimated = date > CALENDAR_THROUGH;
    if (today && min >= today.open && min < today.close) {
        const at = etWallToMs(date, today.close);
        return { state: 'open', at, ms: at - now, date, early: today.early, estimated };
    }
    if (today && min < today.open) {
        const at = etWallToMs(date, OPEN_MIN);
        return { state: min >= PRE_MIN ? 'pre' : 'closed', at, ms: at - now, date, holiday: null, estimated };
    }
    let next = date;
    for (let i = 1; i <= 10; i++) { next = addDays(date, i); if (sessionOf(next)) break; }
    const at = etWallToMs(next, OPEN_MIN);
    return { state: 'closed', at, ms: at - now, date: next, holiday: NYSE_HOLIDAYS[date] || null, estimated };
}

/** "45m", "2h 10m", "1d 3h". */
export function fmtCountdown(ms) {
    const m = Math.max(0, Math.round(ms / 60000));
    if (m < 60) return `${m}m`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ${m % 60}m`;
    return `${Math.floor(h / 24)}d ${h % 24}h`;
}

/** The pill's text. Opens more than 18 h away read as a day + time (ET) instead of a countdown. */
export function clockLabel(st) {
    if (!st) return '';
    const approx = st.estimated ? '≈ ' : '';
    if (st.state === 'open') return `${approx}Open · closes in ${fmtCountdown(st.ms)}${st.early ? ' (1 pm early close)' : ''}`;
    if (st.state === 'pre') return `${approx}Pre-market · opens in ${fmtCountdown(st.ms)}`;
    const why = st.holiday ? ` (${st.holiday})` : '';
    const when = st.ms <= 18 * 3600e3 ? `in ${fmtCountdown(st.ms)}` : `${weekdayOf(st.date)} 9:30 ET`;
    return `${approx}Closed${why} · opens ${when}`;
}

/**
 * The newest NYSE session that has CLOSED (ET): today from its close (16:00, 13:00 on an
 * early close) on a trading day, otherwise the previous trading day. A daily CSV (CBOE's
 * VIX_History) only ever holds completed sessions, so this is the newest date it can have.
 */
export function latestCompletedSessionDate(now = Date.now()) {
    const { date, min } = etParts(now);
    const today = sessionOf(date);
    if (today && min >= today.close) return date;
    let d = date;
    for (let i = 1; i <= 10; i++) { d = addDays(date, -i); if (sessionOf(d)) break; }
    return d;
}

/**
 * Is a daily CLOSE dated `closeDate` the current level right now? Only outside regular
 * hours, and only when it is the latest completed session: during the session the
 * newest close is yesterday's, not the level the market is printing.
 * @returns {{current:boolean, expected:string, inSession:boolean, closeMs:number|null}}
 *   closeMs = epoch ms of that session's close (an honest `savedAt` for the value).
 */
export function dailyCloseStatus(closeDate, now = Date.now()) {
    const { date, min } = etParts(now);
    const today = sessionOf(date);
    const inSession = !!(today && min >= today.open && min < today.close);
    const expected = latestCompletedSessionDate(now);
    const s = closeDate ? sessionOf(closeDate) : null;
    const closeMs = s ? etWallToMs(closeDate, s.close) : null;
    return { current: !inSession && !!closeDate && closeDate >= expected, expected, inSession, closeMs };
}
