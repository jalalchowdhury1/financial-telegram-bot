import { SERIES, BLS_THROUGH, FED_THROUGH, WINDOW_DAYS, FED_REMINDER_DAYS, upcomingEvents, econLine } from '../econCalendar';

const at = (iso) => Date.parse(iso);
const ISO = /^\d{4}-\d{2}-\d{2}$/;

describe('the hand-copied calendar', () => {
    test('copied dates: Fed FOMC (statement day) and BLS CPI / Employment Situation', () => {
        // federalreserve.gov/monetarypolicy/fomccalendars.htm, read 2026-10-04 (last day of each meeting)
        expect(SERIES.fomc.dates).toEqual([
            '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17', '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09',
            '2027-01-27', '2027-03-17', '2027-04-28', '2027-06-09', '2027-07-28', '2027-09-15', '2027-10-27', '2027-12-08',
        ]);
        // bls.gov/schedule/news_release/cpi.htm + empsit.htm, read 2026-10-04
        expect(SERIES.cpi.dates.slice(-3)).toEqual(['2026-10-14', '2026-11-10', '2026-12-10']);
        expect(SERIES.jobs.dates.slice(-3)).toEqual(['2026-10-02', '2026-11-06', '2026-12-04']);
        expect(SERIES.fomc.min).toBe(14 * 60);
        expect(SERIES.cpi.min).toBe(8 * 60 + 30);
        expect(SERIES.jobs.min).toBe(8 * 60 + 30);
    });

    test('every date is a real, sorted weekday; FOMC statements land on a Tuesday or Wednesday', () => {
        for (const s of Object.values(SERIES)) {
            expect([...s.dates].sort()).toEqual(s.dates);
            for (const d of s.dates) {
                expect(d).toMatch(ISO);
                const wd = new Date(`${d}T12:00:00Z`).getUTCDay();
                expect(wd >= 1 && wd <= 5).toBe(true);
            }
        }
        for (const d of SERIES.fomc.dates) expect([2, 3]).toContain(new Date(`${d}T12:00:00Z`).getUTCDay());
    });

    test('BLS has not posted 2027 yet: no CPI or jobs date is guessed past BLS_THROUGH', () => {
        // bls.gov's own release calendar (bls.ics) runs to 2026-12-30; its last CPI / jobs dates are Dec 10 / Dec 4
        expect(BLS_THROUGH).toBe('2026-12-30');
        expect(FED_THROUGH).toBe('2027-12-31');
        for (const d of [...SERIES.cpi.dates, ...SERIES.jobs.dates]) expect(d <= BLS_THROUGH).toBe(true);
        for (const d of SERIES.fomc.dates) expect(d <= FED_THROUGH).toBe(true);
        // each series knows where its own copied schedule ends
        expect(SERIES.cpi.through).toBe(BLS_THROUGH);
        expect(SERIES.jobs.through).toBe(BLS_THROUGH);
        expect(SERIES.fomc.through).toBe(FED_THROUGH);
    });

    const dayPlus = (n) => new Date(Date.now() + n * 864e5).toISOString().slice(0, 10);
    // BLS (exception): goes red on the first day the 14-day line reaches past BLS_THROUGH (from 2026-12-17).
    // An earlier warning would already be red today with nothing to copy: BLS has not posted 2027 yet.
    test('REMINDER: the 14-day line has reached the end of the BLS dates — copy next year\'s CPI + jobs schedule from bls.gov', () => {
        expect(BLS_THROUGH >= dayPlus(WINDOW_DAYS)).toBe(true);
    });

    // Fed: goes red 6 months before FED_THROUGH runs out (from 2027-07-01) — the Fed posts the next year well ahead.
    test('REMINDER: under 6 months of Fed dates left — copy the next year\'s FOMC calendar from federalreserve.gov', () => {
        expect(FED_REMINDER_DAYS).toBeGreaterThanOrEqual(182);
        expect(FED_THROUGH >= dayPlus(FED_REMINDER_DAYS)).toBe(true);
    });
});

describe('upcomingEvents', () => {
    test('only the next 14 days (ET), in time order, at most 3', () => {
        // Tue 1 Dec 2026, 10:00 ET
        const evs = upcomingEvents(at('2026-12-01T15:00:00Z'));
        expect(evs.map((e) => `${e.name} ${e.date}`)).toEqual(['Jobs 2026-12-04', 'FOMC 2026-12-09', 'CPI 2026-12-10']);
        expect(upcomingEvents(at('2026-12-01T15:00:00Z'), { max: 2 })).toHaveLength(2);
    });

    test('a release time is New York wall clock, across the DST change', () => {
        // CPI 14 Oct 2026 8:30 EDT = 12:30Z; CPI 10 Nov 2026 8:30 EST = 13:30Z
        expect(upcomingEvents(at('2026-10-14T05:00:00Z'))[0].at).toBe(at('2026-10-14T12:30:00Z'));
        expect(upcomingEvents(at('2026-11-10T05:00:00Z'))[0].at).toBe(at('2026-11-10T13:30:00Z'));
    });

    test('each series stops at its own copied schedule: never a guessed date, never a hidden known one', () => {
        // Tue 1 Dec 2026: Jobs Fri, FOMC Wed 9th, CPI Thu 10th — cut at the 9th, CPI is not listed
        expect(upcomingEvents(at('2026-12-01T15:00:00Z'), { through: '2026-12-09' }).map((e) => e.name)).toEqual(['Jobs', 'FOMC']);
        // 20 Jan 2027: BLS's 2027 dates are not copied yet, the Fed's are → FOMC Jan 27 still shows
        expect(upcomingEvents(at('2027-01-20T15:00:00Z')).map((e) => `${e.name} ${e.date}`)).toEqual(['FOMC 2027-01-27']);
    });

    test('the ET day decides "today", whatever the device clock zone', () => {
        // 02:00Z on 14 Oct is still 13 Oct (22:00) in New York: CPI is tomorrow, not today
        const [cpi] = upcomingEvents(at('2026-10-14T02:00:00Z'));
        expect(cpi.name).toBe('CPI');
        expect(cpi.today).toBe(false);
    });
});

describe('econLine — the one quiet line under the market clock', () => {
    test('a normal day: "Next · …" with weekdays this week, dates further out, one ET', () => {
        expect(econLine(at('2026-12-01T15:00:00Z'))).toEqual({
            text: 'Next · Jobs Fri 8:30 · FOMC Dec 9 2:00 · CPI Dec 10 8:30 ET', alert: false,
            // one segment per release, so the line can only ever wrap BETWEEN releases
            segments: [
                { text: 'Next · Jobs Fri 8:30', hot: false },
                { text: 'FOMC Dec 9 2:00', hot: false },
                { text: 'CPI Dec 10 8:30 ET', hot: false },
            ],
        });
        expect(econLine(at('2026-09-28T17:50:00Z'))).toMatchObject({ text: 'Next · Jobs Fri 8:30 ET', alert: false });
    });

    test('release day before the time: amber, with a countdown', () => {
        // Wed 14 Oct 2026 07:20 EDT
        expect(econLine(at('2026-10-14T11:20:00Z'))).toEqual({
            text: '🔔 CPI today 8:30 ET · in 1h 10m · Next · FOMC Oct 28 2:00 ET', alert: true,
            segments: [{ text: '🔔 CPI today 8:30 ET · in 1h 10m', hot: true }, { text: 'Next · FOMC Oct 28 2:00 ET', hot: false }],
        });
    });

    test('after the time: "CPI out 8:30", no longer amber', () => {
        expect(econLine(at('2026-10-14T12:30:00Z')).text).toBe('CPI out 8:30 · Next · FOMC Oct 28 2:00 ET');
        expect(econLine(at('2026-10-14T20:00:00Z'))).toMatchObject({ text: 'CPI out 8:30 · Next · FOMC Oct 28 2:00 ET', alert: false });
        // the next ET day it is gone
        expect(econLine(at('2026-10-15T12:00:00Z'))).toMatchObject({ text: 'Next · FOMC Oct 28 2:00 ET', alert: false });
    });

    test('FOMC day reads 2:00 ET (the statement), in EST after the clocks change', () => {
        // Wed 9 Dec 2026 13:00 EST = 18:00Z
        expect(econLine(at('2026-12-09T18:00:00Z')).text).toBe('🔔 FOMC today 2:00 ET · in 1h 0m · Next · CPI Thu 8:30 ET');
    });

    test('release day keeps to one row on a phone: the countdown + only the next release after it', () => {
        // Wed 28 Oct 2026 13:00 EDT: FOMC today, then Jobs Nov 6 and CPI Nov 10 are both within 14 days
        expect(econLine(at('2026-10-28T17:00:00Z'))).toEqual({
            text: '🔔 FOMC today 2:00 ET · in 1h 0m · Next · Jobs Nov 6 8:30 ET', alert: true,
            segments: [{ text: '🔔 FOMC today 2:00 ET · in 1h 0m', hot: true }, { text: 'Next · Jobs Nov 6 8:30 ET', hot: false }],
        });
        // Fri 4 Dec 2026 08:00 EST: Jobs today, then FOMC Wed and CPI Thu
        expect(econLine(at('2026-12-04T13:00:00Z')).text).toBe('🔔 Jobs today 8:30 ET · in 30m · Next · FOMC Wed 2:00 ET');
        // once it is out, the quiet line has room for both again
        expect(econLine(at('2026-12-04T14:00:00Z')).text).toBe('Jobs out 8:30 · Next · FOMC Wed 2:00 · CPI Thu 8:30 ET');
    });

    test('past the BLS dates: the known FOMC still shows, without "Next" (a jobs or CPI date it cannot see may come first)', () => {
        // Wed 20 Jan 2027 10:00 EST
        expect(econLine(at('2027-01-20T15:00:00Z'))).toEqual({
            text: 'FOMC Jan 27 2:00 ET', alert: false, segments: [{ text: 'FOMC Jan 27 2:00 ET', hot: false }],
        });
        // ...and on the day, the amber countdown as usual
        expect(econLine(at('2027-01-27T18:00:00Z'))).toMatchObject({ text: '🔔 FOMC today 2:00 ET · in 1h 0m', alert: true });
        // the window still inside BLS's dates: "Next" as normal
        expect(econLine(at('2026-12-01T15:00:00Z')).text.startsWith('Next · ')).toBe(true);
    });

    test('nothing within 14 days → null (the line hides)', () => {
        expect(econLine(at('2026-12-11T15:00:00Z'))).toBeNull();
        expect(econLine(at('2027-01-01T15:00:00Z'))).toBeNull();
        expect(econLine(NaN)).toBeNull();
    });
});
