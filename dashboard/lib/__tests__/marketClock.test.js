import {
    marketStatus, clockLabel, fmtCountdown, etWallToMs, etParts, sessionOf,
    NYSE_HOLIDAYS, EARLY_CLOSES, CALENDAR_THROUGH,
} from '../marketClock';

const at = (iso) => Date.parse(iso);

test('etWallToMs is DST-correct (EST vs EDT)', () => {
    expect(new Date(etWallToMs('2026-03-06', 570)).toISOString()).toBe('2026-03-06T14:30:00.000Z'); // EST
    expect(new Date(etWallToMs('2026-03-09', 570)).toISOString()).toBe('2026-03-09T13:30:00.000Z'); // EDT
    expect(etParts(at('2026-09-26T21:41:00Z'))).toEqual({ date: '2026-09-26', min: 17 * 60 + 41 });
});

test('weekend: closed, opens Monday (a day+ away → weekday and ET)', () => {
    const st = marketStatus(at('2026-09-26T21:41:00Z')); // Sat 17:41 ET
    expect(st).toMatchObject({ state: 'closed', date: '2026-09-28', holiday: null });
    expect(new Date(st.at).toISOString()).toBe('2026-09-28T13:30:00.000Z');
    expect(clockLabel(st)).toBe('Closed · opens Mon 9:30 ET');
});

test('pre-market, open, and after the close', () => {
    expect(clockLabel(marketStatus(at('2026-09-28T12:45:00Z')))).toBe('Pre-market · opens in 45m'); // 08:45 ET
    expect(clockLabel(marketStatus(at('2026-09-28T05:00:00Z')))).toBe('Closed · opens in 8h 30m'); // 01:00 ET: night, not pre-market
    expect(marketStatus(at('2026-09-28T08:00:00Z')).state).toBe('pre'); // 04:00 ET sharp
    expect(clockLabel(marketStatus(at('2026-09-28T17:50:00Z')))).toBe('Open · closes in 2h 10m'); // 13:50 ET
    expect(clockLabel(marketStatus(at('2026-09-28T20:30:00Z')))).toBe('Closed · opens in 17h 0m'); // 16:30 ET
    expect(marketStatus(at('2026-09-28T20:00:00Z')).state).toBe('closed'); // 16:00 sharp = closed
});

test('holidays and 1 pm early closes', () => {
    expect(clockLabel(marketStatus(at('2026-11-26T15:00:00Z')))).toBe('Closed (Thanksgiving) · opens Fri 9:30 ET');
    expect(clockLabel(marketStatus(at('2026-11-27T17:00:00Z')))).toBe('Open · closes in 1h 0m (1 pm early close)');
    expect(marketStatus(at('2026-11-27T18:30:00Z')).state).toBe('closed'); // 13:30 ET after an early close
    expect(marketStatus(at('2027-07-02T21:00:00Z')).date).toBe('2027-07-06'); // Fri → skips Mon Jul 5
});

test('past the calendar: weekday rule still runs, flagged ≈', () => {
    expect(clockLabel(marketStatus(at('2029-01-03T16:00:00Z')))).toMatch(/^≈ Open/);
});

test('calendar sanity: holidays and early closes fall on weekdays and never overlap', () => {
    for (const d of Object.keys(NYSE_HOLIDAYS)) expect([1, 2, 3, 4, 5]).toContain(new Date(`${d}T12:00:00Z`).getUTCDay());
    for (const d of EARLY_CLOSES) {
        expect(NYSE_HOLIDAYS[d]).toBeUndefined();
        expect(sessionOf(d)).toMatchObject({ early: true, close: 13 * 60 });
    }
});

test('REMINDER: the holiday calendar must reach a year ahead — extend it from nyse.com', () => {
    const yearAhead = new Date(Date.now() + 365 * 864e5).toISOString().slice(0, 10);
    expect(CALENDAR_THROUGH >= yearAhead).toBe(true);
});

test('fmtCountdown', () => {
    expect(fmtCountdown(0)).toBe('0m');
    expect(fmtCountdown(59 * 60e3)).toBe('59m');
    expect(fmtCountdown(26 * 3600e3)).toBe('1d 2h');
});
