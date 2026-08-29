import {
    fearGreedTag,
    parseVixObservations,
    computeVixFearGreedTag,
    resolveVixFearGreedTag,
    computeVixFearGreedTagFromCboe,
} from '../vixFearGreed';

// Builds a FRED /fred/series/observations-shaped payload from a plain array
// of numbers (or the '.' missing-value sentinel), dated sequentially from
// startDate — mirrors what api.stlouisfed.org actually returns.
function fredPayload(values, startDate = '2026-01-01') {
    const observations = values.map((value, i) => {
        const d = new Date(startDate);
        d.setUTCDate(d.getUTCDate() + i);
        return { date: d.toISOString().slice(0, 10), value: value === '.' ? '.' : String(value) };
    });
    return { observations };
}

describe('fearGreedTag (must match vix-fear-greed/fear_greed.py exactly)', () => {
    // 49 identical closes + 1 different "latest" close — isolates the effect
    // of the most recent print on a known 50-day SMA.
    const closesWithLatest = (base, latest) => Array(49).fill(base).concat([latest]);

    test('FEAR side when the latest close sits above its 50d SMA', () => {
        // sma50=20.1, latest=25, pct_diff=+0.2438 -> score 24
        expect(fearGreedTag(closesWithLatest(20, 25))).toBe('FEAR24');
    });

    test('GREED side when the latest close sits below its 50d SMA', () => {
        // sma50=19.9, latest=15, pct_diff=-0.2462 -> score 25
        expect(fearGreedTag(closesWithLatest(20, 15))).toBe('GREED25');
    });

    test('NEUTRAL00 when the latest close equals its 50d SMA exactly', () => {
        expect(fearGreedTag(closesWithLatest(20, 20))).toBe('NEUTRAL00');
    });

    test('score is capped at 99 for an extreme deviation', () => {
        // sma50=29.8, latest=1000, pct_diff=+32.56 (3256%) -> capped at 99
        expect(fearGreedTag(closesWithLatest(10, 1000))).toBe('FEAR99');
    });

    test('score is zero-padded to two digits', () => {
        // sma50=100.02, latest=101, pct_diff=+0.0098 -> score 1 -> "01"
        expect(fearGreedTag(closesWithLatest(100, 101))).toBe('FEAR01');
    });

    test('reproduces ~ the live GREED13 reading from real VIXCLS closes (2026-06-19 -> 2026-08-27)', () => {
        const closes = [
            16.78, 17.28, 19.49, 18.63, 18.89, 18.41, 17.65, 16.45, 16.59, 16.15,
            15.81, 15.57, 16.13, 16.9, 15.84, 15.03, 17.16, 16.5, 15.67, 16.73,
            18.77, 18.65, 17.05, 16.64, 18.7, 18.58, 18.67, 18.21, 20.66, 17.09,
            15.99, 15.86, 16.5, 15.81, 15.15, 14.9, 15.46, 15.28, 14.55, 14.63,
            14.25, 15.19, 15.84, 14.89, 16.01, 15.13, 15.85, 15.45, 15.21, 14.51,
        ];
        expect(fearGreedTag(closes)).toBe('GREED12');
    });

    test('returns null when fewer than 50 closes are available (the 50d SMA needs 50 points)', () => {
        expect(fearGreedTag(Array(49).fill(20))).toBeNull();
        expect(fearGreedTag([])).toBeNull();
        expect(fearGreedTag(null)).toBeNull();
        expect(fearGreedTag(undefined)).toBeNull();
    });
});

describe('parseVixObservations (missing-value filtering)', () => {
    test('drops FRED\'s "." missing-value sentinel', () => {
        const data = {
            observations: [
                { date: '2026-08-24', value: '15.85' },
                { date: '2026-08-25', value: '.' },
                { date: '2026-08-26', value: '15.21' },
            ],
        };
        expect(parseVixObservations(data)).toEqual([
            { date: '2026-08-24', value: 15.85 },
            { date: '2026-08-26', value: 15.21 },
        ]);
    });

    test('sorts ascending by date regardless of the API response order', () => {
        const data = {
            observations: [
                { date: '2026-08-26', value: '15.21' },
                { date: '2026-08-24', value: '15.85' },
                { date: '2026-08-25', value: '15.45' },
            ],
        };
        expect(parseVixObservations(data).map((o) => o.date)).toEqual([
            '2026-08-24', '2026-08-25', '2026-08-26',
        ]);
    });

    test('handles malformed/missing payloads without throwing', () => {
        expect(parseVixObservations({})).toEqual([]);
        expect(parseVixObservations(null)).toEqual([]);
        expect(parseVixObservations(undefined)).toEqual([]);
        expect(parseVixObservations({ observations: [{ date: '2026-08-24', value: 'not-a-number' }] })).toEqual([]);
    });
});

describe('computeVixFearGreedTag (fetch + compute)', () => {
    test('computes the tag from a FRED-shaped payload', async () => {
        const values = Array(49).fill(20).concat([25]); // -> FEAR24
        const fetchJson = jest.fn().mockResolvedValue(fredPayload(values));
        const tag = await computeVixFearGreedTag('fake-key', { fetchJson });
        expect(tag).toBe('FEAR24');
        expect(fetchJson).toHaveBeenCalledTimes(1);
        const [url] = fetchJson.mock.calls[0];
        expect(url).toContain('VIXCLS');
        expect(url).toContain('fake-key');
    });

    test('throws when no FRED API key is configured', async () => {
        await expect(computeVixFearGreedTag(undefined, { fetchJson: jest.fn() }))
            .rejects.toThrow(/FRED_API_KEY/);
    });

    test('throws when FRED returns too few valid observations', async () => {
        const fetchJson = jest.fn().mockResolvedValue(fredPayload(Array(10).fill(20)));
        await expect(computeVixFearGreedTag('fake-key', { fetchJson }))
            .rejects.toThrow(/Insufficient/);
    });

    test('filters "." values before checking sufficiency (60 raw rows, 15 missing -> 45 valid, still insufficient)', async () => {
        const payload = fredPayload(Array(60).fill(20));
        payload.observations.slice(0, 15).forEach((o) => { o.value = '.'; });
        const fetchJson = jest.fn().mockResolvedValue(payload);
        await expect(computeVixFearGreedTag('fake-key', { fetchJson }))
            .rejects.toThrow(/Insufficient/);
    });

    test('propagates a fetch failure', async () => {
        const fetchJson = jest.fn().mockRejectedValue(new Error('network down'));
        await expect(computeVixFearGreedTag('fake-key', { fetchJson })).rejects.toThrow('network down');
    });
});

describe('resolveVixFearGreedTag (fallback chain: FRED-computed -> sheet C2 -> N/A)', () => {
    test('prefers the FRED-computed tag when it succeeds, and says so in the message', async () => {
        const values = Array(49).fill(20).concat([15]); // -> GREED25
        const fetchJson = jest.fn().mockResolvedValue(fredPayload(values));
        const { tag, message } = await resolveVixFearGreedTag({ fredApiKey: 'k', fetchJson, sheetValue: 'GREED07' });
        expect(tag).toBe('GREED25');
        expect(message).toMatch(/computed from FRED/i);
    });

    test('falls back to the sheet C2 value when the FRED computation fails, and says so in the message', async () => {
        const fetchJson = jest.fn().mockRejectedValue(new Error('FRED 429'));
        const { tag, message } = await resolveVixFearGreedTag({ fredApiKey: 'k', fetchJson, sheetValue: 'GREED07' });
        expect(tag).toBe('GREED07');
        expect(message).toMatch(/FRED failed/i);
        expect(message).toMatch(/GREED07/);
    });

    test('falls back to N/A when FRED fails and no sheet value is available', async () => {
        const fetchJson = jest.fn().mockRejectedValue(new Error('FRED down'));
        const { tag, message } = await resolveVixFearGreedTag({ fredApiKey: 'k', fetchJson });
        expect(tag).toBe('N/A');
        expect(message).toMatch(/N\/A/);
    });

    test('falls back to the sheet value when no FRED API key is configured', async () => {
        const { tag, message } = await resolveVixFearGreedTag({ fredApiKey: undefined, fetchJson: jest.fn(), sheetValue: 'FEAR03' });
        expect(tag).toBe('FEAR03');
        expect(message).toMatch(/FRED failed/i);
    });

    test('honors the vix_fred fault-injection gate (?_fail=vix_fred forces the sheet fallback for prod verification)', async () => {
        const values = Array(50).fill(20);
        const fetchJson = jest.fn().mockResolvedValue(fredPayload(values));
        const faults = new Set(['vix_fred']);
        const { tag, message } = await resolveVixFearGreedTag({ fredApiKey: 'k', fetchJson, sheetValue: 'GREED07', faults });
        expect(tag).toBe('GREED07');
        expect(fetchJson).not.toHaveBeenCalled();
        expect(message).toMatch(/injected fault/i);
    });
});

// --- CBOE primary tier -------------------------------------------------
// FRED's VIXCLS runs one trading day behind (verified 2026-08-29: FRED's
// newest print was Thu 08/27 -> GREED12 while Fri's real close 14.43 gives
// GREED13). CBOE's own daily-history CSV is same-day, keyless and
// authoritative, so it sits AHEAD of FRED in the cascade.
function cboeCsv(values, startDate = '2026-01-01') {
    const rows = values.map((v, i) => {
        const d = new Date(startDate);
        d.setUTCDate(d.getUTCDate() + i);
        const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
        const dd = String(d.getUTCDate()).padStart(2, '0');
        return `${mm}/${dd}/${d.getUTCFullYear()},0,0,0,${v}`;
    });
    return ['DATE,OPEN,HIGH,LOW,CLOSE', ...rows].join('\n');
}

describe('computeVixFearGreedTagFromCboe', () => {
    const closes = Array(49).fill(20).concat([15]); // sma50=19.9, pct=-24.62% -> GREED25

    test('computes the tag from the CBOE daily-history CSV', async () => {
        const fetchText = jest.fn().mockResolvedValue(cboeCsv(closes));
        await expect(computeVixFearGreedTagFromCboe({ fetchText })).resolves.toBe('GREED25');
        expect(fetchText.mock.calls[0][0]).toContain('cdn.cboe.com');
    });

    test('throws when the CSV has fewer than 50 usable closes', async () => {
        const fetchText = jest.fn().mockResolvedValue(cboeCsv(Array(10).fill(20)));
        await expect(computeVixFearGreedTagFromCboe({ fetchText })).rejects.toThrow(/Insufficient/);
    });
});

describe('resolveVixFearGreedTag cascade: CBOE -> FRED -> sheet', () => {
    const cboeCloses = Array(49).fill(20).concat([15]);   // sma50=19.9, pct=-24.62% -> GREED25
    const fredCloses = Array(49).fill(20).concat([25]);   // FEAR24

    test('CBOE wins when it succeeds, and FRED is never called', async () => {
        const fetchText = jest.fn().mockResolvedValue(cboeCsv(cboeCloses));
        const fetchJson = jest.fn();
        const { tag, message } = await resolveVixFearGreedTag({
            fredApiKey: 'k', fetchJson, fetchText, sheetValue: 'GREED99',
        });
        expect(tag).toBe('GREED25');
        expect(message).toMatch(/CBOE/i);
        expect(fetchJson).not.toHaveBeenCalled();
    });

    test('falls back to FRED when CBOE fails, and says so', async () => {
        const fetchText = jest.fn().mockRejectedValue(new Error('CBOE down'));
        const fetchJson = jest.fn().mockResolvedValue(fredPayload(fredCloses));
        const { tag, message } = await resolveVixFearGreedTag({
            fredApiKey: 'k', fetchJson, fetchText, sheetValue: 'GREED99',
        });
        expect(tag).toBe('FEAR24');
        expect(message).toMatch(/CBOE down/);
        expect(message).toMatch(/FRED/);
    });

    test('falls back to the sheet when BOTH CBOE and FRED fail', async () => {
        const { tag, message } = await resolveVixFearGreedTag({
            fredApiKey: 'k',
            fetchText: jest.fn().mockRejectedValue(new Error('CBOE down')),
            fetchJson: jest.fn().mockRejectedValue(new Error('FRED down')),
            sheetValue: 'GREED99',
        });
        expect(tag).toBe('GREED99');
        expect(message).toMatch(/sheet value \(GREED99\)/);
    });

    test('?_fail=vix_cboe forces the FRED tier', async () => {
        const { tag } = await resolveVixFearGreedTag({
            fredApiKey: 'k',
            fetchText: jest.fn().mockResolvedValue(cboeCsv(cboeCloses)),
            fetchJson: jest.fn().mockResolvedValue(fredPayload(fredCloses)),
            sheetValue: 'GREED99',
            faults: new Set(['vix_cboe']),
        });
        expect(tag).toBe('FEAR24');
    });

    test('?_fail=vix_cboe,vix_fred forces the sheet tier', async () => {
        const { tag } = await resolveVixFearGreedTag({
            fredApiKey: 'k',
            fetchText: jest.fn().mockResolvedValue(cboeCsv(cboeCloses)),
            fetchJson: jest.fn().mockResolvedValue(fredPayload(fredCloses)),
            sheetValue: 'GREED99',
            faults: new Set(['vix_cboe', 'vix_fred']),
        });
        expect(tag).toBe('GREED99');
    });
});
