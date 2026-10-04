import { yearTicks, indexFromPointer, tfAvailable, fmtDay, readChoice, saveChoice, unitsForPx, spreadLabels, axisPos, gutterFor, clampUnits } from '../chartAxis';

const quarterly = (from, to) => {
    const out = [];
    for (let y = from; y <= to; y++) for (const m of ['01', '04', '07', '10']) out.push(`${y}-${m}-01`);
    return out;
};
const toX = (n) => (i) => 42 + (i / (n - 1)) * 430;

describe('yearTicks', () => {
    it('thins an 80-year axis (Profit Margin 1947→2026) to round decades', () => {
        const d = quarterly(1947, 2026);
        const t = yearTicks(d, toX(d.length));
        expect(t.map((x) => x.label)).toEqual(['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020']);
    });
    it('keeps every year on a short axis but drops a crowded partial first year', () => {
        const d = ['2021-12-01', ...quarterly(2022, 2026)]; // 2021 sits ~21 units left of 2022
        const t = yearTicks(d, toX(d.length));
        expect(t.map((x) => x.label)).toEqual(['2022', '2023', '2024', '2025', '2026']);
    });
    it('never returns overlapping labels', () => {
        for (const [a, b] of [[1947, 2026], [2010, 2026], [1990, 2026], [2016, 2026]]) {
            const d = quarterly(a, b);
            const t = yearTicks(d, toX(d.length));
            for (let i = 1; i < t.length; i++) expect(t[i].x - t[i - 1].x).toBeGreaterThanOrEqual(26);
            expect(t.length).toBeLessThanOrEqual(8);
        }
    });
    it('handles one year and empty input', () => {
        expect(yearTicks(['2026-01-01', '2026-02-01'], toX(2))).toHaveLength(1);
        expect(yearTicks([], toX(2))).toEqual([]);
    });
});

describe('indexFromPointer', () => {
    const geo = { w: 480, padL: 42, padR: 8 };
    const rect = { left: 100, width: 960 };
    it('maps the plot edges to the first and last points', () => {
        expect(indexFromPointer(100 + 84, rect, 11, geo)).toBe(0);
        expect(indexFromPointer(100 + 944, rect, 11, geo)).toBe(10);
        expect(indexFromPointer(100 + 514, rect, 11, geo)).toBe(5);
    });
    it('clamps outside the plot and refuses unknowable geometry', () => {
        expect(indexFromPointer(0, rect, 11, geo)).toBe(0);
        expect(indexFromPointer(5000, rect, 11, geo)).toBe(10);
        expect(indexFromPointer(300, { left: 0, width: 0 }, 11, geo)).toBeNull();
        expect(indexFromPointer(300, rect, 1, geo)).toBeNull();
        expect(indexFromPointer(NaN, rect, 11, geo)).toBeNull();
    });
});

it('tfAvailable needs ~90% coverage; ALL (null) is always available', () => {
    expect(tfAvailable(1260, 302)).toBe(false);
    expect(tfAvailable(252, 302)).toBe(true);
    expect(tfAvailable(1260, 1134)).toBe(true);
    expect(tfAvailable(null, 2)).toBe(true);
});

it('fmtDay formats ISO dates without timezone drift', () => {
    expect(fmtDay('2026-09-25')).toBe('Sep 25, 2026');
    expect(fmtDay('1947-01-01')).toBe('Jan 1, 1947');
    expect(fmtDay('garbage')).toBe('garbage');
});

describe('readChoice / saveChoice', () => {
    beforeEach(() => window.localStorage.clear());
    it('round-trips a timeframe and rejects junk', () => {
        saveChoice('k', '10Y');
        expect(readChoice('k')).toBe('10Y');
        window.localStorage.setItem('k', '<script>');
        expect(readChoice('k')).toBeNull();
    });
    it('never throws when storage is blocked', () => {
        const spy = jest.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
        const spy2 = jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
        expect(readChoice('k')).toBeNull();
        expect(() => saveChoice('k', '1Y')).not.toThrow();
        spy.mockRestore(); spy2.mockRestore();
    });
});

describe('readable axis labels (HTML over the SVG)', () => {
    it('unitsForPx: viewBox units a run of CSS px spans; null until the chart is measured', () => {
        expect(unitsForPx(34, 480, 336)).toBeCloseTo(48.57, 2); // phone: 480 units drawn 336px wide
        expect(unitsForPx(34, 480, 620)).toBeCloseTo(26.32, 2); // desk
        expect(unitsForPx(34, 480, null)).toBeNull();
        expect(unitsForPx(34, 480, 0)).toBeNull();
        expect(unitsForPx(34, 480, NaN)).toBeNull();
    });

    it('yearTicks with a phone-sized minGap drops a partial first year that would touch its neighbour', () => {
        const d = ['2021-11-01', '2021-12-01', ...quarterly(2022, 2026)];
        const x = toX(d.length);
        // 2021 sits ~41 units left of 2022: fine on a desk (26-unit gap), a collision at 336px wide
        expect(yearTicks(d, x).map((t) => t.label)[0]).toBe('2021');
        expect(yearTicks(d, x, { minGap: unitsForPx(34, 480, 336) }).map((t) => t.label)).toEqual(['2022', '2023', '2024', '2025', '2026']);
    });

    it('spreadLabels keeps labels in priority order and drops any closer than the gap', () => {
        const items = [{ id: 'zero', pos: 30 }, { id: 'hi', pos: 4 }, { id: 'lo', pos: 40 }];
        expect(spreadLabels(items, 12).map((i) => i.id)).toEqual(['zero', 'hi']); // lo is 10px from zero
        expect(spreadLabels(items, 8).map((i) => i.id)).toEqual(['zero', 'hi', 'lo']);
        expect(spreadLabels([], 12)).toEqual([]);
        expect(spreadLabels([{ id: 'a', pos: NaN }, { id: 'b', pos: 5 }], 12).map((i) => i.id)).toEqual(['b']);
    });

    it('axisPos places a label by % of the viewBox (so it never stretches with the SVG)', () => {
        expect(axisPos(240, 90, 480, 180)).toEqual({ left: '50%', top: '50%' });
        expect(axisPos(38, 180, 480, 180)).toEqual({ left: '7.9167%', top: '100%' });
    });

    it('gutterFor: a left gutter wide enough for the longest y label at 10px, never under the old one', () => {
        // '1234.5' = 6 chars ≈ 6 × 6.1 + 8 = 44.6px → 63.7 units on a 480-unit chart drawn 336px wide
        expect(gutterFor(['1.0', '1234.5'], 480, 336, 42)).toBe(64);
        expect(gutterFor(['1.0', '1234.5'], 480, 620, 42)).toBe(42); // desk: the old gutter already fits
        expect(gutterFor(['1.0'], 480, null, 42)).toBe(42); // not measured yet → old layout
        expect(gutterFor([], 480, 336, 42)).toBe(42);
    });

    it('clampUnits keeps a centred label of `halfPx` inside the chart', () => {
        expect(clampUnits(475, 12, 480, 336)).toBeCloseTo(480 - unitsForPx(12, 480, 336), 6);
        expect(clampUnits(2, 12, 480, 336)).toBeCloseTo(unitsForPx(12, 480, 336), 6);
        expect(clampUnits(240, 12, 480, 336)).toBe(240);
        expect(clampUnits(475, 12, 480, null)).toBe(475); // not measured: left alone
    });
});
