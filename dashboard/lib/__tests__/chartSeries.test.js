import { buildChartSeries, chartFor, SHEET_METRICS, CHART_DAYS } from '../marks';

const NOW = new Date('2026-09-26T16:00:00Z');
const day = (i) => new Date(Date.parse('2026-09-26T00:00:00Z') - i * 86400000).toISOString().slice(0, 10);
const row = (date, cols) => { const r = Array(71).fill(''); r[0] = date; for (const [c, v] of Object.entries(cols)) r[c] = String(v); return r; };
const claimsCol = SHEET_METRICS.claims.col, housingCol = SHEET_METRICS.housing.col;

function rows() {
    const out = [['Date']];
    for (let i = 120; i >= 0; i--) {
        out.push(row(day(i), { [claimsCol]: 200 + (i % 7), [housingCol]: 1300 + i }));
        out.push(row(day(i), { [claimsCol]: 210 + (i % 7), [housingCol]: i === 30 ? 1330000 : 1300 + i })); // 2nd run: last row wins
    }
    out.push(row('2026-10-01', { [claimsCol]: 999 })); // after today: ignored
    return out;
}

test('buildChartSeries: 90 days on one axis, last row per date, unit jumps dropped', () => {
    const s = buildChartSeries(rows(), NOW);
    expect(s.from).toBe(day(89));
    expect(s.days).toBe(CHART_DAYS);
    expect(s.v.claims).toHaveLength(90);
    expect(s.v.claims[89]).toBe(210); // today, second run
    expect(s.v.housing[89 - 30]).toBeNull(); // 1330000 vs 1300: ×1000 unit jump
    expect(s.v.sahmRule).toBeUndefined(); // no data → not tappable
});

test('chartFor: points with dates, and null when the sheet is not the number on screen', () => {
    const s = buildChartSeries(rows(), NOW);
    const c = chartFor(s, 'claims', 211);
    expect(c.label).toBe('Initial Claims (4wk)');
    expect(c.points).toHaveLength(90);
    expect(c.points[89]).toEqual({ date: '2026-09-26', value: 210 });
    expect(chartFor(s, 'claims', 210000)).toBeNull(); // ×1000 basis
    expect(chartFor(s, 'claims', 500)).toBeNull(); // > 50% apart
    expect(chartFor(s, 'claims', NaN)).toBeNull(); // N/A on screen: no chart
    expect(chartFor(s, 'nope', 1)).toBeNull();
    expect(chartFor(null, 'claims', 1)).toBeNull();
    expect(chartFor({ from: day(89), v: { claims: [1, 2, 3] } }, 'claims', 3)).toBeNull(); // < 10 points
});
