import fs from 'fs';
import path from 'path';
import { parseWestmetallTable, westmetallCopperLeg, resolveLeg, buildCopperGold, LB_PER_TONNE } from '../copperGold';

// Real page, fetched 2026-10-09:
//   https://www.westmetall.com/en/markdaten.php?action=table&field=LME_Cu_cash
const HTML = fs.readFileSync(path.join(__dirname, 'fixtures', 'westmetall-cu-cash-2026-10-09.html'), 'utf8');
const NOW = new Date('2026-10-09T18:00:00Z');

describe('parseWestmetallTable', () => {
    const hist = parseWestmetallTable(HTML);

    test('parses the whole year, ascending, in $/lb', () => {
        expect(hist.length).toBeGreaterThan(180);
        expect(hist[0].date).toBe('2026-01-02');
        const last = hist[hist.length - 1];
        expect(last.date).toBe('2026-10-09');
        // 14,689.00 $/t ÷ 2204.6226 = 6.663 $/lb (gold-api HG spot read 6.643 the same day)
        expect(last.price).toBeCloseTo(14689 / LB_PER_TONNE, 6);
        expect(last.price).toBeCloseTo(6.663, 2);
        for (let i = 1; i < hist.length; i++) expect(hist[i].date > hist[i - 1].date).toBe(true);
    });

    test('reads the CASH column by header, never the 3-month or the stock column', () => {
        const row = hist.find((p) => p.date === '2026-10-08');
        expect(row.price * LB_PER_TONNE).toBeCloseTo(14526, 6);   // cash, not 14,417 (3-month)
        for (const p of hist) expect(p.price).toBeLessThan(20);      // stock (~235,000 t) would be ~100 "$/lb"
    });

    test('columns located by header text: a reordered table still reads cash', () => {
        const html = '<table><tr><th>LME Copper stock</th><th>date</th><th>LME Copper Cash-Settlement</th></tr>'
            + '<tr><td>233,025</td><td>09. October 2026</td><td>14,689.00</td></tr></table>';
        expect(parseWestmetallTable(html)).toEqual([{ date: '2026-10-09', price: 14689 / LB_PER_TONNE }]);
    });

    test('no cash header → [] (no data, not wrong data)', () => {
        const html = '<table><tr><th>date</th><th>LME Copper stock</th></tr><tr><td>09. October 2026</td><td>233,025</td></tr></table>';
        expect(parseWestmetallTable(html)).toEqual([]);
        expect(parseWestmetallTable('')).toEqual([]);
        expect(parseWestmetallTable(null)).toEqual([]);
    });
});

describe('westmetallCopperLeg', () => {
    test('mid-year: one page is enough (no prior-year request)', async () => {
        const calls = [];
        const leg = await westmetallCopperLeg(async (year) => { calls.push(year); return HTML; }, NOW);
        expect(calls).toEqual([undefined]);
        expect(leg.currentDate).toBe('2026-10-09');
        expect(leg.historyAsc.length).toBeGreaterThan(180);
    });

    test('early in the year the prior year is added so the 3-month window spans', async () => {
        const jan = '<table><tr><th>date</th><th>LME Copper Cash-Settlement</th></tr>'
            + '<tr><td>06. January 2027</td><td>13,000.00</td></tr></table>';
        const calls = [];
        const leg = await westmetallCopperLeg(async (year) => { calls.push(year); return year ? HTML : jan; }, new Date('2027-01-07T12:00:00Z'));
        expect(calls).toEqual([undefined, 2026]);
        expect(leg.currentDate).toBe('2027-01-06');
        expect(leg.historyAsc[0].date).toBe('2026-01-02');
    });

    test('prior-year failure only shortens history; empty current page throws', async () => {
        const jan = '<table><tr><th>date</th><th>LME Copper Cash-Settlement</th></tr><tr><td>06. January 2027</td><td>13,000.00</td></tr></table>';
        const leg = await westmetallCopperLeg(async (year) => { if (year) throw new Error('503'); return jan; }, new Date('2027-01-07T12:00:00Z'));
        expect(leg.historyAsc).toHaveLength(1);
        await expect(westmetallCopperLeg(async () => '<html>blocked</html>', NOW)).rejects.toThrow(/no rows/);
    });
});

describe('copper cascade with Westmetall as tier 2', () => {
    // The live situation on 2026-10-09: CNBC @HG.1 still answers, but its quote froze at 2026-05-27.
    const staleCnbc = { name: 'cnbc', freshnessDays: 7, fetch: async () => ({ current: 4.9, currentDate: '2026-05-27', historyAsc: [{ date: '2026-05-27', price: 4.9 }] }) };
    const westmetall = { name: 'westmetall', freshnessDays: 7, fetch: () => westmetallCopperLeg(async () => HTML, NOW) };
    const goldapi = { name: 'goldapi', freshnessDays: 7, fetch: async () => ({ current: 6.64, currentDate: '2026-10-09', historyAsc: [] }) };

    test('a stale CNBC is DEMOTED (skipped), never served; Westmetall answers with history', async () => {
        const leg = await resolveLeg([staleCnbc, westmetall, goldapi], new Set(), NOW);
        expect(leg.source).toBe('westmetall');
        expect(leg.tried).toEqual(['cnbc:stale(2026-05-27)', 'westmetall:ok']);
        expect(leg.historyAsc.length).toBeGreaterThan(180);
    });

    test('1mo and 3mo deltas come back (were null on gold-api spot alone)', async () => {
        const copper = await resolveLeg([staleCnbc, westmetall, goldapi], new Set(), NOW);
        const goldHist = [];
        for (let d = 120; d >= 0; d--) goldHist.push({ date: new Date(NOW.getTime() - d * 864e5).toISOString().slice(0, 10), price: 4200 });
        const gold = { current: 4200, currentDate: '2026-10-09', historyAsc: goldHist, source: 'polygon', tried: ['polygon:ok'] };
        const cg = buildCopperGold(copper, gold);
        expect(cg.change).not.toBeNull();
        expect(cg.change3mo).not.toBeNull();
        expect(cg.source).toBe('copper:westmetall · gold:polygon');
    });

    test('?_fail=cg_westmetall switches it off → spot-only gold-api, deltas null', async () => {
        const leg = await resolveLeg([staleCnbc, westmetall, goldapi], new Set(['cg_westmetall']), NOW);
        expect(leg.source).toBe('goldapi');
        expect(leg.tried).toEqual(['cnbc:stale(2026-05-27)', 'westmetall:off', 'goldapi:ok']);
    });

    test('a Westmetall page that stopped updating is itself demoted', async () => {
        const later = new Date('2026-10-30T12:00:00Z');
        const wm = { name: 'westmetall', freshnessDays: 7, fetch: () => westmetallCopperLeg(async () => HTML, later) };
        const leg = await resolveLeg([wm, goldapi], new Set(), later);
        expect(leg.tried[0]).toBe('westmetall:stale(2026-10-09)');
    });
});
