import { copperGoldRatio, priceAtAgo, ratioChange } from '../finance';

describe('copperGoldRatio', () => {
    test('computes copper($/lb) / gold($/oz), scaled x1000', () => {
        // copper ~$6.43/lb, gold ~$4570/oz -> ~1.407
        expect(copperGoldRatio(6.43, 4570)).toBeCloseTo(1.407, 2);
    });

    test('returns null when either leg is missing or zero', () => {
        expect(copperGoldRatio(null, 4570)).toBeNull();
        expect(copperGoldRatio(6.43, 0)).toBeNull();
        expect(copperGoldRatio(undefined, undefined)).toBeNull();
        expect(copperGoldRatio(0, 4570)).toBeNull();
    });

    test('rises when copper outperforms gold', () => {
        const before = copperGoldRatio(6.0, 4570);
        const after = copperGoldRatio(6.5, 4570);
        expect(after).toBeGreaterThan(before);
    });
});

describe('priceAtAgo', () => {
    // A daily series spanning ~100 days, price = 100 + dayIndex.
    const daily = [];
    for (let i = 0; i < 100; i++) {
        const d = new Date('2026-06-06').getTime() - (99 - i) * 86400000;
        daily.push({ date: new Date(d).toISOString().slice(0, 10), price: 100 + i });
    }
    const latest = daily[daily.length - 1].date; // 2026-06-06, price 199

    test('finds the value ~30 days before the latest (daily series)', () => {
        // 30 days before index 99 is index 69 -> price 169
        expect(priceAtAgo(daily, latest, 30)).toBe(169);
    });

    test('finds the value ~90 days before the latest (daily series)', () => {
        // 90 days before index 99 is index 9 -> price 109
        expect(priceAtAgo(daily, latest, 90)).toBe(109);
    });

    test('snaps a monthly series to the prior month for a 30-day window', () => {
        const monthly = [
            { date: '2026-01-01', price: 11000 },
            { date: '2026-02-01', price: 12000 },
            { date: '2026-03-01', price: 12500 },
            { date: '2026-04-01', price: 12900 },
            { date: '2026-05-01', price: 13480 }, // latest
        ];
        expect(priceAtAgo(monthly, '2026-05-01', 30)).toBe(12900); // April
        expect(priceAtAgo(monthly, '2026-05-01', 90)).toBe(12000); // Feb (~89d)
    });

    test('returns null when history is too short to span the window', () => {
        const short = [
            { date: '2026-06-01', price: 5 },
            { date: '2026-06-06', price: 6 },
        ];
        expect(priceAtAgo(short, '2026-06-06', 30)).toBeNull(); // closest is 5d ago, >tol from 30d target
    });

    test('guards against bad input', () => {
        expect(priceAtAgo(null, '2026-06-06', 30)).toBeNull();
        expect(priceAtAgo([{ date: '2026-06-06', price: 1 }], '2026-06-06', 30)).toBeNull();
        expect(priceAtAgo(daily, 'not-a-date', 30)).toBeNull();
    });
});

describe('ratioChange', () => {
    test('computes value + pct', () => {
        const c = ratioChange(1.44, 1.40);
        expect(c.value).toBeCloseTo(0.04, 5);
        expect(c.pct).toBeCloseTo(2.857, 2);
    });
    test('negative when falling', () => {
        expect(ratioChange(1.36, 1.44).value).toBeLessThan(0);
    });
    test('null on missing/zero base', () => {
        expect(ratioChange(1.44, null)).toBeNull();
        expect(ratioChange(null, 1.44)).toBeNull();
        expect(ratioChange(1.44, 0)).toBeNull();
    });
});
