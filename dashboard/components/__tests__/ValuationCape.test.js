import { valuationRow } from '../EconomicIndicatorGrid';

describe('Market Valuation row — CAPE fallback is labelled CAPE, never P/E', () => {
    test('TTM P/E from multpl reads "P/E ~26.7" and is marked/charted as peRatio', () => {
        const r = valuationRow({ peRatio: 26.66, peSource: 'multpl', peIsCape: false, peRatioAsOf: '2026-10-09T12:00:00Z' });
        expect(r.value).toBe('P/E ~26.7');
        expect(r.markKey).toBe('peRatio');
        expect(r.raw).toBe(26.66);
        expect(r.benchmark).toBe('Fair at ~20');
    });

    test('FRED PE10 fallback reads "CAPE ~38.4", with CAPE\'s own norm and no P/E mark', () => {
        const r = valuationRow({ peRatio: 38.4, peSource: 'cape', peIsCape: true, peRatioAsOf: '2026-09-01' });
        expect(r.value).toBe('CAPE ~38.4');
        expect(r.value).not.toMatch(/P\/E ~/);
        expect(r.benchmark).toMatch(/CAPE, not P\/E/);
        expect(r.tooltip).toMatch(/Shiller CAPE/);
        // Kept out of the since-last-visit mark and the P/E history chart.
        expect(r.markKey).toBeNull();
        expect(r.raw).toBeUndefined();
        expect(r.metric.asOf).toBe('2026-09-01');
    });

    test('no P/E at all → "P/E N/A"', () => {
        expect(valuationRow({ peRatio: null }).value).toBe('P/E N/A');
        expect(valuationRow(undefined).value).toBe('P/E N/A');
    });
});
