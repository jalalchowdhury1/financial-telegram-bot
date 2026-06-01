import { copperGoldRatio } from '../finance';

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
