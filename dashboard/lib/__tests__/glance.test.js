import { glanceNumbers, glanceAge, shouldShowGlance } from '../glance';

const spy = { current: 769.64, dailyChange: { value: 5.65, pct: 0.7395 }, rsi: 56.48 };
const fg = { score: 31.2, rating: 'fear' };

describe('glanceNumbers — the same numbers the cards show, never a guess', () => {
    test('SPY price + the spy-daily-move % + the F&G score', () => {
        expect(glanceNumbers({ spy, spyDailyMove: { value: '0.74%' }, fg })).toEqual({
            price: '769.64', move: { up: true, text: '▲0.74%' }, fg: 31, fgScore: 31.2,
        });
        expect(glanceNumbers({ spy, spyDailyMove: { value: '-0.31%' }, fg }).move).toEqual({ up: false, text: '▼0.31%' });
    });
    test("no spy-daily-move: SPY's own daily change; neither: no move at all (not a fake 0.00%)", () => {
        expect(glanceNumbers({ spy, spyDailyMove: { value: null }, fg }).move).toEqual({ up: true, text: '▲0.74%' });
        expect(glanceNumbers({ spy: { current: 700 }, spyDailyMove: null, fg }).move).toBeNull();
        expect(glanceNumbers({ spy: { current: 700 }, spyDailyMove: { value: 'N/A' }, fg }).move).toBeNull();
    });
    test('an F&G score that is not a number is left out', () => {
        expect(glanceNumbers({ spy, fg: { score: 'N/A' } }).fg).toBeNull();
    });
    test('nothing at all when SPY or F&G is missing, errored, or SPY has no price (as Market Pulse)', () => {
        expect(glanceNumbers({ spy: null, fg })).toBeNull();
        expect(glanceNumbers({ spy, fg: null })).toBeNull();
        expect(glanceNumbers({ spy: { error: 'down' }, fg })).toBeNull();
        expect(glanceNumbers({ spy, fg: { error: 'down' } })).toBeNull();
        expect(glanceNumbers({ spy: {}, fg })).toBeNull();
        expect(glanceNumbers({ spy: { current: 'N/A' }, fg })).toBeNull();
        expect(glanceNumbers()).toBeNull();
    });
});

describe('glanceAge — the header badge rule', () => {
    test('a saved copy with no live answer yet shows its time; otherwise the live age', () => {
        expect(glanceAge({ updatedAt: null, saved: '10:42', loading: true })).toEqual({ saved: '10:42' });
        expect(glanceAge({ updatedAt: null, saved: '10:42', loading: false })).toEqual({ saved: '10:42' });
        expect(glanceAge({ updatedAt: 1000, saved: '10:42', loading: true })).toEqual({ saved: '10:42' });
        expect(glanceAge({ updatedAt: 1000, saved: '10:42', loading: false })).toEqual({ at: 1000 });
        expect(glanceAge({ updatedAt: 1000, saved: undefined, loading: false })).toEqual({ at: 1000 });
        expect(glanceAge({ updatedAt: null, saved: undefined, loading: true })).toBeNull();
    });
});

test('shouldShowGlance: only once the anchor is wholly above the top of the screen', () => {
    expect(shouldShowGlance(-1)).toBe(true);
    expect(shouldShowGlance(0)).toBe(false);
    expect(shouldShowGlance(120)).toBe(false);
    expect(shouldShowGlance(undefined)).toBe(false);
    expect(shouldShowGlance(NaN)).toBe(false);
});
