import { glanceNumbers, glanceAge, shouldShowGlance } from '../glance';

const spy = { current: 769.64, dailyChange: { value: 5.65, pct: 0.7395 }, rsi: 56.48 };
const fg = { score: 31.2, rating: 'fear' };

describe('glanceNumbers — the same numbers the cards show, never a guess', () => {
    test('SPY price + the spy-daily-move % + the F&G score', () => {
        expect(glanceNumbers({ spy, spyDailyMove: { value: '0.74%' }, fg })).toEqual({
            price: '769.64', move: { up: true, text: '▲0.74%', when: null }, fg: 31, fgScore: 31.2,
        });
        expect(glanceNumbers({ spy, spyDailyMove: { value: '-0.31%' }, fg }).move).toEqual({ up: false, text: '▼0.31%', when: null });
    });
    test("no spy-daily-move: SPY's own daily change; neither: no move at all (not a fake 0.00%)", () => {
        expect(glanceNumbers({ spy, spyDailyMove: { value: null }, fg }).move).toEqual({ up: true, text: '▲0.74%', when: null });
        expect(glanceNumbers({ spy: { current: 700 }, spyDailyMove: null, fg }).move).toBeNull();
        expect(glanceNumbers({ spy: { current: 700 }, spyDailyMove: { value: 'N/A' }, fg }).move).toBeNull();
    });
    test('an F&G score that is not a number is left out', () => {
        expect(glanceNumbers({ spy, fg: { score: 'N/A' } }).fg).toBeNull();
    });
    test('the move names its session like the SPY card: "Fri" all weekend, nothing extra on the day itself', () => {
        expect(glanceNumbers({ spy, spyDailyMove: { value: '0.74%' }, fg, when: 'Fri' }).move).toEqual({ up: true, text: '▲0.74%', when: 'Fri' });
        expect(glanceNumbers({ spy, spyDailyMove: { value: '0.74%' }, fg, when: 'today' }).move.when).toBeNull();
        expect(glanceNumbers({ spy, spyDailyMove: { value: '0.74%' }, fg, when: '' }).move.when).toBeNull();
        expect(glanceNumbers({ spy: { current: 700 }, fg, when: 'Fri' }).move).toBeNull(); // no move: no lone "Fri"
    });
    test('F&G missing or errored: SPY and the ↻ stay (a failed feed is when ↻ matters most)', () => {
        for (const bad of [null, undefined, { error: 'down' }, {}]) {
            const n = glanceNumbers({ spy, spyDailyMove: { value: '0.74%' }, fg: bad });
            expect(n).toMatchObject({ price: '769.64', fg: null });
            expect(n.move.text).toBe('▲0.74%');
        }
    });
    test('nothing at all when SPY is missing, errored, or has no price', () => {
        expect(glanceNumbers({ spy: null, fg })).toBeNull();
        expect(glanceNumbers({ spy: { error: 'down' }, fg })).toBeNull();
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
