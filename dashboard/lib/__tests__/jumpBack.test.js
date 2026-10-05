/**
 * ↩ Back pill (lib/jumpBack.js): every jump remembers where he was as "section + offset",
 * so the way back survives cards landing (and pushing the page down) while he peeks.
 */
import { pickAnchor, resolveBack, movedTooFar, rememberJump, clearJump, currentJump, onJump } from '../jumpBack';

const SECTIONS = [
    { label: 'SPY overview', top: 900 },
    { label: 'Fear & Greed', top: 1500 },
    { label: 'Economy', top: 3000 },
    { label: 'Volatility', top: 5000 },
];
const VH = 844;

afterEach(() => clearJump());

test('the anchor is the section he is reading (top third of the screen), else Top', () => {
    expect(pickAnchor(0, SECTIONS, VH)).toEqual({ label: 'Top', offset: 0 });
    expect(pickAnchor(300, SECTIONS, VH)).toEqual({ label: 'Top', offset: 300 });         // 900 > 300 + 281
    expect(pickAnchor(700, SECTIONS, VH)).toEqual({ label: 'SPY overview', offset: -200 }); // header just below the top
    expect(pickAnchor(3100, SECTIONS, VH)).toEqual({ label: 'Economy', offset: 100 });
    expect(pickAnchor(9000, SECTIONS, VH)).toEqual({ label: 'Volatility', offset: 4000 });
    // order of the list does not matter; junk rows are skipped
    expect(pickAnchor(3100, [...SECTIONS].reverse().concat([{ label: 'x', top: NaN }, null]), VH)).toEqual({ label: 'Economy', offset: 100 });
    expect(pickAnchor(500, [], VH)).toEqual({ label: 'Top', offset: 500 });
});

test('the way back follows the section when cards above it landed meanwhile', () => {
    const spot = { label: 'Economy', offset: 100, y: 3100 };
    expect(resolveBack(spot, () => 3000)).toBe(3100);
    expect(resolveBack(spot, () => 3392)).toBe(3492);       // the Jev card landed above: +392 px
    expect(resolveBack(spot, () => null)).toBe(3100);        // section gone: the raw spot
    expect(resolveBack({ label: 'Top', offset: 0, y: 0 }, () => 999)).toBe(0);
    expect(resolveBack({ label: 'SPY overview', offset: -200, y: 700 }, () => 100)).toBe(0); // never below 0
    expect(resolveBack(null, () => 0)).toBeNull();
});

test('a manual scroll of more than one screen from where he landed dismisses it', () => {
    expect(movedTooFar(5000, null, VH)).toBe(false);         // still flying to the card
    expect(movedTooFar(5400, 5000, VH)).toBe(false);
    expect(movedTooFar(5000 + VH + 1, 5000, VH)).toBe(true);
    expect(movedTooFar(5000 - VH - 1, 5000, VH)).toBe(true);
});

test('rememberJump stores one spot (a second jump overwrites it) and tells subscribers', () => {
    const seen = [];
    const off = onJump((s) => seen.push(s && s.label));
    rememberJump({ y: 3100, vh: VH, sections: SECTIONS });
    expect(currentJump()).toMatchObject({ label: 'Economy', offset: 100, y: 3100 });
    const first = currentJump().id;
    rememberJump({ y: 5100, vh: VH, sections: SECTIONS });
    expect(currentJump()).toMatchObject({ label: 'Volatility', offset: 100, y: 5100 });
    expect(currentJump().id).not.toBe(first);
    clearJump();
    expect(currentJump()).toBeNull();
    off();
    rememberJump({ y: 0, vh: VH, sections: SECTIONS });
    expect(seen).toEqual(['Economy', 'Volatility', null]);
    expect(rememberJump({ y: NaN, vh: VH, sections: SECTIONS })).toBeNull();
});
