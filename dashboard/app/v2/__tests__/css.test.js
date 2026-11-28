/** Stylesheet guards: things jsdom cannot see but a real browser broke on. */
import fs from 'fs';
import path from 'path';

const css = fs.readFileSync(path.join(__dirname, '..', 'v2.css'), 'utf8');

it('the hero / section rise-in never keeps its layer (14 Nov 2026: `both` put the hero on top of the top line\'s links)', () => {
    const rule = css.match(/\.v2-hero, \.v2-section \{ animation:[^}]*\}/)?.[0];
    expect(rule).toMatch(/backwards/);
    expect(rule).not.toMatch(/\b(both|forwards)\b/);
});

it('the chart draw-in is switched on the chart\'s box, never on the svg it clips (the observer looped)', () => {
    expect(css).toMatch(/\)\.is-drawn > svg \{ animation: v2-draw/);
    expect(css).not.toMatch(/svg[^{,]*\.is-drawn\s*\{/);
});

it('the Today card never widens the page: a long verdict scrolls inside the card, cells wrap whole below 1080 px, buttons show focus', () => {
    const verdict = css.match(/\.v2-tc-verdict \{[^}]*\}/)?.[0];
    expect(verdict).toMatch(/min-width: 0/);
    expect(verdict).toMatch(/overflow-x: auto/);
    expect(css.match(/@media \(max-width: 1080px\) \{\n  \.v2-tc \{[\s\S]*?\n\}/)?.[0]).toMatch(/\.v2-tc-cells \{ flex-wrap: wrap;/);
    expect(css).toMatch(/\.v2-tc button \{ all: unset; cursor: pointer; white-space: nowrap; \}/);   // a cell's own text never wraps
    expect(css).toMatch(/\.v2-tc button:focus-visible \{ outline: 1px solid var\(--line-2\); \}/);
});
