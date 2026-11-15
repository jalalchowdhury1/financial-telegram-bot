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
