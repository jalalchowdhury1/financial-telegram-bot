/**
 * 📡 Market Pulse verdict chips (lib/pulseVerdicts.js): one chip per below-the-fold card,
 * built from the SAME numbers and thresholds that card uses. Missing data = no chip;
 * a saved or stale source = a chip marked old. Fixtures are the real routes' answers.
 */
import React from 'react';
import { render, screen } from '@testing-library/react';
import { pulseVerdicts, horsemenRiding } from '../pulseVerdicts';
import FourHorsemen from '../../components/FourHorsemen';

const FRED = require('./fixtures/pulse-fred-2026-10-04.json');
const VOL = require('./fixtures/pulse-vol-2026-10-04.json');
const RB = require('./fixtures/pulse-rubber-band-2026-10-04.json');

const clone = (o) => JSON.parse(JSON.stringify(o));
const texts = (chips) => chips.map((c) => c.text);

test('the real answers of 4 Oct give the five verdicts in a fixed order, each pointing at its card', () => {
    const chips = pulseVerdicts({ fred: FRED, vol: VOL, rubberBand: RB });
    expect(texts(chips)).toEqual(['Vol calm', 'Horsemen 1/4', 'Curve +0.45%', 'Bull 7/8', 'Dips pay ✓']);
    expect(chips.map((c) => c.tone)).toEqual(['good', 'caution', 'good', 'good', 'good']);
    expect(chips.map((c) => c.jump)).toEqual(['Volatility', 'Recession watch', 'Yield curve', 'Bull checklist', 'Rubber band']);
    expect(chips.every((c) => c.old === null)).toBe(true);
    expect(chips.every((c) => typeof c.why === 'string' && c.why.length > 0)).toBe(true);
    expect(horsemenRiding(FRED)).toEqual({ riding: 1, known: 4 });
});

test('tones follow each card\'s own colour rule', () => {
    const rb = (colour) => pulseVerdicts({ rubberBand: { ...RB, verdict: { colour, text: 'x' } } })[0];
    expect(rb('amber')).toMatchObject({ text: 'Dips: watch', tone: 'watch' });
    expect(rb('red')).toMatchObject({ text: 'Dips: stop', tone: 'bad' });

    const vol = (state) => {
        const v = clone(VOL);
        v.regime.curve.state = state;
        return pulseVerdicts({ vol: v })[0];
    };
    expect(vol('watch')).toMatchObject({ text: 'Vol watch', tone: 'watch' });
    expect(vol('stress')).toMatchObject({ text: 'Vol stress', tone: 'bad' });

    const f = clone(FRED);
    f.yieldCurve.current = -0.12;                       // inverted: also a horseman riding
    f.indicators.sahmRule.value = 0.5;                  // Sahm rule triggered
    const chips = pulseVerdicts({ fred: f });
    expect(texts(chips)).toEqual(['Horsemen 3/4', 'Curve −0.12%', 'Bull 7/8']); // a real minus, as What moved prints it
    expect(chips.map((c) => c.tone)).toEqual(['bad', 'bad', 'good']);

    const calm = clone(FRED);
    calm.horsemen.bankruptcies.changePct = 4;
    expect(pulseVerdicts({ fred: calm })[0]).toMatchObject({ text: 'Horsemen 0/4', tone: 'good' });
    // 1-2 riding / Bull 50-74 % = the cards' badge-yellow ('caution', --yellow); Dips amber and
    // Vol watch are orange on their own cards ('watch', --orange)
    const two = clone(FRED);
    two.yieldCurve.current = -0.12;
    expect(pulseVerdicts({ fred: two })[0]).toMatchObject({ text: 'Horsemen 2/4', tone: 'caution' });

    const bull = (n) => {
        const g = clone(FRED);
        Object.keys(g.checklist).forEach((k, i) => { g.checklist[k].bullish = i < n; });
        return pulseVerdicts({ fred: g }).find((c) => c.key === 'bull');
    };
    expect(bull(4)).toMatchObject({ text: 'Bull 4/8', tone: 'caution' }); // 50 % — badge-yellow on the card
    expect(bull(3)).toMatchObject({ text: 'Bull 3/8', tone: 'bad' });
});

test('the Rubber Band verdict arrives after the others (its card fetches it): it lands at the END, never in front of a chip on screen', () => {
    // On a warm open fred + vol paint from saved copies at once; Dips comes ~0.2 s later.
    // Inserted first, it slid every chip sideways under a thumb aimed at "Vol calm".
    const before = texts(pulseVerdicts({ fred: FRED, vol: VOL, rubberBand: undefined }));
    const after = texts(pulseVerdicts({ fred: FRED, vol: VOL, rubberBand: RB }));
    expect(after.slice(0, before.length)).toEqual(before);
    expect(after[after.length - 1]).toBe('Dips pay ✓');
});

test('missing or failed sources leave their chip out — never a guessed 0 or NaN', () => {
    for (const bad of [null, undefined, {}, { error: 'x' }, 'x', []]) {
        expect(pulseVerdicts({ fred: bad, vol: bad, rubberBand: bad })).toEqual([]);
    }
    // the routes' own fallback bodies
    expect(pulseVerdicts({
        vol: { updated_at: null, tickers: [], _meta: { source: 'Unavailable' } },
        rubberBand: { dials: null, verdict: null, _meta: { messages: ['gist down'] } },
    })).toEqual([]);
    expect(pulseVerdicts({ rubberBand: { ...RB, verdict: { colour: 'grey', text: 'no data' } } })).toEqual([]);
    const v = clone(VOL);
    v.regime.curve = { points: [], state: null };
    expect(pulseVerdicts({ vol: v })).toEqual([]);

    const f = clone(FRED);
    f.yieldCurve.current = null;
    f.checklist = {};
    f.indicators = {};
    f.horsemen = {};
    expect(pulseVerdicts({ fred: f })).toEqual([]);         // nothing known: no "Horsemen 0/4"
    expect(horsemenRiding(f)).toBeNull();

    const all = JSON.stringify(pulseVerdicts({ fred: { yieldCurve: { current: 'N/A' }, checklist: { a: {} } } }));
    expect(all).not.toMatch(/NaN|undefined/);
});

test('a saved copy or a stale source marks its chip old instead of passing for live', () => {
    const chips = pulseVerdicts({ fred: FRED, vol: VOL, rubberBand: RB, saved: { fred: '19:43', vol: 'Fri 16:10' } });
    const old = Object.fromEntries(chips.map((c) => [c.key, c.old]));
    expect(old.dips).toBeNull();
    expect(old.vol).toEqual({ kind: 'saved', note: 'saved copy Fri 16:10' });
    expect(old.horsemen).toEqual({ kind: 'saved', note: 'saved copy 19:43' });
    expect(old.curve).toEqual({ kind: 'saved', note: 'saved copy 19:43' });
    expect(old.bull).toEqual({ kind: 'saved', note: 'saved copy 19:43' });

    const staleRb = { ...RB, _meta: { ...RB._meta, stale: true, ageDays: 6 } };
    expect(pulseVerdicts({ rubberBand: staleRb })[0].old).toEqual({ kind: 'stale', note: 'stale · 6 days old' });
    const staleVol = clone(VOL);
    staleVol.regime.curve.stale = true;
    expect(pulseVerdicts({ vol: staleVol })[0].old).toEqual({ kind: 'stale', note: 'saved copy from 2026-10-02' });
    const staleFred = clone(FRED);
    staleFred.yieldCurve.stale = true;
    expect(pulseVerdicts({ fred: staleFred }).find((c) => c.key === 'curve').old).toEqual({ kind: 'stale', note: 'stale · as of 2026-10-02' });
});

describe('Horsemen chip = the Recession watch card\'s own "N of 4 riding" badge', () => {
    const variants = {
        'real 4 Oct': FRED,
        'claims rising': (() => {
            const f = clone(FRED);
            f.horsemen.claims.history = f.horsemen.claims.history.map((p, i) => ({ ...p, value: 150000 + i * 2000 }));
            return f;
        })(),
        'everything riding': (() => {
            const f = clone(FRED);
            f.horsemen.claims.history = f.horsemen.claims.history.map((p, i) => ({ ...p, value: 150000 + i * 2000 }));
            f.indicators.sahmRule.value = 0.62;
            f.yieldCurve.current = -0.3;
            return f;
        })(),
        'nothing riding': (() => {
            const f = clone(FRED);
            f.horsemen.bankruptcies.changePct = -3;
            return f;
        })(),
    };
    for (const [name, fred] of Object.entries(variants)) {
        it(name, () => {
            const { unmount } = render(<FourHorsemen fred={fred} loading={false} />);
            const badge = screen.getByText(/\d of 4 riding/).textContent;
            const r = horsemenRiding(fred);
            expect(badge).toBe(`${r.riding} of 4 riding`);
            unmount();
        });
    }
});
