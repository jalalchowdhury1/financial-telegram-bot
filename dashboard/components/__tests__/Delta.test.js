import { render, screen, fireEvent, act } from '@testing-library/react';
import Delta from '../Delta';

const printMark = {
    kind: 'print', value: 5.0, prev: 8.19, dir: -1,
    heldFrom: '2026-06-03', heldDays: 22, runs: [4.47, -2.71, 8.12, 8.19, 5.0],
};
const moveMark = {
    kind: 'move', value: 1.50, prev: 1.56, dir: -1, sigma: 0.02, move: -0.06,
    runs: [1.61, 1.58, 1.60, 1.55, 1.57, 1.56, 1.50],
};

beforeAll(() => {
    // jsdom has no IntersectionObserver; Delta falls back to marking `seen` immediately.
    global.IntersectionObserver = undefined;
});

describe('rendering', () => {
    test('an unmarked number renders as plain text with no glyph and no button role', () => {
        const { container } = render(<Delta mark={null}>5.0%</Delta>);
        expect(screen.getByText('5.0%')).toBeInTheDocument();
        expect(container.querySelector('.mark-glyph')).toBeNull();
        expect(screen.queryByRole('button')).toBeNull();
    });

    test('a print mark renders a dot', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        expect(container.querySelector('.mark-dot')).toBeInTheDocument();
        expect(container.querySelector('[data-mark="print"]')).toBeInTheDocument();
    });

    test('a move mark renders a directional chevron, not a dot', () => {
        const { container } = render(<Delta mark={moveMark}>1.50</Delta>);
        expect(container.querySelector('.mark-dot')).toBeNull();
        expect(container.querySelector('.mark-glyph').textContent).toBe('⌄');
        const up = render(<Delta mark={{ ...moveMark, dir: 1 }}>1.62</Delta>);
        expect(up.container.querySelector('.mark-glyph').textContent).toBe('⌃');
    });

    test('the current value is always rendered, marked or not', () => {
        render(<Delta mark={printMark}>5.0%</Delta>);
        expect(screen.getByText('5.0%')).toBeInTheDocument();
    });
});

describe('the reveal', () => {
    test('a tap opens exactly one popover, and it mounts on document.body', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(0);

        fireEvent.click(container.querySelector('[data-mark]'));
        const pops = document.querySelectorAll('.mark-pop');
        expect(pops).toHaveLength(1);

        // it must escape the card's stacking context, so it cannot be inside the trigger
        expect(container.contains(pops[0])).toBe(false);
        expect(document.body.contains(pops[0])).toBe(true);
    });

    test('shows the previous value, the delta, and how long it held', () => {
        const { container } = render(<Delta mark={printMark} format={(v) => `${v.toFixed(2)}%`}>5.0%</Delta>);
        fireEvent.click(container.querySelector('[data-mark]'));
        expect(screen.getByText('8.19%')).toBeInTheDocument();
        expect(screen.getByText(/held 22 days/)).toBeInTheDocument();
        expect(screen.getByText(/Jun 3/)).toBeInTheDocument();
        expect(screen.getByText(/▼/)).toBeInTheDocument();
    });

    test('a move mark explains itself as a 2 sigma move, not a print', () => {
        const { container } = render(<Delta mark={moveMark}>1.50</Delta>);
        fireEvent.click(container.querySelector('[data-mark]'));
        expect(screen.getByText('Yesterday')).toBeInTheDocument();
        expect(screen.getByText(/2σ of its own daily range/)).toBeInTheDocument();
    });

    test('clicking the dot opens it too', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        fireEvent.click(container.querySelector('.mark-glyph'));
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(1);
    });

    test('tapping again closes it', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        const trigger = container.querySelector('[data-mark]');
        fireEvent.click(trigger);
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(1);
        fireEvent.click(trigger);
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(0);
    });

    test('a real double-click (click, click, dblclick) leaves it open, once', () => {
        const { container } = render(<Delta mark={printMark}>218</Delta>);
        const t = container.querySelector('[data-mark]');
        fireEvent.click(t, { detail: 1 });
        fireEvent.click(t, { detail: 2 });
        fireEvent.doubleClick(t);
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(1);
    });

    test('Escape closes it', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        fireEvent.click(container.querySelector('[data-mark]'));
        fireEvent.keyDown(document, { key: 'Escape' });
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(0);
    });

    test('a click outside closes it, a click inside does not', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        fireEvent.click(container.querySelector('[data-mark]'));
        fireEvent.mouseDown(document.querySelector('.mark-pop'));
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(1);
        fireEvent.mouseDown(document.body);
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(0);
    });

    test('keyboard: Enter opens it', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        fireEvent.keyDown(container.querySelector('[data-mark]'), { key: 'Enter' });
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(1);
    });
});

describe('resilience', () => {
    test('omits the sparkline rather than breaking when there are too few points', () => {
        const { container } = render(<Delta mark={{ ...printMark, runs: [5.0] }}>5.0%</Delta>);
        fireEvent.click(container.querySelector('[data-mark]'));
        expect(document.querySelector('.mark-pop')).toBeInTheDocument();
        expect(document.querySelector('.mark-spark')).toBeNull();
    });

    test('renders with no runs array at all', () => {
        const { container } = render(<Delta mark={{ ...printMark, runs: undefined }}>5.0%</Delta>);
        expect(() => fireEvent.click(container.querySelector('[data-mark]'))).not.toThrow();
        expect(document.querySelector('.mark-pop')).toBeInTheDocument();
    });
});

/**
 * REGRESSION — caught in the browser on 2026-08-25. The cards refuse to mark a stale
 * value, but the header chip counted one anyway, so a Sheet-last-known-good load showed
 * "4 new prints" above a page with no marks on it. The two must apply the same rule.
 */
describe('collectLiveValues freshness', () => {
    const { collectLiveValues } = require('../MarkProvider');

    const metric = (over) => ({ value: 5, stale: false, unavailable: false, ...over });

    test('passes a fresh value through', () => {
        const v = collectLiveValues({ checklist: { nfci: metric({ value: -0.56 }) } }, null, null);
        expect(v.nfci).toBe(-0.56);
    });

    test('withholds a stale value, so the chip cannot count what no card will mark', () => {
        const v = collectLiveValues({ checklist: { nfci: metric({ stale: true }) } }, null, null);
        expect(v.nfci).toBeUndefined();
    });

    test('withholds an unavailable value', () => {
        const v = collectLiveValues({ indicators: { claims: metric({ unavailable: true }) } }, null, null);
        expect(v.claims).toBeUndefined();
    });

    test('withholds a stale yield curve and profit margin', () => {
        const v = collectLiveValues(
            { yieldCurve: { current: 0.5, stale: true }, profitMargin: { current: 14.9, stale: true } },
            null, null,
        );
        expect(v.yieldCurve).toBeUndefined();
        expect(v.profitMargin).toBeUndefined();
    });

    test('survives entirely absent payloads', () => {
        expect(() => collectLiveValues(null, null, null)).not.toThrow();
    });
});

describe('📈 tap for the 90-day chart', () => {
    const { MarkProvider } = require('../MarkProvider');
    const series = { from: '2026-06-29', days: 90, v: { claims: Array.from({ length: 90 }, (_, i) => 200 + (i % 5)) } };
    const wrap = (ui) => render(<MarkProvider history={null} series={series}>{ui}</MarkProvider>);

    test('a number with history but no mark is tappable and opens its chart', async () => {
        global.fetch = jest.fn(() => Promise.resolve({ ok: false, status: 404 }));
        wrap(<Delta mark={null} chartKey="claims" raw={204}>204</Delta>);
        const btn = screen.getByRole('button');
        expect(btn).toHaveClass('chartable');
        await act(async () => { fireEvent.click(btn); });
        require('../../lib/longHistory').resetLong();
        delete global.fetch;
        const pop = document.querySelector('.mark-pop');
        expect(pop.textContent).toMatch(/Initial Claims \(4wk\) · 90 days/);
        expect(pop.textContent).toMatch(/low 200/);
        expect(pop.textContent).toMatch(/Sep 26: 204/);
        expect(pop.querySelector('svg polyline')).not.toBeNull();
        fireEvent.click(btn);
        expect(document.querySelector('.mark-pop')).toBeNull();
    });

    test('range chips: 3M by default, a tap re-slices without closing, and the pick carries to the next popover', async () => {
        require('../../lib/chartRange').resetRange();
        window.localStorage.clear();
        global.fetch = jest.fn(() => Promise.resolve({ ok: false, status: 404 })); // no baked file → sheet only
        wrap(<Delta mark={null} chartKey="claims" raw={204}>204</Delta>);
        const btn = document.querySelector('.chartable');
        await act(async () => { fireEvent.click(btn); });
        const on = () => document.querySelector('.series-tf.is-on').textContent;
        expect(on()).toBe('3M');
        fireEvent.click(screen.getByRole('button', { name: '1M' }));
        expect(document.querySelector('.mark-pop')).not.toBeNull();
        expect(on()).toBe('1M');
        expect(document.querySelector('.mark-pop').textContent).toMatch(/· 30 days/);
        fireEvent.click(screen.getByRole('button', { name: 'MAX' }));
        expect(document.querySelector('.mark-pop').textContent).toMatch(/· since Jun 29/);
        expect(document.querySelector('.mark-pop-foot').textContent).toBe('daily snapshots · history sheet');
        fireEvent.click(btn); await act(async () => { fireEvent.click(btn); }); // close, reopen
        expect(on()).toBe('MAX');
        require('../../lib/chartRange').resetRange();
        window.localStorage.clear();
        require('../../lib/longHistory').resetLong();
        delete global.fetch;
    });

    test('📜 baked long history: fetched on open, drawn BEFORE the sheet, 1Y/5Y chips, source in the footer', async () => {
        require('../../lib/chartRange').resetRange();
        window.localStorage.clear();
        require('../../lib/longHistory').resetLong();
        // weekly claims from 1990-01-05 to 2026-09-25: the part from Jun 29 on must be dropped
        const t = Array.from({ length: 1900 }, (_, i) => i * 7);
        const body = { key: 'claims', from: '1990-01-05', t, v: t.map((d) => 300 + (d % 50)) };
        global.fetch = jest.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(body) }));
        wrap(<Delta mark={null} chartKey="claims" raw={204}>204</Delta>);
        await act(async () => { fireEvent.click(document.querySelector('.chartable')); });
        expect(global.fetch).toHaveBeenCalledWith('/history/claims.json');
        expect([...document.querySelectorAll('.series-tf')].map((b) => b.textContent)).toEqual(['1M', '3M', '6M', '1Y', '5Y', 'MAX']);
        const pop = () => document.querySelector('.mark-pop').textContent;
        expect(pop()).toMatch(/· 90 days/);                       // 3M: the sheet alone
        expect(document.querySelector('.mark-pop-foot').textContent).toBe('daily snapshots · history sheet');
        fireEvent.click(screen.getByRole('button', { name: '5Y' }));
        expect(pop()).toMatch(/· 5 years/);
        fireEvent.click(screen.getByRole('button', { name: 'MAX' }));
        expect(pop()).toMatch(/· since Jan 5, 1990/);
        expect(pop()).toMatch(/Jan 1990: 3\d\d/);                  // month + year once the span is long
        expect(document.querySelector('.mark-pop-foot').textContent).toBe('FRED ICSA (4-week avg) · snapshots from Jun 29');
        expect(document.querySelector('.series-join')).not.toBeNull(); // where the snapshots take over
        // the sheet's own points are all still there, unchanged, after the baked ones
        expect(pop()).toMatch(/Sep 2026: 204/);
        require('../../lib/chartRange').resetRange();
        window.localStorage.clear();
        require('../../lib/longHistory').resetLong();
        delete global.fetch;
    });

    test('📜 a stat with no baked file shows only the chips its history can fill; a remembered 5Y falls back to MAX', () => {
        require('../../lib/chartRange').resetRange();
        window.localStorage.setItem('ftb:chartRange', '5Y');
        const s = { from: '2026-03-12', days: 199, v: { usdbdt: Array.from({ length: 199 }, (_, i) => 120 + (i % 3)) } };
        render(<MarkProvider history={null} series={s}><Delta mark={null} chartKey="usdbdt" raw={122}>122</Delta></MarkProvider>);
        fireEvent.click(document.querySelector('.chartable'));
        expect([...document.querySelectorAll('.series-tf')].map((b) => b.textContent)).toEqual(['1M', '3M', '6M', 'MAX']);
        expect(document.querySelector('.series-tf.is-on').textContent).toBe('MAX');
        expect(document.querySelector('.mark-pop').textContent).toMatch(/· since Mar 12/);
        require('../../lib/chartRange').resetRange();
        window.localStorage.clear();
    });

    test('📜 the old ALL pick (saved before 1Y/5Y existed) opens as MAX', () => {
        require('../../lib/chartRange').resetRange();
        window.localStorage.setItem('ftb:chartRange', 'ALL');
        expect(require('../../lib/chartRange').getRange()).toBe('MAX');
        require('../../lib/chartRange').resetRange();
        window.localStorage.clear();
    });

    test('a marked number opens with a single tap and shows the chart under the mark', async () => {
        global.fetch = jest.fn(() => Promise.resolve({ ok: false, status: 404 }));
        const { container } = wrap(<Delta mark={printMark} chartKey="claims" raw={204}>204</Delta>);
        await act(async () => { fireEvent.click(container.querySelector('[data-mark]')); });
        require('../../lib/longHistory').resetLong();
        delete global.fetch;
        const pop = document.querySelector('.mark-pop');
        expect(pop.textContent).toMatch(/Before this print/);
        expect(pop.textContent).toMatch(/90 days/);
    });

    test('no history, or history of a different basis → plain text, not tappable', () => {
        wrap(<Delta mark={null} chartKey="sahmRule" raw={0.03}>0.03</Delta>);
        wrap(<Delta mark={null} chartKey="claims" raw={204000}>204k</Delta>);
        expect(screen.queryByRole('button')).toBeNull();
    });
});
