import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SpyChart, { joinBaked } from '../SpyChart';
import { decodeLong } from '../../lib/longHistory';

if (!window.PointerEvent) {
    window.PointerEvent = class PointerEvent extends MouseEvent {
        constructor(type, params = {}) { super(type, params); this.pointerType = params.pointerType || 'mouse'; }
    };
}

// n daily bars ending 2026-09-25, price p0 + i*step
const bars = (n, p0, step = 0.1) => {
    const out = [];
    const end = Date.UTC(2026, 8, 25);
    for (let i = 0; i < n; i++) {
        const d = new Date(end - (n - 1 - i) * 864e5).toISOString().slice(0, 10);
        const p = p0 + i * step;
        out.push({ date: d, price: p, ma50: p, ma200: p });
    }
    return out;
};

beforeEach(() => {
    window.localStorage.clear();
    global.fetch = jest.fn(() => Promise.reject(new Error('no baked file')));
});

// the baked file's shape: {from, t: day offsets, v}
const bakedFile = (from, n, v) => ({ from, t: [...Array(n).keys()], v: Array(n).fill(v) });

test('the baked daily file unlocks 5Y / 10Y / ALL', async () => {
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, json: () => bakedFile('2010-01-01', 6000, 600) }));
    render(<SpyChart chartHistory={bars(302, 600)} current={630} />);
    await waitFor(() => expect(screen.getByRole('button', { name: '5Y' })).not.toBeDisabled());
    await waitFor(() => expect(screen.getByRole('button', { name: '10Y' })).not.toBeDisabled());
    expect(global.fetch).toHaveBeenCalledWith('/history/spy-daily.json');
});

test('joinBaked: baked closes only before the live window, MAs over the whole run; index path keeps live only', () => {
    const live = bars(302, 600, 0);
    const baked = Array.from({ length: 1100 }, (_, i) => ({ date: new Date(Date.UTC(2024, 0, 1) + i * 864e5).toISOString().slice(0, 10), value: 600 }));
    const out = joinBaked(baked, live);
    expect(out.length).toBe(baked.filter((p) => p.date < live[0].date).length + 302);
    expect(out[out.length - 1]).toEqual({ date: live[301].date, price: 600, ma50: 600, ma200: 600 });
    expect(joinBaked(baked, bars(302, 6000, 0))).toHaveLength(302); // S&P index points ≠ SPY dollars
    expect(joinBaked(null, live)).toBe(live);
});

test('14 months of bars (the Polygon path) cannot pose as a 5Y chart', () => {
    render(<SpyChart chartHistory={bars(302, 600)} current={630} />);
    expect(screen.getByRole('button', { name: '1Y' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: '5Y' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '10Y' })).toBeDisabled();
    // the active tab fell back to ALL
    expect(screen.getByRole('button', { name: 'ALL' }).style.color).toBe('rgb(56, 189, 248)');
    expect(document.querySelector('.chart-note')).toBeNull();
});

test('index-point history (Sheet path) is labelled as index points, not dollars', () => {
    const { container } = render(<SpyChart chartHistory={bars(1500, 6000, 1)} current={771.35} />);
    const labels = [...container.querySelectorAll('.axis-lbl')].map((t) => t.textContent);
    expect(labels.length).toBeGreaterThan(4); // the y ticks are there, just not in dollars
    expect(labels.some((l) => l.startsWith('$'))).toBe(false);
    expect(container.querySelector('.chart-note').textContent).toMatch(/S&P 500 index points/);
});

test('tap readout shows date and SPY price; the choice is remembered', () => {
    const { container, unmount } = render(<SpyChart chartHistory={bars(1500, 400)} current={549.9} />);
    fireEvent.click(screen.getByRole('button', { name: '1Y' }));
    expect(window.localStorage.getItem('ftb:tf:spy')).toBe('1Y');
    const svg = container.querySelector('svg');
    svg.getBoundingClientRect = () => ({ left: 0, width: 540, top: 0, height: 220 });
    fireEvent.pointerDown(svg, { clientX: 532, pointerType: 'touch' });
    expect(container.querySelector('.chart-readout').textContent).toBe('Sep 25, 2026 · $549.90');
    unmount();
    render(<SpyChart chartHistory={bars(1500, 400)} current={549.9} />);
    expect(screen.getByRole('button', { name: '1Y' }).style.color).toBe('rgb(56, 189, 248)');
});

test('axis labels are HTML spans over the chart, placed by % of the viewBox, never SVG text', () => {
    const { container } = render(<SpyChart chartHistory={bars(1500, 400)} current={549.9} />);
    expect(container.querySelectorAll('svg text').length).toBe(0);
    const labels = [...container.querySelectorAll('.axis-layer .axis-lbl')];
    const texts = labels.map((l) => l.textContent);
    expect(texts).toContain('2026'); // a year label
    expect(texts.some((t) => /^\$\d+$/.test(t))).toBe(true); // a y tick: SPY dollars
    for (const l of labels) {
        expect(l.style.left).toMatch(/%$/);
        expect(l.style.top).toMatch(/%$/);
    }
    expect(container.querySelector('.axis-layer').getAttribute('aria-hidden')).toBe('true');
});

// The committed bake (scripts/bake_spy_daily.py). /api/spy carries 302 bars (~437 days), so
// a bake older than that leaves a hole before the live window; joinBaked then drops it and
// 5Y/10Y/ALL go grey again. The quarterly rebake keeps it ≤ ~3 months old: red here at 300
// days = three missed quarters, before the chart breaks.
test('baked SPY file: daily since 1993, sane closes, fresh enough to meet the live window', () => {
    const body = require('../../public/history/spy-daily.json');
    const pts = decodeLong(body);
    expect(pts.length).toBeGreaterThan(8000);
    expect(pts[0].date).toBe('1993-01-29');
    expect(pts.every((p, i) => i === 0 || p.date > pts[i - 1].date)).toBe(true);
    expect(pts.every((p) => p.value > 20 && p.value < 5000)).toBe(true); // SPY dollars, not index points
    expect((Date.now() - Date.parse(body.to)) / 864e5).toBeLessThan(300);
});

test('1M / 3M / 6M tabs work on the live 14 months and label weeks / months, not years', () => {
    const { container } = render(<SpyChart chartHistory={bars(302, 600)} current={630} />);
    for (const tf of ['1M', '3M', '6M']) expect(screen.getByRole('button', { name: tf })).not.toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: '1M' }));
    expect(screen.getByRole('button', { name: '1M' }).style.color).toBe('rgb(56, 189, 248)');
    const x = [...container.querySelectorAll('.axis-lbl')].map((t) => t.textContent).filter((t) => !t.startsWith('$'));
    expect(x.length).toBeGreaterThan(1);
    expect(x.every((t) => /^[A-Z][a-z]{2} \d{1,2}$/.test(t))).toBe(true); // 'Sep 8', never '2026'
    expect(window.localStorage.getItem('ftb:tf:spy')).toBe('1M');
});

test('📅 picks any from / to dates; a range with no trading days says so and keeps the tab', () => {
    // flat 600 until 2026-06-30, then 660: the return over picked dates shows exactly which bars were used
    const hist = bars(302, 600, 0).map((d) => (d.date > '2026-06-30' ? { ...d, price: 660 } : d));
    render(<SpyChart chartHistory={hist} current={660} />);
    fireEvent.click(screen.getByRole('button', { name: 'Pick dates' }));
    fireEvent.change(screen.getByLabelText('From'), { target: { value: '2026-06-01' } });
    fireEvent.change(screen.getByLabelText('To'), { target: { value: '2026-06-30' } });
    expect(screen.getByText(/% return/).textContent).toBe('+0.0% return');
    fireEvent.change(screen.getByLabelText('To'), { target: { value: '2026-07-31' } });
    expect(screen.getByText(/% return/).textContent).toBe('+10.0% return');
    expect(screen.getByRole('button', { name: 'ALL' }).style.background).toBe('rgba(255, 255, 255, 0.05)'); // no preset lit
    fireEvent.change(screen.getByLabelText('From'), { target: { value: '2027-01-01' } });
    expect(screen.getByText(/Under 2 trading days/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '1Y' })); // a tab closes the picker
    expect(screen.queryByLabelText('From')).toBeNull();
});

// FactSet's file for the lens tests: one shared copy, since loadFwd keeps the first answer for the whole file
const fwdFile = { from: '2026-01-02', to: '2026-09-18', t: [0, 259], v: [22, 19], a5: [19.8, 19.8], a10: [19.1, 19.1] };
const fwdFetch = () => jest.fn((url) => Promise.resolve(url === '/history/fwdPE.json' ? { ok: true, json: async () => fwdFile } : { ok: false }));

test('the chip rows never slide up or down: one scroll over them hid the lens buttons (1 Nov 2026)', async () => {
    // each chip's tap band sits 12px below it, so a sideways-scrolling row is taller inside than out
    global.fetch = fwdFetch();
    render(<SpyChart chartHistory={bars(302, 600)} current={630} lenses={{ peNow: null }} />);
    fireEvent.click(screen.getByRole('tab', { name: 'Fwd P/E' }));
    expect(screen.getByRole('tablist', { name: 'Chart' }).style.overflowY).toBe('hidden');
    await waitFor(() => expect(screen.getByRole('button', { name: '1Y' }).parentElement.style.overflowY).toBe('hidden')); // the lens's own row
    fireEvent.click(screen.getByRole('tab', { name: 'Price' }));
    expect(screen.getByRole('button', { name: '1Y' }).parentElement.style.overflowY).toBe('hidden');
});

test('Earnings view: SPY\'s move drawn as earnings × P/E, bars that add up on a log scale, since the window start or a tapped day', async () => {
    global.fetch = fwdFetch();
    const { container } = render(<SpyChart chartHistory={bars(302, 600)} current={630} lenses={{ peNow: null }} />);
    fireEvent.click(screen.getByRole('tab', { name: 'Earnings' }));
    const split = await screen.findByRole('img', { name: /^Since / });
    // SPY 603.4 → 630.1 while the P/E went 22 → 19.0: earnings did more than all of it
    expect(split.getAttribute('aria-label')).toBe('Since Jan 1, 2026: earnings +21%, P/E −14%, SPY +4%');
    expect(split.textContent).toBe('Earnings+21%× P/E−14%= SPY+4.4%');
    const [e, p, spy] = [...split.querySelectorAll('span[style*="border-radius"]')].map((b) => parseFloat(b.style.width));
    expect(e - p).toBeCloseTo(spy, 1); // the P/E bar runs back from where earnings ended, to SPY's end
    expect(screen.getByText('Since Jan 1, 2026')).toBeTruthy();
    const svg = container.querySelector('.chart-plot svg');
    svg.getBoundingClientRect = () => ({ left: 0, width: 540, top: 0, height: 220 });
    fireEvent.pointerDown(svg, { clientX: 300, pointerType: 'touch' });
    const readout = container.querySelector('.chart-readout').textContent;
    expect(readout).toMatch(/^Since \w{3} \d+, 2026 · SPY \$\d+$/);
    expect(split.getAttribute('aria-label').startsWith(`${readout.split(' · ')[0]}:`)).toBe(true); // the picture follows the tap
});

test('Fwd P/E lens: round y ticks, start → end up top, SPY 6 months later on tap, "est." once FactSet goes quiet', async () => {
    global.fetch = fwdFetch();
    const { container, unmount } = render(<SpyChart chartHistory={bars(302, 600)} current={630} lenses={{ peNow: null }} />);
    fireEvent.click(screen.getByRole('tab', { name: 'Fwd P/E' }));
    await waitFor(() => expect(container.querySelector('.axis-layer')).not.toBeNull());
    fireEvent.click(screen.getByRole('button', { name: 'ALL' }));
    const ticks = [...container.querySelectorAll('.axis-layer .axis-lbl')].map((l) => l.textContent).filter((t) => /x$/.test(t));
    expect(ticks).toEqual(['19x', '20x', '21x', '22x']); // each label on its own gridline, never 21.4x
    expect(screen.getByText('22.0x → 19.0x')).toBeTruthy();
    expect(screen.getByText('━ Fwd P/E')).toBeTruthy(); // the newest report is a week old: no "est."
    const svg = container.querySelector('.chart-plot svg'); // not a view button's sparkline
    svg.getBoundingClientRect = () => ({ left: 0, width: 540, top: 0, height: 220 });
    fireEvent.pointerDown(svg, { clientX: 60, pointerType: 'touch' });
    expect(container.querySelector('.chart-readout').textContent).toMatch(/^Jan \d+, 2026 · \d+\.\dx · SPY \+3% in 6M$/);
    unmount();
    // ten more days of SPY and no new report: the line still moves with SPY, and says so
    const later = bars(302, 600).map((b) => ({ ...b, date: new Date(Date.parse(b.date) + 10 * 864e5).toISOString().slice(0, 10) }));
    render(<SpyChart chartHistory={later} current={630} lenses={{ peNow: null }} />);
    fireEvent.click(screen.getByRole('tab', { name: 'Fwd P/E' }));
    await waitFor(() => expect(screen.getByText('━ Fwd P/E · est. since Sep 18')).toBeTruthy());
});
