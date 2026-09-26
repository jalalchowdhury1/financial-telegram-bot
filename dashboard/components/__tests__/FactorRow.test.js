import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import FactorRow, { rankFactors, isStale, isLongStale, fmtPct, perYear, WINDOWS } from '../FactorRow';

const win = (rel, f, b, from = '2026-03-25') => ({ rel, f, b, from, to: '2026-09-25', spark: [0, rel / 2, rel] });
const factor = (key, label, short, ticker, rels, extra = {}) => ({
    key, label, short, ticker, what: `${label} explained`, asOf: '2026-09-25', stale: false,
    windows: Object.fromEntries(WINDOWS.map((w, i) => [w, rels[i] == null ? null : win(rels[i], rels[i] + 5, 5)])),
    ...extra,
});

//                                   1M    3M    6M    YTD   1Y    3Y    5Y    10Y
const payload = {
    asOf: '2026-09-25',
    bench: 'SPY',
    windows: WINDOWS,
    factors: [
        factor('value', 'Value', 'Value', 'VLUE', [1.2, -2.0, -3.1, -4.0, -6.0, -10, -15, null]),
        factor('momentum', 'Momentum', 'Mom.', 'MTUM', [0.5, 3.0, 6.2, 7.0, 9.9, 12, 20, null]),
        factor('quality', 'Quality', 'Quality', 'QUAL', [0.1, 1.0, 0.4, 1.1, 2.2, 3, 4, null]),
        factor('size', 'Small caps', 'Size', 'IWM', [-0.3, -1.0, -1.2, -2.0, -8.0, -20, -30, null]),
        factor('lowvol', 'Low vol', 'Low vol', 'USMV', [-1.0, -3.0, -4.0, -5.0, -7.0, -9, -12, null]),
    ],
    _meta: { source: 'SPY:cnbc', hasErrors: false, stale: false },
};

const vals = () => [...document.querySelectorAll('.factor-chip .factor-val')].map((e) => e.textContent);
const SIX_M = ['−3.1%', '+6.2%', '+0.4%', '−1.2%', '−4.0%'];
const ready = () => waitFor(() => expect(vals()).toEqual(SIX_M));

beforeEach(() => {
    window.localStorage.clear();
    global.fetch = jest.fn().mockResolvedValue({ json: async () => payload });
});
afterEach(() => jest.restoreAllMocks());

it('fetches and shows every factor for the default 6M window, leader first in the caption', async () => {
    render(<FactorRow />);
    await ready();
    expect(screen.getByRole('radio', { name: '6M' })).toHaveAttribute('aria-checked', 'true');
    const caption = document.querySelector('.factor-caption');
    expect(caption.textContent).toMatch(/6M leader Momentum \+6\.2%/);
    expect(caption.textContent).toMatch(/laggard Low vol −4\.0%/);
    expect(caption.textContent).toMatch(/price ratio, through Sep 25/);
    expect(document.querySelector('.factor-chip.is-leader').textContent).toMatch(/Momentum/);
});

it('switching the timeline re-renders every chip and is remembered', async () => {
    render(<FactorRow />);
    await ready();
    fireEvent.click(screen.getByRole('radio', { name: '5Y' }));
    expect(vals()).toEqual(['−15.0%', '+20.0%', '+4.0%', '−30.0%', '−12.0%']);
    expect(window.localStorage.getItem('ftb:factorWindow')).toBe('5Y');
});

it('restores the remembered window', async () => {
    window.localStorage.setItem('ftb:factorWindow', '1Y');
    render(<FactorRow />);
    await waitFor(() => expect(vals()).toEqual(['−6.0%', '+9.9%', '+2.2%', '−8.0%', '−7.0%']));
});

it('a window no factor has is disabled', async () => {
    render(<FactorRow />);
    await ready();
    expect(screen.getByRole('radio', { name: '10Y' })).toBeDisabled();
    expect(screen.getByRole('radio', { name: '3Y' })).not.toBeDisabled();
});

it('falls back to the nearest shorter window when the remembered one has no data', async () => {
    window.localStorage.setItem('ftb:factorWindow', '10Y');
    render(<FactorRow />);
    await waitFor(() => expect(vals()[1]).toBe('+20.0%')); // 5Y
    expect(screen.getByRole('radio', { name: '5Y' })).toHaveAttribute('aria-checked', 'true');
});

it('tapping a chip explains it in the caption; tapping again returns to the summary', async () => {
    render(<FactorRow />);
    await ready();
    const chip = screen.getByRole('button', { name: /Value: −3\.1% versus the S&P 500 over 6M/ });
    fireEvent.click(chip);
    const caption = document.querySelector('.factor-caption');
    expect(caption.textContent).toMatch(/Value \(VLUE\) lagged the S&P by 3\.1% since Mar 25, 2026: VLUE \+1\.9% vs SPY \+5\.0%\. Value explained\./);
    expect(chip).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(chip);
    expect(caption.textContent).toMatch(/leader Momentum/);
});

it('marks a stale payload in the caption', async () => {
    global.fetch = jest.fn().mockResolvedValue({ json: async () => ({ ...payload, _meta: { ...payload._meta, stale: true } }) });
    render(<FactorRow />);
    await ready();
    expect(document.querySelector('.factor-stale')).toBeInTheDocument();
});

it('renders nothing when the route has no factors or the fetch fails', async () => {
    global.fetch = jest.fn().mockResolvedValue({ json: async () => ({ factors: [] }) });
    const { container } = render(<FactorRow />);
    await waitFor(() => expect(container.querySelector('.factor-row')).toBeNull());

    // A failed fetch is retried once (lib/loadJson.js, 1.5 s later) before the row gives up.
    global.fetch = jest.fn().mockRejectedValue(new Error('down'));
    const second = render(<FactorRow />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2), { timeout: 3000 });
    await waitFor(() => expect(second.container.querySelector('.factor-row')).toBeNull());
});

it('keeps the last good data when a refresh comes back empty', async () => {
    jest.useFakeTimers({ now: Date.parse('2026-09-26T12:00:00Z') });
    const { rerender } = render(<FactorRow refreshKey="a" />);
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(vals()).toEqual(SIX_M);
    global.fetch = jest.fn().mockResolvedValue({ json: async () => ({ factors: [] }) });
    jest.setSystemTime(Date.parse('2026-09-26T12:10:00Z'));
    rerender(<FactorRow refreshKey="b" />);
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(vals()).toEqual(SIX_M);
    jest.useRealTimers();
});

describe('helpers', () => {
    it('rankFactors picks leader and laggard among factors that have the window', () => {
        const { leader, laggard } = rankFactors(payload.factors, '6M');
        expect(leader.key).toBe('momentum');
        expect(laggard.key).toBe('lowvol');
        expect(rankFactors(payload.factors, '10Y')).toEqual({ leader: null, laggard: null });
    });
    it('isStale by flag or by age of asOf', () => {
        expect(isStale({ asOf: '2026-09-25', _meta: {} }, new Date('2026-09-28T12:00:00Z'))).toBe(false);
        expect(isStale({ asOf: '2026-09-18', _meta: {} }, new Date('2026-09-28T12:00:00Z'))).toBe(true);
        expect(isStale({ asOf: '2026-09-25', _meta: { stale: true } })).toBe(true);
    });
});

describe('20Y / 30Y / 40Y (research portfolios)', () => {
    const longWin = (rel, f, b) => ({ rel, f, b, from: '2006-08-31', to: '2026-08-31', spark: [0, rel], basis: 'research' });
    const withLong = {
        ...payload,
        windows: WINDOWS,
        long: { through: '2026-08', source: 'live', basis: 'total return, monthly' },
        factors: payload.factors.map((f, i) => ({
            ...f,
            longProxy: `${f.label} proxy`,
            windows: {
                ...f.windows,
                '20Y': longWin(-15.09 + i, 641.05, 772.74),
                '30Y': longWin(25.87 + i, 2474.57, 1945.47),
                '40Y': longWin(34.56 + i, 12887.2, 6759),
            },
        })),
    };

    it('shows the long windows with their own basis in the caption', async () => {
        global.fetch = jest.fn().mockResolvedValue({ json: async () => withLong });
        render(<FactorRow />);
        await ready();
        fireEvent.click(screen.getByRole('radio', { name: '20Y' }));
        expect(vals()).toEqual(['−15.1%', '−14.1%', '−13.1%', '−12.1%', '−11.1%']);
        const caption = document.querySelector('.factor-caption').textContent;
        expect(caption).toMatch(/research portfolios vs whole market, total return, through Aug 2026/);
        expect(caption).not.toMatch(/price ratio/);
    });

    it('a tapped chip explains the long window: proxy, per-year gap, dividends included', async () => {
        global.fetch = jest.fn().mockResolvedValue({ json: async () => withLong });
        render(<FactorRow />);
        await ready();
        fireEvent.click(screen.getByRole('radio', { name: '40Y' }));
        fireEvent.click(document.querySelector('.factor-chip'));
        const caption = document.querySelector('.factor-caption').textContent;
        expect(caption).toMatch(/Value \(Value proxy\) beat the whole US market by 34\.6%/);
        expect(caption).toMatch(/≈\+0\.7%\/yr/);
        expect(caption).toMatch(/\+12,887% vs \+6,759%, dividends included/);
    });

    it('long buttons are disabled when the route has no long history', async () => {
        render(<FactorRow />);
        await ready();
        for (const w of ['20Y', '30Y', '40Y']) expect(screen.getByRole('radio', { name: w })).toBeDisabled();
    });

    it('a remembered 40Y falls back to the longest window that exists', async () => {
        window.localStorage.setItem('ftb:factorWindow', '40Y');
        render(<FactorRow />);
        await waitFor(() => expect(screen.getByRole('radio', { name: '5Y' })).toHaveAttribute('aria-checked', 'true'));
    });
});

describe('keyboard', () => {
    it('arrow keys move through the enabled windows only; Home/End jump', async () => {
        render(<FactorRow />);
        await ready();
        const group = screen.getByRole('radiogroup');
        const checked = () => screen.getAllByRole('radio').find((r) => r.getAttribute('aria-checked') === 'true').textContent;
        fireEvent.keyDown(group, { key: 'ArrowRight' });
        expect(checked()).toBe('YTD');
        fireEvent.keyDown(group, { key: 'End' });
        expect(checked()).toBe('5Y'); // 10Y and 20Y+ have no data in this payload
        fireEvent.keyDown(group, { key: 'ArrowRight' });
        expect(checked()).toBe('5Y');
        fireEvent.keyDown(group, { key: 'Home' });
        expect(checked()).toBe('1M');
        expect(document.activeElement.textContent).toBe('1M');
        expect(window.localStorage.getItem('ftb:factorWindow')).toBe('1M');
    });
    it('only the checked window is in the tab order', async () => {
        render(<FactorRow />);
        await ready();
        const tabbable = screen.getAllByRole('radio').filter((r) => r.tabIndex === 0);
        expect(tabbable.map((r) => r.textContent)).toEqual(['6M']);
    });
});

describe('edge cache', () => {
    it('an automatic load may hit the edge cache; a manual refresh busts it', async () => {
        const { rerender } = render(<FactorRow refreshKey="a" />);
        await ready();
        expect(global.fetch.mock.calls[0][0]).toBe('/api/factors');
        jest.spyOn(Date, 'now').mockReturnValue(Date.now() + 120e3);
        rerender(<FactorRow refreshKey="b" bust />);
        await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
        expect(global.fetch.mock.calls[1][0]).toMatch(/^\/api\/factors\?_t=\d+$/);
    });
    it('a manual refresh right after a load still refetches (skips the 60 s floor); an automatic one does not', async () => {
        const { rerender } = render(<FactorRow refreshKey={1} />);
        await ready();
        rerender(<FactorRow refreshKey={2} />);
        expect(global.fetch).toHaveBeenCalledTimes(1);
        rerender(<FactorRow refreshKey={3} bust />);
        await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
    });
});

describe('long helpers', () => {
    it('fmtPct drops decimals once they are noise', () => {
        expect(fmtPct(12887.2)).toBe('+12,887%');
        expect(fmtPct(-150.4)).toBe('−150%');
        expect(fmtPct(3.14)).toBe('+3.1%');
        expect(fmtPct(0)).toBe('0.0%');
        expect(fmtPct(null)).toBe('—');
    });
    it('perYear compounds the gap back to a yearly rate', () => {
        expect(perYear(-15.09, 20)).toBeCloseTo(-0.815, 2);
        expect(perYear(100, 10)).toBeCloseTo(7.177, 2);
        expect(perYear(-100, 10)).toBeNull();
        expect(perYear(null, 10)).toBeNull();
    });
    it('isLongStale only past 4 months behind', () => {
        expect(isLongStale({ long: { through: '2026-08' } }, new Date('2026-09-26T12:00:00Z'))).toBe(false);
        expect(isLongStale({ long: { through: '2026-05' } }, new Date('2026-09-26T12:00:00Z'))).toBe(false);
        expect(isLongStale({ long: { through: '2026-04' } }, new Date('2026-09-26T12:00:00Z'))).toBe(true);
        expect(isLongStale({})).toBe(false);
    });
});
