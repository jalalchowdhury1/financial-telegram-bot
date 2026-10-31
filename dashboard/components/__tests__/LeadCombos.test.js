import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import LeadCombos from '../LeadCombos';

jest.mock('../MarkProvider', () => ({ useSeries: () => null }));

// 501 daily readings ending today; `steps` = [[days ago, value], …] changes the value from that day on
const DAY = 86400000;
const from = new Date(Date.now() - 500 * DAY).toISOString().slice(0, 10);
const file = (base, steps = []) => ({
    from, t: Array.from({ length: 501 }, (_, i) => i),
    v: Array.from({ length: 501 }, (_, i) => steps.reduce((v, [ago, x]) => (i >= 500 - ago ? x : v), base)),
});
const FILES = {
    aaiiDiff: file(5), cl: file(100), // fear + oil: neither leg lit
    claims: file(100, [[420, 88], [360, 100], [20, 90], [5, 100]]), usdcad: file(1.3, [[420, 1.4], [360, 1.3], [20, 1.4]]), // fired 420 and 20 days ago, claims back since
    vixCurrent: file(30), housing: file(1500, [[60, 1600]]), // VIX lit, housing +6.7% = ¾ of the way to +8%
    peRatio: file(25), tnx: file(4),
    'spy-daily': { from, t: Array.from({ length: 501 }, (_, i) => i), v: Array.from({ length: 501 }, (_, i) => 100 + i / 10) },
};

beforeEach(() => {
    global.fetch = jest.fn((url) => {
        const key = String(url).match(/\/history\/(.+)\.json/)?.[1];
        return Promise.resolve(key in FILES ? { ok: true, json: async () => FILES[key] } : { ok: false });
    });
});

const row = (name) => screen.getByText(name).closest('button');

it('shows only the pairs near their lines or fired this month, each with its record and today', async () => {
    render(<LeadCombos spy={null} />);
    await waitFor(() => expect(row('Jobs fine + CAD slump').textContent).toMatch(/Bullish to/));
    const cu = row('Jobs fine + CAD slump');
    expect(cu.className).toMatch(/is-on/);
    expect(cu.querySelectorAll('.v2-cb-leg.is-lit')).toHaveLength(1); // claims slipped back: still shown, it fired 20 days ago
    expect(cu.querySelector('.v2-cb-rec').textContent).toMatch(/^\+\d+% 1×$/); // the 20-day-old one is still live, out of the record
    expect(row('Panic + housing boom').querySelector('.v2-cb-now').textContent).toBe('1 of 2');
    expect(screen.queryByText('Fear + oil crash')).toBeNull();
    expect(screen.queryByText('Selloff + falling rates')).toBeNull();
});

it('a tap opens the proof for that pair', async () => {
    render(<LeadCombos spy={null} />);
    await waitFor(() => expect(row('Jobs fine + CAD slump')).not.toBeDisabled());
    fireEvent.click(row('Jobs fine + CAD slump'));
    expect(await screen.findByRole('dialog', { name: 'Jobs fine + CAD slump' })).toBeInTheDocument();
});

it('renders nothing when no pair is near, or a file is missing', async () => {
    for (const k of ['claims', 'usdcad', 'housing']) FILES[k] = file(FILES[k].v[0]);
    const quiet = render(<LeadCombos spy={null} />);
    await act(() => new Promise((r) => setTimeout(r, 20)));
    expect(quiet.container).toBeEmptyDOMElement();
    quiet.unmount();
    FILES.vixCurrent = file(30); FILES.housing = file(1500, [[60, 1700]]); // hv lit…
    delete FILES.tnx; // …but a file is missing
    const { container } = render(<LeadCombos spy={null} />);
    await act(() => new Promise((r) => setTimeout(r, 20)));
    expect(container).toBeEmptyDOMElement();
});
