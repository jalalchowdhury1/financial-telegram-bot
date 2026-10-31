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
    claims: file(100, [[420, 88], [360, 100], [30, 90]]), usdcad: file(1.3, [[420, 1.4], [360, 1.3], [30, 1.4]]), // fired 420 and 30 days ago
    vixCurrent: file(30), housing: file(1500), // VIX lit, housing flat
    peRatio: file(25), tnx: file(4),
    'spy-daily': { from, t: Array.from({ length: 501 }, (_, i) => i), v: Array.from({ length: 501 }, (_, i) => 100 + i / 10) },
};

const serve = (files) => {
    global.fetch = jest.fn((url) => {
        const key = String(url).match(/\/history\/(.+)\.json/)?.[1];
        return Promise.resolve(key in files ? { ok: true, json: async () => files[key] } : { ok: false });
    });
};
beforeEach(() => serve(FILES));
// The same files, ending 8 days ago: the sheet that should carry them to today has stopped (or not landed yet).
const OLD = Object.fromEntries(Object.entries(FILES).map(([k, f]) => [k, { ...f, from: new Date(Date.parse(f.from) - 8 * DAY).toISOString().slice(0, 10) }]));

const row = (name) => screen.getByText(name).closest('button');

it('shows each pair: both legs against their lines, its record, and today', async () => {
    render(<LeadCombos spy={null} />);
    await waitFor(() => expect(row('Jobs fine + CAD slump').textContent).toMatch(/Bullish to/));
    const cu = row('Jobs fine + CAD slump');
    expect(cu.className).toMatch(/is-on/);
    expect(cu.querySelectorAll('.v2-cb-leg.is-lit')).toHaveLength(2);
    expect(cu.querySelector('.v2-cb-rec').textContent).toMatch(/^\+\d+% 1×$/); // the 30-day-old one is still live, out of the record
    expect(row('Fear + oil crash').querySelector('.v2-cb-now').textContent).toBe('0 of 2');
    expect(row('Panic + housing boom').querySelector('.v2-cb-now').textContent).toBe('1 of 2');
    expect(row('Selloff + falling rates').querySelector('.v2-cb-now').textContent).toBe('0 of 2');
});

it('a tap opens the proof for that pair', async () => {
    render(<LeadCombos spy={null} />);
    await waitFor(() => expect(row('Jobs fine + CAD slump')).not.toBeDisabled());
    fireEvent.click(row('Jobs fine + CAD slump'));
    expect(await screen.findByRole('dialog', { name: 'Jobs fine + CAD slump' })).toBeInTheDocument();
});

it('a pair whose newest reading is over 5 days old says N/A and is unread for the top line; a fire stays true', async () => {
    serve(OLD);
    const onState = jest.fn();
    render(<LeadCombos spy={null} onState={onState} />);
    await waitFor(() => expect(screen.getAllByText('N/A')).toHaveLength(3));
    expect(row('Jobs fine + CAD slump').querySelector('.v2-cb-now').textContent).toMatch(/^Bullish to /); // fired 38 days ago
    const [fo, cu, hv, pt] = onState.mock.calls.at(-1)[0];
    expect([fo, hv, pt]).toEqual([null, null, null]);
    expect(cu).toEqual({ jump: 'Lead combos', fire: expect.stringMatching(/^Jobs fine \+ CAD slump bullish to /) });
});

it('waits (…) while the sheet is on its way, instead of N/A on old files', async () => {
    serve(OLD);
    const { rerender } = render(<LeadCombos spy={null} waiting />);
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(screen.getAllByText('…')).toHaveLength(3);
    rerender(<LeadCombos spy={null} />);
    expect(screen.getAllByText('N/A')).toHaveLength(3);
});

it('unmounting (a crash, a refresh) leaves the top line unread, never the last state', async () => {
    const onState = jest.fn();
    const { unmount } = render(<LeadCombos spy={null} onState={onState} />);
    await waitFor(() => expect(onState.mock.calls.at(-1)?.[0]?.[1]?.fire).toMatch(/^Jobs fine \+ CAD slump bullish to /));
    unmount();
    expect(onState).toHaveBeenLastCalledWith(null);
});

it('says N/A when a file is missing, and never crashes', async () => {
    delete FILES.tnx;
    render(<LeadCombos spy={null} />);
    await waitFor(() => expect(screen.getAllByText('N/A')).toHaveLength(4));
});
