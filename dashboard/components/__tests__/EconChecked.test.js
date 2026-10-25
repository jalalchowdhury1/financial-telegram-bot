import { render, screen, fireEvent } from '@testing-library/react';
import EconChecked, { newestPrint } from '../EconChecked';
import { savedLabel } from '../../lib/snapshot';

const m = (value, asOf, stale = false) => ({ value, asOf, stale, unavailable: value == null });
const fred = {
    indicators: { claims: m(198, '2026-10-17'), creditSpread: m(1.11, '2026-10-22'), sentiment: m(48, '2026-09-01') },
    checklist: { nfci: { ...m(-0.5, '2026-10-16'), label: 'Financial Conditions' } },
    yieldCurve: { current: 0.31, asOf: '2026-10-21', stale: false },
    _meta: { fetchedAt: '2026-10-25T20:58:03.605Z' },
};

it('names the newest print across the econ numbers, its date as printed (no timezone shift)', () => {
    expect(newestPrint(fred)).toEqual({ name: 'BBB Credit Spread', date: 'Oct 22' });
});

it('skips stale and missing numbers: a stale print never counts as the newest', () => {
    const f = { ...fred, indicators: { ...fred.indicators, creditSpread: m(1.11, '2026-10-24', true), claims: m(null, '2026-10-23') } };
    expect(newestPrint(f)).toEqual({ name: 'Yield Curve', date: 'Oct 21' });
    expect(newestPrint({ indicators: { claims: m(null, null) } })).toBeNull();
    expect(newestPrint(null)).toBeNull();
});

const checked = `Econ checked ${savedLabel(Date.parse(fred._meta.fetchedAt))}`;

it('says "Econ checked" with the device time, and a tap reveals the newest print on one line', () => {
    render(<EconChecked fred={fred} />);
    const btn = screen.getByRole('button', { name: checked });
    expect(document.body.textContent).not.toMatch(/as of/);
    expect(screen.queryByText(/newest print/)).toBeNull();
    fireEvent.click(btn);
    expect(screen.getByText('newest print: BBB Credit Spread · Oct 22')).toBeInTheDocument();
    fireEvent.click(btn);
    expect(screen.queryByText(/newest print/)).toBeNull();
});

it('with no usable print there is nothing to reveal: plain text, no button', () => {
    render(<EconChecked fred={{ _meta: fred._meta }} />);
    expect(screen.getByText(checked)).toBeInTheDocument();
    expect(screen.queryByRole('button')).toBeNull();
});

it('a copy checked days ago names its day, so a saved or server-stale FRED never reads as today', () => {
    const at = Date.now() - 3 * 864e5;
    render(<EconChecked fred={{ _meta: { fetchedAt: new Date(at).toISOString() } }} />);
    const d = new Date(at);
    const hm = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
    expect(screen.getByText(`Econ checked ${d.toLocaleDateString('en-US', { weekday: 'short' })} ${hm}`)).toBeInTheDocument();
});
