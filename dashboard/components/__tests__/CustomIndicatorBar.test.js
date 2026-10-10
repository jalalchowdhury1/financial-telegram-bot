import { render, screen } from '@testing-library/react';
import CustomIndicatorBar, { aaiiAsOf, aaiiCaption, staleNote } from '../CustomIndicatorBar';

describe('AAII as-of line', () => {
    test('fresh survey shows its date and source', () => {
        render(<div>{aaiiAsOf({ as_of: '2026-09-23', source: 'aaii.com', stale: false })}</div>);
        expect(screen.getByTestId('aaii-asof').textContent).toBe('Survey Sep 23 · aaii.com');
    });
    test('a stale survey is labelled STALE, never shown as fresh', () => {
        render(<div>{aaiiAsOf({ as_of: '2026-09-09', source: 'substack', stale: true })}</div>);
        expect(screen.getByTestId('aaii-asof').textContent).toBe('⚠️ STALE · survey Sep 9 · substack');
    });
    test('no AAII block (old payload / AAII down) renders nothing', () => {
        expect(aaiiAsOf(null)).toBeNull();
        expect(aaiiAsOf({ as_of: 'junk' })).toBeNull();
    });
});

// Phones hide every .pill-detail (globals.css ≤500px), so the AAII pill needs its own
// one-line caption: who leads + the survey date, orange when the survey is stale.
// Live /api/sheets 2026-10-04: AAIIDiff "11.90%", AAII bear 46.5 − bull 34.6, as_of 2026-09-30.
const LIVE_AAII = { bull: 34.6, neutral: 18.9, bear: 46.5, as_of: '2026-09-30', source: 'aaii.com', stale: false, lastGood: false };

describe('AAII phone caption', () => {
    test('says who leads (diff = bear − bull) and the survey date', () => {
        render(<div>{aaiiCaption(LIVE_AAII, '11.90%')}</div>);
        const c = screen.getByTestId('aaii-caption');
        expect(c.textContent).toBe('Bears +11.9 · Sep 30');
        expect(c).toHaveClass('pill-caption');
        expect(c).not.toHaveClass('pill-detail');          // .pill-detail is display:none on phones
        expect(c).not.toHaveClass('pill-caption-stale');
    });
    test('bulls ahead reads as bulls', () => {
        render(<div>{aaiiCaption({ ...LIVE_AAII, bull: 45, bear: 30 }, '-15.00%')}</div>);
        expect(screen.getByTestId('aaii-caption').textContent).toBe('Bulls +15.0 · Sep 30');
    });
    test('a stale survey turns into an orange STALE warning, still visible on phones', () => {
        render(<div>{aaiiCaption({ ...LIVE_AAII, as_of: '2026-09-23', stale: true }, '11.90%')}</div>);
        const c = screen.getByTestId('aaii-caption');
        expect(c.textContent).toBe('⚠ STALE · Sep 23');
        expect(c).toHaveClass('pill-caption');
        expect(c).toHaveClass('pill-caption-stale');
        expect(c).not.toHaveClass('pill-detail');
    });
    test('no date, junk date or no number -> no caption (never a guess)', () => {
        expect(aaiiCaption(null, '11.90%')).toBeNull();
        expect(aaiiCaption({ ...LIVE_AAII, as_of: 'junk' }, '11.90%')).toBeNull();
        expect(aaiiCaption(LIVE_AAII, 'N/A')).toBeNull();
        expect(aaiiCaption(LIVE_AAII, undefined)).toBeNull();
    });
    test('the bar renders the caption under the AAII number; desktop detail lines stay', () => {
        render(<CustomIndicatorBar loading={false}
            sheets={{ AAIIDiff: '11.90%', AAII: LIVE_AAII, VIX: { current: '15.31', threeMonth: '18.01', fearGreed: 'GREED03' } }} />);
        expect(screen.getByTestId('aaii-caption').textContent).toBe('Bears +11.9 · Sep 30');
        expect(screen.getByTestId('aaii-asof').textContent).toBe('Survey Sep 30 · aaii.com');
    });
    test('AAII down (N/A) or a payload without the AAII block: no caption, nothing throws', () => {
        const { rerender } = render(<CustomIndicatorBar loading={false} sheets={{ AAIIDiff: 'N/A' }} />);
        expect(screen.queryByTestId('aaii-caption')).toBeNull();
        rerender(<CustomIndicatorBar loading={false} sheets={{ AAIIDiff: '11.90%' }} />);
        expect(screen.queryByTestId('aaii-caption')).toBeNull();
        rerender(<CustomIndicatorBar loading={false} sheets={{}} />);
        expect(screen.queryByTestId('aaii-caption')).toBeNull();
    });
});

// /api/sheets per-field last-good (lib/sheetsCascade.js): a value served from a cached
// copy or a lagged source must carry a visible STALE line, on every screen size.
describe('stale pill line', () => {
    const meta = (staleFields, fields) => ({ _meta: { staleFields, fields } });
    test('a KV/tmp copy shows "⚠ STALE · cached <date>" with its source as the tooltip', () => {
        const s = meta(['NotSoBoring'], { NotSoBoring: { source: 'KV last-good (2026-10-08T20:00:00.000Z) ← Google Sheets (Live)', stale: true, savedAt: '2026-10-08T20:00:00.000Z' } });
        render(<div>{staleNote(s, ['NotSoBoring'])}</div>);
        const el = screen.getByTestId('stale-NotSoBoring');
        expect(el.textContent).toBe('⚠ STALE · cached Oct 8');
        expect(el.getAttribute('title')).toMatch(/^KV last-good/);
    });
    test('a FRED-lagged VIX says so', () => {
        const s = meta(['vixCurrent'], { vixCurrent: { source: 'FRED VIXCLS (close 2026-10-08; lags a trading day)', stale: true } });
        render(<div>{staleNote(s, ['vixCurrent', 'vixThreeMonth'])}</div>);
        expect(screen.getByTestId('stale-vixCurrent').textContent).toBe('⚠ STALE · FRED, lags a day');
    });
    test('the frozen FrontRunner sheet fallback says "frozen sheet"', () => {
        const s = meta(['FrontRunner'], { FrontRunner: { source: 'Google Sheets (Live), backup only: its RSI inputs stopped updating on 2026-08-24', stale: true } });
        render(<div>{staleNote(s, ['FrontRunner'])}</div>);
        expect(screen.getByTestId('stale-FrontRunner').textContent).toBe('⚠ STALE · frozen sheet');
    });
    test('nothing stale (or an old payload without _meta) renders nothing', () => {
        expect(staleNote(meta([], {}), ['FrontRunner'])).toBeNull();
        expect(staleNote({}, ['FrontRunner'])).toBeNull();
        expect(staleNote(null, ['FrontRunner'])).toBeNull();
    });
    test('the bar renders the line under the right pill', () => {
        const sheets = {
            NotSoBoring: 'ON', FrontRunner: 'BIL (T-Bill ETF)', AAIIDiff: 'N/A', VIX: { current: '14.84', threeMonth: '17.77', fearGreed: 'GREED04' },
            _meta: { staleFields: ['FrontRunner'], fields: { FrontRunner: { source: '/tmp last-good (2026-10-09T13:00:00.000Z) ← Google Sheets (Live)', stale: true, savedAt: '2026-10-09T13:00:00.000Z' } } },
        };
        render(<CustomIndicatorBar sheets={sheets} loading={false} />);
        expect(screen.getByTestId('stale-FrontRunner').textContent).toBe('⚠ STALE · cached Oct 9');
        expect(screen.queryByTestId('stale-NotSoBoring')).toBeNull();
        expect(screen.queryByTestId('stale-vixCurrent')).toBeNull();
    });
});
