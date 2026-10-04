import { render, screen } from '@testing-library/react';
import CustomIndicatorBar, { aaiiAsOf, aaiiCaption } from '../CustomIndicatorBar';

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
