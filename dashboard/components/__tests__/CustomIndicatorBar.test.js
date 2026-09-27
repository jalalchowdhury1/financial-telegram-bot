import { render, screen } from '@testing-library/react';
import { aaiiAsOf } from '../CustomIndicatorBar';

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
