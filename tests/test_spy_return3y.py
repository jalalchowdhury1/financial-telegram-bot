"""3-year return on the SPY card must never be a shorter return in disguise.

Polygon's free tier returns ~2 years of bars; the old code computed the return over
whatever it got and labelled it "3Y" (2026-09-26: +34.8% shown vs the true ~+81%)."""
import bot.fetchers as f


def _rows(n, last='2026-10-09'):
    from datetime import date, timedelta
    end = date.fromisoformat(last)
    days = [end - timedelta(days=i) for i in range(n)][::-1]
    return [{'date': d.isoformat(), 'close': 100.0 + i} for i, d in enumerate(days)]


def test_three_full_years_uses_1095_calendar_days():
    # Real SPY 2026-10-09: 1095 days back = 2023-10-10 (434.54) -> 79.17%, matching
    # the Sheet; 756 rows back was 2023-10-05 (424.5) -> 83.41%.
    rows = [{'date': '2023-10-05', 'close': 424.5}, {'date': '2023-10-09', 'close': 432.29},
            {'date': '2023-10-10', 'close': 434.54}, {'date': '2026-10-09', 'close': 778.57}]
    assert round(f._return_3y_from_rows(rows, 778.57), 2) == 79.17


def test_two_years_of_bars_is_not_a_three_year_return():
    assert f._return_3y_from_rows(_rows(730), 771.35) is None
    assert f._return_3y_from_rows([], 771.35) is None
    assert f._return_3y_from_rows(_rows(1200), 771.35) is not None


def test_pct_cell_parses_sheet_formats():
    assert f._pct_cell('81.19%') == 81.19
    assert f._pct_cell(' 81.05 ') == 81.05
    assert f._pct_cell('#N/A') is None
    assert f._pct_cell(None) is None


def test_sheet_fallback_reads_the_indicator_row(monkeypatch):
    csv_text = 'Current SPY,771.35\nPrice from Three Years Ago,426.05\nThree-Year Return,81.05%\n'
    monkeypatch.setattr(f, '_get_sheet_csv', lambda url, **kw: csv_text)
    assert f._sheet_return_3y() == 81.05


def test_sheet_fallback_uses_the_daily_move_row_when_indicators_lack_it(monkeypatch):
    daily = '\n'.join(['x,y'] * 10 + ['3 YR Return,81.19%'])

    def fake(url, **kw):
        return 'Current SPY,771.35\n' if url == f.URLS['SPY_INDICATORS'] else daily
    monkeypatch.setattr(f, '_get_sheet_csv', fake)
    assert f._sheet_return_3y() == 81.19


def test_sheet_fallback_is_none_when_both_sheets_fail(monkeypatch):
    def boom(url, **kw):
        raise TimeoutError('sheet stalled')
    monkeypatch.setattr(f, '_get_sheet_csv', boom)
    assert f._sheet_return_3y() is None


def test_sheet_layer_reads_a_percent_3y_cell(monkeypatch):
    # "79.17%" made float() fail, so the 3Y silently came from the daily-move sheet
    # (80.17%, a different anchor) — 2026-10-09.
    csv_text = ('200d MA SPY,722.9284\n9d RSI SPY,63.1\nSPY 52 week high,781.62\n'
                'Current SPY,778.57\nPrice from Three Years Ago,434.54\nThree-Year Return,79.17%\n')

    class R:
        text = csv_text
    monkeypatch.setattr(f, '_fetch_yfinance', lambda *a, **k: None)
    monkeypatch.setattr(f.requests, 'get', lambda *a, **k: R())
    monkeypatch.setattr(f, '_get_sheet_csv', lambda url, **kw: '\n'.join(['x,y'] * 10 + ['3 YR Return,80.17%']))
    out = f.fetch_spy_with_fallback()
    assert out['return3y'] == 79.17
