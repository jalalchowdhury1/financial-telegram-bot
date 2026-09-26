"""3-year return on the SPY card must never be a shorter return in disguise.

Polygon's free tier returns ~2 years of bars; the old code computed the return over
whatever it got and labelled it "3Y" (2026-09-26: +34.8% shown vs the true ~+81%)."""
import bot.fetchers as f


def _rows(n, start=100.0, step=1.0):
    return [{'date': f'd{i}', 'close': start + i * step} for i in range(n)]


def test_three_full_years_gives_the_real_return():
    rows = _rows(800)
    base = rows[-756]['close']
    assert f._return_3y_from_rows(rows, 200.0) == f._calc_pct(200.0, base)


def test_two_years_of_bars_is_not_a_three_year_return():
    assert f._return_3y_from_rows(_rows(501), 771.35) is None


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
