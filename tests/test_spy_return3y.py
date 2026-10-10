"""3-year return on the SPY card must never be a shorter return in disguise.

Polygon's free tier returns ~2 years of bars; the old code computed the return over
whatever it got and labelled it "3Y" (2026-09-26: +34.8% shown vs the true ~+81%)."""
import bot.fetchers as f


def _rows(n, last='2026-10-09'):
    from datetime import date, timedelta
    end = date.fromisoformat(last)
    days = [end - timedelta(days=i) for i in range(n)][::-1]
    return [{'date': d.isoformat(), 'close': 100.0 + i} for i, d in enumerate(days)]


def test_three_full_years_uses_the_same_date_three_years_back():
    # Real SPY 2026-10-09 -> 2023-10-09 (432.29) -> 80.10%; a Sunday anniversary takes the Friday.
    rows = [{'date': '2023-10-06', 'close': 429.54}, {'date': '2023-10-09', 'close': 432.29},
            {'date': '2023-10-10', 'close': 434.54}, {'date': '2026-10-09', 'close': 778.57}]
    assert round(f._return_3y_from_rows(rows, 778.57), 2) == 80.10
    rows[-1]['date'] = '2026-10-08'
    assert f._return_3y_from_rows(rows, 778.57) == f._calc_pct(778.57, 429.54)


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


def test_sheet_layer_prefers_the_same_date_3y_then_parses_a_percent_cell(monkeypatch):
    # Owner's pick 2026-10-09: same date 3 years back = the daily-move sheet (80.17%).
    # When that sheet fails, the indicators' "79.17%" must still parse (float() choked on '%').
    csv_text = ('200d MA SPY,722.9284\n9d RSI SPY,63.1\nSPY 52 week high,781.62\n'
                'Current SPY,778.57\nPrice from Three Years Ago,434.54\nThree-Year Return,79.17%\n')

    class R:
        text = csv_text
    monkeypatch.setattr(f, '_fetch_yfinance', lambda *a, **k: None)
    monkeypatch.setattr(f.requests, 'get', lambda *a, **k: R())
    monkeypatch.setattr(f, '_get_sheet_csv', lambda url, **kw: '\n'.join(['x,y'] * 10 + ['3 YR Return,80.17%']))
    assert f.fetch_spy_with_fallback()['return3y'] == 80.17

    def boom(url, **kw):
        raise TimeoutError('sheet stalled')
    monkeypatch.setattr(f, '_get_sheet_csv', boom)
    assert f.fetch_spy_with_fallback()['return3y'] == 79.17


def test_short_polygon_history_takes_the_3y_base_from_nasdaq_anchored_on_the_spot_date(monkeypatch):
    from datetime import date, timedelta
    end = date(2026, 10, 8)                                   # Polygon ends yesterday
    poly = [{'date': (end - timedelta(days=i)).isoformat(), 'price': 700.0} for i in range(600)][::-1]
    monkeypatch.setattr(f, '_fetch_yfinance', lambda *a, **k: None)
    monkeypatch.setattr(f, '_fetch_polygon_aggs', lambda *a, **k: {'history': poly})
    monkeypatch.setattr(f, '_fetch_finnhub_quote', lambda *a, **k: {
        'current': 778.57, 'dailyChange': {'value': 4.64, 'pct': 0.6}, 'lastDate': '2026-10-09'})
    nq = [{'date': '2023-10-06', 'close': 429.54}, {'date': '2023-10-09', 'close': 432.29},
          {'date': '2023-10-10', 'close': 434.54}]
    monkeypatch.setattr(f, '_nasdaq_rows', lambda s: nq)
    monkeypatch.setattr(f, '_sheet_return_3y', lambda: 1 / 0)  # must not be reached
    out = f.fetch_spy_with_fallback(polygon_api_key='k', finnhub_api_key='k')
    assert round(out['return3y'], 2) == 80.10


def test_yfinance_and_polygon_down_uses_nasdaq_bars_before_the_sheet(monkeypatch):
    from datetime import date, timedelta
    end = date(2026, 10, 9)
    nq = [{'date': (end - timedelta(days=i)).isoformat(), 'close': 700.0 - i * 0.2} for i in range(1200)][::-1]
    monkeypatch.setattr(f, '_fetch_yfinance', lambda *a, **k: None)
    monkeypatch.setattr(f, '_fetch_polygon_aggs', lambda *a, **k: None)
    monkeypatch.setattr(f, '_fetch_finnhub_quote', lambda *a, **k: None)
    monkeypatch.setattr(f, '_nasdaq_rows', lambda *a, **k: nq)
    monkeypatch.setattr(f, '_sheet_return_3y', lambda: 1 / 0)  # must not be reached
    out = f.fetch_spy_with_fallback(polygon_api_key='k', finnhub_api_key='k')
    assert out['_meta']['source'].startswith('Nasdaq')
    assert out['current'] == 700.0 and out['rsi'] is not None
    base = next(r['close'] for r in reversed(nq) if r['date'] <= '2023-10-09')
    assert round(out['return3y'], 4) == round((700.0 - base) / base * 100, 4)


def test_rsi_counts_todays_spot_when_polygon_bars_end_yesterday(monkeypatch):
    import pandas as pd
    from datetime import date, timedelta
    end = date(2026, 10, 8)
    poly = [{'date': (end - timedelta(days=i)).isoformat(), 'price': 700.0 + (i % 7) - i * 0.1} for i in range(600)][::-1]
    monkeypatch.setattr(f, '_fetch_yfinance', lambda *a, **k: None)
    monkeypatch.setattr(f, '_fetch_polygon_aggs', lambda *a, **k: {'history': poly})
    monkeypatch.setattr(f, '_fetch_finnhub_quote', lambda *a, **k: {
        'current': 720.0, 'dailyChange': {'value': 3.0, 'pct': 0.4}, 'lastDate': '2026-10-09'})
    monkeypatch.setattr(f, '_nasdaq_rows', lambda *a, **k: [])
    monkeypatch.setattr(f, '_sheet_return_3y', lambda: None)
    out = f.fetch_spy_with_fallback(polygon_api_key='k', finnhub_api_key='k')
    want = f.calculate_rsi(pd.Series([p['price'] for p in poly] + [720.0]), period=9)
    assert out['rsi'] == want
    assert out['chartHistory'][-1]['date'] == '2026-10-09'
