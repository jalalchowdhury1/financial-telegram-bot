"""/api/spy must survive a NaN close from Yahoo.

CloudWatch 2026-10-09 00:34 UTC and 2026-10-10 01:23-01:29 UTC (evening ET): every time
yfinance answered, /api/spy died with "cannot convert float NaN to integer" -- a NaN close
poisoned the rolling MA50/MA200 and round() blew up on the chart history. The dashboard
then fell back to its own tiers, so the Lambda was silently unused those evenings.
yfinance's keepna=False drops a row only when EVERY column is NaN/0, so a bar with a NaN
close but a real volume gets through. The fix drops such bars; it never invents a price."""
import math
import sys
import types
from datetime import date, timedelta

import pandas as pd
import pytest

import bot.fetchers as f


def _days(n, last='2026-10-09'):
    end = date.fromisoformat(last)
    return [end - timedelta(days=i) for i in range(n)][::-1]


def _yf_metric(prices, last='2026-10-09'):
    days = _days(len(prices), last)
    return {'history': [{'date': d.isoformat(), 'price': p} for d, p in zip(days, prices)]}


def _isolate(monkeypatch):
    monkeypatch.setattr(f, '_fetch_polygon_aggs', lambda *a, **k: None)
    monkeypatch.setattr(f, '_nasdaq_rows', lambda *a, **k: [])
    monkeypatch.setattr(f, '_sheet_return_3y', lambda *a, **k: None)


@pytest.mark.parametrize('where', ['middle', 'last'])
def test_spy_survives_a_nan_close_from_yfinance(monkeypatch, where):
    prices = [400.0 + i * 0.3 for i in range(1200)]
    nan_at = 600 if where == 'middle' else len(prices) - 1
    prices[nan_at] = float('nan')
    monkeypatch.setattr(f, '_fetch_yfinance', lambda *a, **k: _yf_metric(prices))
    _isolate(monkeypatch)

    out = f.fetch_spy_with_fallback()  # raised "cannot convert float NaN to integer" before the fix

    assert out['_meta']['source'] == 'yfinance'
    for key in ('current', 'rsi', 'return3y'):
        assert math.isfinite(out[key]), (key, out[key])
    assert math.isfinite(out['ma200']['value']) and math.isfinite(out['week52High']['value'])
    assert all(math.isfinite(p['price']) and math.isfinite(p['ma50']) and math.isfinite(p['ma200'])
               for p in out['chartHistory'])
    nan_date = _days(len(prices))[nan_at].isoformat()
    assert nan_date not in {p['date'] for p in out['chartHistory']}  # dropped, not invented
    if where == 'last':
        assert out['current'] == prices[-2]  # the last REAL close, never a made-up number


def test_fetch_yfinance_drops_nan_close_bars(monkeypatch):
    idx = pd.to_datetime(['2026-10-07', '2026-10-08', '2026-10-09']).tz_localize('America/New_York')
    hist = pd.DataFrame({'Close': [777.22, float('nan'), 778.57], 'Volume': [30989200, 40431100, 22596500]},
                        index=idx)

    class _Ticker:
        def __init__(self, symbol):
            pass

        def history(self, **kw):
            return hist

    fake = types.SimpleNamespace(Ticker=_Ticker, set_tz_cache_location=lambda path: None)
    monkeypatch.setitem(sys.modules, 'yfinance', fake)

    out = f._fetch_yfinance('SPY')

    assert [r['date'] for r in out['history']] == ['2026-10-07', '2026-10-09']
    assert out['current'] == 778.57 and math.isfinite(out['dailyChange']['pct'])
