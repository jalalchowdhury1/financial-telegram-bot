"""/api/spy's FRED layer must stay on one scale.

Audit 1 Nov 2026: when yfinance, Polygon, Nasdaq and the sheet all fail, the Lambda falls to
FRED's SP500 series (S&P 500 index points, ~6,500) and then laid Finnhub's SPY spot (~680) on
top as `current` and as the newest bar: ma200 -89.5%, 52w -90%, 3Y -86%, RSI 0.35, served as
"FRED S&P 500 Index + Finnhub Spot" and saved by the dashboard as its last-good copy."""
import math
from datetime import date, timedelta

import bot.fetchers as f


class _Resp:
    status_code = 200

    def __init__(self, text='', js=None):
        self.text, self._js = text, js

    def json(self):
        return self._js

    def raise_for_status(self):
        pass


def test_fred_index_rows_never_take_the_spy_spot(monkeypatch):
    end = date(2026, 10, 30)
    days = [end - timedelta(days=i) for i in range(1300)][::-1]
    obs = [{'date': d.isoformat(), 'value': f'{5000 + i * 1.5:.2f}'} for i, d in enumerate(days)]

    def fake_get(url, *a, **k):
        if 'stlouisfed' in url:
            return _Resp(js={'observations': obs})
        return _Resp(text='')  # the sheet layer finds nothing

    monkeypatch.setattr(f, '_fetch_yfinance', lambda *a, **k: None)
    monkeypatch.setattr(f, '_fetch_polygon_aggs', lambda *a, **k: None)
    monkeypatch.setattr(f, '_nasdaq_rows', lambda *a, **k: [])
    monkeypatch.setattr(f, '_sheet_return_3y', lambda *a, **k: None)
    monkeypatch.setattr(f.requests, 'get', fake_get)
    monkeypatch.setattr(f, '_fetch_finnhub_quote', lambda *a, **k: {
        'current': 680.0, 'dailyChange': {'value': 2.0, 'pct': 0.3}, 'lastDate': '2026-11-02'})

    out = f.fetch_spy_with_fallback(fred_api_key='k', finnhub_api_key='fh')

    assert out['_meta']['source'] == 'FRED S&P 500 Index'  # no "+ Finnhub Spot"
    assert out['current'] == float(obs[-1]['value'])
    assert abs(out['ma200']['pct']) < 50 and abs(out['week52High']['pct']) < 50
    assert 0 < out['rsi'] <= 100 and math.isfinite(out['return3y']) and out['return3y'] > 0
    assert out['chartHistory'][-1] == {**out['chartHistory'][-1], 'date': '2026-10-30', 'price': out['current']}
