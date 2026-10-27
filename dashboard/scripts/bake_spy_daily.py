"""Bake SPY's full daily closes (1993 on) into public/history/spy-daily.json.

/api/spy carries only the last 302 bars (~14 months), so the SPY chart's 5Y / 10Y / ALL tabs
draw this file before them (components/SpyChart.js). Raw closes (auto_adjust=False), like the
bot's yfinance tier. Same {from, t:[day offsets], v} shape as bake_long_history.py, so
lib/longHistory.js decodeLong reads it. Re-run every quarter by rebake-long-history.yml:
the 14-month live window covers the gap between bakes.
"""
import json
import sys
from datetime import date
from pathlib import Path

import yfinance as yf

OUT = Path(__file__).resolve().parent.parent / 'public' / 'history' / 'spy-daily.json'


def main():
    h = yf.Ticker('SPY').history(period='max', auto_adjust=False)
    pts = [(ts.date(), round(float(c), 2)) for ts, c in h['Close'].items() if c == c]
    if len(pts) < 8000:
        sys.exit(f'SPY daily: only {len(pts)} bars, keeping the old file')
    start = pts[0][0]
    body = {'key': 'spy', 'source': 'Yahoo Finance (raw closes)', 'from': start.isoformat(), 'to': pts[-1][0].isoformat(),
            't': [(d - start).days for d, _ in pts], 'v': [v for _, v in pts]}
    OUT.write_text(json.dumps(body, separators=(',', ':')) + '\n')
    print(f'wrote {OUT.name}: {len(pts)} bars {body["from"]} → {body["to"]}')


if __name__ == '__main__':
    main()
