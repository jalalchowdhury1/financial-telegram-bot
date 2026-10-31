"""Bake VIX ÷ VIX 3M, every trading day since Dec 2007, into public/history/vix-ratio-daily.json.

The v2 VIX tile's drill-in (components/SignalStudy.js) finds each time VIX closed above VIX 3M and
draws what SPY did next. The vixCurrent / vix3m long-history files are weekly before 2021, which
misses inversions that last a few days, so this keeps every day. FRED VIXCLS and VXVCLS closes,
same {from, t:[day offsets], v} shape as bake_long_history.py. Re-run every quarter by
rebake-long-history.yml; the history sheet's daily rows cover the gap between bakes.
"""
import json
import sys
from pathlib import Path

from bake_long_history import fred

OUT = Path(__file__).resolve().parent.parent / 'public' / 'history' / 'vix-ratio-daily.json'


def main():
    v3 = dict(fred('VXVCLS'))
    pts = [(d, round(v / v3[d], 4)) for d, v in fred('VIXCLS') if v3.get(d)]
    if len(pts) < 4500:
        sys.exit(f'VIX ratio: only {len(pts)} days, keeping the old file')
    start = pts[0][0]
    body = {'key': 'vixRatio', 'source': 'FRED VIXCLS ÷ VXVCLS', 'from': start.isoformat(), 'to': pts[-1][0].isoformat(),
            't': [(d - start).days for d, _ in pts], 'v': [v for _, v in pts]}
    OUT.write_text(json.dumps(body, separators=(',', ':')) + '\n')
    print(f'wrote {OUT.name}: {len(pts)} days {body["from"]} → {body["to"]}')


if __name__ == '__main__':
    main()
