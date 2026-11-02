"""Bake VIX ÷ VIX 3M, every trading day since Dec 2007, into public/history/vix-ratio-daily.json,
and VIX itself, every trading day since 1990, into public/history/vix-daily.json.

The v2 VIX tile's drill-in (components/SignalStudy.js) finds each time VIX closed above VIX 3M and
draws what SPY did next; the Volatility card's "After spikes" (components/VixSpikes.js) finds each
time VIX first closed at 30+. The vixCurrent / vix3m long-history files are weekly before 2021, which
misses inversions and spikes that last a few days, so these keep every day. FRED VIXCLS and VXVCLS closes,
same {from, t:[day offsets], v} shape as bake_long_history.py. Re-run every quarter by
rebake-long-history.yml; the history sheet's daily rows (the card's own year, for VIX) cover the gap between bakes.
"""
import json
import sys
from pathlib import Path

from bake_long_history import fred

OUT = Path(__file__).resolve().parent.parent / 'public' / 'history' / 'vix-ratio-daily.json'
VIX_OUT = OUT.with_name('vix-daily.json')


def write(out, key, source, pts):
    start = pts[0][0]
    body = {'key': key, 'source': source, 'from': start.isoformat(), 'to': pts[-1][0].isoformat(),
            't': [(d - start).days for d, _ in pts], 'v': [v for _, v in pts]}
    out.write_text(json.dumps(body, separators=(',', ':')) + '\n')
    print(f'wrote {out.name}: {len(pts)} days {body["from"]} → {body["to"]}')


def main():
    vix = fred('VIXCLS')
    if len(vix) < 9000:
        sys.exit(f'VIX: only {len(vix)} days, keeping the old files')
    write(VIX_OUT, 'vixDaily', 'FRED VIXCLS', [(d, round(v, 2)) for d, v in vix])
    v3 = dict(fred('VXVCLS'))
    pts = [(d, round(v / v3[d], 4)) for d, v in vix if v3.get(d)]
    if len(pts) < 4500:
        sys.exit(f'VIX ratio: only {len(pts)} days, keeping the old file')
    write(OUT, 'vixRatio', 'FRED VIXCLS ÷ VXVCLS', pts)


if __name__ == '__main__':
    main()
