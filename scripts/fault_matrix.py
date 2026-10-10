#!/usr/bin/env python3
"""Nightly fault matrix for the dashboard's layered backups (added 2026-10-09).

Switches sources off on PRODUCTION with ?_fail= (test mode never writes /tmp or KV),
one layer at a time, and checks the one rule that matters: a number is either live
and labelled live, a saved copy labelled stale, or "Unavailable" — never an old or
wrong number dressed as live. Alerts the 📡 thread ONLY when a check fails.

  .venv/bin/python scripts/fault_matrix.py            # run + alert on failure
  .venv/bin/python scripts/fault_matrix.py --no-alert # run, print only

Each case expects one of:
  live  — answered by a live tier (no saved-copy label, not stale-as-copy)
  copy  — answered by a saved copy (/tmp, KV, Sheet LKG), which MUST be stale-labelled
  none  — every tier off: no numbers at all (Unavailable / N/A / HTTP 5xx)
  any   — live or copy (a later live tier may or may not answer at night), still labelled right
"""
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import concurrent.futures as cf

BASE = os.environ.get('FTB_BASE', 'https://financial-telegram-bot-beryl.vercel.app') + '/api/'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPY_ALL = 'lambda,polygon,finnhub,nasdaq,cnbc,yahoo'
MOVE_ALL = 'lambda,finnhub,cnbc,nasdaq,polygon,yahoo'
ME_ALL = ('lambda,polygon,fred,treasury,pmms,cnbc,gold_api,coinbase,coingecko,kraken,erapi,'
          'frankfurter,fawaz,fawaz_bdt,dxy_computed')
HM_ALL = 'fred,hm_treasury,hm_bls,hm_dol,hm_fredcsv,fredcsv'
CG_ALL = 'cg_cnbc,cg_westmetall,cg_fred,cg_goldapi,cg_yahoo,cg_polygon'
SH_LIVE = 'sheets_main,sheets_alt,sheets_cboe,sheets_fred,vix_cboe,vix_fred,signals'
FG_LIVE = 'cnn,rapidapi,fg_yahoo,fg_cboe,fg_fred'
FX_LIVE = 'fx_cnbc,fx_cnbcw,fx_nasdaq,fx_polygon,fx_yahoo'
VOL_LIVE = 'vol_cboe,vol_cnbc,vol_fred,vol_yahoo,vol_polygon,vol_curve'

CASES = [
    ('spy', '', 'live'), ('spy', 'lambda', 'live'), ('spy', 'lambda,polygon,finnhub', 'live'),
    ('spy', SPY_ALL, 'copy'), ('spy', SPY_ALL + ',tmplg', 'copy'), ('spy', SPY_ALL + ',lastgood', 'none'),
    ('spy-daily-move', '', 'live'), ('spy-daily-move', 'lambda', 'live'), ('spy-daily-move', 'lambda,finnhub', 'live'),
    ('spy-daily-move', 'lambda,finnhub,cnbc', 'any'),
    ('spy-daily-move', MOVE_ALL, 'copy'), ('spy-daily-move', MOVE_ALL + ',lastgood', 'none'),
    ('market-extra', '', 'live'), ('market-extra', 'lambda', 'any'), ('market-extra', 'lambda,fred', 'any'),
    ('market-extra', ME_ALL, 'copy'), ('market-extra', ME_ALL + ',lastgood', 'none'),
    ('fred', '', 'live'), ('fred', 'cg_cnbc', 'live'), ('fred', 'pe_multpl,pe_yahoo', 'live'),
    ('fred', HM_ALL, 'copy'), ('fred', HM_ALL + ',' + CG_ALL + ',lastgood,sheetlkg', 'any'),
    ('sheets', '', 'live'), ('sheets', 'vix_cboe,vix_fred', 'any'), ('sheets', 'sheets_main,sheets_alt', 'any'),
    ('sheets', SH_LIVE + ',sheets_cache', 'copy'), ('sheets', SH_LIVE + ',sheets_cache,sheets_kvlg', 'none'),
    ('fear-greed', '', 'live'), ('fear-greed', 'cnn', 'any'), ('fear-greed', FG_LIVE, 'copy'),
    ('fear-greed', FG_LIVE + ',fg_cache,fg_kvlg', 'none'),
    ('factors', '', 'any'), ('factors', FX_LIVE + ',tmplg', 'copy'), ('factors', FX_LIVE + ',lastgood,fx_kv,fx_baked', 'none'),
    ('vol', '', 'live'), ('vol', VOL_LIVE + ',tmplg', 'copy'), ('vol', VOL_LIVE + ',lastgood,vol_curvelg,vol_curvekv', 'none'),
    ('breadth', '', 'live'), ('breadth', 'breadth_polygon', 'live'), ('breadth', 'breadth_polygon,breadth_cnbc', 'copy'),
    ('breadth', 'breadth_polygon,breadth_cnbc,lastgood', 'none'),
    ('polymarket', '', 'live'), ('polymarket', 'lambda,gamma', 'copy'), ('polymarket', 'lambda,gamma,lastgood', 'none'),
    ('rubber-band', '', 'live'), ('rubber-band', 'gist', 'copy'), ('rubber-band', 'gist,lastgood', 'none'),
    ('history', '', 'live'), ('history', 'history_sheet', 'copy'), ('history', 'history_sheet,lastgood', 'none'),
    ('aaii', '', 'live'), ('aaii', 'aaii_http', 'live'), ('aaii', 'aaii_http,aaii_substack,aaii_rss,aaii_lastgood', 'none'),
]

COPY_RE = re.compile(r'last-good|last-known-good|^Stale', re.I)
NONE_RE = re.compile(r'^(Unavailable|Failed|Static Defaults|none)$', re.I)
# A number the card would print. Excludes years/ids by only looking at known value keys.
VALUE_KEYS = {
    'spy': ['current', 'return3y'], 'spy-daily-move': ['value'], 'fear-greed': ['score'],
    'sheets': ['NotSoBoring', 'FrontRunner'], 'market-extra': [], 'fred': [],
}


def fetch(route, faults):
    q = {'_t': str(random.randint(1, 10 ** 9))}
    if faults:
        q['_fail'] = faults
    url = BASE + route + '?' + urllib.parse.urlencode(q, safe=',')
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers={'user-agent': 'fault-matrix/1.0'}), timeout=70)
        status, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read()
    except Exception as e:  # network / timeout
        return None, None, round(time.time() - t, 1), str(e)[:80]
    try:
        return status, json.loads(body), round(time.time() - t, 1), None
    except ValueError:
        return status, None, round(time.time() - t, 1), 'non-JSON body'


def has_value(v):
    return v not in (None, '', 'N/A', 'n/a') and not (isinstance(v, str) and v.startswith('N/A'))


def classify(d):
    """-> 'copy' | 'live' | 'none', plus the label."""
    m = (d or {}).get('_meta') or {}
    src = str(m.get('source') or (d or {}).get('source') or '')
    if d is None or d.get('error') or NONE_RE.match(src):
        return 'none', src
    if m.get('lastGoodAt') or COPY_RE.search(src):
        return 'copy', src
    return 'live', src


def check(route, faults, expect):
    status, d, secs, err = fetch(route, faults)
    if err and d is None and status is None:
        return False, f'no answer ({err})', secs
    kind, src = classify(d)
    stale = bool(((d or {}).get('_meta') or {}).get('stale'))
    problems = []
    if kind == 'copy' and not stale and route not in ('aaii',):
        problems.append('saved copy NOT flagged stale')
    if expect == 'none':
        if kind != 'none' and status and status < 500:
            problems.append(f'expected Unavailable, got {kind} "{src[:50]}"')
        for k in VALUE_KEYS.get(route, []):
            if has_value((d or {}).get(k)):
                problems.append(f'everything off but {k}={d.get(k)!r} still shown')
    elif expect in ('live', 'copy', 'any'):
        if status != 200:
            problems.append(f'HTTP {status}')
        if kind == 'none':
            problems.append(f'Unavailable although a {"saved copy" if expect == "copy" else "tier"} should answer')
        elif expect == 'live' and kind != 'live':
            problems.append(f'expected live, got {kind} "{src[:50]}"')
        elif expect == 'copy' and kind != 'copy':
            problems.append(f'every live tier off but answer labelled live: "{src[:50]}"')
    return not problems, '; '.join(problems) or f'{kind}: {src[:60]}', secs


def spy_3y_agrees():
    """The 3Y return must match between the main source and the first backup (±0.6 pt)."""
    _, a, _, _ = fetch('spy', '')
    _, b, _, _ = fetch('spy', 'lambda')
    ra, rb = (a or {}).get('return3y'), (b or {}).get('return3y')
    if not isinstance(ra, (int, float)) or not isinstance(rb, (int, float)):
        return False, f'3Y missing (main {ra}, backup {rb})'
    return abs(ra - rb) <= 0.6, f'3Y main {ra:.2f}% vs backup {rb:.2f}%'


SHEET_3Y = [
    # (name, csv url, row label, pinned to TODAY()?) — the bot's last-resort 3Y layers (bot/fetchers.py
    # _sheet_return_3y). A TODAY()-pinned sheet only lines up on a trading day's evening.
    ('SPY_INDICATORS', 'https://docs.google.com/spreadsheets/d/1FPxydetBtxFIm-qxrF5BR-sMZAUnbdA09LPbSu5lUCs/export?format=csv&gid=941079229',
     'Three-Year Return', False),
    ('SPY_DAILY_MOVE', 'https://docs.google.com/spreadsheets/d/1T99550TEo19JB6I3aKnRRGXAblB8mWNBsM-48jrDGe4/export?format=csv&gid=0',
     '3 YR Return', False),  # B11 computed in-sheet, anchored on SPY's last trade since 2026-10-10
]


def sheets_3y_agree():
    """Each Google Sheet's 3Y must match the dashboard's (±0.6 pt), so a sheet fallback never
    shows a different number. -> [(ok, msg)]"""
    _, d, _, _ = fetch('spy', '')
    main_3y = (d or {}).get('return3y')
    session = (((d or {}).get('chartHistory') or [{}])[-1]).get('date')
    ny_today = time.strftime('%Y-%m-%d', time.gmtime(time.time() - 4 * 3600))  # EDT; EST shifts it an hour, fine at 21:40
    out = []
    for name, url, label, pinned in SHEET_3Y:
        if pinned and session != ny_today:
            out.append((True, f'{name} 3Y skipped (TODAY()-pinned, no session today)'))
            continue
        try:
            body = urllib.request.urlopen(urllib.request.Request(url + f'&_t={random.randint(1, 10 ** 9)}',
                                                                 headers={'user-agent': 'fault-matrix/1.0'}), timeout=30).read().decode()
            cell = next(l.split(',', 1)[1] for l in body.splitlines() if l.split(',', 1)[0].strip() == label)
            v = float(cell.strip().strip('"').rstrip('%'))
        except Exception as e:
            out.append((False, f'{name} 3Y unreadable ({str(e)[:60]})'))
            continue
        if not isinstance(main_3y, (int, float)):
            out.append((False, f'{name} {v:.2f}% but dashboard 3Y missing'))
            continue
        out.append((abs(v - main_3y) <= 0.6, f'{name} 3Y {v:.2f}% vs dashboard {main_3y:.2f}%'))
    return out


def signals_agree():
    """NotSoBoring/FrontRunner are computed from daily prices (dashboard lib/signals.js) with
    the sheets as backups. Checks: both computed and current; NotSoBoring equals the live
    sheet's; the Yahoo backup gives the same answers as Nasdaq. -> [(ok, msg)]"""
    out = []
    _, d, _, _ = fetch('sheets', '')
    f = ((d or {}).get('_meta') or {}).get('fields') or {}
    for k in ('NotSoBoring', 'FrontRunner'):
        src = str((f.get(k) or {}).get('source') or '')
        ok = src.startswith('Computed') and not (f.get(k) or {}).get('stale')
        out.append((ok, f'{k} {(d or {}).get(k)!r} ← {src[:70]}'))
    _, sh, _, _ = fetch('sheets', 'signals')
    sh_src = str(((((sh or {}).get('_meta') or {}).get('fields') or {}).get('NotSoBoring') or {}).get('source') or '')
    a, b = (d or {}).get('NotSoBoring'), (sh or {}).get('NotSoBoring')
    if sh_src.startswith('Google Sheets'):
        out.append((a == b, f'NotSoBoring computed {a!r} vs sheet {b!r}'))
    else:
        out.append((False, f'NotSoBoring sheet unreadable ({sh_src[:60]})'))
    _, y, _, _ = fetch('sheets', 'signals_nasdaq')
    yf = ((y or {}).get('_meta') or {}).get('fields') or {}
    via = str((yf.get('FrontRunner') or {}).get('source') or '')
    same = all((y or {}).get(k) == (d or {}).get(k) for k in ('NotSoBoring', 'FrontRunner'))
    out.append(('Yahoo' in via and same, f'Yahoo backup: {(y or {}).get("NotSoBoring")!r}/{(y or {}).get("FrontRunner")!r} ← {via[:60]}'))
    return out


def main():
    alert = '--no-alert' not in sys.argv
    # normal loads first (warms caches), then fault cases in modest parallel
    normal = [c for c in CASES if not c[1]]
    rest = [c for c in CASES if c[1]]
    results = []
    with cf.ThreadPoolExecutor(6) as ex:
        results += list(zip(normal, ex.map(lambda c: check(*c), normal)))
    with cf.ThreadPoolExecutor(4) as ex:
        results += list(zip(rest, ex.map(lambda c: check(*c), rest)))
    ok3, msg3 = spy_3y_agrees()
    sheet_checks = sheets_3y_agree()
    signal_checks = signals_agree()

    fails = [(c, r) for c, r in results if not r[0]]
    for (route, faults, expect), (ok, msg, secs) in results:
        print(f"{'PASS' if ok else 'FAIL'} {route:15} {expect:5} {faults[:48]:48} {secs:5}s  {msg}")
    print(f"{'PASS' if ok3 else 'FAIL'} spy-3y-agree  {msg3}")
    for ok, msg in sheet_checks:
        print(f"{'PASS' if ok else 'FAIL'} sheet-3y      {msg}")
    for ok, msg in signal_checks:
        print(f"{'PASS' if ok else 'FAIL'} signals       {msg}")
    sheet_bad = [m for ok, m in sheet_checks if not ok]
    signal_bad = [m for ok, m in signal_checks if not ok]
    total = len(results) + 1 + len(sheet_checks) + len(signal_checks)
    bad = len(fails) + (0 if ok3 else 1) + len(sheet_bad) + len(signal_bad)
    print(f'=== {total - bad}/{total} passed ===')

    if bad and alert:
        lines = [f'• <b>{c[0]}</b> <code>{c[1] or "normal"}</code>: {r[1][:120]}' for c, r in fails[:8]]
        if not ok3:
            lines.append(f'• <b>spy 3Y</b>: {msg3}')
        lines += [f'• <b>sheet 3Y</b>: {m}' for m in sheet_bad]
        lines += [f'• <b>signals</b>: {m}' for m in signal_bad]
        send(f'🧪 <b>Dashboard backup test: {bad} of {total} failed</b>\n' + '\n'.join(lines)
             + '\n<blockquote><i>scripts/fault_matrix.py · nightly · run it again to recheck</i></blockquote>')
    return 1 if bad else 0


def send(text):
    env = {}
    try:
        with open(os.path.join(REPO, '.env')) as f:
            for line in f:
                if '=' in line and not line.lstrip().startswith('#'):
                    k, v = line.strip().split('=', 1)
                    env[k] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    token = env.get('TELEGRAM_TOKEN') or os.environ.get('TELEGRAM_TOKEN')
    chat = os.environ.get('FAULT_MATRIX_ALERT_CHAT') or env.get('TELEGRAM_CHAT_ID')
    if not token or not chat:
        print('alert skipped: no TELEGRAM_TOKEN / chat id')
        return
    data = urllib.parse.urlencode({'chat_id': chat, 'text': text, 'parse_mode': 'HTML',
                                   'disable_web_page_preview': 'true'}).encode()
    try:
        urllib.request.urlopen(f'https://api.telegram.org/bot{token}/sendMessage', data=data, timeout=20)
    except Exception as e:
        print(f'alert failed: {e}')


if __name__ == '__main__':
    sys.exit(main())
