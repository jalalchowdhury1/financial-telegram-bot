#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["xlrd==2.0.1"]
# ///
"""
bake_long_history.py — decades of look-back for the 📈 tap-a-number popover.

The popover's own data is the history sheet: daily snapshots, which only start on
2026-03-12. This bake adds what came BEFORE, one static file per stat:

    public/history/<key>.json        fetched by the popover when it opens (lib/longHistory.js)
    lib/data/longHistoryIndex.json   bundled: which stats have one, how far back, the source

The sheet is never touched. The popover draws baked points only BEFORE the sheet's first
point for that stat, so the newest part of every chart is still the sheet's record of what
the dashboard showed.

Every value is computed the way the dashboard computes the live number (same series, same
transform — see SPECS and app/api/fred/route.js buildResponse), then dated by when it was
PUBLISHED (observation date + the series' usual release lag): the sheet records a print on
the day it reached us, so a long line dated by observation period would zig-zag at the join.
Daily series older than 5 years are thinned to one point a week (the week's last close).

Self-check (fails closed): each stat is compared with the sheet wherever both exist. A stat
whose values do not match the sheet (a units or formula slip) is REFUSED and its old file
kept. A bake that reaches back less far, or has far fewer points, than the file it would
replace is refused too, except from a ROLLING source (FRED serves only the last 3 years of
ICE BofA series): there the file keeps its older points and the download is added after
them, once the two agree where they overlap. BDT pairs have no long free source and stay
sheet-only.

Run (from dashboard/):  uv run scripts/bake_long_history.py [--dry] [--retry-after=SECONDS] [key ...]
--retry-after: stats whose DOWNLOAD failed get one more try after that wait. A refused stat
is not retried: it would only refuse again.
.github/workflows/rebake-long-history.yml re-bakes every quarter: the sheet's chart window
is capped at 730 days (lib/marks.js CHART_MAX_DAYS), so a bake left alone ~2 years would
leave a gap between its last day and the sheet. A stat that kept its old file is fine until
then. The last lines are for that workflow: NOTHING BAKED when every stat failed, and
RUNWAY LOW naming each stat whose chart would show a gap within a year (a file kept too
long, or a source that still answers but stopped updating).
"""
import csv
import functools
import io
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / 'public' / 'history'
INDEX = HERE.parent / 'lib' / 'data' / 'longHistoryIndex.json'
SHEET_CSV = ('https://docs.google.com/spreadsheets/d/1lA-_yjLMc3qDTt9sogSPQrCohNULIk5wwJYfb5wIHfc'
             '/export?format=csv&gid=0')
BROWSER = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
DAILY_YEARS = 5          # keep every day this far back from the newest point; weekly before
MATCH_FRAC = 0.2         # refuse when the median |baked − sheet| exceeds 20% of the stat's 10-year p10–p90 range
CHART_DAYS = 730         # lib/marks.js CHART_MAX_DAYS: the days of sheet /api/history serves
RUNWAY_ALERT_DAYS = 365  # RUNWAY LOW names a stat whose chart gap is nearer than this
SHEET_WAITS = (5, 15, 30, 60, 120)   # the sheet is read once, before any stat: a Google blip must not end the run
JOIN_FRAC = 0.05         # a ROLLING join needs the median |download − file| where they overlap within 5% of that range
JOIN_MIN_DAYS = 90       # ...over at least this many shared days


# ── downloads ──────────────────────────────────────────────────────────────────

def curl(url, ua=None, headers=(), waits=(2, 5)):
    """FRED resets connections from most user agents but answers curl's own, so FRED calls
    pass ua=None. One try more than there are waits; raises on failure."""
    cmd = ['curl', '-sfL', '-m', '90', '--compressed']
    if ua:
        cmd += ['-A', ua]
    for h in headers:
        cmd += ['-H', h]
    last = None
    for wait in (*waits, None):
        r = subprocess.run(cmd + [url], capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        last = f'curl exit {r.returncode}'
        if wait is not None:
            time.sleep(wait)
    raise RuntimeError(f'{last}: {url}')


def fred(sid):
    text = curl(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}').decode()
    out = []
    for line in text.splitlines()[1:]:
        d, _, v = line.partition(',')
        try:
            out.append((date.fromisoformat(d), float(v)))
        except ValueError:
            pass  # '.' or '' = no observation that day
    if len(out) < 10:
        raise RuntimeError(f'FRED {sid}: only {len(out)} observations')
    return out


def add_months(d, n):
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


@functools.lru_cache(maxsize=None)
def cnbc_bars(symbol, rng):
    """One CNBC bar series, fetched once per run (gold and copper/gold share @GC.1).
    CNBC's chart API caches its answer per EXACT URL, an error too: when its backend fails
    it answers 200 with no bars and "503 Backend fetch failed", then replays that for hours
    (.DXY ALL replayed a 23:42 error until at least 09:42 the next day, 2026-10-11). Retrying
    the same URL could never get past it, and sank all four of PR #61's dry runs. So every
    try carries a `_=` the cache has never seen and reaches the backend, whose own misses
    are brief."""
    url = f'https://ts-api.cnbc.com/harmony/app/charts/{rng}.json?symbol={quote(symbol)}'
    for attempt in range(4):
        if attempt:
            time.sleep(10 * 2 ** (attempt - 1))   # 10, 20, 40 s; main() retries once more later
        j = json.loads(curl(f'{url}&_={time.time_ns()}{attempt}', ua=BROWSER))
        out = []
        for b in (j.get('barData') or {}).get('priceBars') or []:
            try:
                out.append((datetime.strptime(b['tradeTime'][:8], '%Y%m%d').date(), float(b['close'])))
            except (KeyError, ValueError, TypeError):
                pass
        if out:
            return tuple(out)
    raise RuntimeError(f'CNBC {symbol} {rng}: no bars after 4 tries ({j.get("statusMessage", "")[:80]})')


def cnbc(symbol):
    """CNBC chart bars: 1Y = daily (≈2 years), 10Y = weekly (bar dated the week's Sunday,
    close = Friday), ALL = quarterly (dated the quarter's first day, close = quarter end)."""
    bars = functools.partial(cnbc_bars, symbol)
    daily = list(bars('1Y'))
    weekly = [(d + timedelta(days=5), v) for d, v in bars('10Y')]
    quarterly = [(add_months(d, 3) - timedelta(days=1), v) for d, v in bars('ALL')]
    return ([p for p in quarterly if p[0] < weekly[0][0]]
            + [p for p in weekly if p[0] < daily[0][0]] + daily)


def cnn_fear_greed():
    j = json.loads(curl('https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2020-07-15',
                        ua=BROWSER, headers=('Referer: https://edition.cnn.com/', 'Accept: application/json')))
    by = {}
    for p in j['fear_and_greed_historical']['data']:
        by[datetime.fromtimestamp(p['x'] / 1000, tz=timezone.utc).date()] = float(p['y'])
    pts = sorted(by.items())
    # CNN's feed opens with ~6 months of filler (2020-07-15 → 2021-01-21): a flat 50 broken
    # by a few junk prints (3.8, 1.9). Not data: start after the filler's last 50.
    first = pts[0][0] if pts else None
    filler = [i for i, (d, v) in enumerate(pts) if v == 50.0 and (d - first).days <= 366]
    return pts[filler[-1] + 1:] if len(filler) >= 20 else pts


def aaii_spread():
    """AAII weekly survey, BEAR − bull in percentage points: the pill's AAIIDiff
    (lib/aaii.js) and so the sheet's AAII DIFF are bear − bull, positive = bears lead."""
    import xlrd
    book = xlrd.open_workbook(file_contents=curl('https://www.aaii.com/files/surveys/sentiment.xls', ua=BROWSER))
    s = book.sheet_by_name('SENTIMENT')
    out = []
    for r in range(s.nrows):
        row = s.row_values(r)
        if not isinstance(row[0], float) or not isinstance(row[1], float) or not isinstance(row[3], float):
            continue
        d = xlrd.xldate_as_datetime(row[0], book.datemode).date()
        out.append((d, (row[3] - row[1]) * 100))
    return sorted(dict(out).items())


def multpl_pe():
    """S&P 500 trailing P/E by month (multpl.com — the dashboard's own first P/E source)."""
    html = curl('https://www.multpl.com/s-p-500-pe-ratio/table/by-month', ua=BROWSER).decode()
    out = {}
    for d, cell in re.findall(r'<td[^>]*>\s*([A-Z][a-z]{2} \d{1,2}, \d{4})\s*</td>\s*<td[^>]*>(.*?)</td>', html, re.S):
        num = re.findall(r'-?\d+(?:\.\d+)?', re.sub(r'<[^>]+>|&#x?[0-9a-fA-F]+;', ' ', cell))
        if num:
            out[datetime.strptime(d, '%b %d, %Y').date()] = float(num[-1])
    return sorted(out.items())


# ── transforms (each mirrors app/api/fred/route.js buildResponse) ──────────────

def month_offset_pct(obs, n):
    """% change vs the observation n months earlier (findByMonthOffset on monthly data)."""
    by = {(d.year, d.month): v for d, v in obs}
    out = []
    for d, v in obs:
        p = add_months(d, -n)
        base = by.get((p.year, p.month))
        if base:
            out.append((d, (v - base) / base * 100))
    return out


def sahm(unrate):
    """FRED SAHMCURRENT, as lib/jevInputs.js sahmFromHistory: the 3-month average minus the lowest
    3-month average of the prior 12 months. By calendar month, so a month FRED skipped (Oct 2025)
    is a gap, not the month before; an average needs 2 of its 3 months."""
    by = {d.year * 12 + d.month - 1: v for d, v in unrate}
    avg = {}
    for k in by:
        vals = [by[j] for j in (k - 2, k - 1, k) if j in by]
        if len(vals) >= 2:
            avg[k] = sum(vals) / len(vals)
    out = []
    for d, _ in unrate:
        k = d.year * 12 + d.month - 1
        prior = [avg[j] for j in range(k - 12, k) if j in avg]
        if k in avg and len(prior) >= 11:
            out.append((d, avg[k] - min(prior)))
    return out


def claims_4wk(icsa):
    return [(icsa[i][0], sum(v for _, v in icsa[i - 3:i + 1]) / 4 / 1000) for i in range(3, len(icsa))]


def ratio(a, b, mult=1.0):
    bb = dict(b)
    return [(d, v / bb[d] * mult) for d, v in a if bb.get(d)]


def mortgage_payment(rates, mspus, mspus_lag):
    """Freddie 30-yr rate × the median home price PUBLISHED by then, 80% LTV, 360 payments
    (bot/fetchers.py: principal = MSPUS × 0.80)."""
    prices = [(d + timedelta(days=mspus_lag), v) for d, v in mspus]
    out, j = [], -1
    for d, rate in rates:
        while j + 1 < len(prices) and prices[j + 1][0] <= d:
            j += 1
        if j < 0:
            continue
        principal = prices[j][1] * 0.80
        r = rate / 100 / 12
        out.append((d, principal * r * (1 + r) ** 360 / ((1 + r) ** 360 - 1) if r > 0 else principal / 360))
    return out


# ── the stats ──────────────────────────────────────────────────────────────────
# key: sheet col (lib/marks.js SHEET_METRICS — keep in step), release lag in days
# (observation date → first publication), source label, builder.
# Lags: weekly FRED series are dated by the week's end and print ~5 days later; monthly
# ones are dated the 1st and print the following month (jobs ~day 35, rent CPI ~42, retail
# ~45, industrial ~46, housing ~47, M2 ~55, UMich on FRED ~56 — a month behind the survey,
# savings ~60, durable goods M3 ~60, JOLTS ~62); quarterly corporate profits ride the 2nd GDP
# estimate (~150), the FHFA index ~150. Measured 2026-10-10 against the days the sheet's
# value changed (the median of each series' release events).
SPECS = {
    'yieldCurve':      (1,  0,   'FRED T10Y2Y', lambda: fred('T10Y2Y')),
    'profitMargin':    (2,  150, 'FRED corporate profits ÷ GDP', lambda: ratio(fred('A053RC1Q027SBEA'), fred('GDP'), 100)),
    'sahmRule':        (3,  35,  'FRED UNRATE (Sahm rule)', lambda: sahm(fred('UNRATE'))),
    'sentiment':       (4,  56,  'FRED UMCSENT', lambda: fred('UMCSENT')),
    'claims':          (5,  5,   'FRED ICSA (4-week avg)', lambda: claims_4wk(fred('ICSA'))),
    'creditSpread':    (6,  0,   'FRED BAMLC0A4CBBB', lambda: fred('BAMLC0A4CBBB')),
    'realYields':      (7,  0,   'FRED DFII10', lambda: fred('DFII10')),
    'peRatio':         (9,  0,   'multpl S&P 500 P/E', multpl_pe),
    'nfci':            (10, 5,   'FRED NFCI', lambda: fred('NFCI')),
    'm2':              (11, 55,  'FRED M2SL (year-on-year)', lambda: month_offset_pct(fred('M2SL'), 12)),
    'retail':          (12, 45,  'FRED RSXFS (3-month change)', lambda: month_offset_pct(fred('RSXFS'), 3)),
    'housing':         (13, 47,  'FRED HOUST', lambda: fred('HOUST')),
    'indpro':          (14, 46,  'FRED INDPRO (6-month change)', lambda: month_offset_pct(fred('INDPRO'), 6)),
    'jolts':           (15, 62,  'FRED JTSJOL', lambda: fred('JTSJOL')),
    'durable':         (16, 60,  'FRED DGORDER (3-month change)', lambda: month_offset_pct(fred('DGORDER'), 3)),
    'savings':         (17, 60,  'FRED PSAVERT', lambda: fred('PSAVERT')),
    'rentIndex':       (18, 42,  'FRED rent CPI × 4.41', lambda: [(d, v * 4.41) for d, v in fred('CUUR0000SEHA')]),
    'mortgagePayment': (19, 0,   'FRED MORTGAGE30US + MSPUS', lambda: mortgage_payment(fred('MORTGAGE30US'), fred('MSPUS'), 118)),
    'mortgageRate':    (20, 0,   'FRED MORTGAGE30US', lambda: fred('MORTGAGE30US')),
    'tnx':             (21, 0,   'FRED DGS10', lambda: fred('DGS10')),
    't2y':             (22, 0,   'FRED DGS2', lambda: fred('DGS2')),
    'dxy':             (23, 0,   'CNBC .DXY', lambda: cnbc('.DXY')),
    'cl':              (24, 0,   'FRED DCOILWTICO', lambda: fred('DCOILWTICO')),
    'usdcad':          (25, 0,   'FRED DEXCAUS', lambda: fred('DEXCAUS')),
    'usdinr':          (26, 0,   'FRED DEXINUS', lambda: fred('DEXINUS')),
    'cadinr':          (29, 0,   'FRED DEXINUS ÷ DEXCAUS', lambda: ratio(fred('DEXINUS'), fred('DEXCAUS'))),
    'gold':            (30, 0,   'CNBC gold futures', lambda: cnbc('@GC.1')),
    'btc':             (31, 0,   'FRED CBBTCUSD', lambda: fred('CBBTCUSD')),
    'copperGold':      (32, 0,   'CNBC copper ÷ gold × 1000', lambda: ratio(cnbc('@HG.1'), cnbc('@GC.1'), 1000)),
    'atnhpi':          (33, 150, 'FRED ATNHPIUS39300Q', lambda: fred('ATNHPIUS39300Q')),
    'aaiiDiff':        (35, 0,   'AAII survey', aaii_spread),
    'vixCurrent':      (36, 0,   'FRED VIXCLS', lambda: fred('VIXCLS')),
    'vix3m':           (37, 0,   'FRED VXVCLS', lambda: fred('VXVCLS')),
    'cnnFearGreed':    (66, 0,   'CNN Fear & Greed', cnn_fear_greed),
}

# The sheet's own history is a DIFFERENT number before these dates, so the baked line is
# drawn up to them instead of up to the sheet's first point (the popover reads `sheetFrom`
# from the index). P/E: until 2026-09-05 the dashboard's P/E came from another source
# (~29-32 while multpl said ~25-26); from that day it is multpl's, the source baked here.
# Savings: BEA's annual update (live 2026-09-30) lifted the whole PSAVERT history ~1.5
# points, so the sheet's earlier snapshots (2.6-3.6) are numbers BEA has since replaced.
SHEET_FROM = {'peRatio': date(2026, 9, 5), 'savings': date(2026, 9, 30)}

# Sources that serve a moving window, not their whole history: FRED's licence for the ICE
# BofA indices covers only the last 3 years, so creditSpread's download starts a quarter
# later every quarter. Refused as "starts later", its file would have aged until its chart
# opened a gap (from Jan 2027 on). Their files keep their older points (see join).
ROLLING = {'creditSpread'}

# ── the sheet (same parse as lib/marks.js buildSeries) ─────────────────────────

def parse_value(raw):
    s = (raw or '').strip().replace(',', '')
    if not s or s.upper() == 'N/A' or not re.fullmatch(r'-?\d*\.?\d+', s):
        return None
    v = float(s)
    return None if abs(v) < 1e-9 else v


def unit_jump(a, b):
    lo, hi = sorted((abs(a), abs(b)))
    if lo < 1e-9:
        return False
    r = hi / lo
    return 900 <= r <= 1100 or 900000 <= r <= 1100000


def sheet_series(rows, col):
    by = {}
    for r in rows:
        d = (r[0] if r else '').strip()[:10]
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            continue
        v = parse_value(r[col] if col < len(r) else '')
        if v is None:
            by.pop(d, None)
            continue
        by[d] = v
    s = sorted(by.items())
    if not s:
        return []
    last = s[-1][1]
    return [(date.fromisoformat(d), v) for d, v in s if not unit_jump(v, last)]


# ── checks, thinning, output ───────────────────────────────────────────────────

def median_diff(pts, sheet):
    """Median |baked − sheet|, the baked value taken as of the day BEFORE each sheet day
    (the sheet's 10:00 ET snapshot shows the previous close / the latest print)."""
    diffs, j = [], -1
    for d, s in sheet:
        while j + 1 < len(pts) and pts[j + 1][0] < d:
            j += 1
        if j >= 0 and (d - pts[j][0]).days <= 400:
            diffs.append(abs(pts[j][1] - s))
    return (statistics.median(diffs), len(diffs)) if diffs else (None, 0)


def spread_of(pts):
    recent = sorted(v for d, v in pts if d >= pts[-1][0] - timedelta(days=3653))
    p10, p90 = recent[len(recent) // 10], recent[(len(recent) * 9) // 10]
    return (p90 - p10) or abs(statistics.median(recent)) or 1.0



def thin(pts):
    """Every point for the last DAILY_YEARS; before that the last point of each ISO week —
    only for daily series (weekly / monthly / quarterly ones are already sparse)."""
    gaps = [(b[0] - a[0]).days for a, b in zip(pts, pts[1:])]
    if not gaps or statistics.median(gaps) > 3:
        return pts
    cut = pts[-1][0] - timedelta(days=round(365.25 * DAILY_YEARS))
    weeks = {}
    for d, v in pts:
        if d < cut:
            weeks[d.isocalendar()[:2]] = (d, v)
    return sorted(weeks.values()) + [p for p in pts if p[0] >= cut]


def sig(v):
    return float(f'{v:.5g}')


def file_points(body):
    start = date.fromisoformat(body['from'])
    return [(start + timedelta(days=t), v) for t, v in zip(body['t'], body['v'])]


def join(old, new, spread):
    """A ROLLING source's download `new` added after its file's points `old` that it no
    longer serves: (points, how many old ones were kept), or (None, why) when the two do not
    agree where they overlap (a unit or series change must not be spliced onto the old line)
    or barely overlap (nothing to check the join against, and a hole between them)."""
    by_day = dict(old)
    diffs = [abs(sig(v) - by_day[d]) for d, v in new if d in by_day]
    if len(diffs) < JOIN_MIN_DAYS:
        return None, f'shares {len(diffs)} days with the existing file, {JOIN_MIN_DAYS} needed to join'
    md = statistics.median(diffs)
    if md > JOIN_FRAC * spread:
        return None, f'differs from the existing file where they overlap (median {md:.4g}, limit {JOIN_FRAC * spread:.4g})'
    kept = [p for p in old if p[0] < new[0][0]]
    return thin(kept + new), len(kept)


def runway_low(entries, keys, today):
    """Of `keys`, the stats worth the owner's time. The popover draws a baked file only
    before the sheet window, which /api/history caps at CHART_DAYS, so a file leaves a gap
    in its chart from (its last day + CHART_DAYS): one missed quarter still leaves ~2 years.
    `entries` is the index after this run (a re-baked stat's new last day, a failed one's
    old). Named when that gap is less than RUNWAY_ALERT_DAYS away, or with no file at all."""
    low = []
    for k in keys:
        if k not in entries:
            low.append(f'{k} (no baked file)')
            continue
        gap = date.fromisoformat(entries[k]['to']) + timedelta(days=CHART_DAYS)
        if (gap - today).days < RUNWAY_ALERT_DAYS:
            low.append(f'{k} (chart gap from {gap})')
    return low


def main(argv):
    dry = '--dry' in argv
    retry_after = next((int(a.split('=', 1)[1]) for a in argv if a.startswith('--retry-after=')), 0)
    only = [a for a in argv if not a.startswith('--')]
    unknown = [k for k in only if k not in SPECS]
    if unknown:
        sys.exit(f'unknown key(s): {unknown}')
    rows = list(csv.reader(io.StringIO(curl(SHEET_CSV, ua='financial-dashboard-bake/1.0', waits=SHEET_WAITS).decode())))
    try:
        index = json.loads(INDEX.read_text())
    except FileNotFoundError:
        index = {'keys': {}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'{"key":16} {"from":10} {"to":10} {"pts":>5}  {"vs sheet":>9} {"n":>4} {"limit":>8}  lag')

    def bake(key):
        """'ok' (written), 'download' (source failed) or 'refused' (self-check)."""
        col, lag, source, build = SPECS[key]
        try:
            raw = sorted(build())
        except Exception as e:  # one dead source must not sink the others
            print(f'{key:16} FAILED download: {e}')
            return 'download'
        pts = [(d + timedelta(days=lag), v) for d, v in raw]
        sheet = [p for p in sheet_series(rows, col) if p[0] >= SHEET_FROM.get(key, date.min)]
        md, n = median_diff(pts, sheet)
        spread = spread_of(pts)
        limit = MATCH_FRAC * spread
        pts = thin(pts)
        verdict, note = 'ok', ''
        if md is not None and md > limit:
            verdict = 'REFUSED: does not match the sheet'
        prev_path = OUT_DIR / f'{key}.json'
        if prev_path.exists() and verdict == 'ok':
            prev = json.loads(prev_path.read_text())
            if key in ROLLING:
                joined, kept = join(file_points(prev), pts, spread)
                if joined is None:
                    verdict = f'REFUSED: {kept}'
                else:
                    pts, note = joined, f'kept {kept} older points' if kept else ''
            if verdict != 'ok':
                pass
            elif pts[0][0].isoformat() > prev['from'] and (pts[0][0] - date.fromisoformat(prev['from'])).days > 31:
                verdict = f'REFUSED: starts {pts[0][0]}, the existing file starts {prev["from"]}'
            elif len(pts) < 0.9 * len(prev['v']):
                verdict = f'REFUSED: {len(pts)} points, the existing file has {len(prev["v"])}'
        mds = f'{md:9.4g}' if md is not None else '        —'
        print(f'{key:16} {pts[0][0]} {pts[-1][0]} {len(pts):5d}  {mds} {n:4d} {limit:8.4g}  {lag:3d}  {note if verdict == "ok" else verdict}')
        if verdict != 'ok':
            return 'refused'
        start = pts[0][0]
        body = {'key': key, 'source': source, 'from': start.isoformat(), 'to': pts[-1][0].isoformat(),
                't': [(d - start).days for d, _ in pts], 'v': [sig(v) for _, v in pts]}
        index['keys'][key] = {'from': body['from'], 'to': body['to'], 'n': len(pts), 'source': source}
        if key in SHEET_FROM:
            index['keys'][key]['sheetFrom'] = SHEET_FROM[key].isoformat()
        if not dry:
            prev_path.write_text(json.dumps(body, separators=(',', ':'), ensure_ascii=False) + '\n')
        return 'ok'

    def attempt(key):
        try:
            return bake(key)
        except Exception as e:  # nor may a bad answer that trips a self-check (no points at all)
            print(f'{key:16} FAILED: {type(e).__name__}: {e}')
            return 'error'

    results = {k: attempt(k) for k in SPECS if not only or k in only}
    retry = [k for k, r in results.items() if r == 'download']
    if retry and retry_after:
        print(f'\n{len(retry)} download(s) failed, one more try in {retry_after} s: {retry}')
        time.sleep(retry_after)
        for k in retry:
            results[k] = attempt(k)
    failed = [k for k, r in results.items() if r != 'ok']
    index['bakedAt'] = date.today().isoformat()
    index['keys'] = dict(sorted(index['keys'].items()))
    if not dry:
        INDEX.write_text(json.dumps(index, indent=1, ensure_ascii=False) + '\n')
    print(f'\n{"dry run — nothing written" if dry else f"wrote {OUT_DIR.relative_to(HERE.parent)}/ + {INDEX.relative_to(HERE.parent)}"}'
          f'{f"; FAILED/REFUSED: {failed}" if failed else ""}')
    if failed and len(failed) == len(results):
        print('NOTHING BAKED: every stat kept its old file')
    low = runway_low(index['keys'], list(results), date.today())
    if low:
        print(f'RUNWAY LOW: {", ".join(low)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
