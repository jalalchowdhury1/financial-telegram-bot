# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "xlrd==2.0.1"]
# ///
"""Regenerate dashboard/lib/data/bankruptciesBaked.json — the baked historical
US bankruptcy-filings series (12-month totals ending each quarter) behind the
Four Horsemen card's bankruptcies panel (see dashboard/lib/bankruptcies.js).

Walks every uscourts.gov quarterly F-2 landing page (2001 -> now), downloads the
linked XLSX/XLS, and extracts the national Total row (total + business filings),
handling every table vintage: modern column-anchored XLSX (formula cells,
absent zero-cells), 2013-2018 layouts whose business/nonbusiness anchoring can
be incomplete (pair-rule fallback: the unique pair summing to the total), and
pre-2013 .xls files that store large numbers as comma-formatted TEXT.

Run it with uv (deps declared inline): `uv run scripts/build_bankruptcies_history.py`
Downloads are cached in $TMPDIR/f2cache so re-runs are fast and polite.
Only needed when extending the baked history; the dashboard's live tier keeps
the newest quarter fresh on its own. .github/workflows/rebake-bankruptcies.yml
runs it monthly and ships what it adds.

It never loses a quarter: one this run could not fetch keeps its baked value,
so a uscourts blip or outage changes nothing. Exit 1 (the workflow pings the
owner) only when the bake stops GROWING, see write_bake. `--strict` (the PR and
branch dry run) also exits 1 when the run could not re-read the baked quarters,
which that never-lose merge would otherwise hide, see check_strict."""
import datetime
import io
import json
import os
import tempfile
import re
import sys
import time
import zipfile

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
BASE = "https://www.uscourts.gov"
OUT = os.path.join(os.path.dirname(__file__), "..", "dashboard", "lib", "data", "bankruptciesBaked.json")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "f2cache")
RETRY_WAITS = (5, 20)    # a uscourts blip gets two more tries; a 404 is an answer (not published yet)
# A uscourts outage must end the run quickly and quietly: a quarter with no answer costs at
# least 27 s (retry waits + politeness sleep), so ~100 of them reach the step's 45-min cap,
# and that timeout would ping the owner.
BREAKER = 5              # this many quarters in a row with no answer = uscourts is down: stop
BUDGET_S = 30 * 60       # stop fetching after 30 min (a normal run takes ~6), inside that cap
STRICT_MISSES = 2        # --strict: at most this many baked quarters not re-read this run
clock = time.monotonic


class NoAnswer(Exception):
    """uscourts did not answer: a connection error, or a 5xx/429 on every try."""


def latest_quarter_end(today=None):
    """The most recent quarter end that is already published.

    AOUSC posts a quarter's F-2 about 4-6 weeks after the quarter closes, so we
    stop one quarter short of "now" — asking for an unpublished quarter just
    yields a 404 and a pointless retry loop.
    """
    today = today or datetime.date.today()
    ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
    candidates = [
        (y, m, d)
        for y in range(today.year - 1, today.year + 1)
        for (m, d) in ends
        if (datetime.date(y, m, d) + datetime.timedelta(days=45)) <= today
    ]
    return max(candidates)


def quarters(start_year=2001, end=None):
    """Every quarter end from `start_year` through `end` (default: the newest
    published quarter, computed from today — NOT a hardcoded date, which would
    silently stop extending the baked history as time passed)."""
    end = end or latest_quarter_end()
    q = [(3, 31), (6, 30), (9, 30), (12, 31)]
    for y in range(start_year, end[0] + 1):
        for m, d in q:
            if (y, m, d) > end:
                return
            yield y, m, d


def parse_xlsx_total_row(data):
    """Column-anchored: find Business/Nonbusiness header columns, read the Total
    row at those columns. Tolerates formula cells and absent zero-cells.
    Returns (total, business, nonbusiness, all_numeric_cells)."""
    z = zipfile.ZipFile(io.BytesIO(data))
    shared = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
    strs = []
    for si in re.findall(r"<si>([\s\S]*?)</si>", shared):
        txt = "".join(re.findall(r"<t[^>]*>([^<]*)</t>", si))
        strs.append(re.sub(r"\s+", " ", txt).strip())
    sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8", "ignore")
    rows = []
    for r in re.findall(r"<row[^>]*>([\s\S]*?)</row>", sheet):
        cells = []
        for m in re.finditer(r'<c\b([^>]*)>(?:<f[^>]*>[\s\S]*?</f>)?(?:<v>([^<]*)</v>)?', r):
            attrs, v = m.group(1), m.group(2)
            cm = re.search(r'\br="([A-Z]+)\d+"', attrs)
            if not cm or v is None:
                continue
            col = cm.group(1)
            if 't="s"' in attrs:
                cells.append((col, strs[int(v)]))
            else:
                try:
                    cells.append((col, float(v)))
                except ValueError:
                    pass
        rows.append(cells)
    biz = nonbiz = None
    for cells in rows:
        for col, val in cells:
            if isinstance(val, str) and re.match(r'(?i)^business', val.strip()) and biz is None:
                biz = col
            if isinstance(val, str) and re.match(r'(?i)^(total )?non-?business', val.strip()) and nonbiz is None:
                nonbiz = col
    for cells in rows:
        if not cells or cells[0][0] != 'A':
            continue
        label = cells[0][1]
        if not (isinstance(label, str) and re.match(r'^total\.*$', label.strip().lower())):
            continue
        nums = [v for _, v in cells if isinstance(v, float)]
        if len(nums) < 4:
            continue
        d = dict(cells)
        return (nums[0], d.get(biz), d.get(nonbiz), nums)
    return None


def _xls_num(cell):
    """Old .xls vintages store big numbers as TEXT with commas ('1,202,503')."""
    import xlrd   # here, not at the top: the tests import this module without it
    if cell.ctype == xlrd.XL_CELL_NUMBER:
        return cell.value
    if cell.ctype == xlrd.XL_CELL_TEXT:
        s = cell.value.strip().replace(',', '')
        if s.isdigit():
            return float(s)
    return None


def parse_xls_total_row(data):
    import xlrd
    wb = xlrd.open_workbook(file_contents=data)
    sh = wb.sheet_by_index(0)
    biz = nonbiz = None
    for r in range(min(14, sh.nrows)):
        for c in range(sh.ncols):
            v = sh.cell_value(r, c)
            if isinstance(v, str):
                s = v.strip()
                if re.match(r'(?i)^business', s) and biz is None:
                    biz = c
                if re.match(r'(?i)^(total )?non-?business', s) and nonbiz is None:
                    nonbiz = c
    for r in range(sh.nrows):
        first = sh.cell_value(r, 0)
        if not (isinstance(first, str) and re.match(r'^total\.*$', first.strip().lower())):
            continue
        row = sh.row(r)
        numcells = [(c, _xls_num(row[c])) for c in range(1, sh.ncols) if _xls_num(row[c]) is not None]
        if len(numcells) < 4:
            continue
        b = _xls_num(row[biz]) if biz is not None else None
        nb = _xls_num(row[nonbiz]) if nonbiz is not None else None
        return (numcells[0][1], b, nb, [v for _, v in numcells])
    return None


def extract(res):
    """Header-anchored first; else the pair rule: exactly business + nonbusiness
    sums to the total (business is always the smaller of the pair)."""
    if not res:
        return None
    total, business, nonbusiness, nums = res
    if total and business and nonbusiness and 0 < business < total \
            and abs(business + nonbusiness - total) / total < 0.02:
        return int(round(total)), int(round(business))
    if total:
        rest = [v for v in nums[1:] if 0 < v < total]
        pairs = [(x, y) for i, x in enumerate(rest) for y in rest[i + 1:]
                 if abs(x + y - total) / total < 0.002 and x != y]
        if len({tuple(sorted(p)) for p in pairs}) == 1:
            x, y = pairs[0]
            return int(round(total)), int(round(min(x, y)))
    return None


def get(sess, url, timeout):
    """sess.get, with two more tries after a connection error, a 5xx or a 429; NoAnswer if
    every try failed that way. Any other status is an answer, a 404 included."""
    for wait in (*RETRY_WAITS, None):
        try:
            r = sess.get(url, timeout=timeout)
        except requests.RequestException as e:
            why = f"{type(e).__name__}: {e}"
        else:
            if r.status_code < 500 and r.status_code != 429:
                return r
            why = f"HTTP {r.status_code}"
        if wait is None:
            raise NoAnswer(why)
        time.sleep(wait)


def fetch_one(sess, date, page_url):
    """One quarter as {date, total, business}, or None with the reason printed. Raises
    NoAnswer when uscourts does not answer."""
    r = get(sess, page_url, 30)
    if r.status_code != 200:
        print(f"{date} page {r.status_code}", file=sys.stderr)
        return None
    links = re.findall(r'href="([^"]*f[-_]?2[^"]*\.(?:xlsx|xls))"', r.text) or \
            re.findall(r'href="([^"]*\.(?:xlsx|xls))"', r.text)
    links = [l for l in links if "guide" not in l]
    if not links:
        print(f"{date} no file link", file=sys.stderr)
        return None
    furl = links[0] if links[0].startswith("http") else BASE + links[0]
    cache = os.path.join(CACHE_DIR, f"{date}_{os.path.basename(furl)}")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        content = open(cache, "rb").read()
    else:
        fr = get(sess, furl, 60)
        if fr.status_code != 200:
            print(f"{date} file {fr.status_code} {furl}", file=sys.stderr)
            return None
        content = fr.content
        open(cache, "wb").write(content)
    nums = parse_xlsx_total_row(content) if furl.endswith("xlsx") else parse_xls_total_row(content)
    got = extract(nums)
    if not got:
        print(f"{date} PARSE FAIL nums={nums[:8] if nums else None} {furl}", file=sys.stderr)
        return None
    total, business = got
    if not (200000 < total < 2400000 and 8000 < business < 100000):
        print(f"{date} IMPLAUSIBLE total={total:,} business={business:,} — skipped", file=sys.stderr)
        return None
    print(f"{date} total={total:,} business={business:,}")
    return {"date": date, "total": total, "business": business}


def fetch_quarters(sess):
    """Every published quarter uscourts answers for, as ([{date, total, business}], why the
    run stopped early or None). Newest first, so a run cut short still gets the new ones;
    every quarter it did not reach keeps its baked value."""
    out, silent, stopped, start = [], 0, None, clock()
    todo = list(quarters())
    for y, m, d in reversed(todo):
        if clock() - start > BUDGET_S:
            stopped = f"the {BUDGET_S // 60}-min time budget ran out"
            break
        date = f"{y}-{m:02d}-{d:02d}"
        page_url = f"{BASE}/data-news/data-tables/{y}/{m:02d}/{d:02d}/bankruptcy-filings/f-2"
        try:
            q = fetch_one(sess, date, page_url)
            silent = 0
        except NoAnswer as e:
            q, silent = None, silent + 1
            print(f"{date} no answer ({e})", file=sys.stderr)
            if silent >= BREAKER:
                stopped = f"uscourts did not answer {BREAKER} quarters in a row ({e})"
                break
        except Exception as e:     # an answer it could not use (a broken file): still an answer
            q, silent = None, 0
            print(f"{date} ERROR {e}", file=sys.stderr)
        if q:
            out.append(q)
        time.sleep(2)
    print(f"fetched {len(out)} of {len(todo)} published quarters")
    if stopped:
        print(f"STOPPED EARLY: {stopped}. Every quarter it did not reach keeps its baked value.")
    return sorted(out, key=lambda x: x["date"]), stopped


def write_bake(fetched, path=None, today=None):
    """Merge this run's quarters into the baked file and write it if anything changed.

    NEVER lose a quarter. The file is the bankruptcies panel's permanent floor, and a run
    can come back short (uscourts down, a page reshuffled, one file that will not parse):
    a quarter it did not get keeps its baked value, one it did get wins (a correction). So
    a blip changes nothing and stays silent. Until 11 Oct 2026 a run one quarter short
    REFUSED to write and failed, which pinged the owner over a hiccup.

    Exit 1 only when the bake stops GROWING: uscourts published the second-newest quarter
    ~3 months ago, so every monthly run since has missed it. That is a broken scraper, not
    bad luck, and the owner must act. (The newest quarter alone can be late or caught by a
    blip; next month retries it.) An unreadable file is never overwritten either."""
    path = path or OUT
    existing = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception as e:
            print(f"REFUSING TO WRITE: the existing bake is unreadable ({e}); restore it from git first.",
                  file=sys.stderr)
            return 1
    merged = {q["date"]: q for q in existing}
    new = [q["date"] for q in fetched if q["date"] not in merged]
    revised = [q["date"] for q in fetched if q["date"] in merged and merged[q["date"]] != q]
    merged.update((q["date"], q) for q in fetched)
    out = [merged[k] for k in sorted(merged)]
    kept = len(out) - len(fetched)

    published = [f"{y}-{m:02d}-{d:02d}" for y, m, d in quarters(end=latest_quarter_end(today))]
    if published[-2] not in merged:
        print(f"NOT GROWING: no {published[-2]} quarter, which uscourts published months ago; "
              f"the newest baked quarter is {out[-1]['date'] if out else 'none'}. The scraper needs a look.",
              file=sys.stderr)
        return 1
    if kept:
        print(f"kept {kept} quarter(s) from the existing bake that this run could not fetch")
    if published[-1] not in merged:
        print(f"{published[-1]} is not in yet (published late, or a blip): next month retries it")

    if out == existing:
        print(f"no change ({len(out)} quarters)")
        return 0
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(out)} quarters -> {path} (was {len(existing)}; new: {new or 'none'}, "
          f"revised: {revised or 'none'})")
    return 0


def check_strict(fetched, stopped, path=None):
    """--strict, for a PR or another branch: its dry run must show the scraper still reads
    uscourts, and the never-lose merge above would pass a scraper that reads nothing. Exit 1
    when the run stopped early or did not re-read more than STRICT_MISSES baked quarters."""
    with open(path or OUT) as f:
        baked = {q["date"] for q in json.load(f)}
    missed = sorted(baked - {q["date"] for q in fetched})
    if stopped or len(missed) > STRICT_MISSES:
        why = stopped or f"{len(missed)} baked quarters not re-read (first: {', '.join(missed[:3])})"
        print(f"STRICT: {why}, so this dry run cannot show the scraper works.", file=sys.stderr)
        return 1
    print(f"strict: re-read all but {len(missed)} of the {len(baked)} baked quarters")
    return 0


def main(argv=None):
    strict = "--strict" in (sys.argv[1:] if argv is None else argv)
    sess = requests.Session()
    sess.headers.update(UA)
    fetched, stopped = fetch_quarters(sess)
    rc = write_bake(fetched)
    if rc == 0 and strict:
        rc = check_strict(fetched, stopped)
    return rc


if __name__ == "__main__":
    sys.exit(main())
