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
the newest quarter fresh on its own."""
import io
import json
import os
import tempfile
import re
import sys
import time
import zipfile

import requests
import xlrd

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
BASE = "https://www.uscourts.gov"
OUT = os.path.join(os.path.dirname(__file__), "..", "dashboard", "lib", "data", "bankruptciesBaked.json")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "f2cache")


def quarters(start_year=2001, end=(2026, 3, 31)):
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
    if cell.ctype == xlrd.XL_CELL_NUMBER:
        return cell.value
    if cell.ctype == xlrd.XL_CELL_TEXT:
        s = cell.value.strip().replace(',', '')
        if s.isdigit():
            return float(s)
    return None


def parse_xls_total_row(data):
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


def main():
    out = []
    sess = requests.Session()
    sess.headers.update(UA)
    for y, m, d in quarters():
        date = f"{y}-{m:02d}-{d:02d}"
        page_url = f"{BASE}/data-news/data-tables/{y}/{m:02d}/{d:02d}/bankruptcy-filings/f-2"
        try:
            r = sess.get(page_url, timeout=30)
            if r.status_code != 200:
                print(f"{date} page {r.status_code}", file=sys.stderr)
                time.sleep(2)
                continue
            links = re.findall(r'href="([^"]*f[-_]?2[^"]*\.(?:xlsx|xls))"', r.text) or \
                    re.findall(r'href="([^"]*\.(?:xlsx|xls))"', r.text)
            links = [l for l in links if "guide" not in l]
            if not links:
                print(f"{date} no file link", file=sys.stderr)
                time.sleep(2)
                continue
            furl = links[0] if links[0].startswith("http") else BASE + links[0]
            cache = os.path.join(CACHE_DIR, f"{date}_{os.path.basename(furl)}")
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            if os.path.exists(cache):
                content = open(cache, "rb").read()
            else:
                fr = sess.get(furl, timeout=60)
                if fr.status_code != 200:
                    print(f"{date} file {fr.status_code} {furl}", file=sys.stderr)
                    time.sleep(2)
                    continue
                content = fr.content
                open(cache, "wb").write(content)
            nums = parse_xlsx_total_row(content) if furl.endswith("xlsx") else parse_xls_total_row(content)
            got = extract(nums)
            if not got:
                print(f"{date} PARSE FAIL nums={nums[:8] if nums else None} {furl}", file=sys.stderr)
            else:
                total, business = got
                if not (200000 < total < 2400000 and 8000 < business < 100000):
                    print(f"{date} IMPLAUSIBLE total={total:,} business={business:,} — skipped", file=sys.stderr)
                else:
                    out.append({"date": date, "total": total, "business": business})
                    print(f"{date} total={total:,} business={business:,}")
        except Exception as e:
            print(f"{date} ERROR {e}", file=sys.stderr)
        time.sleep(2)
    out.sort(key=lambda x: x["date"])
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(out)} quarters -> {OUT}")


if __name__ == "__main__":
    main()
