/**
 * US Bankruptcies (Administrative Office of the U.S. Courts) — the 4th horseman.
 *
 * The AOUSC publishes bankruptcy filings QUARTERLY as "Table F-2" (business and
 * nonbusiness cases filed, by chapter, for the 12-month period ending each
 * quarter). There is no FRED series and no JSON API — the durable machine
 * artifact is a small XLSX at a predictable URL, e.g.
 *   https://www.uscourts.gov/sites/default/files/document/bf_f2_0331.2026.xlsx
 * with a stable layout: a national "Total" row whose numeric cells are three
 * equal-width groups (Total | Business | Nonbusiness), each group starting with
 * its "All Chapters" column.
 *
 * Layers of protection (the series can never just vanish from the dashboard):
 *   1. LIVE  — try the predictable XLSX URL for the most recent quarter-ends
 *              (newest first); if a file 404s/renames, discover the link from
 *              that quarter's F-2 landing page instead.
 *   2. BAKED — `lib/data/bankruptciesBaked.json`, the full quarterly history
 *              scraped from the same tables (regenerate with
 *              `scripts/build_bankruptcies_history.py`). Always present, so a
 *              uscourts outage only costs freshness, never the chart.
 *   3. The whole block lives inside /api/fred's serve() → /tmp last-known-good.
 * Fault gates: ?_fail=bk_uscourts (kill live), bk_baked (kill baked history).
 *
 * The XLSX parse uses a minimal, dependency-free ZIP reader (Node's zlib does
 * the DEFLATE) — deliberately NOT a SheetJS dependency for one tiny stable file.
 */
import { inflateRawSync } from 'zlib';
import { withFreshness } from './freshness';

// The 12-month-ending value only changes once a quarter and publishes ~4-6
// weeks after quarter end, so the newest point legitimately ages to ~120 days
// before the next print. Past the deadline the value is shown in orange
// (graceful staleness) rather than hidden — an old real filings number beats
// an N/A (same call as S&P EPS).
export const BANKRUPTCY_FRESHNESS_DAYS = 150;

const EOCD_SIG = 0x06054b50, CDIR_SIG = 0x02014b50;

/** Extract one entry (by exact name) from a ZIP buffer. Returns Buffer|null. */
export function unzipEntry(buf, entryName) {
    // Find End-of-Central-Directory (scan back ≤64KB for the signature).
    let i = buf.length - 22;
    const floor = Math.max(0, buf.length - 22 - 65535);
    while (i >= floor && buf.readUInt32LE(i) !== EOCD_SIG) i--;
    if (i < floor) throw new Error('xlsx: not a zip (no EOCD)');
    const count = buf.readUInt16LE(i + 10);
    let off = buf.readUInt32LE(i + 16);
    for (let n = 0; n < count; n++) {
        if (buf.readUInt32LE(off) !== CDIR_SIG) throw new Error('xlsx: bad central directory');
        const method = buf.readUInt16LE(off + 10);
        const csize = buf.readUInt32LE(off + 20);
        const nameLen = buf.readUInt16LE(off + 28);
        const extraLen = buf.readUInt16LE(off + 30);
        const commentLen = buf.readUInt16LE(off + 32);
        const localOff = buf.readUInt32LE(off + 42);
        const name = buf.toString('utf8', off + 46, off + 46 + nameLen);
        if (name === entryName) {
            // Local header repeats name/extra with its OWN lengths — trust those.
            const lNameLen = buf.readUInt16LE(localOff + 26);
            const lExtraLen = buf.readUInt16LE(localOff + 28);
            const start = localOff + 30 + lNameLen + lExtraLen;
            const data = buf.subarray(start, start + csize);
            return method === 8 ? inflateRawSync(data) : Buffer.from(data);
        }
        off += 46 + nameLen + extraLen + commentLen;
    }
    return null;
}

/**
 * Parse an F-2 XLSX and return the national totals from its "Total" row:
 * { total, business, nonbusiness }. The parse is COLUMN-ANCHORED: it locates
 * the "Business" / "Nonbusiness" header cells and reads the Total row at those
 * columns (cells can be formulas, and zero cells are simply absent from the
 * XML, so positional/width rules are unreliable). Throws unless
 * business + nonbusiness ≈ total, so a reshuffled table can never ship a
 * wrong number.
 */
export function parseF2Totals(buf) {
    const ssXml = unzipEntry(buf, 'xl/sharedStrings.xml')?.toString('utf8') ?? '';
    const strs = [...ssXml.matchAll(/<si>([\s\S]*?)<\/si>/g)].map((m) =>
        [...m[1].matchAll(/<t[^>]*>([^<]*)<\/t>/g)].map((t) => t[1]).join('').replace(/\s+/g, ' ').trim(),
    );
    const sheetXml = unzipEntry(buf, 'xl/worksheets/sheet1.xml')?.toString('utf8');
    if (!sheetXml) throw new Error('xlsx: sheet1.xml missing');

    // Rows as [{col:'G', str|num}] — tolerate formula cells (<f> before <v>).
    const rows = [...sheetXml.matchAll(/<row[^>]*>([\s\S]*?)<\/row>/g)].map((r) => {
        const cells = [];
        for (const c of r[1].matchAll(/<c\b([^>]*)>(?:<f[^>]*>[\s\S]*?<\/f>)?(?:<v>([^<]*)<\/v>)?/g)) {
            const [, attrs, v] = c;
            const col = attrs.match(/\br="([A-Z]+)\d+"/)?.[1];
            if (!col || v === undefined) continue;
            if (/\bt="s"/.test(attrs)) cells.push({ col, str: strs[parseInt(v, 10)] ?? '' });
            else { const f = parseFloat(v); if (Number.isFinite(f)) cells.push({ col, num: f }); }
        }
        return cells;
    });

    let bizCol = null, nonbizCol = null;
    for (const row of rows) {
        for (const c of row) {
            if (c.str && /^business/i.test(c.str)) bizCol = bizCol ?? c.col;
            if (c.str && /^nonbusiness/i.test(c.str)) nonbizCol = nonbizCol ?? c.col;
        }
        if (bizCol && nonbizCol) break;
    }
    if (!bizCol || !nonbizCol) throw new Error('F-2: Business/Nonbusiness header not found');

    for (const row of rows) {
        // The national row: label "Total" in column A (older vintages also have a
        // "Total" SUB-header cell in the business group — column A + a numeric
        // row disambiguates) with the numbers alongside.
        if (row[0]?.col !== 'A' || !/^total\.*$/.test(row[0]?.str?.trim().toLowerCase() ?? '')) continue;
        if (row.filter((c) => c.num != null).length < 4) continue;
        const at = (col) => row.find((c) => c.col === col)?.num ?? null;
        const total = row.find((c) => c.num != null)?.num ?? null; // first numeric = Total All Chapters
        return validateTotals({ total, business: at(bizCol), nonbusiness: at(nonbizCol) });
    }
    throw new Error('F-2: no Total row found');
}

/** Cross-check the three extracted numbers before letting them near the UI. */
export function validateTotals({ total, business, nonbusiness }) {
    const ok = total > 0 && business > 0 && nonbusiness > 0 && business < total
        && Math.abs(business + nonbusiness - total) / total < 0.02;
    if (!ok) throw new Error(`F-2: Total row failed sanity check (${total}/${business}/${nonbusiness})`);
    return { total: Math.round(total), business: Math.round(business), nonbusiness: Math.round(nonbusiness) };
}

/** The most recent n calendar-quarter ends strictly before `now`, newest first. */
export function quarterEndsBefore(now, n = 5) {
    const out = [];
    for (let y = now.getUTCFullYear(); out.length < n; y--) {
        for (const [m, d] of [[12, 31], [9, 30], [6, 30], [3, 31]]) {
            if (out.length >= n) break;
            if (new Date(Date.UTC(y, m - 1, d)) < now) out.push({ y, m, d });
        }
    }
    return out;
}

const pad = (x) => String(x).padStart(2, '0');
export const quarterDateStr = (q) => `${q.y}-${pad(q.m)}-${pad(q.d)}`;
export const f2XlsxUrl = (q) => `https://www.uscourts.gov/sites/default/files/document/bf_f2_${pad(q.m)}${pad(q.d)}.${q.y}.xlsx`;
export const f2PageUrl = (q) => `https://www.uscourts.gov/data-news/data-tables/${q.y}/${pad(q.m)}/${pad(q.d)}/bankruptcy-filings/f-2`;

/** Pick the F-2 spreadsheet link out of a quarter's landing page HTML. */
export function findF2Link(html) {
    const links = [...html.matchAll(/href="([^"]*\.xlsx?)"/gi)].map((m) => m[1])
        .filter((l) => !/guide/i.test(l));
    const f2 = links.find((l) => /f[-_]?2/i.test(l)) ?? links[0] ?? null;
    if (!f2) return null;
    return f2.startsWith('http') ? f2 : `https://www.uscourts.gov${f2}`;
}

/**
 * Resolve the bankruptcies series: newest live quarter from uscourts.gov merged
 * over the baked history. Never throws. `fetchBuffer(url) -> Buffer` and
 * `fetchText(url) -> string` are injected by the route (network-free in tests).
 */
export async function resolveBankruptcies({ now = new Date(), faults, fetchBuffer, fetchText, baked, deadlineMs = 15000 }) {
    const tried = [];
    const deadline = Date.now() + deadlineMs;
    let live = null;

    if (faults?.has('bk_uscourts')) {
        tried.push('uscourts:off');
    } else {
        for (const q of quarterEndsBefore(now, 5)) {
            // Time-box the live tier: when uscourts is down/slow, stop probing
            // older quarters and serve baked instead of stalling the whole route.
            if (Date.now() > deadline) { tried.push('uscourts:deadline'); break; }
            const dateStr = quarterDateStr(q);
            try {
                let buf;
                try {
                    buf = await fetchBuffer(f2XlsxUrl(q));
                } catch {
                    // Predictable URL missed (renamed file?) — discover the link
                    // from the quarter's landing page instead.
                    const link = findF2Link(await fetchText(f2PageUrl(q)));
                    if (!link) throw new Error('no spreadsheet link on F-2 page');
                    buf = await fetchBuffer(link);
                }
                live = { date: dateStr, ...parseF2Totals(buf) };
                tried.push(`uscourts:${dateStr}:ok`);
                break;
            } catch {
                tried.push(`uscourts:${dateStr}:miss`);
            }
        }
    }

    let history = faults?.has('bk_baked') ? [] : (baked ?? []).slice();
    tried.push(faults?.has('bk_baked') ? 'baked:off' : `baked:${history.length}q`);
    if (live) {
        history = history.filter((h) => h.date !== live.date);
        history.push(live);
    }
    history.sort((a, b) => (a.date < b.date ? -1 : 1));

    const last = history[history.length - 1] ?? null;
    // Year-over-year: each point is a 12-month-ending sum, so compare to the
    // same quarter-end one year earlier (exact date match — gaps stay honest).
    const yearAgoDate = last ? `${parseInt(last.date.slice(0, 4), 10) - 1}${last.date.slice(4)}` : null;
    const yearAgo = yearAgoDate ? history.find((h) => h.date === yearAgoDate) : null;
    const changePct = last && yearAgo?.business ? ((last.business - yearAgo.business) / yearAgo.business) * 100 : null;

    const fresh = withFreshness(last?.business, last?.date ?? null, BANKRUPTCY_FRESHNESS_DAYS, now);
    return {
        current: fresh.value,               // business filings, 12-mo ending (the classic recession line)
        total: last?.total ?? null,         // business + nonbusiness
        asOf: fresh.asOf,
        stale: fresh.stale,
        staleDays: fresh.staleDays,
        unavailable: fresh.unavailable,
        changePct,                          // YoY, null when the prior-year quarter is absent
        status: fresh.unavailable ? 'unknown' : changePct == null ? 'neutral' : changePct > 0 ? 'rising' : 'falling',
        source: live ? 'uscourts' : history.length ? 'baked' : null,
        history: history.map((h) => ({ date: h.date, value: h.business, total: h.total })),
        tried,
    };
}
