#!/usr/bin/env node

/**
 * bake-fwd-pe.mjs — refresh public/history/fwdPE.json, the SPY chart's forward P/E lens (lib/fwdPe.js).
 *
 * Adds every FactSet Earnings Insight report newer than the file's last row (Wednesday–Friday PDFs, plain
 * then 'A'), parsed by the same parseFactSet as /api/fwd-pe. `--dir <folder>` reads `<YYYY-MM-DD>.pdf`
 * files instead: how the 2011–2016 reports (Wayback Machine copies) went in. REFUSES to write fewer rows
 * than the file already has.
 *
 * Usage (from dashboard/): node scripts/bake-fwd-pe.mjs [--dir <folder>]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { getDocumentProxy, extractText } from 'unpdf';
import { parseFactSet, reportUrl, reportCandidates, decodeFwd, encodeFwd, joinFwd } from '../lib/fwdPe.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, '..', 'public', 'history', 'fwdPE.json');
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

async function textOf(bytes) {
    const { text } = await extractText(await getDocumentProxy(new Uint8Array(bytes)), { mergePages: true });
    return text;
}

async function fetchPdf(url) {
    const res = await fetch(url, { headers: { 'user-agent': UA }, signal: AbortSignal.timeout(60000) });
    if (!res.ok || !(res.headers.get('content-type') || '').includes('pdf')) return null;
    return new Uint8Array(await res.arrayBuffer());
}

const old = fs.existsSync(OUT) ? decodeFwd(JSON.parse(fs.readFileSync(OUT, 'utf8'))) || [] : [];
const dirArg = process.argv.indexOf('--dir');
const found = [];
if (dirArg > 0) {
    const dir = process.argv[dirArg + 1];
    for (const f of fs.readdirSync(dir).filter((x) => /^\d{4}-\d{2}-\d{2}\.pdf$/.test(x)).sort()) {
        const got = parseFactSet(await textOf(fs.readFileSync(path.join(dir, f))));
        if (got) found.push({ date: f.slice(0, 10), ...got });
        else console.log(`no forward P/E in ${f}`);
    }
} else {
    const after = old.length ? old[old.length - 1].date : '2016-12-31';
    for (const day of reportCandidates(new Date().toISOString().slice(0, 10), after).reverse()) {
        for (const suffix of ['', 'A']) {
            const bytes = await fetchPdf(reportUrl(day, suffix)).catch(() => null);
            if (!bytes) continue;
            const got = parseFactSet(await textOf(bytes));
            if (got) found.push({ date: day, ...got });
            else console.log(`no forward P/E in the ${day} report`);
            break;
        }
    }
}
// A --dir row replaces the file's row for that date; fetched rows are all newer.
const byDate = new Map([...old, ...found].map((r) => [r.date, r]));
const rows = joinFwd([...byDate.values()].sort((a, b) => (a.date < b.date ? -1 : 1)), []);
if (rows.length < old.length) {
    console.log(`FAILED/REFUSED: ${rows.length} rows < the file's ${old.length}`);
    process.exit(1);
}
fs.writeFileSync(OUT, JSON.stringify(encodeFwd(rows, { source: 'FactSet Earnings Insight (weekly)' })));
console.log(`wrote ${path.relative(process.cwd(), OUT)}: ${rows.length} reports ${rows[0].date} → ${rows[rows.length - 1].date} (+${found.length})`);
