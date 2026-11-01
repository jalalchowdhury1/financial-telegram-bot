/**
 * /api/fwd-pe — the S&P 500 forward P/E reports FactSet posted since public/history/fwdPE.json was baked.
 *
 *   200 { rows: [{date, pe, a5, a10}], latest, seen, checkedAt, _meta: {bakedTo, messages} }
 *       rows = reports newer than the bake (the SPY chart joins them on); latest = the newest of bake + rows;
 *       seen = the newest report found, even one that would not parse (/api/freshness grades both).
 *
 * Never 5xx: with FactSet or KV down it answers with what it has (the bake at worst). The work and the
 * KV list live in lib/fwdPe.js resolveFwdPe. Fault gates: fwdpe_http, fwdpe_kv.
 */
import { faultsFrom } from '../../../lib/faults';
import { cacheHeaders } from '../../../lib/cdn';
import { defaultKv } from '../../../lib/kv';
import { etParts } from '../../../lib/marketClock';
import { resolveFwdPe, decodeFwd } from '../../../lib/fwdPe';
import baked from '../../../public/history/fwdPE.json';

export const dynamic = 'force-dynamic';
export const maxDuration = 60;

const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';
const bakedRows = decodeFwd(baked) || [];

/** The PDF's bytes, null when that week has none (404, or a page that is not a PDF). */
async function fetchPdf(url) {
    const res = await fetch(url, { cache: 'no-store', headers: { 'user-agent': UA }, signal: AbortSignal.timeout(10000) });
    if (res.status === 404 || (res.ok && !(res.headers.get('content-type') || '').includes('pdf'))) return null;
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
}

async function pdfText(bytes) {
    const { getDocumentProxy, extractText } = await import('unpdf');
    const { text } = await extractText(await getDocumentProxy(bytes), { mergePages: true });
    return text;
}

export async function GET(request) {
    request.headers.get('user-agent');
    const faults = faultsFrom(request);
    const now = Date.now();
    const { rows, seen, checkedAt, messages } = await resolveFwdPe({
        fetchPdf, pdfText, kv: defaultKv, bakedTo: baked.to, today: etParts(now).date, now, faults,
    });
    const body = {
        rows,
        latest: rows[rows.length - 1] || bakedRows[bakedRows.length - 1] || null,
        seen,
        checkedAt: checkedAt ? new Date(checkedAt).toISOString() : null,
        _meta: { bakedTo: baked.to, messages },
    };
    return Response.json(body, { headers: cacheHeaders('fwd-pe', { payload: body, testMode: faults.size > 0 }) });
}
