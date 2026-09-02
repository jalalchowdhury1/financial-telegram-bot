import { EXTERNAL_URLS } from '../../../lib/constants';
import { serve } from '../../../lib/store';
import { faultsFrom, trip } from '../../../lib/faults';
import { validateSnapshot, snapshotAge } from '../../../lib/rubberBand';

export const dynamic = 'force-dynamic';

/**
 * /api/rubber-band — the nightly Rubber Band Radar snapshot (five regime dials).
 *
 * The Mac mini computes it after each close and publishes to a secret gist; this
 * route just relays it. Never-throws via serve(): gist down → last-known-good
 * (flagged stale) → an empty payload with `_meta.hasErrors`. A snapshot older
 * than 4 days is served but stamped `_meta.stale` so the UI and the health check
 * both show it as a missed run rather than a current reading.
 */
export async function GET(request) {
    request.headers.get('user-agent');
    const faults = faultsFrom(request);
    const messages = [];
    return serve('rubber-band', async () => {
        trip('gist', faults);
        const res = await fetch(`${EXTERNAL_URLS.RUBBER_BAND_GIST}?t=${Date.now()}`, { cache: 'no-store', next: { revalidate: 0 } });
        if (!res.ok) throw new Error(`gist returned ${res.status}`);
        const snap = await res.json();
        if (!validateSnapshot(snap)) throw new Error('snapshot failed validation');
        const { ageDays, stale } = snapshotAge(snap.asOf);
        if (stale) messages.push(`snapshot is ${ageDays} days old (as of ${snap.asOf}) — the nightly run may have missed`);
        return { ...snap, _meta: { source: 'gist', hasErrors: stale, stale, ageDays, messages } };
    }, {
        isGood: (x) => validateSnapshot(x),
        fallback: { asOf: null, dials: null, verdict: null },
        faults,
    });
}
