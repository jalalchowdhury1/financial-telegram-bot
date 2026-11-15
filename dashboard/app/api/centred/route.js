import { EXTERNAL_URLS } from '../../../lib/constants';
import { serve } from '../../../lib/store';
import { faultsFrom, trip } from '../../../lib/faults';
import { validateCentred, publicCentred } from '../../../lib/centred';
import { centredStale } from '../../../lib/servedFreshness';

export const dynamic = 'force-dynamic';

/**
 * /api/centred — the nightly Centred check (counts, state, dates, whole-algo deltas only), published by the
 * Mac mini to a secret gist; this route relays it, as /api/split.
 */
export async function GET(request) {
    request.headers.get('user-agent');
    const faults = faultsFrom(request);
    return serve('centred', async () => {
        trip('gist', faults);
        const res = await fetch(`${EXTERNAL_URLS.CENTRED_GIST}?t=${Date.now()}`, { cache: 'no-store', next: { revalidate: 0 }, signal: AbortSignal.timeout(8000) });
        if (!res.ok) throw new Error(`gist returned ${res.status}`);
        const snap = await res.json();
        if (!validateCentred(snap)) throw new Error('snapshot failed validation');
        const stale = centredStale(snap);
        return { ...publicCentred(snap), _meta: { source: 'gist', hasErrors: stale, stale, messages: stale ? [`centred is as of ${snap.generated_at}: the nightly run may have missed`] : [] } };
    }, {
        isGood: validateCentred,
        fallback: { v: 1, generated_at: null, state: 'unknown', note: 'no feed', checked: 0, off: 0, history: [] },
        faults,
    });
}
