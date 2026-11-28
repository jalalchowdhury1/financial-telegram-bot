import { EXTERNAL_URLS } from '../../../lib/constants';
import { serve } from '../../../lib/store';
import { faultsFrom, trip } from '../../../lib/faults';
import { publicOwn, validateOwn } from '../../../lib/own';
import { ownStale } from '../../../lib/servedFreshness';

export const dynamic = 'force-dynamic';

/**
 * /api/own — the morning own-watch (alarms on what he owns + the race), published by the Mac mini to the split's
 * secret gist; this route relays it, as /api/centred, rebuilt from an allow-list (lib/own.js).
 */
export async function GET(request) {
    request.headers.get('user-agent');
    const faults = faultsFrom(request);
    return serve('own', async () => {
        trip('gist', faults);
        const res = await fetch(`${EXTERNAL_URLS.OWN_GIST}?t=${Date.now()}`, { cache: 'no-store', next: { revalidate: 0 }, signal: AbortSignal.timeout(8000) });
        if (!res.ok) throw new Error(`gist returned ${res.status}`);
        const snap = publicOwn(await res.json());
        if (!snap) throw new Error('snapshot failed validation');
        const stale = ownStale(snap);
        return { ...snap, _meta: { source: 'gist', hasErrors: stale, stale, messages: stale ? [`own is as of ${snap.made}: the morning run may have missed`] : [] } };
    }, {
        isGood: (x) => validateOwn(x),
        fallback: { v: 1, made: null, asof: null, alarms: [], race: null },
        faults,
    });
}
