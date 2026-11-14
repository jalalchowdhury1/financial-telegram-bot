import { EXTERNAL_URLS } from '../../../lib/constants';
import { serve } from '../../../lib/store';
import { faultsFrom, trip } from '../../../lib/faults';
import { validateSplit } from '../../../lib/split';

export const dynamic = 'force-dynamic';

/**
 * /api/split — the Main / Shartino split (percentages only), computed each morning on
 * the Mac mini and published to a secret gist; this route relays it, as /api/rubber-band.
 */
export async function GET(request) {
    request.headers.get('user-agent');
    const faults = faultsFrom(request);
    return serve('split', async () => {
        trip('gist', faults);
        const res = await fetch(`${EXTERNAL_URLS.SPLIT_GIST}?t=${Date.now()}`, { cache: 'no-store', next: { revalidate: 0 } });
        if (!res.ok) throw new Error(`gist returned ${res.status}`);
        const snap = await res.json();
        if (!validateSplit(snap)) throw new Error('snapshot failed validation');
        return { ...snap, _meta: { source: 'gist', hasErrors: false, messages: [] } };
    }, {
        isGood: (x) => validateSplit(x),
        fallback: { asOf: null, main: null, target: null, band: null },
        faults,
    });
}
