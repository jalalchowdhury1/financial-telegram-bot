import { EXTERNAL_URLS, DEFAULT_HEADERS } from '../../../lib/constants';
import { proxyFetch } from '../../../lib/fetcher';

export const dynamic = 'force-dynamic';

function getRatingFromScore(score) {
    if (score < 25) return 'EXTREME FEAR';
    if (score < 45) return 'FEAR';
    if (score <= 55) return 'NEUTRAL';
    if (score <= 75) return 'GREED';
    return 'EXTREME GREED';
}

export async function GET() {
    let score = 'N/A';
    let rating = 'N/A';
    let previousClose = 'N/A';
    let previousWeek = 'N/A';
    let previousMonth = 'N/A';
    let previousYear = 'N/A';
    let source = 'Unknown';
    let hasErrors = false;
    let messages = [];

    // Tier 1: CNN Business API
    try {
        const res = await proxyFetch(EXTERNAL_URLS.CNN_FEAR_GREED, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Referer': 'https://edition.cnn.com/',
                'Accept': 'application/json'
            },
            next: { revalidate: 0 }
        });

        if (!res.ok) throw new Error(`CNN API returned ${res.status}`);

        const data = await res.json();
        const fg = data.fear_and_greed;

        score = fg?.score ?? 'N/A';
        rating = fg?.rating?.toUpperCase() || getRatingFromScore(score);
        previousClose = fg?.previous_close ?? 'N/A';
        previousWeek = fg?.previous_1_week ?? 'N/A';
        previousMonth = fg?.previous_1_month ?? 'N/A';
        previousYear = fg?.previous_1_year ?? 'N/A';

        source = 'CNN';
        messages.push('CNN data parsed successfully.');

        // Ensure success if we reached here
        return Response.json({
            score, rating, previousClose, previousWeek, previousMonth, previousYear,
            _meta: { source, hasErrors, messages }
        });
    } catch (error) {
        messages.push(`Attempt 1 (CNN) failed: ${error.message}`);
    }

    // Tier 2: RapidAPI Fear & Greed Index
    try {
        const rapidApiKey = process.env.RAPIDAPI_KEY || '7a22060824mshabd4fb2494530d3p1cee20jsnb5257930e869';
        const res = await proxyFetch(EXTERNAL_URLS.RAPIDAPI_FEAR_GREED, {
            headers: {
                'X-RapidAPI-Key': rapidApiKey,
                'X-RapidAPI-Host': 'fear-and-greed-index.p.rapidapi.com'
            },
            next: { revalidate: 0 }
        });

        if (!res.ok) throw new Error(`RapidAPI returned ${res.status}`);

        const data = await res.json();
        const fg = data.fgi;

        if (!fg || fg.now === undefined) throw new Error('RapidAPI data malformed');

        score = fg.now?.value ?? 'N/A';
        rating = fg.now?.valueText?.toUpperCase() || getRatingFromScore(score);
        previousClose = fg.previousClose?.value ?? 'N/A';
        previousWeek = fg.oneWeekAgo?.value ?? 'N/A';
        previousMonth = fg.oneMonthAgo?.value ?? 'N/A';
        previousYear = fg.oneYearAgo?.value ?? 'N/A';

        source = 'RapidAPI';
        messages.push('RapidAPI data parsed successfully.');

        return Response.json({
            score, rating, previousClose, previousWeek, previousMonth, previousYear,
            _meta: { source, hasErrors, messages }
        });
    } catch (error) {
        messages.push(`Attempt 2 (RapidAPI) failed: ${error.message}`);
    }

    // Tier 3: VIX Proxy Fallback
    try {
        const res = await proxyFetch(EXTERNAL_URLS.YAHOO_VIX, {
            headers: DEFAULT_HEADERS,
            next: { revalidate: 0 }
        });

        if (!res.ok) throw new Error(`Yahoo VIX API returned ${res.status}`);

        const data = await res.json();
        const result = data.chart.result[0];
        const timestamps = result.timestamp;
        const quotes = result.indicators.quote[0].close;

        // VIX to Score proxy calculation:
        // VIX ~ 10-12 is Extreme Greed (~80-100 score). VIX > 35 is Extreme Fear (~0-20 score).
        // Score = 100 - ((VIX - 10) / 25) * 100
        const calculateScore = (vixStr) => {
            if (vixStr === null || vixStr === undefined) return 'N/A';
            const v = parseFloat(vixStr);
            let s = 100 - ((v - 10) / 25) * 100;
            return Math.max(0, Math.min(100, s));
        };

        const vixLen = quotes.length;
        if (vixLen > 0) {
            const currentVix = quotes[vixLen - 1];
            const closeVix = vixLen > 1 ? quotes[vixLen - 2] : null;
            const weekVix = vixLen > 5 ? quotes[vixLen - 6] : null;
            const monthVix = vixLen > 21 ? quotes[vixLen - 22] : null;

            score = calculateScore(currentVix);
            rating = getRatingFromScore(score);
            previousClose = calculateScore(closeVix);
            previousWeek = calculateScore(weekVix);
            previousMonth = calculateScore(monthVix);
            previousYear = 'N/A'; // Need 1y of data for this, keeping payload light

            source = 'Yahoo ^VIX Proxy';
            hasErrors = true; // Technically a fallback, so we flag it
            messages.push('Using simulated Fear & Greed score generated from current CBOE Volatility Index (^VIX).');

            return Response.json({
                score, rating, previousClose, previousWeek, previousMonth, previousYear,
                _meta: { source, hasErrors, messages }
            });
        }
        throw new Error('No valid VIX data returned');
    } catch (error) {
        messages.push(`Attempt 3 (VIX Proxy) failed: ${error.message}`);

        // Final fallback failure state
        return Response.json({
            score: 'N/A', rating: 'N/A', previousClose: 'N/A', previousWeek: 'N/A', previousMonth: 'N/A', previousYear: 'N/A',
            _meta: { source: 'Failed', hasErrors: true, messages }
        }, { status: 500 });
    }
}
