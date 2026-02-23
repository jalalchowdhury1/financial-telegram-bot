// /api/fear-greed - Fetch CNN Fear & Greed Index
export const dynamic = 'force-dynamic';
export async function GET() {
    try {
        const res = await fetch('https://production.dataviz.cnn.io/index/fearandgreed/graphdata', {
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

        const score = fg?.score ?? 'N/A';
        const rating = fg?.rating?.toUpperCase() || 'N/A';
        const previousClose = fg?.previous_close ?? 'N/A';
        const previousWeek = fg?.previous_1_week ?? 'N/A';
        const previousMonth = fg?.previous_1_month ?? 'N/A';
        const previousYear = fg?.previous_1_year ?? 'N/A';

        const missingFields = [];
        if (score === 'N/A') missingFields.push('score');
        if (previousClose === 'N/A') missingFields.push('previousClose');

        return Response.json({
            score,
            rating,
            previousClose,
            previousWeek,
            previousMonth,
            previousYear,
            _meta: {
                source: 'CNN',
                hasErrors: missingFields.length > 0,
                messages: missingFields.length > 0 ? [`Missing fields from CNN: ${missingFields.join(', ')}`] : ['CNN data parsed successfully']
            }
        });
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
