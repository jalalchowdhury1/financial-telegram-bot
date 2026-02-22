// /api/fear-greed - Fetch CNN Fear & Greed Index
export async function GET() {
    try {
        const res = await fetch('https://production.dataviz.cnn.io/index/fearandgreed/graphdata', {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            },
            next: { revalidate: 0 }
        });

        if (!res.ok) throw new Error(`CNN API returned ${res.status}`);

        const data = await res.json();
        const fg = data.fear_and_greed;

        return Response.json({
            score: fg.score,
            rating: fg.rating.toUpperCase(),
            previousClose: fg.previous_close,
            previousWeek: fg.previous_1_week,
            previousMonth: fg.previous_1_month,
            previousYear: fg.previous_1_year
        });
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
