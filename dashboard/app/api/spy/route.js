// /api/spy - Fetch SPY statistics from Stooq

export const dynamic = 'force-dynamic';
export async function GET() {
    try {

        let rows = [];
        let dataSource = 'Stooq';
        try {
            const url = 'https://stooq.com/q/d/l/?s=spy.us&i=d';
            const res = await fetch(url, {
                headers: { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36' },
                next: { revalidate: 0 }
            });
            if (!res.ok) throw new Error('Stooq response not ok');
            const text = await res.text();
            const lines = text.trim().split('\n');
            rows = lines.slice(1).map(line => {
                const [date, open, high, low, close, volume] = line.split(',');
                return { date, open: +open, high: +high, low: +low, close: +close, volume: +volume };
            }).filter(r => !isNaN(r.close));
        } catch (stooqError) {
            dataSource = 'Yahoo Finance (Fallback)';
            console.warn('Stooq fetch failed, falling back to Yahoo Finance...');
            const yUrl = 'https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5y&interval=1d';
            const yRes = await fetch(yUrl, {
                headers: { 'User-Agent': 'Mozilla/5.0' },
                next: { revalidate: 0 }
            });
            if (!yRes.ok) throw new Error('Both Stooq and Yahoo Finance APIs failed');
            const yData = await yRes.json();
            const result = yData.chart.result[0];
            const timestamps = result.timestamp;
            const quotes = result.indicators.quote[0];

            for (let i = 0; i < timestamps.length; i++) {
                if (quotes.close[i] !== null) {
                    const d = new Date(timestamps[i] * 1000);
                    rows.push({
                        date: d.toISOString().split('T')[0],
                        open: quotes.open[i],
                        high: quotes.high[i],
                        low: quotes.low[i],
                        close: quotes.close[i],
                        volume: quotes.volume[i]
                    });
                }
            }
        }

        if (rows.length < 10) throw new Error('Insufficient SPY data');

        const current = rows[rows.length - 1].close;
        const prevClose = rows[rows.length - 2].close;
        const dailyChange = current - prevClose;
        const dailyChangePct = ((dailyChange) / prevClose) * 100;

        // 200-day MA
        const last200 = rows.slice(-200);
        const ma200 = last200.reduce((s, r) => s + r.close, 0) / last200.length;
        const ma200Pct = ((current - ma200) / ma200) * 100;

        // 52-week high (matching Python's closing maximum)
        const last252 = rows.slice(-252);
        const week52High = Math.max(...last252.map(r => r.close));
        const high52wPct = ((current - week52High) / week52High) * 100;

        // 9-day RSI (matching Python's rolling simple mean of periods)
        const period = 9;
        const recentRows = rows.slice(-(period + 1));
        const deltas = [];
        for (let i = 1; i < recentRows.length; i++) {
            deltas.push(recentRows[i].close - recentRows[i - 1].close);
        }

        let avgGain = 0, avgLoss = 0;
        for (let i = 0; i < deltas.length; i++) {
            if (deltas[i] > 0) avgGain += deltas[i];
            else avgLoss += Math.abs(deltas[i]);
        }
        avgGain /= period;
        avgLoss /= period;

        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        const rsi = 100 - (100 / (1 + rs));

        // 3-year return
        const days3y = Math.min(756, rows.length);
        const price3yAgo = rows[rows.length - days3y].close;
        const return3y = ((current - price3yAgo) / price3yAgo) * 100;

        // Historical chart data with 50d + 200d MA
        const chartHistory = [];
        for (let i = 199; i < rows.length; i++) {
            const slice200 = rows.slice(i - 199, i + 1);
            const ma200v = slice200.reduce((s, r) => s + r.close, 0) / 200;
            const slice50 = rows.slice(Math.max(0, i - 49), i + 1);
            const ma50v = slice50.reduce((s, r) => s + r.close, 0) / slice50.length;
            chartHistory.push({
                date: rows[i].date,
                price: rows[i].close,
                ma50: Math.round(ma50v * 100) / 100,
                ma200: Math.round(ma200v * 100) / 100
            });
        }

        return Response.json({
            current,
            dailyChange: { value: dailyChange, pct: dailyChangePct },
            ma200: { value: ma200, pct: ma200Pct },
            week52High: { value: week52High, pct: high52wPct },
            rsi,
            return3y,
            chartHistory,
            _meta: {
                source: dataSource,
                hasErrors: false,
                messages: [`Loaded ${rows.length} days of SPY history from ${dataSource}`]
            }
        });
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
