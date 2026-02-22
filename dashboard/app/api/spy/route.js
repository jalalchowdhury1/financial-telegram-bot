// /api/spy - Fetch SPY statistics from Stooq
export async function GET() {
    try {
        const url = 'https://stooq.com/q/d/l/?s=spy.us&i=d';
        const res = await fetch(url, { next: { revalidate: 0 } });
        const text = await res.text();

        const lines = text.trim().split('\n');
        // Header: Date,Open,High,Low,Close,Volume
        const rows = lines.slice(1).map(line => {
            const [date, open, high, low, close, volume] = line.split(',');
            return { date, open: +open, high: +high, low: +low, close: +close, volume: +volume };
        }).filter(r => !isNaN(r.close));

        if (rows.length < 10) throw new Error('Insufficient SPY data');

        const current = rows[rows.length - 1].close;

        // 200-day MA
        const last200 = rows.slice(-200);
        const ma200 = last200.reduce((s, r) => s + r.close, 0) / last200.length;
        const ma200Pct = ((current - ma200) / ma200) * 100;

        // 52-week high
        const last252 = rows.slice(-252);
        const week52High = Math.max(...last252.map(r => r.high));
        const high52wPct = ((current - week52High) / week52High) * 100;

        // 9-day RSI (Wilder's smoothing)
        const closes = rows.map(r => r.close);
        const period = 9;
        const deltas = [];
        for (let i = 1; i < closes.length; i++) {
            deltas.push(closes[i] - closes[i - 1]);
        }

        let avgGain = 0, avgLoss = 0;
        for (let i = 0; i < period; i++) {
            if (deltas[i] > 0) avgGain += deltas[i];
            else avgLoss += Math.abs(deltas[i]);
        }
        avgGain /= period;
        avgLoss /= period;

        for (let i = period; i < deltas.length; i++) {
            const gain = deltas[i] > 0 ? deltas[i] : 0;
            const loss = deltas[i] < 0 ? Math.abs(deltas[i]) : 0;
            avgGain = (avgGain * (period - 1) + gain) / period;
            avgLoss = (avgLoss * (period - 1) + loss) / period;
        }

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
            ma200: { value: ma200, pct: ma200Pct },
            week52High: { value: week52High, pct: high52wPct },
            rsi,
            return3y,
            chartHistory
        });
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
