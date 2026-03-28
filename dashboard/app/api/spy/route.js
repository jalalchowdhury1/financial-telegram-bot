import { EXTERNAL_URLS } from '../../../lib/constants';
import { fetchText, fetchJson } from '../../../lib/fetcher';
import { calculateRSI, calculateMA, calculatePctChange } from '../../../lib/finance';

export const dynamic = 'force-dynamic';
export async function GET() {
    try {
        let rows = [];
        let dataSource = 'Stooq';

        // --- Source 1: Stooq ---
        try {
            const text = await fetchText(EXTERNAL_URLS.STOOQ_SPY);
            const lines = text.trim().split('\n');
            rows = lines.slice(1).map(line => {
                const [date, open, high, low, close, volume] = line.split(',');
                return { date, open: +open, high: +high, low: +low, close: +close, volume: +volume };
            }).filter(r => !isNaN(r.close));
            if (rows.length < 10) throw new Error('Stooq returned insufficient data');
        } catch (stooqError) {
            console.warn(`[SPY] Stooq failed: ${stooqError.message}`);
            rows = [];

            // --- Source 2: Yahoo Finance (query1 then query2) ---
            const yahooUrls = [
                EXTERNAL_URLS.YAHOO_SPY,
                EXTERNAL_URLS.YAHOO_SPY.replace('query1.finance', 'query2.finance')
            ];
            for (const url of yahooUrls) {
                try {
                    dataSource = `Yahoo Finance (${url.includes('query2') ? 'query2' : 'query1'})`;
                    const yData = await fetchJson(url);
                    const result = yData.chart.result[0];
                    const timestamps = result.timestamp;
                    const quotes = result.indicators.quote[0];
                    for (let i = 0; i < timestamps.length; i++) {
                        if (quotes.close[i] !== null) {
                            const d = new Date(timestamps[i] * 1000);
                            rows.push({
                                date: d.toISOString().split('T')[0],
                                close: quotes.close[i]
                            });
                        }
                    }
                    if (rows.length >= 10) break;
                    rows = [];
                } catch (yErr) {
                    console.warn(`[SPY] Yahoo ${url.includes('query2') ? 'query2' : 'query1'} failed: ${yErr.message}`);
                    rows = [];
                }
            }

            // --- Source 3: FRED SP500 (reliable, uses existing API key) ---
            if (rows.length < 10) {
                try {
                    dataSource = 'FRED S&P 500 Index';
                    console.warn('[SPY] Falling back to FRED SP500...');
                    const fredKey = process.env.FRED_API_KEY;
                    const fredUrl = `https://api.stlouisfed.org/fred/series/observations?series_id=SP500&api_key=${fredKey}&file_type=json&observation_start=2010-01-01&limit=5000&sort_order=asc`;
                    const fredData = await fetchJson(fredUrl);
                    rows = fredData.observations
                        .filter(o => o.value !== '.')
                        .map(o => ({ date: o.date, close: parseFloat(o.value) }));
                } catch (fredErr) {
                    console.warn(`[SPY] FRED fallback failed: ${fredErr.message}`);
                    rows = [];
                }
            }
        }

        if (rows.length < 10) throw new Error('Insufficient SPY data — all sources failed');

        const current = rows[rows.length - 1].close;
        const prevClose = rows[rows.length - 2].close;
        const dailyChange = current - prevClose;
        const dailyChangePct = calculatePctChange(current, prevClose);

        // 200-day MA
        const ma200 = calculateMA(rows, 200);
        const ma200Pct = calculatePctChange(current, ma200);

        // 52-week high
        const last252 = rows.slice(-252);
        const week52High = Math.max(...last252.map(r => r.close));
        const high52wPct = calculatePctChange(current, week52High);

        // 9-day RSI
        const rsi = calculateRSI(rows, 9);

        // 3-year return
        const days3y = Math.min(756, rows.length);
        const price3yAgo = rows[rows.length - days3y].close;
        const return3y = calculatePctChange(current, price3yAgo);

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
                messages: [`Loaded ${rows.length} days from ${dataSource}`]
            }
        });
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
