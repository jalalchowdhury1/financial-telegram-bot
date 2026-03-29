import { EXTERNAL_URLS } from '../../../lib/constants';
import { fetchText, fetchJson } from '../../../lib/fetcher';
import { calculateRSI, calculateMA, calculatePctChange } from '../../../lib/finance';

export const dynamic = 'force-dynamic';

function parseYahooChart(data) {
    const result = data?.chart?.result?.[0];
    if (!result) return [];
    const timestamps = result.timestamp;
    const closes = result.indicators?.quote?.[0]?.close;
    if (!timestamps || !closes) return [];
    const rows = [];
    for (let i = 0; i < timestamps.length; i++) {
        if (closes[i] != null) {
            rows.push({ date: new Date(timestamps[i] * 1000).toISOString().split('T')[0], close: closes[i] });
        }
    }
    return rows;
}

export async function GET() {
    try {
        let rows = [];
        let dataSource = 'Stooq';

        // Layer 1: Stooq CSV (primary - may be intermittent from Vercel)
        try {
            dataSource = 'Stooq';
            const text = await fetchText(EXTERNAL_URLS.STOOQ_SPY);
            if (!text || text.trim().length < 100) throw new Error('Stooq returned empty response');
            const lines = text.trim().split('\n');
            const parsed = lines.slice(1).map(line => {
                const [date, , , , close] = line.split(',');
                return { date, close: parseFloat(close) };
            }).filter(r => r.date && !isNaN(r.close));
            if (parsed.length < 10) throw new Error('Stooq returned insufficient data');
            rows = parsed;
        } catch (e) {
            console.warn(`[SPY] Layer 1 (Stooq) failed: ${e.message}`);
            rows = [];
        }

        // Layer 2: FRED SP500 (fallback - reliable government source)
        if (rows.length < 10) {
            try {
                dataSource = 'FRED S&P 500 Index';
                console.warn('[SPY] Layer 2: Falling back to FRED SP500...');
                const fredKey = process.env.FRED_API_KEY;
                if (!fredKey) throw new Error('FRED_API_KEY not configured');
                const fredUrl = `https://api.stlouisfed.org/fred/series/observations?series_id=SP500&api_key=${fredKey}&file_type=json&observation_start=2010-01-01&limit=5000&sort_order=asc`;
                const fredData = await fetchJson(fredUrl);
                rows = fredData.observations
                    .filter(o => o.value !== '.')
                    .map(o => ({ date: o.date, close: parseFloat(o.value) }));
                if (rows.length < 10) throw new Error('insufficient rows');
            } catch (e) {
                console.warn(`[SPY] Layer 2 (FRED SP500) failed: ${e.message}`);
                rows = [];
            }
        }

        // Layer 3: Yahoo Finance v8 (may be blocked from Vercel)
        if (rows.length < 10) {
            try {
                dataSource = 'Yahoo Finance (query1)';
                rows = parseYahooChart(await fetchJson(EXTERNAL_URLS.YAHOO_SPY));
                if (rows.length < 10) throw new Error('insufficient rows');
            } catch (e) {
                console.warn(`[SPY] Layer 3 (Yahoo query1) failed: ${e.message}`);
                rows = [];
            }
        }

        // Layer 4: Yahoo Finance v8 query2 (backup)
        if (rows.length < 10) {
            try {
                dataSource = 'Yahoo Finance (query2)';
                rows = parseYahooChart(await fetchJson(EXTERNAL_URLS.YAHOO_SPY.replace('query1.finance', 'query2.finance')));
                if (rows.length < 10) throw new Error('insufficient rows');
            } catch (e) {
                console.warn(`[SPY] Layer 4 (Yahoo query2) failed: ${e.message}`);
                rows = [];
            }
        }

        if (rows.length < 10) throw new Error('Insufficient SPY data — all 4 sources failed');

        const current = rows[rows.length - 1].close;
        const prevClose = rows[rows.length - 2].close;
        const dailyChange = current - prevClose;
        const dailyChangePct = calculatePctChange(current, prevClose);
        const ma200 = calculateMA(rows, 200);
        const ma200Pct = calculatePctChange(current, ma200);
        const last252 = rows.slice(-252);
        const week52High = Math.max(...last252.map(r => r.close));
        const high52wPct = calculatePctChange(current, week52High);
        const rsi = calculateRSI(rows, 9);
        const days3y = Math.min(756, rows.length);
        const return3y = calculatePctChange(current, rows[rows.length - days3y].close);

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
            _meta: { source: dataSource, hasErrors: false, messages: [`Loaded ${rows.length} days from ${dataSource}`] }
        });
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
