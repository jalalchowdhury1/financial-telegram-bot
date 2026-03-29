import { EXTERNAL_URLS, GOOGLE_SHEETS } from '../../../lib/constants';
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
        let dataSource = 'Google Sheet';
        let indicators = null;

        // Layer 1: Google Sheet (pre-calculated indicators - most reliable from Vercel)
        try {
            dataSource = 'Google Sheet';
            console.warn('[SPY] Layer 1: Fetching from Google Sheet...');
            const text = await fetchText(GOOGLE_SHEETS.SPY_INDICATORS);
            if (!text || text.trim().length < 50) throw new Error('Google Sheet returned empty');

            // Parse CSV: "200d MA SPY,661.63265" format
            const lines = text.trim().split('\n');
            const parsed = {};
            for (const line of lines) {
                const [key, value] = line.split(',');
                if (key && value) {
                    parsed[key.trim()] = parseFloat(value.trim());
                }
            }

            if (parsed['200d MA SPY'] && parsed['9d RSI SPY'] && parsed['SPY 52 week high'] && parsed['Current SPY']) {
                indicators = {
                    ma200: parsed['200d MA SPY'],
                    rsi: parsed['9d RSI SPY'],
                    week52High: parsed['SPY 52 week high'],
                    current: parsed['Current SPY']
                };
                console.warn(`[SPY] Google Sheet indicators: MA200=${indicators.ma200}, RSI=${indicators.rsi}, 52wHigh=${indicators.week52High}, Current=${indicators.current}`);
            } else {
                throw new Error('Missing required indicators in Google Sheet');
            }
        } catch (e) {
            console.warn(`[SPY] Layer 1 (Google Sheet) failed: ${e.message}`);
            rows = [];
        }

        // Layer 2: Stooq CSV (raw prices - fallback for chart)
        if (rows.length === 0 && !indicators) {
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
                console.warn(`[SPY] Layer 2 (Stooq) failed: ${e.message}`);
                rows = [];
            }
        }

        // Layer 3: FRED SP500 (fallback - reliable government source)
        if (rows.length === 0 && !indicators) {
            try {
                dataSource = 'FRED S&P 500 Index';
                console.warn('[SPY] Layer 3: Falling back to FRED SP500...');
                const fredKey = process.env.FRED_API_KEY;
                if (!fredKey) throw new Error('FRED_API_KEY not configured');
                const fredUrl = `https://api.stlouisfed.org/fred/series/observations?series_id=SP500&api_key=${fredKey}&file_type=json&observation_start=2010-01-01&limit=5000&sort_order=asc`;
                const fredData = await fetchJson(fredUrl);
                rows = fredData.observations
                    .filter(o => o.value !== '.')
                    .map(o => ({ date: o.date, close: parseFloat(o.value) }));
                if (rows.length < 10) throw new Error('insufficient rows');
            } catch (e) {
                console.warn(`[SPY] Layer 3 (FRED SP500) failed: ${e.message}`);
                rows = [];
            }
        }

        // Layer 4: Yahoo Finance v8 (may be blocked from Vercel)
        if (rows.length === 0 && !indicators) {
            try {
                dataSource = 'Yahoo Finance (query1)';
                rows = parseYahooChart(await fetchJson(EXTERNAL_URLS.YAHOO_SPY));
                if (rows.length < 10) throw new Error('insufficient rows');
            } catch (e) {
                console.warn(`[SPY] Layer 4 (Yahoo query1) failed: ${e.message}`);
                rows = [];
            }
        }

        // Layer 5: Yahoo Finance v8 query2 (backup)
        if (rows.length === 0 && !indicators) {
            try {
                dataSource = 'Yahoo Finance (query2)';
                rows = parseYahooChart(await fetchJson(EXTERNAL_URLS.YAHOO_SPY.replace('query1.finance', 'query2.finance')));
                if (rows.length < 10) throw new Error('insufficient rows');
            } catch (e) {
                console.warn(`[SPY] Layer 5 (Yahoo query2) failed: ${e.message}`);
                rows = [];
            }
        }

        // Build response - use indicators from Google Sheet if available
        let current, ma200, ma200Pct, week52High, high52wPct, rsi, return3y, chartHistory;

        if (indicators) {
            // Use pre-calculated indicators from Google Sheet
            current = indicators.current;
            ma200 = indicators.ma200;
            week52High = indicators.week52High;
            rsi = indicators.rsi;

            // Calculate percentages from current values
            ma200Pct = calculatePctChange(current, ma200);
            high52wPct = calculatePctChange(current, week52High);
            return3y = null; // Not available from Google Sheet

            // For chart, we still need price history - will be empty if only Google Sheet worked
            chartHistory = [];
        } else if (rows.length >= 10) {
            // Calculate from raw price data
            current = rows[rows.length - 1].close;
            const prevClose = rows[rows.length - 2].close;
            const dailyChangePct = calculatePctChange(current, prevClose);
            ma200 = calculateMA(rows, 200);
            ma200Pct = calculatePctChange(current, ma200);
            const last252 = rows.slice(-252);
            week52High = Math.max(...last252.map(r => r.close));
            high52wPct = calculatePctChange(current, week52High);
            rsi = calculateRSI(rows, 9);
            const days3y = Math.min(756, rows.length);
            return3y = calculatePctChange(current, rows[rows.length - days3y].close);

            // Build chart history
            chartHistory = [];
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
        } else {
            throw new Error('Insufficient SPY data — all sources failed');
        }

        return Response.json({
            current,
            dailyChange: { value: current - (rows.length >= 2 ? rows[rows.length - 2].close : current), pct: 0 },
            ma200: { value: ma200, pct: ma200Pct },
            week52High: { value: week52High, pct: high52wPct },
            rsi,
            return3y,
            chartHistory,
            _meta: { source: dataSource, hasErrors: false, messages: [`Loaded from ${dataSource}${indicators ? ' (indicators only)' : ''}`] }
        });
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
