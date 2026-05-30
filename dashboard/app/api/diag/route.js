// TEMPORARY diagnostic: probes candidate backup sources from Vercel's network
// to see which are reachable. Remove before final merge.
export const fetchCache = 'default-cache';

async function probe(name, url, kind, headers) {
    try {
        const res = await fetch(url, { cache: 'no-store', headers: headers || { 'User-Agent': 'Mozilla/5.0' } });
        const body = kind === 'text' ? await res.text() : await res.json().catch(() => null);
        let sample;
        if (kind === 'text') sample = String(body).slice(0, 120);
        else sample = JSON.stringify(body).slice(0, 220);
        return { name, ok: res.ok, status: res.status, sample };
    } catch (e) {
        return { name, ok: false, status: 'ERR', sample: e.message.slice(0, 160) };
    }
}

export async function GET(request) {
    request.headers.get('user-agent');
    const fredKey = process.env.FRED_API_KEY;
    const results = await Promise.all([
        probe('yahoo-q1-SPY', 'https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d', 'json'),
        probe('yahoo-q2-SPY', 'https://query2.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d', 'json'),
        probe('stooq-spy', 'https://stooq.com/q/d/l/?s=spy.us&i=d', 'text'),
        probe('stooq-usdcad', 'https://stooq.com/q/d/l/?s=usdcad&i=d', 'text'),
        probe('er-api-USD', 'https://open.er-api.com/v6/latest/USD', 'json'),
        probe('frankfurter-USD', 'https://api.frankfurter.app/latest?from=USD&to=CAD,INR', 'json'),
        probe('coingecko-btc', 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true', 'json'),
        probe('coinbase-btc', 'https://api.coinbase.com/v2/prices/BTC-USD/spot', 'json'),
        probe('dbnomics-filter-T10Y2Y', 'https://api.db.nomics.world/v22/series?provider_code=FRED&series_code=T10Y2Y&observations=1', 'json'),
        probe('dbnomics-path-DGS10', 'https://api.db.nomics.world/v22/series/FRED/DGS10?observations=1', 'json'),
        probe('fred-crude-DCOILWTICO', `https://api.stlouisfed.org/fred/series/observations?series_id=DCOILWTICO&api_key=${fredKey}&file_type=json&sort_order=desc&limit=2`, 'json'),
        probe('fred-gold', `https://api.stlouisfed.org/fred/series/observations?series_id=GOLDPMGBD228NLBM&api_key=${fredKey}&file_type=json&sort_order=desc&limit=2`, 'json'),
        probe('fred-dxy-DTWEXBGS', `https://api.stlouisfed.org/fred/series/observations?series_id=DTWEXBGS&api_key=${fredKey}&file_type=json&sort_order=desc&limit=2`, 'json'),
        probe('fred-DGS10', `https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key=${fredKey}&file_type=json&sort_order=desc&limit=2`, 'json'),
        probe('twelvedata-demo', 'https://api.twelvedata.com/price?symbol=SPY&apikey=demo', 'json'),
        probe('coincap-btc', 'https://api.coincap.io/v2/assets/bitcoin', 'json'),
    ]);
    return Response.json({ results }, { headers: { 'cache-control': 'no-store' } });
}
