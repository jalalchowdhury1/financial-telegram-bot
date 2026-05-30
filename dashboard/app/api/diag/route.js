// TEMPORARY reliability diagnostic: hits each candidate backup source MANY times
// from Vercel's network and reports the success rate (not just a one-off).
// Remove before final merge.
export const fetchCache = 'default-cache';
export const maxDuration = 60;

async function reliability(name, url, { attempts = 8, kind = 'json', headers } = {}) {
    let ok = 0;
    const statuses = {};
    for (let i = 0; i < attempts; i++) {
        try {
            const res = await fetch(url, { cache: 'no-store', headers: headers || { 'User-Agent': 'Mozilla/5.0' } });
            let good = res.ok;
            if (good && kind === 'json') { const j = await res.json().catch(() => null); good = !!j && !j.error_code; }
            if (good && kind === 'csv') { const t = await res.text(); good = /^Date,/i.test(t.trim()); }
            statuses[res.status] = (statuses[res.status] || 0) + 1;
            if (good) ok++;
        } catch (e) {
            const key = (e.message || 'ERR').slice(0, 24);
            statuses[key] = (statuses[key] || 0) + 1;
        }
        await new Promise((r) => setTimeout(r, 120));
    }
    return { name, rate: `${ok}/${attempts}`, ok, attempts, statuses };
}

export async function GET(request) {
    request.headers.get('user-agent');
    const k = process.env.FRED_API_KEY;
    const results = await Promise.all([
        reliability('yahoo-q1 SPY', 'https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d'),
        reliability('yahoo-q2 SPY', 'https://query2.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d'),
        reliability('yahoo GC=F', 'https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=5d&interval=1d'),
        reliability('yahoo CL=F', 'https://query1.finance.yahoo.com/v8/finance/chart/CL=F?range=5d&interval=1d'),
        reliability('yahoo DX-Y.NYB', 'https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?range=5d&interval=1d'),
        reliability('er-api USD', 'https://open.er-api.com/v6/latest/USD'),
        reliability('frankfurter USD', 'https://api.frankfurter.app/latest?from=USD&to=CAD,INR'),
        reliability('coingecko btc', 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'),
        reliability('coinbase btc', 'https://api.coinbase.com/v2/prices/BTC-USD/spot'),
        reliability('FRED DGS10', `https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key=${k}&file_type=json&sort_order=desc&limit=2`),
        reliability('gamma polymarket', 'https://gamma-api.polymarket.com/markets?active=true&closed=false&order=volume24hr&ascending=false&limit=5'),
        reliability('dbnomics FRED/DGS10/DGS10', 'https://api.db.nomics.world/v22/series/FRED/DGS10/DGS10?observations=1'),
    ]);
    results.sort((a, b) => b.ok - a.ok);
    return Response.json({ note: 'success rate per source from Vercel egress', results }, { headers: { 'cache-control': 'no-store' } });
}
