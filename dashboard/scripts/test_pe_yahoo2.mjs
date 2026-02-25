const SPY_URL = 'https://finance.yahoo.com/quote/SPY/key-statistics';

async function checkPE() {
    console.log("Fetching SPY from Yahoo Finance Key Stats HTML...");
    try {
        const yRes = await fetch(SPY_URL, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
        });
        const yHtml = await yRes.text();
        
        // Let's grab the whole table to see what Yahoo is rendering server side
        const tableMatch = yHtml.match(/<table[\s\S]*?<\/table>/ig);
        if (tableMatch) {
            console.log("Found", tableMatch.length, "tables. First table length:", tableMatch[0].length);
            // Search for SPY valuation metrics
            const peMatch = yHtml.match(/trailingPE.*?(\d+\.\d+)/i);
            console.log("PE Match Regex 1:", peMatch ? peMatch[0] : "None");
            
            const pRatioMatch = yHtml.match(/Price\/Earnings[\s\S]{0,100}td>([\d\.]+)/i);
            console.log("PE Match Regex 2:", pRatioMatch ? pRatioMatch[1] : "None");
        } else {
             console.log("No tables rendered server side by Yahoo Finance");
        }
    } catch (e) {
        console.error("Error fetching Yahoo Finance:", e);
    }
}

checkPE();
