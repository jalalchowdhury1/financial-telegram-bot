const yahooFinance = require('yahoo-finance2').default;

async function checkPE() {
    console.log("Fetching from multpl.com...");
    let multplPE = null;
    try {
        const peRes = await fetch('https://www.multpl.com/s-p-500-pe-ratio', {
            headers: { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36' }
        });
        const peHtml = await peRes.text();
        const peMatch = peHtml.match(/Current S&P 500 PE Ratio[^\d]*(\d+\.\d+)/);
        if (peMatch) multplPE = parseFloat(peMatch[1]);
        console.log("multpl.com S&P 500 P/E:", multplPE);
    } catch (e) {
        console.error("Error scraping multpl:", e);
    }

    console.log("\nFetching SPY from Yahoo Finance...");
    try {
        const result = await yahooFinance.quoteSummary('SPY', { modules: ['summaryDetail'] });
        console.log("Yahoo Finance SPY PE Ratio:", result.summaryDetail.trailingPE);
    } catch (e) {
        console.error("Error fetching Yahoo Finance:", e);
    }
}

checkPE();
