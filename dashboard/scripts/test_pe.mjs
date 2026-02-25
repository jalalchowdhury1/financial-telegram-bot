const SPY_URL = 'https://finance.yahoo.com/quote/SPY/key-statistics';

async function checkPE() {
    console.log("Fetching SPY from Yahoo Finance Key Stats HTML...");
    try {
        const yRes = await fetch(SPY_URL, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
        });
        const yHtml = await yRes.text();
        
        let peMatch = yHtml.match(/trailingPE.*?raw":(\d+\.\d+)/);
        if (peMatch) {
            console.log("Yahoo Finance SPY Trailing PE (raw json):", parseFloat(peMatch[1]));
            return;
        }

        peMatch = yHtml.match(/PE Ratio \(TTM\)[\s\S]*?(\d+\.\d+)/i);
        if (peMatch) {
            console.log("Yahoo Finance SPY Trailing PE (text):", parseFloat(peMatch[1]));
        } else {
            console.log("Could not find trailingPE in Yahoo HTML");
        }
    } catch (e) {
        console.error("Error fetching Yahoo Finance:", e);
    }
}

checkPE();
