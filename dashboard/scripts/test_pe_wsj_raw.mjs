async function checkWSJRaw() {
    console.log("Fetching WSJ JSON API...");
    try {
        const url = 'https://www.wsj.com/market-data/quotes/index/SPX?id=%7B%22ticker%22%3A%22SPX%22%2C%22countryCode%22%3A%22US%22%2C%22instrumentType%22%3A%22INDEX%22%7D&type=quotes_historical';
        const res = await fetch(url, {
            headers: { 
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)',
                'Accept': 'application/json'
             }
        });
        const html = await res.text();
        const peMatch = html.match(/"peRatio"[^>]*?([\d\.]+)/i);
        if (peMatch) {
            console.log("WSJ P/E:", peMatch[1]);
        } else {
            console.log("No Match found in the 500kb response");
            const match2 = html.match(/P\/E Ratio \(TTM\)[\s\S]{0,100}/i);
            console.log("Partial:", match2 ? match2[0] : "Nothing");
        }
    } catch (e) {
        console.error("Error:", e);
    }
}

checkWSJRaw();
