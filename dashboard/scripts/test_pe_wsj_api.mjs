async function checkWSJApi() {
    console.log("Fetching from WSJ API...");
    try {
        const url = 'https://www.wsj.com/market-data/quotes/index/SPX';
        const res = await fetch(url, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' }
        });
        const html = await res.text();
        
        let peMatch = html.match(/"peRatio":\s*(\d+\.\d+)/i);
        if (!peMatch) peMatch = html.match(/P\/E Ratio \(TTM\)[\s\S]*?<span[^>]*>([\d\.]+)<\/span>/i);
        
        if (peMatch) {
            console.log("WSJ SPX P/E:", parseFloat(peMatch[1]));
        } else {
             console.log("No Match on WSJ SPX");
        }
    } catch (e) {
        console.error("Error:", e);
    }
}

checkWSJApi();
