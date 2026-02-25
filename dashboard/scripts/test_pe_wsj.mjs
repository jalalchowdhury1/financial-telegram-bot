async function checkWSJSpy() {
    console.log("Fetching from WSJ SPY Quote...");
    try {
        const url = 'https://www.wsj.com/market-data/quotes/etf/SPY';
        const res = await fetch(url, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' }
        });
        const html = await res.text();
        
        // WSJ usually puts P/E in a span class "WSJTheme--data--xlRLEr3r" or standard list item
        let peMatch = html.match(/P\/E Ratio \(TTM\)[\s\S]*?<span[^>]*>\s*([\d\.]+)\s*<\/span>/i);
        
        if (peMatch) {
            console.log("WSJ SPY P/E:", parseFloat(peMatch[1]));
        } else {
             console.log("No Match on WSJ SPY P/E");
             // console.log(html.substring(0, 1500));
        }
    } catch (e) {
        console.error("Error:", e);
    }
}

checkWSJSpy();
