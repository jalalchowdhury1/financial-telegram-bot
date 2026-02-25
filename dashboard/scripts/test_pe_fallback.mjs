async function checkMacroTrends() {
    console.log("Fetching from macrotrends.net...");
    try {
        const url = 'https://www.macrotrends.net/2577/sp-500-pe-ratio-price-to-earnings-chart';
        const res = await fetch(url, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' }
        });
        const html = await res.text();
        
        let peMatch = html.match(/The current P\/E Ratio.+?>([\d\.]+)<\/strong>/i);
        
        if (peMatch) {
            console.log("MacroTrends P/E:", parseFloat(peMatch[1]));
        } else {
             console.log("No Match on MacroTrends");
        }
    } catch (e) {
        console.error("Error:", e);
    }
}

checkMacroTrends();
