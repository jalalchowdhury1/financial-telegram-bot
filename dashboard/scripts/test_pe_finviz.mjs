async function checkFinviz() {
    console.log("Fetching SPY from Finviz...");
    try {
        const url = 'https://finviz.com/quote.ashx?t=SPY';
        const res = await fetch(url, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' }
        });
        const html = await res.text();
        
        let peMatch = html.match(/>P\/E<\/td>\s*<td.*?><b.*?>([\d\.]+)</i);
        
        if (peMatch) {
            console.log("Finviz SPY P/E:", parseFloat(peMatch[1]));
        } else {
             console.log("No Match on Finviz");
        }
    } catch (e) {
        console.error("Error:", e);
    }
}

checkFinviz();
