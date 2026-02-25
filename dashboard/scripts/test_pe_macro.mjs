async function checkMacrotrends2() {
    console.log("Fetching from macrotrends (json approach)...");
    try {
        const url = 'https://www.macrotrends.net/assets/php/chart_iframe_comp.php?id=2577';
        const res = await fetch(url, {
            headers: { 'User-Agent': 'Mozilla/5.0' }
        });
        const html = await res.text();
        
        let peMatch = html.match(/var chartData = (\[.*?\]);/s);
        
        if (peMatch) {
            const data = JSON.parse(peMatch[1]);
            const lastData = data[data.length - 1];
            console.log("MacroTrends SP500 P/E:", lastData);
        } else {
             console.log("No Match on MacroTrends chart data");
        }
    } catch (e) {
        console.error("Error:", e);
    }
}

checkMacrotrends2();
