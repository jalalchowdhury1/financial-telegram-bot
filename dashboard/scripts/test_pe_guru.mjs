async function checkGuruFocus() {
    console.log("Fetching SPY from GuruFocus...");
    try {
        const url = 'https://www.gurufocus.com/etf/SPY/summary';
        const res = await fetch(url, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' }
        });
        const html = await res.text();

        // This regex looks for PE Ratio string followed by the table value
        const peMatch = html.match(/PE Ratio[\s\S]*?<td>\s*([\d\.]+)\s*<\/td>/i);

        if (peMatch) {
            console.log("GuruFocus SPY P/E:", parseFloat(peMatch[1]));
        } else {
            console.log("No Match on GuruFocus");
            const match2 = html.match(/"pe_ratio":([\d\.]+)/i);
            console.log("JSON Match:", match2 ? match2[1] : "None");
        }
    } catch (e) {
        console.error("Error:", e);
    }
}

checkGuruFocus();
