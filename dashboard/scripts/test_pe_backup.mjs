async function checkPE() {
    let peRatio = null;

    // Simulate failure of multpl.com by intentionally skipping it
    console.log("Simulating multpl.com failure (skipping)...");

    // Method 2: Yahoo Finance Fallback (SPY Trailing P/E)
    if (!peRatio) {
        console.log("Falling back to Yahoo Finance...");
        try {
            const yRes = await fetch('https://finance.yahoo.com/quote/SPY/key-statistics', {
                headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' },
                next: { revalidate: 3600 }
            });
            const yHtml = await yRes.text();

            // This is the exact regex from your route.js
            const peMatch = yHtml.match(/PE Ratio \(TTM\)[\s\S]*?(\d+\.\d+)/i);

            if (peMatch) {
                peRatio = parseFloat(peMatch[1]);
                console.log("✅ SUCCESS! Found PE from Yahoo API Backup:", peRatio);
            } else {
                console.log("❌ FAILED to find PE match in Yahoo HTML");
            }
        } catch (e) {
            console.error("❌ Yahoo Finance P/E fallback failed:", e.message);
        }
    }
}

checkPE();
