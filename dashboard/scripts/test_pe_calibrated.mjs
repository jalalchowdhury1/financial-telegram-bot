async function checkYahooCalibrated() {
    console.log("Fetching SPY from Yahoo Finance HTML...");
    try {
        const yUrl = 'https://finance.yahoo.com/quote/SPY/key-statistics';
        const yRes = await fetch(yUrl, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
        });
        const yHtml = await yRes.text();

        let peMatch = yHtml.match(/PE Ratio \(TTM\)[\s\S]*?(\d+\.\d+)/i);
        if (peMatch) {
            let basePE = parseFloat(peMatch[1]);
            // Yahoo uses Operating Earnings. Multpl uses As-Reported (GAAP) Earnings.
            // Historically, GAAP P/E runs about 6-8% higher than Operating P/E for the S&P 500
            // We apply a safe 1.07x multiplier to approximate the GAAP P/E.
            let calibratedPE = basePE * 1.07;
            console.log("Yahoo Finance SPY Operating PE:", basePE);
            console.log("Yahoo Finance SPY Calibrated GAAP PE:", calibratedPE.toFixed(2));
        } else {
            console.log("Could not find trailingPE in Yahoo HTML");
        }
    } catch (e) {
        console.error("Error fetching Yahoo Finance:", e);
    }
}

checkYahooCalibrated();
