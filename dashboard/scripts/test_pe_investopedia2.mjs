async function checkInvestopedia() {
    console.log("Fetching from Multpl Shiller PE...");
    try {
        const url2 = 'https://www.multpl.com/shiller-pe';
         const res2 = await fetch(url2, {
            headers: { 'User-Agent': 'Mozilla/5.0' }
        });
        const html2 = await res2.text();
        const peMatch2 = html2.match(/Current Shiller PE Ratio[^\d]*(\d+\.\d+)/);
        console.log("Shiller P/E (CAPE):", peMatch2 ? parseFloat(peMatch2[1]) : "No match");
    } catch (e) {
        console.error("Error:", e);
    }
}

checkInvestopedia();
