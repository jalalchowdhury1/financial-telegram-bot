async function checkInvestopedia() {
    console.log("Fetching from SSRN/C Schiller data...");
    try {
        const url = 'https://www.multpl.com/s-p-500-pe-ratio';
        const res = await fetch(url, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' }
        });
        const html = await res.text();
        console.log("Multpl is working fine: ", html.length > 100);
        
        // Let's try to find an alternative multpl page for standard PE just in case the URL changes
        const url2 = 'https://www.multpl.com/shiller-pe';
         const res2 = await fetch(url2, {
            headers: { 'User-Agent': 'Mozilla/5.0' }
        });
        const html2 = await res2.text();
        const peMatch2 = html2.match(/Current Shiller PE Ratio[^\d]*(\d+\.\d+)/);
        console.log("Shiller PE:", peMatch2 ? peMatch2[1] : "No match");
        
    } catch (e) {
        console.error("Error:", e);
    }
}

checkInvestopedia();
