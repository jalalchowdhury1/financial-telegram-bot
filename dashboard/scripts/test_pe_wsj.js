const axios = require('axios');
const cheerio = require('cheerio');

async function checkWSJ() {
    console.log("Fetching from WSJ Market Data...");
    try {
        const url = 'https://www.wsj.com/market-data/stocks/peyields';
        const res = await fetch(url, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' }
        });
        const html = await res.text();
        const $ = cheerio.load(html);

        // Find the table row containing "S&P 500"
        let pe = null;
        $('tr').each((i, el) => {
            const text = $(el).text();
            if (text.includes('S&P 500')) {
                const cols = $(el).find('td');
                if (cols.length >= 2) {
                    pe = $(cols[0]).text().trim();
                }
            }
        });

        console.log("WSJ S&P 500 P/E:", pe);
    } catch (e) {
        console.error("Error fetching WSJ:", e);
    }
}

checkWSJ();
