// /api/sheets - Fetch Google Sheets custom indicators
export async function GET() {
    try {
        const sheets = [
            {
                name: 'NotSoBoring',
                url: 'https://docs.google.com/spreadsheets/d/10Y8Jus8_fMwH9H69vWh7thSzl2hH34Ri3BRbDw_GEgw/export?format=csv&gid=0',
                parse: (rows) => rows[2]?.[1]?.trim() || 'N/A'
            },
            {
                name: 'FrontRunner',
                url: 'https://docs.google.com/spreadsheets/d/1vdlPNlT6gRpzMHuQUT7olqUNb455CQM3ab4wPuCE5R0/export?format=csv&gid=1668420064',
                parse: (rows) => (rows[1]?.[0]?.trim() || 'N/A').split('\n')[0].trim()
            },
            {
                name: 'AAIIDiff',
                url: 'https://docs.google.com/spreadsheets/d/1zQQ2am1yhzTwY7nx8xPak4Q0WoNMwxWj7Ekr-fDEIF4/export?format=csv&gid=0',
                parse: (rows) => rows[1]?.[4]?.trim() || 'N/A'
            },
            {
                name: 'VIX',
                url: 'https://docs.google.com/spreadsheets/d/1vdlPNlT6gRpzMHuQUT7olqUNb455CQM3ab4wPuCE5R0/export?format=csv&gid=790638481',
                parse: (rows) => ({
                    current: rows[1]?.[0]?.trim() || 'N/A',
                    threeMonth: rows[1]?.[1]?.trim() || 'N/A',
                    fearGreed: rows[1]?.[2]?.trim() || 'N/A'
                })
            }
        ];

        const results = {};

        for (const sheet of sheets) {
            try {
                const res = await fetch(sheet.url, { next: { revalidate: 0 } });
                const text = await res.text();
                const rows = text.split('\n').map(row => {
                    const result = [];
                    let current = '';
                    let inQuotes = false;
                    for (const char of row) {
                        if (char === '"') { inQuotes = !inQuotes; }
                        else if (char === ',' && !inQuotes) { result.push(current); current = ''; }
                        else { current += char; }
                    }
                    result.push(current);
                    return result;
                });
                results[sheet.name] = sheet.parse(rows);
            } catch (e) {
                results[sheet.name] = 'Error';
            }
        }

        return Response.json(results);
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
