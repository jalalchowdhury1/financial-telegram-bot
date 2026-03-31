import { GOOGLE_SHEETS } from '../../../lib/constants';
import { fetchText } from '../../../lib/fetcher';

export const dynamic = 'force-dynamic';

function parseCSV(text) {
    return text.split('\n').map(row => {
        const result = [];
        let current = '';
        let inQuotes = false;
        for (const char of row) {
            if (char === '"') inQuotes = !inQuotes;
            else if (char === ',' && !inQuotes) { result.push(current); current = ''; }
            else current += char;
        }
        result.push(current);
        return result;
    });
}

export async function GET() {
    try {
        const text = await fetchText(GOOGLE_SHEETS.SPY_DAILY_MOVE);
        const rows = parseCSV(text);
        const value = rows[11]?.[1]?.trim() || null;

        return Response.json({
            value,
            source: 'Google Sheets'
        });
    } catch (e) {
        return Response.json({
            value: null,
            source: 'Failed',
            error: e.message
        });
    }
}
