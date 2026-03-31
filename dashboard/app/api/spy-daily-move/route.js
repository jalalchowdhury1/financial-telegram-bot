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

        // Debug: log first few rows to understand structure
        console.log('[spy-daily-move] Total rows:', rows.length);

        // Return all rows for debugging - check Vercel logs
        const debugRows = {};
        for (let i = 0; i < Math.min(15, rows.length); i++) {
            debugRows[`row${i}`] = rows[i];
        }
        console.log('[spy-daily-move] Debug rows:', JSON.stringify(debugRows));

        // Try multiple row indices to find the value
        let value = null;
        for (let i = 0; i < Math.min(20, rows.length); i++) {
            const cellB = rows[i]?.[1]?.trim();
            if (cellB && cellB !== '' && !isNaN(parseFloat(cellB))) {
                value = cellB;
                console.log(`[spy-daily-move] Found value '${value}' at row ${i}, col B`);
                break;
            }
        }

        return Response.json({
            value,
            source: 'Google Sheets',
            debug: { totalRows: rows.length, debugRows }
        });
    } catch (e) {
        console.error('[spy-daily-move] Error:', e.message);
        return Response.json({
            value: null,
            source: 'Failed',
            error: e.message
        });
    }
}
