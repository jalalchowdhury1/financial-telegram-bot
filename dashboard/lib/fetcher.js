/**
 * Standardized fetcher for the financial dashboard.
 * Includes default headers, status validation, and error reporting.
 */

import { DEFAULT_HEADERS, YAHOO_TICKERS, GOOGLE_SHEETS } from './constants';

export async function proxyFetch(url, options = {}) {
    const { timeout = 8000, ...fetchOptions } = options;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    const finalOptions = {
        ...fetchOptions,
        signal: controller.signal,
        headers: {
            ...DEFAULT_HEADERS,
            ...(fetchOptions.headers || {})
        },
        next: { revalidate: fetchOptions.revalidate !== undefined ? fetchOptions.revalidate : 0 }
    };

    try {
        const response = await fetch(url, finalOptions);
        clearTimeout(timer);

        if (!response.ok) {
            throw new Error(`Fetch failed for ${url}: ${response.status} ${response.statusText}`);
        }

        return response;
    } catch (error) {
        clearTimeout(timer);
        if (error.name === 'AbortError') {
            throw new Error(`Fetch timed out for ${url} after ${timeout}ms`);
        }
        console.error(`[proxyFetch Error]: ${error.message}`);
        throw error;
    }
}

export async function fetchJson(url, options = {}) {
    const response = await proxyFetch(url, options);
    return response.json();
}

export async function fetchText(url, options = {}) {
    const response = await proxyFetch(url, options);
    return response.text();
}

/**
 * Fetch Yahoo Finance chart data and standardize it.
 */
export async function fetchYahooFinanceSeries(ticker) {
    try {
        const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=1mo&interval=1d`;
        const data = await fetchJson(url, { revalidate: 0 });
        
        const result = data?.chart?.result?.[0];
        if (!result) return null;

        const timestamps = result.timestamp;
        const closes = result.indicators?.quote?.[0]?.close;

        if (!timestamps || !closes || timestamps.length < 2) return null;

        let history = [];
        for (let i = 0; i < timestamps.length; i++) {
            if (closes[i] != null) {
                history.push({
                    date: new Date(timestamps[i] * 1000).toISOString().split('T')[0],
                    price: closes[i]
                });
            }
        }

        if (history.length < 2) return null;

        const current = history[history.length - 1].price;
        const prev = history[history.length - 2].price;
        return {
            current,
            dailyChange: {
                value: current - prev,
                pct: prev ? ((current - prev) / prev) * 100 : 0
            },
            history
        };
    } catch (e) {
        console.warn(`[YahooFinance] Failed to fetch ${ticker}: ${e.message}`);
        return null;
    }
}

/**
 * Fetch and parse Google Sheet CSV export.
 * Returns a key-value mapping of the data.
 */
export async function fetchGoogleSheetExport() {
    try {
        const text = await fetchText(GOOGLE_SHEETS.MARKET_BACKUP, { revalidate: 0 });
        const lines = text.trim().split('\n');
        
        // Skip header
        const data = {};
        for (let i = 1; i < lines.length; i++) {
            const [key, valueStr] = lines[i].split(',');
            if (key && valueStr) {
                const val = parseFloat(valueStr.trim());
                if (!isNaN(val)) {
                    data[key.trim()] = val;
                }
            }
        }
        return data;
    } catch (e) {
        console.warn(`[GoogleSheet] Failed to fetch fallback data: ${e.message}`);
        return null;
    }
}
