/**
 * Standardized fetcher for the financial dashboard.
 * Includes default headers, status validation, and error reporting.
 */

import { DEFAULT_HEADERS } from './constants';

export async function proxyFetch(url, options = {}) {
    const finalOptions = {
        ...options,
        headers: {
            ...DEFAULT_HEADERS,
            ...(options.headers || {})
        },
        next: { revalidate: options.revalidate !== undefined ? options.revalidate : 0 }
    };

    try {
        const response = await fetch(url, finalOptions);

        if (!response.ok) {
            throw new Error(`Fetch failed for ${url}: ${response.status} ${response.statusText}`);
        }

        return response;
    } catch (error) {
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
