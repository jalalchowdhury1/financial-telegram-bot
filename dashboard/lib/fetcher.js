/**
 * Standardized fetcher for the financial dashboard.
 * Includes default headers, status validation, and error reporting.
 */

import { DEFAULT_HEADERS } from './constants';

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
