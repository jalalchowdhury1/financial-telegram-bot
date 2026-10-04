'use client';
/**
 * useElementWidth() → [ref, px]: the rendered width of the element `ref` is attached to,
 * in whole CSS px, kept current by a ResizeObserver. null until measured (server render,
 * jsdom, no ResizeObserver) — callers fall back to their old fixed layout.
 * Charts use it to space HTML axis labels in real pixels (lib/chartAxis.js).
 */
import { useCallback, useRef, useState } from 'react';

export default function useElementWidth() {
    const [px, setPx] = useState(null);
    const ro = useRef(null);
    const ref = useCallback((node) => {
        if (ro.current) { ro.current.disconnect(); ro.current = null; }
        if (!node) return;
        const read = () => {
            const w = Math.round(node.getBoundingClientRect().width);
            if (w > 0) setPx(w);
        };
        read();
        if (typeof ResizeObserver === 'function') {
            try { ro.current = new ResizeObserver(read); ro.current.observe(node); } catch { /* fixed layout */ }
        }
    }, []);
    return [ref, px];
}
