'use client';
/**
 * useSheetLock(open) — while a detail sheet is open, the page underneath holds still.
 *
 * Before: a swipe on the dimmed backdrop scrolled the 8,000px page behind the Jev and
 * Polymarket sheets, so closing one dropped him somewhere else.
 *
 * While ANY sheet is open:
 *  - <html> carries class "sheet-open" (SHARED CONTRACT: floating UI such as a glance
 *    bar or back pill hides itself under html.sheet-open in CSS);
 *  - <html> and <body> get overflow:hidden (iOS Safari 16+ honours it on the root);
 *  - a touchmove that does not start inside a scrollable [data-sheet-scroll] box is
 *    cancelled, which also holds the page on older iOS (pinch-zoom is left alone).
 * window.scrollY is never reset (no body position:fixed), so PullToRefresh, JumpNav's
 * >500px check and scroll listeners see no jump. On close the old inline styles come
 * back and, if anything moved the page anyway, it is put back at the exact scrollY.
 *
 * Nested opens are counted: the page unlocks only when the last sheet closes. The
 * effect cleanup runs on unmount too (a refresh resetting a card mid-open).
 */
import { useEffect } from 'react';

export const SHEET_OPEN_CLASS = 'sheet-open';

let depth = 0;
let saved = null;

function guard(e) {
    if (e.touches && e.touches.length > 1) return; // pinch-zoom
    const t = e.target;
    const box = t && typeof t.closest === 'function' ? t.closest('[data-sheet-scroll]') : null;
    if (box && box.scrollHeight > box.clientHeight) return; // the sheet scrolls itself
    if (e.cancelable) e.preventDefault();
}

export function lockPage() {
    if (typeof document === 'undefined') return;
    depth += 1;
    if (depth > 1) return;
    const html = document.documentElement;
    const body = document.body;
    // a desktop scrollbar disappearing would shift the page sideways; pad it back
    const gutter = Math.max(0, (window.innerWidth || 0) - (html.clientWidth || window.innerWidth || 0));
    saved = {
        y: window.scrollY || window.pageYOffset || 0,
        html: html.style.overflow,
        body: body.style.overflow,
        pad: body.style.paddingRight,
    };
    html.classList.add(SHEET_OPEN_CLASS);
    html.style.overflow = 'hidden';
    body.style.overflow = 'hidden';
    if (gutter > 0) body.style.paddingRight = `${gutter}px`;
    document.addEventListener('touchmove', guard, { passive: false });
}

export function unlockPage() {
    if (typeof document === 'undefined' || depth === 0) return;
    depth -= 1;
    if (depth > 0) return;
    const html = document.documentElement;
    const body = document.body;
    document.removeEventListener('touchmove', guard, { passive: false });
    html.classList.remove(SHEET_OPEN_CLASS);
    const s = saved || { y: null, html: '', body: '', pad: '' };
    saved = null;
    html.style.overflow = s.html;
    body.style.overflow = s.body;
    body.style.paddingRight = s.pad;
    const y = window.scrollY || window.pageYOffset || 0;
    if (s.y != null && Math.abs(y - s.y) >= 1) {
        try { window.scrollTo(0, s.y); } catch { /* best effort */ }
    }
}

export default function useSheetLock(open) {
    useEffect(() => {
        if (!open) return undefined;
        lockPage();
        return unlockPage;
    }, [open]);
}
