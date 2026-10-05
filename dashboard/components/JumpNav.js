'use client';
import { useEffect, useState } from 'react';
import { rememberJump } from '../lib/jumpBack';

/**
 * 🧭 Jump menu. On a phone the page is ~8,000 px tall — reaching the vol table or
 * the markets grid meant a long thumb-scroll every time.
 *
 * A small round button (bottom-right, shown once you scroll past the first screen)
 * opens a list of every section; tap one to jump there. The list is built from
 * `[data-jump="Label"]` markers when it opens, so a card that renders nothing (a
 * route that is down, the pills with JEV_PILLS=off) is simply not listed.
 * Esc, the ☰ button again, or a tap outside closes it.
 *
 * A marker may sit on a `display: contents` wrapper (page.js wraps self-contained
 * cards that way so their grid layout is untouched); its first child is the target.
 */
export const SHOW_AFTER_PX = 500;

export function jumpTarget(el) {
    if (!el) return null;
    const t = typeof window !== 'undefined' && window.getComputedStyle?.(el).display === 'contents' ? el.firstElementChild : el;
    return t && t.getClientRects().length ? t : null;
}

export function collectSections(root = typeof document !== 'undefined' ? document : null) {
    if (!root) return [];
    return [...root.querySelectorAll('[data-jump]')]
        .map((el) => ({ label: el.getAttribute('data-jump'), el }))
        .filter((s) => s.label && jumpTarget(s.el));
}

/** Every rendered section's document top (and its sides, for cards sharing a desk row), for the ↩ back pill. */
export function measureSections() {
    if (typeof window === 'undefined') return [];
    return collectSections().map((s) => {
        const r = jumpTarget(s.el).getBoundingClientRect();
        return { label: s.label, top: r.top + window.scrollY, left: r.left, right: r.right };
    });
}

/** Call just before any jump: remembers where he is, so ↩ can bring him back (lib/jumpBack.js). */
export function noteJumpFrom() {
    try {
        rememberJump({ y: window.scrollY, vh: window.innerHeight, cx: window.innerWidth / 2, sections: measureSections() });
    } catch { /* never block the jump itself */ }
}

export default function JumpNav() {
    const [show, setShow] = useState(false);
    const [open, setOpen] = useState(false);
    const [sections, setSections] = useState([]);

    useEffect(() => {
        const onScroll = () => setShow(window.scrollY > SHOW_AFTER_PX);
        onScroll();
        window.addEventListener('scroll', onScroll, { passive: true });
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    useEffect(() => {
        if (!open) return undefined;
        const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
        document.addEventListener('keydown', onKey);
        return () => document.removeEventListener('keydown', onKey);
    }, [open]);

    const toggle = () => {
        if (open) { setOpen(false); return; }
        setSections(collectSections());
        setOpen(true);
    };
    const smooth = () => (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth');
    const go = (el) => {
        setOpen(false);
        const t = jumpTarget(el);
        if (!t) return;
        noteJumpFrom();
        t.scrollIntoView({ behavior: smooth(), block: 'start' });
    };
    const top = () => {
        setOpen(false);
        noteJumpFrom();
        window.scrollTo({ top: 0, behavior: smooth() });
    };

    if (!show && !open) return null;
    return (
        <>
            {open && <div className="jump-backdrop" onClick={() => setOpen(false)} aria-hidden="true" />}
            {open && (
                <nav className="jump-menu" aria-label="Jump to section">
                    <button type="button" className="jump-item jump-top" onClick={top}>↑ Top</button>
                    {sections.map((s) => (
                        <button type="button" key={s.label} className="jump-item" onClick={() => go(s.el)}>{s.label}</button>
                    ))}
                </nav>
            )}
            <button
                type="button"
                className={`jump-fab${open ? ' is-open' : ''}`}
                onClick={toggle}
                aria-haspopup="true"
                aria-expanded={open}
                aria-label={open ? 'Close section menu' : 'Jump to a section'}
                title="Jump to a section"
            >
                {open ? '✕' : '☰'}
            </button>
        </>
    );
}
