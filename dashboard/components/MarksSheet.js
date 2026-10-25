'use client';
/**
 * ● sheet — what the "● N new prints" chip and the glance bar's ● N open (MarkChip.js).
 * One row per lit number, biggest change for its size first: "Sahm Rule −0.03 → 0.00 · Economy".
 * Tap a row: the sheet closes, the page scrolls to that number, flashes it, and the ↩ pill can
 * bring him back. Rows come from the same markFor() rule as the counts (useMarkList), so the
 * chip's number and the list always agree; prev is the mark's own baseline, never a chart point.
 * The card name is read off the page ([data-jump] around the number, Delta's data-mark-key).
 */
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import useSheetLock, { useSheetFocus } from './useSheetLock';
import { useMarkList } from './MarkProvider';
import { onOpenMarks, markSummary } from './MarkChip';
import { noteJumpFrom } from './JumpNav';

const FLASH_MS = 1600; // as jumpToCard (WhatMoved.js)

const size = (m) => (m.prev ? Math.abs(m.value - m.prev) / Math.abs(m.prev) : Infinity);
const decimals = (v) => Math.min(3, (String(v).split('.')[1] || '').length);
const fmt = (v, dp) => v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp }).replace('-', '−');

function target(key) {
    return [...document.querySelectorAll(`[data-mark-key="${key}"]`)].find((el) => el.getClientRects().length)
        || document.querySelector(`[data-mark-key="${key}"]`);
}

export default function MarksSheet({ values }) {
    const [open, setOpen] = useState(false);
    const [go, setGo] = useState(null);
    const boxRef = useRef(null);
    const marks = useMarkList(values);
    useSheetLock(open);
    useSheetFocus(open, boxRef);

    useEffect(() => onOpenMarks(() => setOpen(true)), []);
    useEffect(() => {
        if (!open) return undefined;
        const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
        document.addEventListener('keydown', onKey);
        return () => document.removeEventListener('keydown', onKey);
    }, [open]);
    // The jump runs AFTER the sheet's lock has let go (that cleanup puts scrollY back).
    useEffect(() => {
        if (!go) return;
        setGo(null);
        const el = target(go);
        if (!el) return;
        noteJumpFrom();
        const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
        el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center' });
        el.classList.add('jump-flash');
        setTimeout(() => el.classList.remove('jump-flash'), FLASH_MS);
    }, [go]);

    if (!open || typeof document === 'undefined') return null;
    const rows = [...marks].sort((a, b) => size(b) - size(a));
    const counts = { print: rows.filter((m) => m.kind !== 'move').length, move: rows.filter((m) => m.kind === 'move').length };

    // Portalled to <body>: .card's backdrop-filter would trap a fixed sheet (see JevPillModal).
    return createPortal(
        <div className="jev-modal-backdrop marks-sheet-backdrop" onClick={(e) => { if (e.target === e.currentTarget) setOpen(false); }}
            role="dialog" aria-modal="true" aria-labelledby="marks-sheet-title">
            <div className="jev-modal marks-sheet" ref={boxRef} data-sheet-scroll="">
                <div className="jev-modal-header marks-sheet-head">
                    <h2 id="marks-sheet-title" className="jev-modal-title">{markSummary(counts) || 'Nothing lit now'}</h2>
                    <button type="button" className="jev-modal-close sheet-x" onClick={() => setOpen(false)} aria-label="Close">×</button>
                </div>
                {rows.map((m) => {
                    const dp = Math.max(decimals(m.prev), decimals(m.value));
                    const pair = `${fmt(m.prev, dp)} → ${fmt(m.value, dp)}`;
                    const card = target(m.key)?.closest('[data-jump]')?.getAttribute('data-jump');
                    return (
                        <button key={m.key} type="button" className="marks-row"
                            aria-label={`${m.label} ${pair}${card ? ` · ${card}` : ''} — tap to jump to it`}
                            onClick={() => { setOpen(false); setGo(m.key); }}>
                            <span className="marks-glyph" aria-hidden="true">{m.kind === 'move' ? (m.dir > 0 ? '⌃' : '⌄') : <span className="mark-dot" />}</span>
                            <span className="marks-label">{m.label}</span>
                            <span className="marks-pair">{pair}</span>
                            {card && <span className="marks-card"> · {card}</span>}
                        </button>
                    );
                })}
            </div>
        </div>,
        document.body,
    );
}
