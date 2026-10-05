'use client';
/**
 * 📈 "What moved" — one line under the header: the 5 biggest moves since the last close,
 * ranked by how unusual they are for that series (lib/whatMoved.js). Tap a chip to jump
 * to its card. Renders nothing until there is something real to show.
 */
import { useMemo } from 'react';
import { collectMoves, fmtLevel, movedWhen } from '../lib/whatMoved';
import { jumpTarget, noteJumpFrom } from './JumpNav';

const FLASH_MS = 1600;

/** Scroll to the card with `data-jump="label"` and flash it so the eye lands on it. */
export function jumpToCard(label) {
    if (typeof document === 'undefined') return false;
    const el = [...document.querySelectorAll('[data-jump]')].find((e) => e.getAttribute('data-jump') === label);
    const t = jumpTarget(el);
    if (!t) return false;
    noteJumpFrom(); // ↩ the back pill can return here
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    t.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
    t.classList.add('jump-flash');
    setTimeout(() => t.classList.remove('jump-flash'), FLASH_MS);
    return true;
}

export default function WhatMoved({ spy, fg, extra, vol, history, saved = null, waiting = false }) {
    const moves = useMemo(() => collectMoves({ spy, fg, extra, vol, history }), [spy, fg, extra, vol, history]);
    // Which session the moves belong to: "today", or "Fri" over a weekend / before the open.
    const when = useMemo(() => { try { return movedWhen({ spy, vol }); } catch { return null; } }, [spy, vol]);
    if (!moves.length) {
        // Hold the line while its feeds load, so the cards below do not jump when it fills.
        return waiting ? (
            <nav className="moved-strip is-waiting" aria-label="What moved since the last close" aria-busy="true">
                <span className="moved-title">What moved</span>
                <span className="moved-saved">…</span>
            </nav>
        ) : null;
    }
    return (
        <nav
            className="moved-strip"
            aria-label={when === 'today' ? 'What moved today' : when ? `What moved on ${when}` : 'What moved since the last close'}
            data-cached={saved || undefined}
        >
            <span className="moved-title">What moved{when && <span className="moved-when"> · {when}</span>}</span>
            {/* inline, not a corner tag: the strip scrolls sideways and would clip one */}
            {saved && <span className="moved-saved">🕐 {saved}</span>}
            {moves.map((m) => {
                const why = `${m.label} ${fmtLevel(m.prev)} → ${fmtLevel(m.value)}${m.day ? ` on ${m.day}` : ''} · ${m.z.toFixed(1)}× a normal day`;
                return (
                    <button
                        key={m.key}
                        type="button"
                        className={`moved-chip${m.big ? ' is-big' : ''}`}
                        onClick={() => jumpToCard(m.jump)}
                        title={`${why} · tap for the card`}
                        aria-label={why}
                    >
                        {m.big && <span aria-hidden="true">⚡</span>}
                        <span className="moved-label">{m.label}</span>
                        <span className="moved-val">{m.text}</span>
                        {/* a slower feed's move, one business day behind the market date */}
                        {m.day && <span className="moved-day">{m.day}</span>}
                    </button>
                );
            })}
        </nav>
    );
}
