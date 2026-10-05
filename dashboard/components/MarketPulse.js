'use client';
/**
 * 📡 Market Pulse — one line of verdict chips about the cards far below the fold (Rubber band,
 * Volatility, Recession watch, Yield curve, Bull checklist). Tap a chip to jump to its card.
 * SPY / F&G live in What moved and the glance bar, so they are no longer repeated here.
 * Chips come from lib/pulseVerdicts.js: same numbers and thresholds as each card, a missing
 * source = no chip, a saved/stale source = a dashed chip that says so.
 */
import { useMemo } from 'react';
import { pulseVerdicts } from '../lib/pulseVerdicts';
import { jumpToCard } from './WhatMoved';

export default function MarketPulse({ fred, vol, rubberBand, saved = null, waiting = false }) {
    const savedFred = saved?.fred || null;
    const savedVol = saved?.vol || null;
    const chips = useMemo(
        () => pulseVerdicts({ fred, vol, rubberBand, saved: { fred: savedFred, vol: savedVol } }),
        [fred, vol, rubberBand, savedFred, savedVol],
    );
    const label = (
        <span className="pulse-label">
            <span aria-hidden="true">📡</span><span className="pulse-label-text"> Market Pulse</span>
        </span>
    );
    if (!chips.length) {
        // Hold the line while its feeds load, so the cards below do not jump when it fills.
        return waiting ? (
            <nav className="market-pulse is-waiting" aria-label="Market Pulse" aria-busy="true">
                {label}
                <span className="pulse-items"><span className="pulse-wait">…</span></span>
            </nav>
        ) : null;
    }
    return (
        <nav className="market-pulse" aria-label="Market Pulse: verdicts from the cards below">
            {label}
            <span className="pulse-items">
                {chips.map((c) => {
                    const note = c.old ? ` · ${c.old.note}` : '';
                    return (
                        <button
                            key={c.key}
                            type="button"
                            className={`pulse-chip tone-${c.tone}${c.old ? ' is-old' : ''}`}
                            onClick={() => jumpToCard(c.jump)}
                            title={`${c.why}${note} · tap for the card`}
                            aria-label={`${c.text} — ${c.why}${note}`}
                        >
                            {c.old?.kind === 'stale' && <span className="pulse-old" aria-hidden="true">🕐</span>}
                            {c.text}
                        </button>
                    );
                })}
            </span>
        </nav>
    );
}
