'use client';
/**
 * ↩ Back pill — after any jump (What moved chip, Market Pulse chip, ☰ menu), a small glass
 * capsule sits left of the ☰ button: "↩ Back to Economy". One tap returns to the exact spot
 * (lib/jumpBack.js keeps it as section + offset) and the pill goes away. It also fades by
 * itself after 8 s, or once he scrolls more than a screen by hand from where he landed.
 * Hidden while a sheet or the jump menu is open (CSS). No history entries, no #hash.
 */
import { useEffect, useRef, useState } from 'react';
import { clearJump, currentJump, movedTooFar, onJump, resolveBack } from '../lib/jumpBack';
import { collectSections, jumpTarget } from './JumpNav';

export const BACK_PILL_MS = 8000;
const SETTLE_MS = 250;      // no scroll for this long after the flight = the jump has landed
const FLIGHT_START_MS = 1000; // a smooth scroll can start late on a busy phone; none by then = landed in place

function topOf(label) {
    const s = collectSections().find((x) => x.label === label);
    const t = s && jumpTarget(s.el);
    return t ? t.getBoundingClientRect().top + window.scrollY : null;
}

export default function BackPill() {
    const [spot, setSpot] = useState(() => currentJump());
    const lastLabel = useRef(null);

    useEffect(() => onJump(setSpot), []);

    useEffect(() => {
        if (!spot) return undefined;
        let landY = null;
        let settle = setTimeout(() => { landY = window.scrollY; }, FLIGHT_START_MS);
        const onScroll = () => {
            if (landY == null) {
                // still flying to the card: wait for the scroll to settle
                clearTimeout(settle);
                settle = setTimeout(() => { landY = window.scrollY; }, SETTLE_MS);
                return;
            }
            if (movedTooFar(window.scrollY, landY, window.innerHeight)) clearJump();
        };
        const fade = setTimeout(clearJump, BACK_PILL_MS);
        window.addEventListener('scroll', onScroll, { passive: true });
        return () => {
            clearTimeout(settle);
            clearTimeout(fade);
            window.removeEventListener('scroll', onScroll);
        };
    }, [spot]);

    if (spot) lastLabel.current = spot.name || spot.label; // name = a desk row's two cards
    if (!lastLabel.current) return null; // never jumped yet
    const on = !!spot;
    const name = lastLabel.current === 'Top' ? 'top' : lastLabel.current;

    const back = () => {
        const s = currentJump();
        if (!s) return;
        const y = resolveBack(s, topOf);
        clearJump();
        const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
        window.scrollTo({ top: y, behavior: reduce ? 'auto' : 'smooth' });
    };

    return (
        <button
            type="button"
            className={`back-pill${on ? ' is-on' : ''}`}
            onClick={back}
            aria-label={`Back to ${name}`}
            aria-hidden={on ? undefined : 'true'}
            tabIndex={on ? 0 : -1}
            title="Back to where you were"
        >
            <span aria-hidden="true">↩</span> Back to {name}
        </button>
    );
}
