'use client';
/**
 * The Centred tile's details (tap the tile): what the nightly check says, since when, what moving the off-centre
 * settings would do to the whole algo, and the off count per night. The public feed holds no conditions: those are
 * on the nuts-radar Centred tab, one tap away.
 */
import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import useSheetLock, { useSheetFocus } from './useSheetLock';
import { day } from './SignalStudy';

export const CENTRED_TAB = 'https://nuts-radar.vercel.app/#centred';

const signed = (v, d = 1) => `${v < 0 ? '−' : v > 0 ? '+' : ''}${Math.abs(v).toFixed(d)}`;
/** ["Moving them: gain −0.4 %/yr", "Worst fall 0.9 pts smaller"] — centred minus live, plain words, two short lines (a phone fits each). */
export function compareLines(delta) {
    if (!delta) return ['Moving them: not judged yet', null];
    const fall = Math.abs(delta.maxdd) < 0.05 ? 'the same' : `${Math.abs(delta.maxdd).toFixed(1)} pts ${delta.maxdd > 0 ? 'smaller' : 'bigger'}`;
    return [`Moving them: gain ${signed(delta.cagr)} %/yr`, `Worst fall ${fall}`];
}

/** One line for the tile and the popup head. */
export function centredHeadline(c) {
    if (c.state === 'unknown') return 'Centred · not checked';
    return c.off > 0 ? `Off centre · ${c.off} of ${c.checked}` : `Centred · all ${c.checked} in the middle`;
}

export default function CentredDetails({ centred: c, onClose }) {
    const sheetRef = useRef(null), backRef = useRef(null);
    useSheetLock(true);
    useSheetFocus(true, sheetRef);
    useEffect(() => {
        const k = (e) => { if (e.key === 'Escape') onClose(); };
        document.addEventListener('keydown', k);
        return () => document.removeEventListener('keydown', k);
    }, [onClose]);

    const hist = c.history || [];
    const top = Math.max(1, ...hist.map((r) => r.checked));
    const w = 320, h = 60, bw = hist.length ? w / hist.length : w;
    const worst = hist.reduce((a, b) => (b.off > (a?.off ?? -1) ? b : a), null);
    const verdict = c.delta ? c.verdict || 'not judged' : null;
    const [gain, fall] = compareLines(c.delta);

    if (typeof document === 'undefined') return null;
    return createPortal(
        <div ref={backRef} className="v2-aa-back" role="dialog" aria-modal="true" aria-labelledby="v2-cd-title"
            onClick={(e) => { if (e.target === backRef.current) onClose(); }}>
            <div className="v2-aa" ref={sheetRef} data-sheet-scroll="">
                <div className="v2-aa-head">
                    <div className="v2-aa-title">
                        <h2 id="v2-cd-title">Centred</h2>
                        <p>{centredHeadline(c)}</p>
                    </div>
                    <button type="button" className="v2-aa-close sheet-x" onClick={onClose} aria-label="Close">×</button>
                </div>
                {c.note && <p className="v2-aa-foot v2-cd-note">{c.note}</p>}
                {(c.since || c.prices_to) && (
                    <div className="v2-sh-row">
                        <span>{c.since ? `Since ${day(c.since)}` : ''}</span>
                        <span>{c.prices_to ? `Prices to ${day(c.prices_to)}` : ''}</span>
                    </div>
                )}
                <div className="v2-cd-compare">
                    <span>{gain}</span>
                    {verdict && <span className={`v2-cd-chip is-${verdict.replace(' ', '-')}`}>{verdict}</span>}
                </div>
                {fall && <div className="v2-cd-compare is-sub">{fall}</div>}
                {hist.length > 1 && (
                    <>
                        <div className="v2-sh v2-cd-chart">
                            <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img"
                                aria-label={`Off centre per night, ${day(hist[0].date)} to ${day(hist.at(-1).date)}: most ${worst.off} on ${day(worst.date)}`}>
                                <line className="v2-cd-base" x1="0" x2={w} y1={h - 0.5} y2={h - 0.5} vectorEffect="non-scaling-stroke" />
                                {hist.map((r, i) => r.off > 0 && (
                                    <rect key={r.date} className="v2-cd-bar" x={i * bw + bw * 0.15} width={bw * 0.7} y={h - (r.off / top) * h} height={(r.off / top) * h} />
                                ))}
                            </svg>
                        </div>
                        <div className="v2-sh-row"><span>{day(hist[0].date)}</span><span>Most off {worst.off} · {day(worst.date)}</span><span>{day(hist.at(-1).date)}</span></div>
                    </>
                )}
                <p className="v2-aa-foot">{hist.length > 1 ? 'Off centre per night · the' : 'The'} conditions are on the Centred tab</p>
                <a className="v2-cd-link" href={CENTRED_TAB} target="_blank" rel="noopener noreferrer">Open Centred tab ↗</a>
            </div>
        </div>,
        document.querySelector('.v2') || document.body,
    );
}
