'use client';
/**
 * The Centred tile's details (tap the tile): the table (every setting a marble: in the middle, at an edge, hanging
 * off), three spirit levels for what moving the off-centre settings would do to the whole algo, and a nights strip. The public feed holds no conditions: those are
 * on the nuts-radar Centred tab, one tap away.
 */
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import useSheetLock, { useSheetFocus } from './useSheetLock';
import { day } from './SignalStudy';

export const CENTRED_TAB = 'https://nuts-radar.vercel.app/#centred';
const WHY = { mid: 'Small changes to these move nothing', edge: 'A cliff sits right beside these', off: 'A steadier spot was proven nearby' };   // tap a count
const NIGHTS = 30;                                  // the nights strip: one cell per night, newest last

const signed = (v, d = 1) => `${v < 0 ? '−' : v > 0 ? '+' : ''}${Math.abs(v).toFixed(d)}`;
const LEVELS = [['Gain', 'cagr', 5], ['Worst fall', 'maxdd', 5], ['Sharpe', 'sharpe', 0.25]];   // label, delta key, full tilt
/**
 * Three spirit levels, centred version minus live: the bubble sits dead centre when moving the settings changes
 * nothing, drifts right when it is better, left when worse. x = -1..1 along the vial.
 */
export function levels(delta) {
    if (!delta) return null;
    return LEVELS.map(([label, k, full]) => {
        const v = delta[k], flat = Math.abs(v) < (k === 'sharpe' ? 0.005 : 0.05);
        const text = flat ? 'same' : k === 'cagr' ? `${signed(v)} %/yr` : k === 'maxdd' ? `${Math.abs(v).toFixed(1)} pts ${v > 0 ? 'smaller' : 'bigger'}` : signed(v, 2);
        return { label, text, x: flat ? 0 : Math.max(-1, Math.min(1, v / full)), tone: flat ? '' : v > 0 ? 'is-good' : 'is-bad' };
    });
}

/**
 * The table: every setting is a marble on a flat top. In the middle = small changes move nothing; at an edge = a
 * cliff beside it; off centre = hanging over the lip. Counts only, so which marble is which setting stays private.
 */
export function marbles(c) {
    const off = Math.min(c.off, c.checked), on = c.checked - off, edge = Math.min(c.edge || 0, on);
    const L = 22, W = 276, slot = on ? W / on : W, r = Math.max(1.5, Math.min(4, slot / 2 - 0.6)), left = Math.ceil(edge / 2);
    const out = Array.from({ length: on }, (_, i) => ({ x: L + (i + 0.5) * slot, y: 58 - r, r, kind: i < left || i >= on - (edge - left) ? 'edge' : 'mid' }));
    for (let i = 0; i < off; i++) {                                        // alternate lips; a second one on a lip rides on the first
        const side = i % 2 ? 1 : -1, k = Math.floor(i / 2);
        out.push({ x: (side < 0 ? 14 : 306) + side * (1 + k * 1.5), y: 58 - 4 - k * 8.5, r: 4, kind: 'off', side });
    }
    return { list: out, mid: on - edge, edge, off };
}

/** One line for the tile and the popup head. */
export function centredHeadline(c) {
    if (c.state === 'unknown') return 'Centred · not checked';
    return c.off > 0 ? `Off centre · ${c.off} of ${c.checked}` : c.edge > 0 ? `Centred · ${c.checked - c.edge} of ${c.checked} in the middle` : `Centred · all ${c.checked} in the middle`;
}

export default function CentredDetails({ centred: c, onClose }) {
    const sheetRef = useRef(null), backRef = useRef(null);
    const [sel, setSel] = useState(null);                                  // the count tapped: its marbles light, one line says what it means
    useSheetLock(true);
    useSheetFocus(true, sheetRef);
    useEffect(() => {
        const k = (e) => { if (e.key === 'Escape') onClose(); };
        document.addEventListener('keydown', k);
        return () => document.removeEventListener('keydown', k);
    }, [onClose]);

    const hist = (c.history || []).slice(-NIGHTS);
    const worst = hist.reduce((a, b) => (b.off > (a?.off ?? -1) ? b : a), null);
    const verdict = c.delta ? c.verdict || 'not judged' : null;
    const lv = levels(c.delta), m = c.state !== 'unknown' && c.checked > 0 ? marbles(c) : null;

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
                {m && (
                    <>
                        <svg className={`v2-cd-table${sel ? ` sel-${sel}` : ''}`} viewBox="0 30 320 62" role="img"
                            aria-label={`${m.mid} settings in the middle, ${m.edge} at an edge, ${m.off} off centre`}>
                            <path className="v2-cd-mesa" d="M14 92V62q0-4 4-4h284q4 0 4 4v30" />
                            {m.off > 0 && <line className="v2-cd-fall" x1="8" x2="8" y1="66" y2="90" />}
                            {m.off > 1 && <line className="v2-cd-fall" x1="312" x2="312" y1="66" y2="90" />}
                            <line className="v2-cd-flat" x1="52" x2="268" y1="58" y2="58" />
                            <text className="v2-cd-zone" x="160" y="76" textAnchor="middle">FLAT</text>
                            <text className="v2-cd-zone" x="24" y="76">EDGE</text>
                            <text className="v2-cd-zone" x="296" y="76" textAnchor="end">EDGE</text>
                            {m.list.map((b, i) => (
                                <circle key={i} className={`v2-cd-ball is-${b.kind}${b.side > 0 ? ' is-right' : ''}`} cx={b.x} cy={b.y} r={b.r}
                                    style={{ animationDelay: `${Math.min(i, 40) * 18}ms`, transformOrigin: `${b.side ? (b.side < 0 ? 14 : 306) : b.x}px 58px` }} />
                            ))}
                        </svg>
                        <div className="v2-cd-three">
                            {[['mid', m.mid, 'in the middle'], ['edge', m.edge, 'at an edge'], ['off', m.off, 'off centre']].map(([k, n, t]) => (
                                <button key={k} type="button" className={`is-${k}${sel === k ? ' is-sel' : ''}`} aria-pressed={sel === k} onClick={() => setSel(sel === k ? null : k)}><b>{n}</b>{t}</button>
                            ))}
                        </div>
                        {sel && <p className="v2-cd-why">{WHY[sel]}</p>}
                    </>
                )}
                <div className="v2-cd-compare">
                    <span>{lv ? 'If moved to the middle' : 'If moved: not judged yet'}</span>
                    {verdict && <span className={`v2-cd-chip is-${verdict.replace(' ', '-')}`}>{verdict}</span>}
                    {c.worlds > 0 && <span className="v2-cd-worlds">{c.worlds.toLocaleString('en-US')} worlds</span>}
                </div>
                {lv && (
                    <div className="v2-cd-three v2-cd-levels">
                        {lv.map((l) => (
                            <span key={l.label} className={l.tone}>
                                <i className="v2-cd-vial"><i style={{ left: `${50 + l.x * 40}%` }} /></i>
                                <em>{l.label}</em>{l.text}
                            </span>
                        ))}
                    </div>
                )}
                {hist.length > 0 && (
                    <>
                        <div className="v2-cd-nights" role="img"
                            aria-label={`Off centre per night, ${day(hist[0].date)} to ${day(hist.at(-1).date)}: most ${worst.off} on ${day(worst.date)}`}>
                            {Array.from({ length: NIGHTS }, (_, i) => hist[i]).map((r, i) => (
                                <i key={i} className={!r ? undefined : r.off > 0 ? 'is-off' : 'is-ok'} style={r?.off > 0 ? { opacity: 0.45 + 0.55 * (r.off / Math.max(1, worst.off)) } : undefined} />
                            ))}
                        </div>
                        <div className="v2-sh-row">
                            <span>{hist.length} {hist.length === 1 ? 'night' : 'nights'} checked</span>
                            <span>{worst.off > 0 ? `Most off ${worst.off} · ${day(worst.date)}` : 'Never off centre'}</span>
                        </div>
                    </>
                )}
                <p className="v2-aa-foot">The conditions are on the Centred tab</p>
                <a className="v2-cd-link" href={CENTRED_TAB} target="_blank" rel="noopener noreferrer">Open Centred tab ↗</a>
            </div>
        </div>,
        document.querySelector('.v2') || document.body,
    );
}
