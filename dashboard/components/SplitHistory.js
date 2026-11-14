'use client';
/**
 * The split tile's history (tap the tile): Main's share of the Main + Shartino money, one point a close, against
 * the target. The Mac's log starts on `logFrom`; the days before it are the same totals rolled BACKWARDS on the
 * replay curves, true only if no money moved then, so the chart draws them dashed and says so.
 */
import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import useSheetLock, { useSheetFocus } from './useSheetLock';
import { day } from './SignalStudy';

export default function SplitHistory({ split, onClose }) {
    const backRef = useRef(null), sheetRef = useRef(null);
    useSheetLock(true);
    useSheetFocus(true, sheetRef);
    useEffect(() => {
        const k = (e) => { if (e.key === 'Escape') onClose(); };
        document.addEventListener('keydown', k);
        return () => document.removeEventListener('keydown', k);
    }, [onClose]);

    const pts = split.history.map(([date, value]) => ({ date, value }));
    const vals = pts.map((p) => p.value), T = split.target;
    const lo = Math.min(...vals, T) - 1, hi = Math.max(...vals, T) + 1;
    const w = 320, h = 120;
    const x = (i) => (i / (pts.length - 1)) * w, y = (v) => h - ((v - lo) / (hi - lo)) * h;
    const path = (a, b) => pts.slice(a, b).map((p, i) => `${x(a + i).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ');
    const cut = Math.max(0, pts.findIndex((p) => p.date >= split.logFrom)); // first day of the real log
    const top = pts.reduce((a, b) => (b.value > a.value ? b : a)), low = pts.reduce((a, b) => (b.value < a.value ? b : a));
    const first = pts[0], last = pts[pts.length - 1];
    const pc = (v) => `${v.toFixed(1)}%`;

    if (typeof document === 'undefined') return null;
    return createPortal(
        <div ref={backRef} className="v2-aa-back" role="dialog" aria-modal="true" aria-labelledby="v2-sh-title"
            onClick={(e) => { if (e.target === backRef.current) onClose(); }}>
            <div className="v2-aa" ref={sheetRef} data-sheet-scroll="">
                <div className="v2-aa-head">
                    <div className="v2-aa-title">
                        <h2 id="v2-sh-title">Main / Shartino split</h2>
                        <p>Main {pc(last.value)} now · target {T}%</p>
                    </div>
                    <button type="button" className="v2-aa-close sheet-x" onClick={onClose} aria-label="Close">×</button>
                </div>
                <div className="v2-sh">
                    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img"
                        aria-label={`Main's share from ${day(first.date)} to ${day(last.date)}: ${pc(first.value)} to ${pc(last.value)}, high ${pc(top.value)}, low ${pc(low.value)}`}>
                        <line className="v2-sh-target" x1="0" x2={w} y1={y(T)} y2={y(T)} vectorEffect="non-scaling-stroke" />
                        {cut > 0 && <polyline className="v2-sh-line is-guess" points={path(0, cut + 1)} vectorEffect="non-scaling-stroke" />}
                        {cut < pts.length - 1 && <polyline className="v2-sh-line" points={path(cut, pts.length)} vectorEffect="non-scaling-stroke" />}
                    </svg>
                    <span className="v2-sh-tag" style={{ top: `${(y(T) / h) * 100}%` }}>{T}%</span>
                </div>
                <div className="v2-sh-row"><span>{day(first.date)} · {pc(first.value)}</span><span>{day(last.date)} · {pc(last.value)}</span></div>
                <div className="v2-sh-row"><span>High {pc(top.value)} · {day(top.date)}</span><span>Low {pc(low.value)} · {day(low.date)}</span></div>
                {cut > 0 && <p className="v2-aa-foot">Dashed = before {day(split.logFrom)}: estimate, as if no money moved</p>}
                <p className="v2-aa-foot">Estimate from the night shift&apos;s replay · one point a close</p>
            </div>
        </div>,
        document.querySelector('.v2') || document.body,
    );
}
