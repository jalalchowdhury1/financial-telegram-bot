'use client';
import { useMemo, useState } from 'react';
import { fmtPct, perYear, yahooUrl } from './FactorRow';

/**
 * v2's factor strip as a race (FactorRow `race`): all five factors on ONE chart against the
 * S&P (the dashed 0 line), the spotlit one drawn bright and the rest quiet, beside a ranked
 * board. Hover (or tap) a board row to spotlight it; the leader is lit by default. The foot
 * line spells the spotlit factor out: its own return vs SPY's, since when, and a Yahoo link.
 * Same payload as the chips (/api/factors: windows[win].rel / f / b / from / spark).
 */
const YEARS = { '3Y': 3, '5Y': 5, '10Y': 10, '20Y': 20, '30Y': 30, '40Y': 40 };
const W = 1000, H = 200, PAD = 12;
const sign = (v) => (v > 0.05 ? 'up' : v < -0.05 ? 'down' : 'flat');
const since = (iso) => (iso ? new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' }) : '');

export default function FactorRace({ factors, win, isLong, note, stale, loading }) {
    const [hover, setHover] = useState(null);
    const [pinned, setPinned] = useState(null);
    const ranked = useMemo(() => factors.filter((f) => f.windows?.[win]).sort((a, b) => b.windows[win].rel - a.windows[win].rel), [factors, win]);
    const geo = useMemo(() => {
        const sparks = ranked.map((f) => f.windows[win].spark).filter((s) => Array.isArray(s) && s.length > 1);
        if (!sparks.length) return null;
        const all = sparks.flat();
        const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
        const span = hi - lo || 1;
        const y = (v) => PAD + (1 - (v - lo) / span) * (H - 2 * PAD);
        const path = (s) => s.map((v, i) => `${i ? 'L' : 'M'}${((i / (s.length - 1)) * W).toFixed(1)},${y(v).toFixed(1)}`).join('');
        return { y, zero: y(0), lines: Object.fromEntries(ranked.map((f) => [f.key, f.windows[win].spark?.length > 1 ? path(f.windows[win].spark) : null])) };
    }, [ranked, win]);
    if (loading || !ranked.length) return <div className="fr-race is-waiting" aria-busy="true" />;

    const spot = ranked.find((f) => f.key === (hover || pinned)) || ranked[0];
    const sw = spot.windows[win];
    const max = Math.max(...ranked.map((f) => Math.abs(f.windows[win].rel)), 1);
    const years = YEARS[win];
    const end = geo && sw.spark?.length > 1 ? geo.y(sw.spark[sw.spark.length - 1]) : null;
    // The end tag sits above the dot unless the line's own last stretch climbs into that spot.
    const below = end != null && geo.y(Math.max(...sw.spark.slice(-Math.ceil(sw.spark.length / 4)))) < end - 12;
    return (
        <>
            <div className="fr-race">
                <div className={`fr-plot tone-${sign(sw.rel)}`}>
                    {geo && (
                        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
                            <line className="fr-zero" x1="0" x2={W} y1={geo.zero} y2={geo.zero} />
                            {ranked.map((f) => f.key !== spot.key && geo.lines[f.key] && <path key={f.key} className="fr-line" d={geo.lines[f.key]} />)}
                            {geo.lines[spot.key] && <path key={`a${win}${spot.key}`} className="fr-area" d={`${geo.lines[spot.key]}L${W},${geo.zero}L0,${geo.zero}Z`} />}
                            {geo.lines[spot.key] && <path key={`l${win}${spot.key}`} className="fr-line is-spot" pathLength="1" d={geo.lines[spot.key]} />}
                        </svg>
                    )}
                    {geo && <span className="fr-zero-tag" style={{ top: `${(geo.zero / H) * 100}%` }}>S&amp;P 500</span>}
                    {end != null && <span className={`fr-end${below ? ' is-below' : ''}`} style={{ top: `${(end / H) * 100}%` }}><i />{spot.label} {fmtPct(sw.rel)}</span>}
                </div>
                <ol className="fr-board" onPointerLeave={() => setHover(null)}>
                    {ranked.map((f, i) => {
                        const r = f.windows[win].rel;
                        const w = (Math.abs(r) / max) * 50;
                        const py = years ? perYear(r, years) : null;
                        return (
                            <li key={f.key}>
                                <button type="button" className={`fr-row${f.key === spot.key ? ' is-spot' : ''}`} aria-pressed={f.key === spot.key}
                                    onPointerEnter={(e) => { if (e.pointerType === 'mouse') setHover(f.key); }}
                                    onFocus={() => setHover(f.key)} onBlur={() => setHover(null)}
                                    onClick={() => setPinned(f.key)}>
                                    <span className="fr-rank">{i + 1}</span>
                                    <span className="fr-name">{f.label}</span>
                                    <span className="fr-bar"><i className={`tone-${sign(r)}`} style={{ left: `${r >= 0 ? 50 : 50 - w}%`, width: `${w}%` }} /></span>
                                    <span className={`fr-val tone-${sign(r)}`}>{fmtPct(r)}</span>
                                    {years && <span className="fr-py">{py != null ? `${fmtPct(py)}/yr` : ''}</span>}
                                </button>
                            </li>
                        );
                    })}
                </ol>
            </div>
            <div className="fr-foot">
                {stale && <span className="fr-stale">🕐 stale · </span>}
                <b>{spot.label}</b> <span className="fr-tk">{isLong ? spot.longProxy || 'research portfolio' : spot.ticker} </span>{fmtPct(sw.f)} vs {isLong ? 'market' : 'SPY'} {fmtPct(sw.b)} · since {isLong ? sw.from?.slice(0, 4) : since(sw.from)}
                {!isLong && <> · <a href={yahooUrl(spot.ticker)} target="_blank" rel="noopener noreferrer">{spot.ticker} ↗</a></>}
                {note && <span className="fr-note"> · {note}</span>}
            </div>
        </>
    );
}
