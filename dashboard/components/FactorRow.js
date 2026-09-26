'use client';
import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * 🧬 Factor row — a thin strip under the top indicator bar. Five style factors,
 * each as a price ratio vs the S&P 500: the number is how far $1 in the factor
 * ETF is ahead of (or behind) $1 in SPY over the chosen window; the sparkline is
 * that gap over time (0 = the window start, dashed). One timeline control drives
 * every chip; the choice is remembered on this device.
 *
 * Details live in ONE caption line under the chips (hover or tap a chip) rather
 * than in tooltips — tooltips anchored to 68px-wide phone chips bleed off-screen.
 *
 * Self-fetching (like the vol table) so a factor outage can never touch the rest
 * of the page; re-fetches when the page's refresh cycle ticks (`refreshKey`).
 * Renders nothing if the route has no factors at all.
 */

export const WINDOWS = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', '10Y'];
export const DEFAULT_WINDOW = '6M';
const LS_KEY = 'ftb:factorWindow';
const STALE_DAYS = 5;
const MIN_REFETCH_MS = 60e3;

const fmtPct = (v) => (v == null || !Number.isFinite(v) ? '—' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}%`);
const fmtDate = (iso) => {
    if (!iso) return '';
    const d = new Date(`${iso}T12:00:00Z`);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
};
const fmtDateY = (iso) => {
    if (!iso) return '';
    const d = new Date(`${iso}T12:00:00Z`);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
};
const tone = (v) => (v == null ? 'muted' : v > 0.05 ? 'up' : v < -0.05 ? 'down' : 'flat');

export function isStale(data, now = new Date()) {
    if (!data) return false;
    if (data._meta?.stale) return true;
    if (!data.asOf) return false;
    const age = (now.getTime() - Date.parse(`${data.asOf}T21:00:00Z`)) / 864e5;
    return age > STALE_DAYS;
}

/** Leader and laggard for a window, among factors that have it. */
export function rankFactors(factors, win) {
    const have = (factors || []).filter((f) => f.windows?.[win]);
    if (!have.length) return { leader: null, laggard: null };
    const sorted = [...have].sort((a, b) => b.windows[win].rel - a.windows[win].rel);
    return { leader: sorted[0], laggard: sorted.length > 1 ? sorted[sorted.length - 1] : null };
}

export function Sparkline({ values, toneClass }) {
    if (!Array.isArray(values) || values.length < 2) return <span className="factor-spark factor-spark-empty" aria-hidden="true" />;
    const W = 100;
    const H = 30;
    const pad = 2;
    let lo = Math.min(0, ...values);
    let hi = Math.max(0, ...values);
    if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
    const x = (i) => (i / (values.length - 1)) * W;
    const y = (v) => pad + (1 - (v - lo) / (hi - lo)) * (H - 2 * pad);
    const pts = values.map((v, i) => `${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
    const zero = y(0).toFixed(2);
    return (
        <svg className={`factor-spark ${toneClass}`} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
            <line x1="0" x2={W} y1={zero} y2={zero} className="factor-spark-zero" vectorEffect="non-scaling-stroke" />
            <polygon points={`0,${zero} ${pts} ${W},${zero}`} className="factor-spark-fill" />
            <polyline points={pts} className="factor-spark-line" vectorEffect="non-scaling-stroke" />
        </svg>
    );
}

export default function FactorRow({ initialData = null, refreshKey = null }) {
    const [data, setData] = useState(initialData);
    const [status, setStatus] = useState(initialData ? 'ready' : 'loading');
    const [win, setWin] = useState(DEFAULT_WINDOW);
    const [selected, setSelected] = useState(null);
    const [hovered, setHovered] = useState(null);
    const lastFetch = useRef(initialData ? Date.now() : 0);
    // A mounted flag instead of a per-effect `alive`: under React StrictMode (dev) the
    // effect runs, is torn down, and re-runs inside MIN_REFETCH_MS — a per-effect flag
    // would drop the only response and the row would sit on "Loading" forever.
    const mounted = useRef(true);
    useEffect(() => {
        mounted.current = true;
        return () => { mounted.current = false; };
    }, []);

    // Remembered window (per device). Storage can be blocked — never let it throw.
    useEffect(() => {
        try {
            const w = window.localStorage.getItem(LS_KEY);
            if (WINDOWS.includes(w)) setWin(w);
        } catch { /* private mode etc. */ }
    }, []);

    useEffect(() => {
        if (Date.now() - lastFetch.current < MIN_REFETCH_MS) return undefined;
        lastFetch.current = Date.now();
        fetch(`/api/factors?_t=${Date.now()}`, { cache: 'no-store' })
            .then((r) => r.json())
            .then((d) => {
                if (!mounted.current) return;
                // Keep what we had if a refresh comes back empty.
                if (d && Array.isArray(d.factors) && d.factors.length) { setData(d); setStatus('ready'); }
                else setStatus((s) => (s === 'ready' ? 'ready' : 'empty'));
            })
            .catch(() => { if (mounted.current) setStatus((s) => (s === 'ready' ? 'ready' : 'empty')); });
        return undefined;
    }, [refreshKey]);

    const factors = data?.factors || [];
    const available = useMemo(() => new Set(WINDOWS.filter((w) => factors.some((f) => f.windows?.[w]))), [factors]);
    // If the remembered window has no data (e.g. only 2y of history survived an outage), show the nearest one that does.
    const activeWin = available.has(win) ? win : (WINDOWS.slice().reverse().find((w) => available.has(w) && WINDOWS.indexOf(w) < WINDOWS.indexOf(win)) || [...available][0] || win);
    const { leader, laggard } = rankFactors(factors, activeWin);
    const stale = isStale(data);

    if (status === 'empty' || (status === 'ready' && !factors.length)) return null;

    const pick = (w) => {
        setWin(w);
        try { window.localStorage.setItem(LS_KEY, w); } catch { /* ignore */ }
    };

    const focusKey = hovered || selected;
    const focus = factors.find((f) => f.key === focusKey);
    const asOfText = data?.asOf ? `through ${fmtDate(data.asOf)}` : '';

    let caption;
    if (status === 'loading') {
        caption = 'Loading factor data…';
    } else if (focus) {
        const w = focus.windows?.[activeWin];
        caption = w ? (
            <>
                <strong>{focus.label}</strong> ({focus.ticker}) {w.rel >= 0 ? 'beat' : 'lagged'} the S&amp;P by{' '}
                <span className={`factor-${tone(w.rel)}`}>{Math.abs(w.rel).toFixed(1)}%</span> since {fmtDateY(w.from)}:{' '}
                {focus.ticker} {fmtPct(w.f)} vs SPY {fmtPct(w.b)}. <span className="factor-what">{focus.what}.</span>
            </>
        ) : (
            <><strong>{focus.label}</strong>: not enough history for {activeWin}.</>
        );
    } else if (leader) {
        caption = (
            <>
                {activeWin} leader <strong>{leader.label}</strong>{' '}
                <span className={`factor-${tone(leader.windows[activeWin].rel)}`}>{fmtPct(leader.windows[activeWin].rel)}</span>
                {laggard && (
                    <>
                        {' · '}laggard <strong>{laggard.label}</strong>{' '}
                        <span className={`factor-${tone(laggard.windows[activeWin].rel)}`}>{fmtPct(laggard.windows[activeWin].rel)}</span>
                    </>
                )}
                <span className="hide-sm"> · tap a chip for details</span>
            </>
        );
    } else {
        caption = 'No factor data for this window.';
    }

    return (
        <section className="factor-row" aria-label="Factor performance versus the S&P 500">
            <div className="factor-head">
                <div className="factor-title">
                    <span className="emoji">🧬</span>Factors<span className="factor-sub"> vs S&amp;P 500</span>
                </div>
                <div className="factor-tf" role="radiogroup" aria-label="Timeline">
                    {WINDOWS.map((w) => (
                        <button
                            key={w}
                            type="button"
                            role="radio"
                            aria-checked={activeWin === w}
                            className={`factor-tf-btn${activeWin === w ? ' active' : ''}`}
                            disabled={status === 'ready' && !available.has(w)}
                            title={status === 'ready' && !available.has(w) ? `${w}: history unavailable right now` : `Show ${w}`}
                            onClick={() => pick(w)}
                        >
                            {w}
                        </button>
                    ))}
                </div>
            </div>

            <div className="factor-grid">
                {(factors.length ? factors : PLACEHOLDERS).map((f) => {
                    const w = f.windows?.[activeWin];
                    const t = tone(w?.rel);
                    const isLeader = leader && leader.key === f.key && factors.length > 1;
                    return (
                        <button
                            key={f.key}
                            type="button"
                            className={`factor-chip${selected === f.key ? ' is-selected' : ''}${isLeader ? ' is-leader' : ''}`}
                            aria-pressed={selected === f.key}
                            aria-label={w ? `${f.label}: ${fmtPct(w.rel)} versus the S&P 500 over ${activeWin}` : `${f.label}: no data for ${activeWin}`}
                            onClick={() => setSelected((s) => (s === f.key ? null : f.key))}
                            // Hover preview for a real mouse only: a touch "hover" sticks after the
                            // tap, so tapping a chip again could never clear its caption.
                            onPointerEnter={(e) => { if (e.pointerType === 'mouse') setHovered(f.key); }}
                            onPointerLeave={(e) => { if (e.pointerType === 'mouse') setHovered(null); }}
                        >
                            <span className="factor-label">
                                <span className="factor-name-long">{f.label}</span>
                                <span className="factor-name-short">{f.short}</span>
                                {f.stale && <span className="factor-clock" title={`Data through ${fmtDate(f.asOf)}`}> 🕐</span>}
                            </span>
                            <span className="factor-line">
                                <span className={`factor-val factor-${status === 'loading' ? 'muted' : t}`}>{status === 'loading' ? '…' : fmtPct(w?.rel)}</span>
                                <Sparkline values={w?.spark} toneClass={`factor-${t}`} />
                            </span>
                        </button>
                    );
                })}
            </div>

            <div className="factor-caption" aria-live="polite">
                {stale && <span className="factor-stale">🕐 stale · </span>}
                {caption}
                {status === 'ready' && !focus && asOfText && (
                    <span className="factor-asof"> · price ratio, {asOfText}</span>
                )}
            </div>
        </section>
    );
}

const PLACEHOLDERS = [
    { key: 'value', label: 'Value', short: 'Value' },
    { key: 'momentum', label: 'Momentum', short: 'Mom.' },
    { key: 'quality', label: 'Quality', short: 'Quality' },
    { key: 'size', label: 'Small caps', short: 'Size' },
    { key: 'lowvol', label: 'Low vol', short: 'Low vol' },
];
