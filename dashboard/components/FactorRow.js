'use client';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { getJson } from '../lib/loadJson';
import { readSnap, writeSnap, savedLabel } from '../lib/snapshot';

const useIsoLayoutEffect = typeof window !== 'undefined' ? useLayoutEffect : useEffect;
const hasFactors = (d) => !!(d && Array.isArray(d.factors) && d.factors.length);

/**
 * 🧬 Factor row — a thin strip under the top indicator bar. Five style factors,
 * each as a price ratio vs the S&P 500: the number is how far $1 in the factor
 * ETF is ahead of (or behind) $1 in SPY over the chosen window; the sparkline is
 * that gap over time (0 = the window start, dashed). One timeline control drives
 * every chip; the choice is remembered on this device.
 *
 * Details live in ONE caption line under the chips (mouse hover or keyboard focus)
 * rather than in tooltips — tooltips anchored to 68px-wide phone chips bleed off-screen.
 * Clicking / tapping a chip opens that ETF on Yahoo Finance in a new tab (2026-09-26).
 *
 * 20Y / 30Y / 40Y use a different basis (the ETFs are too young): Ken French
 * research portfolios vs the whole US market, total return, monthly. The caption
 * says so whenever one of those windows is on screen (lib/factorsLong.js).
 *
 * Self-fetching (like the vol table) so a factor outage can never touch the rest
 * of the page; re-fetches when the page's refresh cycle ticks (`refreshKey`; a
 * manual refresh also passes `bust` to skip the edge cache).
 * Renders nothing if the route has no factors at all.
 *
 * Keyboard: the timeline is a radio group — ←/→ (and Home/End) move the choice.
 */

export const WINDOWS = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', '10Y', '20Y', '30Y', '40Y'];
export const LONG_WINDOWS = new Set(['20Y', '30Y', '40Y']);
const LONG_YEARS = { '20Y': 20, '30Y': 30, '40Y': 40 };
export const DEFAULT_WINDOW = '6M';
const LONG_STALE_MONTHS = 4; // Ken French is normally 1–2 months behind
const LS_KEY = 'ftb:factorWindow';
const STALE_DAYS = 5;
const MIN_REFETCH_MS = 60e3;

// 40-year totals run to five digits (+12,887%) — drop the decimal once it is noise.
export const fmtPct = (v) => {
    if (v == null || !Number.isFinite(v)) return '—';
    const a = Math.abs(v);
    const body = a >= 1000 ? Math.round(a).toLocaleString('en-US') : a >= 100 ? a.toFixed(0) : a.toFixed(1);
    return `${v > 0 ? '+' : v < 0 ? '−' : ''}${body}%`;
};
/** Relative return over `years` → the same gap per year, compounded (e.g. −15% over 20Y ≈ −0.8%/yr). */
export const perYear = (rel, years) => (Number.isFinite(rel) && years > 0 && rel > -100 ? (Math.pow(1 + rel / 100, 1 / years) - 1) * 100 : null);
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const fmtMonth = (ym) => {
    const m = /^(\d{4})-(\d{2})/.exec(ym || '');
    return m ? `${MONTHS[Number(m[2]) - 1]} ${m[1]}` : '';
};
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

/** The 20Y+ history is monthly and normally 1–2 months behind; stale only past LONG_STALE_MONTHS. */
export function isLongStale(data, now = new Date()) {
    const m = /^(\d{4})-(\d{2})$/.exec(data?.long?.through || '');
    if (!m) return false;
    const behind = (now.getUTCFullYear() - Number(m[1])) * 12 + (now.getUTCMonth() + 1 - Number(m[2]));
    return behind > LONG_STALE_MONTHS;
}

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

export default function FactorRow({ initialData = null, refreshKey = null, bust = false }) {
    const [data, setData] = useState(initialData);
    const [status, setStatus] = useState(initialData ? 'ready' : 'loading');
    // ⚡ Instant open: this device's saved copy, tagged 🕐 until the live answer lands.
    const [savedAt, setSavedAt] = useState(null);
    useIsoLayoutEffect(() => {
        if (initialData) return;
        const snap = readSnap('factors');
        if (snap && hasFactors(snap.data)) { setData(snap.data); setStatus('ready'); setSavedAt(snap.savedAt); }
    }, []);
    const [win, setWin] = useState(DEFAULT_WINDOW);
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
        // The floor stops the first-load double fetch (mount, then the page's first tick);
        // a manual refresh (`bust`) always goes through.
        if (!bust && Date.now() - lastFetch.current < MIN_REFETCH_MS) return undefined;
        lastFetch.current = Date.now();
        // Automatic loads may be answered by the edge cache (lib/cdn.js); a manual refresh
        // busts it. getJson never throws: null = no answer (timeout / network / bad JSON).
        getJson('/api/factors', { bust }).then((d) => {
            if (!mounted.current) return;
            // Keep what we had if a refresh comes back empty.
            if (hasFactors(d)) {
                setData(d); setStatus('ready'); setSavedAt(null);
                writeSnap('factors', d); // small payload: synchronous is fine
            }
            else setStatus((s) => (s === 'ready' ? 'ready' : 'empty'));
        });
        return undefined;
    }, [refreshKey]); // `bust` changes together with each tick

    const factors = data?.factors || [];
    const available = useMemo(() => new Set(WINDOWS.filter((w) => factors.some((f) => f.windows?.[w]))), [factors]);
    // If the remembered window has no data (e.g. only 2y of history survived an outage), show the nearest one that does.
    const activeWin = available.has(win) ? win : (WINDOWS.slice().reverse().find((w) => available.has(w) && WINDOWS.indexOf(w) < WINDOWS.indexOf(win)) || [...available][0] || win);
    const { leader, laggard } = rankFactors(factors, activeWin);
    const isLong = LONG_WINDOWS.has(activeWin);
    const stale = isLong ? isLongStale(data) : isStale(data);

    // Phones scroll the 11-button timeline sideways: keep the active one in view
    // (scrollLeft only — scrollIntoView would also yank the PAGE up to this row).
    const tfRef = useRef(null);
    const markEnd = () => {
        const box = tfRef.current;
        if (box) box.classList.toggle('at-end', box.scrollLeft + box.clientWidth >= box.scrollWidth - 2);
    };
    useEffect(() => {
        const box = tfRef.current;
        const btn = box?.querySelector('[aria-checked="true"]');
        if (box && btn && box.scrollWidth > box.clientWidth) {
            box.scrollLeft = btn.offsetLeft - (box.clientWidth - btn.offsetWidth) / 2;
        }
        markEnd();
    }, [activeWin, status]);

    if (status === 'empty' || (status === 'ready' && !factors.length)) return null;

    const pick = (w) => {
        setWin(w);
        try { window.localStorage.setItem(LS_KEY, w); } catch { /* ignore */ }
    };
    const enabled = (w) => status !== 'ready' || available.has(w);
    // Radio-group keys: ←/→ step through the ENABLED windows, Home/End jump to the ends.
    const onTfKey = (e) => {
        const list = WINDOWS.filter(enabled);
        const i = list.indexOf(activeWin);
        let next = null;
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = list[Math.min(list.length - 1, i + 1)];
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = list[Math.max(0, i - 1)];
        else if (e.key === 'Home') next = list[0];
        else if (e.key === 'End') next = list[list.length - 1];
        if (!next) return;
        e.preventDefault();
        pick(next);
        tfRef.current?.querySelector(`[data-win="${next}"]`)?.focus();
    };

    const focusKey = hovered;
    const focus = factors.find((f) => f.key === focusKey);
    const asOfText = data?.asOf ? `through ${fmtDate(data.asOf)}` : '';

    let caption;
    if (status === 'loading') {
        caption = 'Loading factor data…';
    } else if (focus) {
        const w = focus.windows?.[activeWin];
        const py = isLong ? perYear(w?.rel, LONG_YEARS[activeWin]) : null;
        caption = w && isLong ? (
            <>
                <strong>{focus.label}</strong> ({focus.longProxy || 'research portfolio'}) {w.rel >= 0 ? 'beat' : 'lagged'} the whole US market by{' '}
                <span className={`factor-${tone(w.rel)}`}>{fmtPct(Math.abs(w.rel)).replace(/^\+/, '')}</span>
                {py != null && <> (≈{fmtPct(py)}/yr)</>} since {fmtDateY(w.from)}: {fmtPct(w.f)} vs {fmtPct(w.b)}, dividends included.{' '}
                <span className="factor-what">{focus.what}.</span>
            </>
        ) : w ? (
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
                <span className="hide-sm"> · hover a chip for details, click for Yahoo Finance</span>
            </>
        );
    } else {
        caption = 'No factor data for this window.';
    }

    return (
        <section className="factor-row" aria-label="Factor performance versus the S&P 500" data-jump="Factors" data-cached={savedAt ? savedLabel(savedAt) : undefined}>
            <div className="factor-head">
                <div className="factor-title">
                    <span className="emoji">🧬</span>Factors<span className="factor-sub"> vs S&amp;P 500</span>
                </div>
                <div className="factor-tf" role="radiogroup" aria-label="Timeline" ref={tfRef} onKeyDown={onTfKey} onScroll={markEnd}>
                    {WINDOWS.map((w) => (
                        <button
                            key={w}
                            type="button"
                            role="radio"
                            data-win={w}
                            aria-checked={activeWin === w}
                            tabIndex={activeWin === w ? 0 : -1}
                            className={`factor-tf-btn${activeWin === w ? ' active' : ''}${LONG_WINDOWS.has(w) ? ' is-long' : ''}`}
                            disabled={!enabled(w)}
                            title={!enabled(w) ? `${w}: history unavailable right now`
                                : LONG_WINDOWS.has(w) ? `Show ${w} (research portfolios, total return, monthly)` : `Show ${w}`}
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
                        <a
                            key={f.key}
                            href={yahooUrl(f.ticker)}
                            target="_blank"
                            rel="noopener noreferrer"
                            className={`factor-chip${focusKey === f.key ? ' is-selected' : ''}${isLeader ? ' is-leader' : ''}`}
                            title={`${f.ticker} on Yahoo Finance`}
                            aria-label={`${w ? `${f.label}: ${fmtPct(w.rel)} versus the S&P 500 over ${activeWin}` : `${f.label}: no data for ${activeWin}`}. Opens ${f.ticker} on Yahoo Finance`}
                            // Caption preview for a real mouse or the keyboard only: a touch
                            // "hover" would stick after the tap that opens Yahoo.
                            onPointerEnter={(e) => { if (e.pointerType === 'mouse') setHovered(f.key); }}
                            onPointerLeave={(e) => { if (e.pointerType === 'mouse') setHovered(null); }}
                            onFocus={() => setHovered(f.key)}
                            onBlur={() => setHovered(null)}
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
                        </a>
                    );
                })}
            </div>

            <div className="factor-caption" aria-live="polite">
                {stale && <span className="factor-stale">🕐 stale · </span>}
                {caption}
                {status === 'ready' && !focus && isLong && data?.long?.through && (
                    <span className="factor-asof"> · research portfolios vs whole market, total return, through {fmtMonth(data.long.through)}</span>
                )}
                {status === 'ready' && !focus && !isLong && asOfText && (
                    <span className="factor-asof"> · price ratio, {asOfText}</span>
                )}
            </div>
        </section>
    );
}

/** The ETF's Yahoo Finance quote page. */
export const yahooUrl = (ticker) => `https://finance.yahoo.com/quote/${encodeURIComponent(ticker || 'SPY')}/`;

// Same tickers as lib/factors.js FACTORS, so a chip links correctly before the data lands.
const PLACEHOLDERS = [
    { key: 'value', label: 'Value', short: 'Value', ticker: 'VLUE' },
    { key: 'momentum', label: 'Momentum', short: 'Mom.', ticker: 'MTUM' },
    { key: 'quality', label: 'Quality', short: 'Quality', ticker: 'QUAL' },
    { key: 'size', label: 'Small caps', short: 'Size', ticker: 'IWM' },
    { key: 'lowvol', label: 'Low vol', short: 'Low vol', ticker: 'USMV' },
];
