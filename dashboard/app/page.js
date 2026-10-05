'use client';
import { useState, useEffect, useLayoutEffect, useRef, useMemo } from 'react';
import ErrorBoundary from '../components/ErrorBoundary';
import { freshnessNote, formatAsOf } from '../lib/freshness';

import Gauge from '../components/Gauge';
import Skeleton from '../components/Skeleton';
import SpyChart from '../components/SpyChart';
import MiniChart from '../components/MiniChart';
import MarketPulse from '../components/MarketPulse';
import JevPills from '../components/JevPills';
import CustomIndicatorBar from '../components/CustomIndicatorBar';
import EconomicIndicatorGrid from '../components/EconomicIndicatorGrid';
import FourHorsemen from '../components/FourHorsemen';
import BullChecklist from '../components/BullChecklist';
import ExtraMarketsGrid from '../components/ExtraMarketsGrid';
import PolymarketTable from '../components/PolymarketTable';
import VolMetricsTable from '../components/VolMetricsTable';
import RubberBandRadar from '../components/RubberBandRadar';
import FactorRow from '../components/FactorRow';
import Delta from '../components/Delta';
import MarkChip from '../components/MarkChip';
import { MarkProvider, useMark, collectLiveValues } from '../components/MarkProvider';
import JumpNav from '../components/JumpNav';
import GlanceBar from '../components/GlanceBar';
import BackPill from '../components/BackPill';
import WhatMoved from '../components/WhatMoved';
import { UpdatedAgo, OfflineBanner, PullToRefresh } from '../components/PhonePolish';
import { readSnap, writeSnap, savedLabel, purgeOldSnaps, isLiveAnswer } from '../lib/snapshot';
import { readSeen, writeSeen, mergeSeen, pickSeen, SINCE_MIN_GAP_MS } from '../lib/lastVisit';
import SinceLastVisit from '../components/SinceLastVisit';
import MarketClock from '../components/MarketClock';
import { movedWhen } from '../lib/whatMoved';
import { fgHistoryCells } from '../lib/fgHistory';

// useLayoutEffect warns during the static prerender; on the server nothing runs anyway.
const useIsoLayoutEffect = typeof window !== 'undefined' ? useLayoutEffect : useEffect;
import { getJson } from '../lib/loadJson';

// The page's own feeds (self-fetching cards — factors, vol, rubber band, polymarket —
// load themselves). Order is only the order requests start in.
const FEEDS = [
    { key: 'spy', path: '/api/spy' },
    { key: 'sheets', path: '/api/sheets' },
    { key: 'spyDailyMove', path: '/api/spy-daily-move' },
    { key: 'fg', path: '/api/fear-greed' },
    { key: 'fred', path: '/api/fred' },
    { key: 'extra', path: '/api/market-extra' },
    // Baselines for the fresh-print marks: if it fails the digest is null, no marks
    // render, and every number reads exactly as it does without them.
    { key: 'history', path: '/api/history' },
    // Jev pills: a failure hides the row, nothing else.
    { key: 'jev', path: '/api/jev-pills' },
    // VIX since its last close, for the "What moved" strip. The vol card fetches its own
    // copy; automatic loads of both are answered by the edge cache.
    { key: 'vol', path: '/api/vol' },
];

// Fresh-print marks compare live numbers with the history sheet; while any of these
// is a saved copy, a mark would dress up an old number as news — marks stay off.
const MARK_INPUTS = ['history', 'fred', 'extra', 'sheets'];

/**
 * A hero number that can carry a fresh-print mark. Lives here rather than in the JSX
 * because `useMark` is a hook and the cards are rendered inline in Dashboard.
 * A stale value is never marked — an old number reappearing is not a new print.
 */
function HeroValue({ markKey, raw, stale, format, style, children }) {
    const mark = useMark(markKey, raw);
    return (
        <div className="hero-price" style={style}>
            <Delta mark={stale ? null : mark} format={format} chartKey={markKey} raw={raw}>{children}</Delta>
        </div>
    );
}

// ============ MAIN DASHBOARD ============
export default function Dashboard() {
    const [sheets, setSheets] = useState(null);
    const [spyDailyMove, setSpyDailyMove] = useState(null);
    const [spy, setSpy] = useState(null);
    const [fg, setFg] = useState(null);
    const [fred, setFred] = useState(null);
    const [extraMarkets, setExtraMarkets] = useState(null);
    const [loading, setLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState(null);
    // +1 per finished fetchAll. `lastUpdated` is minute-resolution, so two refreshes in
    // the same minute would not re-key the factor row or the card error boundaries.
    const [refreshTick, setRefreshTick] = useState(0);
    const [systemStatus, setSystemStatus] = useState(null);
    const [apiErrors, setApiErrors] = useState([]);
    const [refreshing, setRefreshing] = useState(false);
    const [history, setHistory] = useState(null);
    // Jev regime pills (2026-09-19). Null = the route failed or JEV_PILLS=off, and the
    // component renders nothing — the page then reads exactly as it did before.
    const [jevPills, setJevPills] = useState(null);
    const [vol, setVol] = useState(null);
    // 📡 Rubber Band verdict for Market Pulse: undefined = not answered yet, null = failed.
    // RubberBandRadar fetches its own route and hands its answer up (no extra FEEDS row).
    const [rubberBand, setRubberBand] = useState(undefined);
    // ⚡ Instant open: feed key → savedAt (ms) while that feed is showing this device's
    // saved copy (lib/snapshot.js). A key leaves the map when its live answer lands.
    const [savedAt, setSavedAt] = useState({});
    const savedRef = useRef({});
    const [updatedAt, setUpdatedAt] = useState(null); // ms, drives "3 min ago"
    // 👋 The numbers seen on the last visit (lib/lastVisit.js), read once before any live
    // answer can overwrite them; re-read when the tab comes back after an hour away. `at`
    // = when it was read: only feeds that land live AFTER it count as seen on this visit.
    const [seenBase, setSeenBase] = useState({ rec: null, at: 0 });
    const readBase = () => setSeenBase({ rec: readSeen(), at: Date.now() });
    // feed key → ms its last LIVE answer landed (the time a seen number is stamped with).
    const [landedAt, setLandedAt] = useState({});
    // Refresh behaviour: `loading` is the FIRST load only (the header badge reads
    // "Loading live data..." until every feed has answered once); cards key off their
    // own feed via `pending` below. Every
    // later fetch is a background refresh — the page keeps showing what it has,
    // and only the header spinner moves. Before this, the 5-minute auto-refresh
    // collapsed all 12 cards to skeletons for ~10s while you were reading.
    const hasLoadedRef = useRef(false);
    const lastFetchRef = useRef(0);
    const inFlightRef = useRef(false);
    // A refresh asked for while one is in flight (the reconnect after a dropped signal, a
    // pull) runs once when it finishes, instead of being dropped until the 5-min tick.
    const queuedRef = useRef(false);
    // Which feeds have not answered even once yet. Each card waits only for ITS OWN
    // feed: before this, one Promise.all held every card on skeletons until the
    // slowest route (market-extra, ~5 s cold) answered — the SPY price took 5.6–9.6 s
    // to appear on a phone although /api/spy answers in ~1.5 s.
    const [pending, setPending] = useState(() => Object.fromEntries(FEEDS.map((f) => [f.key, true])));
    // A manual refresh skips the edge cache; the factor row follows the same choice.
    const [bustKey, setBustKey] = useState(false);

    const setters = {
        sheets: setSheets, spy: setSpy, spyDailyMove: setSpyDailyMove, fg: setFg, fred: setFred,
        extra: setExtraMarkets, history: setHistory, jev: setJevPills, vol: setVol,
    };
    const dropSaved = (key) => {
        if (!(key in savedRef.current)) return;
        const next = { ...savedRef.current };
        delete next[key];
        savedRef.current = next;
        setSavedAt(next);
    };

    // Paint the last visit's numbers before the first frame; live ones replace them feed
    // by feed. Runs before fetchAll (a layout effect precedes every plain effect).
    useIsoLayoutEffect(() => {
        readBase();
        purgeOldSnaps();
        const got = {};
        for (const { key } of FEEDS) {
            const snap = readSnap(key);
            if (!snap) continue;
            setters[key](snap.data);
            got[key] = snap.savedAt;
        }
        if (Object.keys(got).length) { savedRef.current = got; setSavedAt(got); }
    }, []);

    async function fetchAll({ bust = false } = {}) {
        if (inFlightRef.current) { if (bust) queuedRef.current = true; return; }
        inFlightRef.current = true;
        if (!hasLoadedRef.current) setLoading(true);
        setRefreshing(true);
        setApiErrors([]);
        setBustKey(bust);
        lastFetchRef.current = Date.now();
        const got = {};
        let live = 0;
        try {
            // Every feed lands on its own. null = the fetch itself failed (the routes
            // answer 200 even when degraded): keep the previous payload rather than
            // blanking a card that had data. history + jev-pills failures are silent by
            // design — no fresh-print marks / no pills row, everything else unchanged.
            await Promise.all(FEEDS.map(async ({ key, path }) => {
                const res = await getJson(path, { bust });
                got[key] = res;
                // A failure answer (an error, or a route's 200 fallback body) never replaces
                // a saved copy on screen — it stays, tagged 🕐 — and is never saved.
                if (isLiveAnswer(key, res)) {
                    live++;
                    setters[key](res);
                    dropSaved(key);
                    setLandedAt((p) => ({ ...p, [key]: Date.now() }));
                    setTimeout(() => writeSnap(key, res), 0); // off the render path
                } else if (res != null && !(key in savedRef.current)) {
                    setters[key](res); // no saved copy: the card shows the route's own degraded state
                }
                setPending((p) => (p[key] ? { ...p, [key]: false } : p));
            }));
            hasLoadedRef.current = true;
            const { sheets: sheetsRes, spy: spyRes, fg: fgRes, fred: fredRes, extra: extraRes } = got;

            setSystemStatus({
                spy: spyRes?._meta,
                fred: fredRes?._meta,
                fg: fgRes?._meta,
                sheets: sheetsRes?._meta,
                extra: extraRes?._meta
            });

            const errors = [];
            if (sheetsRes?.error) errors.push(`[SHEETS] ${sheetsRes.error}`);
            if (spyRes?.error) errors.push(`[SPY] ${spyRes.error}`);
            if (fgRes?.error) errors.push(`[F&G] ${fgRes.error}`);
            if (fredRes?.error) errors.push(`[FRED] ${fredRes.error}`);
            if (extraRes?.error) errors.push(`[EXTRA] ${extraRes.error}`);
            setApiErrors(errors);

            // "Updated …" only when something live actually landed: a cycle where every feed
            // failed (offline, a phone waking up) must not make saved numbers look fresh.
            const now = new Date();
            if (live) {
                const year = now.getFullYear();
                const month = String(now.getMonth() + 1).padStart(2, '0');
                const day = String(now.getDate()).padStart(2, '0');
                const hours = String(now.getHours()).padStart(2, '0');
                const minutes = String(now.getMinutes()).padStart(2, '0');
                setLastUpdated(`${year}-${month}-${day} ${hours}:${minutes}`);
                setUpdatedAt(now.getTime());
            }
            setRefreshTick((t) => t + 1);
        } catch (e) {
            console.error('Dashboard fetch error:', e);
            setApiErrors(prev => [...prev, `[NETWORK] ${e.toString()}`]);
        }
        setLoading(false);
        setRefreshing(false);
        inFlightRef.current = false;
        if (queuedRef.current) { queuedRef.current = false; setTimeout(() => fetchAll({ bust: true }), 0); }
    }
    const refreshNow = () => fetchAll({ bust: true });

    // The explanatory tooltips are pure CSS :hover, which does not exist on touch —
    // so on a phone every as-of date and metric explanation was unreachable, while the
    // header cheerfully said "hover any number for its date". Tapping a trigger now
    // toggles the same tooltip; tapping elsewhere, or Esc, dismisses it.
    useEffect(() => {
        const closeAll = (except) => document.querySelectorAll('.tooltip-trigger.tooltip-open')
            .forEach((el) => { if (el !== except) el.classList.remove('tooltip-open'); });
        const onClick = (e) => {
            const trigger = e.target.closest?.('.tooltip-trigger');
            closeAll(trigger);
            if (trigger) trigger.classList.toggle('tooltip-open');
        };
        const onKey = (e) => { if (e.key === 'Escape') closeAll(null); };
        document.addEventListener('click', onClick);
        document.addEventListener('keydown', onKey);
        return () => {
            document.removeEventListener('click', onClick);
            document.removeEventListener('keydown', onKey);
        };
    }, []);

    // Desktop shortcut: R refreshes (same as the button, so it skips the edge cache) —
    // unless you are typing, or it is Cmd/Ctrl+R (the browser's own reload).
    useEffect(() => {
        const onKey = (e) => {
            if ((e.key !== 'r' && e.key !== 'R') || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
            const t = e.target;
            if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || ''))) return;
            fetchAll({ bust: true });
        };
        document.addEventListener('keydown', onKey);
        return () => document.removeEventListener('keydown', onKey);
    }, []);

    useEffect(() => {
        const REFRESH_MS = 5 * 60 * 1000;
        fetchAll();
        // Auto-refresh every 5 minutes — but not while the tab is hidden. /api/fred
        // alone is ~650KB, so an idle background tab was pulling ~8MB/hour. On
        // returning to the tab, refresh immediately if the data has gone stale.
        // Offline: skip the tick — the offline banner refreshes on reconnect.
        const interval = setInterval(() => {
            if (!document.hidden && navigator.onLine !== false) fetchAll();
        }, REFRESH_MS);
        const onVisible = () => {
            if (document.hidden) return;
            // Back after an hour or more = a new visit: compare against what was last seen.
            if (Date.now() - lastFetchRef.current >= SINCE_MIN_GAP_MS) readBase();
            if (navigator.onLine !== false && Date.now() - lastFetchRef.current > REFRESH_MS) fetchAll();
        };
        document.addEventListener('visibilitychange', onVisible);
        return () => {
            clearInterval(interval);
            document.removeEventListener('visibilitychange', onVisible);
        };
    }, []);

    const fgSegments = [
        { start: 0, end: 25, color: '#dc2626' },
        { start: 25, end: 45, color: '#f97316' },
        { start: 45, end: 55, color: '#525252' },
        { start: 55, end: 75, color: '#22c55e' },
        { start: 75, end: 100, color: '#15803d' },
    ];

    const rsiSegments = [
        { start: 0, end: 30, color: '#22c55e' },
        { start: 30, end: 70, color: '#3f3f46' },
        { start: 70, end: 100, color: '#dc2626' },
    ];

    const statusColor = (s) => s === 'safe' || s === 'healthy' || s === 'strong' || s === 'tight' || s === 'easy' || s === 'rising' ? 'stat-positive' : s === 'danger' || s === 'weak' || s === 'stressed' || s === 'restrictive' || s === 'falling' ? 'stat-negative' : 'stat-neutral';

    const fgColor = (score) => score < 25 ? 'var(--red)' : score < 45 ? '#f97316' : score < 55 ? 'var(--text-muted)' : score < 75 ? 'var(--green)' : '#15803d';

    // "10:42" label for the OLDEST saved copy among `keys` (undefined = all live).
    const saved = (...keys) => {
        const ts = keys.map((k) => savedAt[k]).filter(Number.isFinite);
        return ts.length ? savedLabel(Math.min(...ts)) : undefined;
    };
    // Same, as the CSS variable a `display: contents` wrapper hands to its card.
    const savedVar = (...keys) => {
        const l = saved(...keys);
        return l ? { '--saved': JSON.stringify(l) } : null;
    };
    const anySaved = saved(...Object.keys(savedAt));
    // 👋 What is on screen now, LIVE only and landed on THIS visit — a saved copy, or the
    // numbers left on screen while the tab was hidden, are not a new look.
    const liveOnly = (key, v) => (!(key in savedAt) && landedAt[key] > seenBase.at ? v : null);
    const seenNow = useMemo(() => pickSeen(
        { spy: liveOnly('spy', spy), vol: liveOnly('vol', vol), fg: liveOnly('fg', fg), extra: liveOnly('extra', extraMarkets) },
        collectLiveValues(liveOnly('fred', fred), liveOnly('extra', extraMarkets), liveOnly('sheets', sheets)),
    ), [spy, vol, fg, extraMarkets, fred, sheets, savedAt, landedAt, seenBase.at]); // eslint-disable-line react-hooks/exhaustive-deps
    useEffect(() => {
        if (!updatedAt || !Object.values(seenNow.v).some((x) => x != null)) return;
        writeSeen(mergeSeen(readSeen(), seenNow, Date.now(), landedAt));
    }, [seenNow, updatedAt]); // eslint-disable-line react-hooks/exhaustive-deps
    const marksOff = MARK_INPUTS.some((k) => k in savedAt);
    // The SPY move's session, named the way What moved names it: "today" during the
    // session, "Fri" all weekend and before Monday's open. Unknown market date = "today".
    const spyWhen = useMemo(() => { try { return movedWhen({ spy, vol }) || 'today'; } catch { return 'today'; } }, [spy, vol]);

    return (
        <MarkProvider history={marksOff ? null : history} series={history?.series || null}>
        <OfflineBanner onBack={refreshNow} since={anySaved || lastUpdated?.slice(11)} />
        <PullToRefresh onRefresh={refreshNow} busy={refreshing} />
        <div className="dashboard">
            {/* Auto-Refresh Visualizer */}
            {lastUpdated && <div key={lastUpdated} className="auto-refresh-bar" style={{ animation: 'progress-fill 300s linear forwards' }}></div>}

            {/* HEADER */}
            <header className="dashboard-header">
                <h1>Jalal's Financial Dashboard</h1>
                <p className="subtitle">Live market data, economic indicators & AI-powered assessment</p>
                <div className="header-status">
                    {/* amber, not live-green, while the page shows a saved copy */}
                    <div className={`live-badge${anySaved && (loading || !updatedAt) ? ' is-saved' : ''}`}>
                        <span className="live-dot" />
                        {loading ? (anySaved ? `🕐 Saved ${anySaved} · loading live…` : 'Loading live data...') : !updatedAt ? (
                            anySaved ? `🕐 Saved ${anySaved} · no live data yet` : 'No live data yet'
                        ) : (
                            <>
                                Updated
                                {/* the clock is hidden on phones — "3 min ago" says it in less
                                    room; the full string pushed this badge into the refresh button */}
                                <span className="upd-date">{lastUpdated} · </span>
                                <UpdatedAgo at={updatedAt} />
                            </>
                        )}
                    </div>
                    <ErrorBoundary resetKey={refreshTick}><MarkChip values={collectLiveValues(fred, extraMarkets, sheets)} /></ErrorBoundary>
                    <button className="refresh-btn" onClick={refreshNow} disabled={refreshing} title="Refresh all data (R)" aria-label="Refresh all data">
                        <svg className={refreshing ? 'spinning' : ''} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="23 4 23 10 17 10" />
                            <polyline points="1 20 1 14 7 14" />
                            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                        </svg>
                    </button>
                </div>
                {/* 🕰️ NYSE open/closed + countdown, computed on the device */}
                <div className="mkt-clock-row"><ErrorBoundary><MarketClock /></ErrorBoundary></div>
                {fred?._meta?.fetchedAt && (
                    <p className="subtitle" style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: '6px' }}>
                        Economic data as of {new Date(fred._meta.fetchedAt).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}<span className="hide-sm"> · refreshes every 30 min</span> · tap a label for its date, a number for its 90-day chart
                    </p>
                )}
            </header>

            {/* 👋 SINCE LAST VISIT — what changed since the numbers you last saw (≥ 1 h ago) */}
            <ErrorBoundary resetKey={refreshTick}><SinceLastVisit base={seenBase.rec} live={seenNow} /></ErrorBoundary>

            {/* 📈 WHAT MOVED — the 5 most unusual moves since the last close; tap → card */}
            <ErrorBoundary resetKey={refreshTick}>
                <WhatMoved
                    spy={spy} fg={fg} extra={extraMarkets} vol={vol} history={history}
                    saved={saved('spy', 'fg', 'extra', 'vol', 'history')}
                    waiting={['spy', 'fg', 'extra', 'vol', 'history'].some((k) => pending[k])}
                />
            </ErrorBoundary>

            {/* CUSTOM INDICATOR BAR */}
            <div className="saved-wrap" data-cached={saved('sheets')} style={{ display: 'contents', ...savedVar('sheets') }}>
                <ErrorBoundary resetKey={refreshTick}><CustomIndicatorBar sheets={sheets} loading={pending.sheets && !sheets} /></ErrorBoundary>
            </div>

            {/* 🧬 FACTOR ROW — style factors vs the S&P 500 with one shared timeline.
                Self-fetching (/api/factors); renders nothing if the route has no data. */}
            <ErrorBoundary resetKey={refreshTick}>
                <FactorRow refreshKey={refreshTick} bust={bustKey} />
            </ErrorBoundary>

            {/* MARKET PULSE - Quick summary at top */}
            <div className="saved-wrap" data-cached={saved('fred', 'vol')} style={{ display: 'contents', ...savedVar('fred', 'vol') }}>
                {/* hold = a cold open where fred or vol has nothing on screen yet: paint the chips in
                    one go, never one landing in front of another under his thumb */}
                <ErrorBoundary resetKey={refreshTick}><MarketPulse fred={fred} vol={vol} rubberBand={rubberBand}
                    saved={{ fred: saved('fred'), vol: saved('vol') }}
                    waiting={pending.fred || pending.vol}
                    hold={(pending.fred && !fred) || (pending.vol && !vol)} /></ErrorBoundary>
            </div>

            {/* JEV REGIME PILLS — hidden entirely when JEV_PILLS=off or the route is unreachable.
                No loading skeleton on purpose: with the kill switch on, a skeleton would flash
                for a few seconds on every load and the page would NOT be exactly the old site. */}
            <ErrorBoundary resetKey={refreshTick}>
                <div className="saved-wrap" data-cached={saved('jev')} style={{ display: 'contents', ...savedVar('jev') }}>
                    <JevPills data={jevPills} />
                </div>
            </ErrorBoundary>

            {/* MAIN GRID */}
            <div className="dashboard-grid">

                {/* ========== REDESIGNED SPY CARD ========== */}
                <div className={`card${spy && !spy.error && (spy.rsi < 30 || spy.rsi > 70) ? ' card-alert' : ''}`} style={{ animationDelay: '0.2s' }} data-jump="SPY overview" data-cached={saved('spy', 'spyDailyMove')}>
                    <div className="card-header">
                        <h2>📊 SPY Market Overview</h2>
                        {spy && !spy.error && <span className={`badge ${spy.rsi > 70 ? 'badge-red' : spy.rsi < 30 ? 'badge-green' : 'badge-blue'}`}>{spy.rsi > 70 ? 'Overbought' : spy.rsi < 30 ? 'Oversold' : 'Neutral'}</span>}
                    </div>
                    <ErrorBoundary resetKey={refreshTick}>
                        {!spy || spy.error ? <Skeleton count={5} /> : (
                            <>
                                {/* Hero price */}
                                <div className="hero-price-section">
                                    <div className="hero-price">${spy.current.toFixed(2)}</div>
                                    {spyDailyMove?.value ? (
                                        <div className={`daily-change-badge ${parseFloat(spyDailyMove.value) >= 0 ? 'daily-up' : 'daily-down'}`}>
                                            {parseFloat(spyDailyMove.value) >= 0 ? '▲' : '▼'} {spyDailyMove.value} {spyWhen}
                                        </div>
                                    ) : spy.dailyChange && (
                                        <div className={`daily-change-badge ${spy.dailyChange.pct >= 0 ? 'daily-up' : 'daily-down'}`}>
                                            {spy.dailyChange.pct >= 0 ? '▲' : '▼'} ${Math.abs(spy.dailyChange.value).toFixed(2)} ({spy.dailyChange.pct >= 0 ? '+' : ''}{spy.dailyChange.pct.toFixed(2)}%) {spyWhen}
                                        </div>
                                    )}
                                    <div className={`hero-change ${spy.ma200.pct >= 0 ? 'stat-positive' : 'stat-negative'}`} style={{ marginTop: '6px' }}>
                                        {spy.ma200.pct >= 0 ? '▲' : '▼'} {Math.abs(spy.ma200.pct).toFixed(2)}% {spy.ma200.pct >= 0 ? 'above' : 'below'} 200d MA
                                    </div>
                                    <div className={`hero-change`} style={{ color: spy.week52High.pct >= -1 ? 'var(--green)' : 'var(--yellow)', fontSize: '0.78rem', marginTop: '2px' }}>
                                        {spy.week52High.pct >= 0 ? '🔥 At 52-week high' : `${spy.week52High.pct.toFixed(2)}% from 52wk high ($${spy.week52High.value.toFixed(2)})`}
                                    </div>
                                </div>

                                {/* Stats grid */}
                                <div className="stats-mini-grid">
                                    <div className="stat-mini">
                                        <span className="stat-mini-label">200d MA</span>
                                        <span className="stat-mini-value">${spy.ma200.value.toFixed(2)}</span>
                                    </div>
                                    <div className="stat-mini">
                                        <span className="stat-mini-label">52w High</span>
                                        <span className={`stat-mini-value ${spy.week52High.pct >= 0 ? 'stat-positive' : 'stat-negative'}`}>${spy.week52High.value.toFixed(2)}</span>
                                    </div>
                                    <div className="stat-mini">
                                        <span className="stat-mini-label">3Y Return</span>
                                        <span className={`stat-mini-value ${spy.return3y == null ? '' : spy.return3y >= 0 ? 'stat-positive' : 'stat-negative'}`}>{spy.return3y == null ? 'N/A' : `${spy.return3y >= 0 ? '+' : ''}${spy.return3y.toFixed(2)}%`}</span>
                                    </div>
                                    <div className="stat-mini">
                                        <span className="stat-mini-label">9d RSI</span>
                                        <span className={`stat-mini-value ${spy.rsi > 70 ? 'stat-negative' : spy.rsi < 30 ? 'stat-positive' : ''}`}>{spy.rsi.toFixed(2)}</span>
                                    </div>
                                </div>

                                {/* RSI Gauge */}
                                <div className="gauge-section">
                                    <Gauge score={spy.rsi} segments={rsiSegments} labels={[0, 30, 50, 70, 100]} />
                                    <div className="gauge-inline-label">
                                        RSI: <strong>{spy.rsi.toFixed(2)}</strong>
                                        <span style={{ marginLeft: '8px', color: spy.rsi > 70 ? 'var(--red)' : spy.rsi < 30 ? 'var(--green)' : 'var(--text-muted)', fontSize: '0.7rem' }}>
                                            {spy.rsi > 70 ? 'OVERBOUGHT' : spy.rsi < 30 ? 'OVERSOLD' : 'NEUTRAL'}
                                        </span>
                                    </div>
                                </div>
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* ========== REDESIGNED FEAR & GREED CARD ========== */}
                <div className="card" style={{ animationDelay: '0.3s' }} data-jump="Fear & Greed" data-cached={saved('fg')}>
                    <div className="card-header">
                        <h2>😨 Fear & Greed Index</h2>
                        {fg && !fg.error && <span className={`badge ${fg.score < 45 ? 'badge-red' : fg.score > 55 ? 'badge-green' : 'badge-yellow'}`}>{fg.rating}</span>}
                    </div>
                    <ErrorBoundary resetKey={refreshTick}>
                        {!fg || fg.error ? <Skeleton type="gauge" /> : (
                            <>
                                {/* Hero score */}
                                <div className="hero-price-section">
                                    <div className="hero-price" style={{ color: fgColor(fg.score) }}>{Math.round(fg.score)}</div>
                                    <div className="hero-change" style={{ color: fgColor(fg.score) }}>{fg.rating}</div>
                                </div>

                                {/* Gauge */}
                                <div className="gauge-section">
                                    <Gauge score={fg.score} segments={fgSegments} labels={[0, 25, 50, 75, 100]} />
                                </div>

                                {/* Historical — a cell the backup tiers leave empty ('N/A', null) reads "—" */}
                                <div className="fg-history">
                                    {fgHistoryCells(fg).map(({ label, val, diff }) => {
                                        const arrow = diff > 0 ? '▲' : diff < 0 ? '▼' : '—';
                                        const arrowColor = diff > 0 ? 'var(--green)' : diff < 0 ? 'var(--red)' : 'var(--text-muted)';
                                        return (
                                            <div key={label} className="fg-history-item">
                                                <div className="fg-history-label">{label}</div>
                                                <div className="fg-history-value">
                                                    {val ?? '—'}
                                                    {diff != null && (
                                                        <span style={{ marginLeft: '6px', fontSize: '0.7rem', color: arrowColor, fontWeight: 600 }}>
                                                            {arrow}{Math.abs(diff)}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* YIELD CURVE */}
                <div className="card" style={{ animationDelay: '0.4s' }} data-jump="Yield curve" data-cached={saved('fred')}>
                    <div className="card-header">
                        <h2><span className="tooltip-trigger" data-tooltip={`When the 2-year yield is higher than the 10-year, it is a classic recession warning.${freshnessNote({ value: fred?.yieldCurve?.current, asOf: fred?.yieldCurve?.asOf, stale: fred?.yieldCurve?.stale }).suffix}`}>📈 Yield Curve (10Y-2Y)</span></h2>
                        {fred?.yieldCurve?.current != null && <span className={`badge ${fred.yieldCurve.current >= 0 ? 'badge-green' : 'badge-red'}`}>{fred.yieldCurve.current >= 0 ? 'Positive' : 'Inverted'}</span>}
                    </div>
                    <ErrorBoundary resetKey={refreshTick}>
                        {!fred ? <Skeleton count={2} /> : (fred.error || fred.yieldCurve?.current == null) ? (
                            <div className="hero-price-section">
                                <div className="hero-price" style={{ fontSize: '2.2rem', color: 'var(--yellow)' }}>N/A</div>
                                <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                    Unavailable — source busy, try again shortly
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="hero-price-section">
                                    <HeroValue markKey="yieldCurve" raw={fred.yieldCurve.current}
                                        stale={fred.yieldCurve.stale}
                                        format={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`}
                                        style={{ fontSize: '2.2rem', color: fred.yieldCurve.stale ? 'var(--orange)' : fred.yieldCurve.current >= 0 ? 'var(--green)' : 'var(--red)' }}>
                                        {fred.yieldCurve.stale ? '🕐 ' : ''}{fred.yieldCurve.current >= 0 ? '+' : ''}{fred.yieldCurve.current.toFixed(2)}%
                                    </HeroValue>
                                    {fred.yieldCurve.stale && (
                                        <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                            Last data {formatAsOf(fred.yieldCurve.asOf)} (stale)
                                        </div>
                                    )}
                                </div>
                                <MiniChart history={fred.yieldCurve.history} color="#818cf8" gradientId="yieldGrad" showZero={true} recessions={fred.recessions || []} />
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* PROFIT MARGIN */}
                <div className="card" style={{ animationDelay: '0.45s' }} data-jump="Profit margin" data-cached={saved('fred')}>
                    <div className="card-header">
                        <h2><span className="tooltip-trigger" data-tooltip={`Corporate Profits / GDP: High margins indicate strong corporate pricing power.${freshnessNote({ value: fred?.profitMargin?.current, asOf: fred?.profitMargin?.asOf, stale: fred?.profitMargin?.stale }).suffix}`}>💰 Profit Margin</span></h2>
                        {fred?.profitMargin?.current != null && <span className="badge badge-blue">Corp Profits / GDP</span>}
                    </div>
                    <ErrorBoundary resetKey={refreshTick}>
                        {!fred ? <Skeleton count={2} /> : (fred.error || fred.profitMargin?.current == null) ? (
                            <div className="hero-price-section">
                                <div className="hero-price" style={{ fontSize: '2.2rem', color: 'var(--yellow)' }}>N/A</div>
                                <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                    Unavailable — source busy, try again shortly
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="hero-price-section">
                                    <HeroValue markKey="profitMargin" raw={fred.profitMargin.current}
                                        stale={fred.profitMargin.stale} format={(v) => `${v.toFixed(2)}%`}
                                        style={{ fontSize: '2.2rem', color: fred.profitMargin.stale ? 'var(--orange)' : 'var(--green)' }}>
                                        {fred.profitMargin.stale ? '🕐 ' : ''}{fred.profitMargin.current.toFixed(2)}%
                                    </HeroValue>
                                    {fred.profitMargin.stale && (
                                        <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                            Last data {formatAsOf(fred.profitMargin.asOf)} (stale)
                                        </div>
                                    )}
                                </div>
                                <MiniChart history={fred.profitMargin.history} color="#22c55e" gradientId="profitGrad" recessions={fred.recessions || []} />
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* S&P 500 EPS */}
                <div className="card" style={{ animationDelay: '0.5s' }} data-jump="S&P 500 EPS" data-cached={saved('fred')}>
                    <div className="card-header">
                        <h2><span className="tooltip-trigger" data-tooltip={`S&P 500 earnings per share, trailing 12 months (as-reported) — the E in P/E. Rising EPS means corporate America is earning more. History is inflation-adjusted (today's dollars).${freshnessNote({ value: fred?.spEps?.current, asOf: fred?.spEps?.asOf, stale: fred?.spEps?.stale }).suffix}`}>🧾 S&P 500 EPS</span></h2>
                        {fred?.spEps?.current != null && <span className="badge badge-blue">Trailing 12M</span>}
                    </div>
                    <ErrorBoundary resetKey={refreshTick}>
                        {!fred ? <Skeleton count={2} /> : (fred.error || fred.spEps?.current == null) ? (
                            <div className="hero-price-section">
                                <div className="hero-price" style={{ fontSize: '2.2rem', color: 'var(--yellow)' }}>N/A</div>
                                <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                    Unavailable — source busy, try again shortly
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="hero-price-section">
                                    {/* spEps has no daily snapshot, so it carries no mark — see lib/marks.js */}
                                    <HeroValue markKey={undefined} raw={fred.spEps.current}
                                        stale={fred.spEps.stale} format={(v) => `$${v.toFixed(2)}`}
                                        style={{ fontSize: '2.2rem', color: fred.spEps.stale ? 'var(--orange)' : 'var(--green)' }}>
                                        {fred.spEps.stale ? '🕐 ' : ''}${fred.spEps.current.toFixed(2)}
                                    </HeroValue>
                                    {fred.spEps.stale && (
                                        <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                            Last data {formatAsOf(fred.spEps.asOf)} (stale)
                                        </div>
                                    )}
                                </div>
                                <MiniChart history={fred.spEps.history} color="#38bdf8" gradientId="epsGrad" recessions={fred.recessions || []} cadence="monthly" />
                            </>
                        )}
                    </ErrorBoundary>
                </div>

                {/* ECONOMIC INDICATORS */}
                <div className="saved-wrap" data-jump="Economy" data-cached={saved('fred')} style={{ display: 'contents', ...savedVar('fred') }}><ErrorBoundary resetKey={refreshTick}><EconomicIndicatorGrid fred={fred} loading={pending.fred && !fred} statusColor={statusColor} /></ErrorBoundary></div>

                {/* FOUR HORSEMEN — RECESSION WATCH (full width) */}
                <div className="saved-wrap" data-jump="Recession watch" data-cached={saved('fred')} style={{ display: 'contents', ...savedVar('fred') }}><ErrorBoundary resetKey={refreshTick}><FourHorsemen fred={fred} loading={pending.fred && !fred} /></ErrorBoundary></div>

                {/* RUBBER BAND RADAR — is the dip-buying regime alive? (full width, nightly from the Mac mini) */}
                <div data-jump="Rubber band" style={{ display: 'contents' }}><ErrorBoundary resetKey={refreshTick}><RubberBandRadar onVerdict={setRubberBand} /></ErrorBoundary></div>

                {/* SPY HISTORICAL CHART */}
                <div className="card" style={{ animationDelay: '0.55s' }} data-jump="SPY chart" data-cached={saved('spy')}>
                    <div className="card-header">
                        <h2>📈 SPY Historical</h2>
                        <span className="badge badge-blue">Price + 200d MA</span>
                    </div>
                    <ErrorBoundary resetKey={refreshTick}>
                        {!spy || spy.error || !spy.chartHistory ? <Skeleton count={4} /> : (
                            <SpyChart chartHistory={spy.chartHistory} recessions={fred?.recessions || []} current={spy.current} />
                        )}
                    </ErrorBoundary>
                </div>

                {/* VOLATILITY METRICS (IV rank / percentile / VRP) */}
                <div data-jump="Volatility" style={{ display: 'contents' }}><ErrorBoundary resetKey={refreshTick}><VolMetricsTable refreshKey={refreshTick} bust={bustKey} /></ErrorBoundary></div>

                {/* BULL MARKET CHECKLIST */}
                <div className="saved-wrap" data-jump="Bull checklist" data-cached={saved('fred')} style={{ display: 'contents', ...savedVar('fred') }}><ErrorBoundary resetKey={refreshTick}><BullChecklist fred={fred} loading={pending.fred && !fred} /></ErrorBoundary></div>

                {/* EXTRA MARKETS GRID */}
                <div className="saved-wrap" data-jump="Markets" data-cached={saved('extra')} style={{ display: 'contents', ...savedVar('extra') }}><ErrorBoundary resetKey={refreshTick}><ExtraMarketsGrid data={extraMarkets} loading={pending.extra && !extraMarkets} /></ErrorBoundary></div>

                {/* POLYMARKET TABLE - Integrated in grid naturally */}
                <div style={{ gridColumn: '1 / -1' }} data-jump="Polymarket">
                    <ErrorBoundary resetKey={refreshTick}><PolymarketTable /></ErrorBoundary>
                </div>

                {/* FINANCIAL DASHBOARD HISTORY LINK */}
                <div style={{ gridColumn: '1 / -1', textAlign: 'center', marginTop: '1rem' }}>
                    <a
                        href="https://docs.google.com/spreadsheets/d/1lA-_yjLMc3qDTt9sogSPQrCohNULIk5wwJYfb5wIHfc/edit?gid=0#gid=0"
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textDecoration: 'none', opacity: 0.5 }}
                    >
                        Financial Dashboard History
                    </a>
                </div>
            </div>

            {/* 🧭 Jump menu — floating, appears once you scroll past the first screen */}
            <JumpNav />

            {/* 🔝 Glance bar — SPY · F&G · age · ↻, floats in once Market Pulse scrolls off */}
            <ErrorBoundary resetKey={refreshTick}>
                <GlanceBar
                    spy={spy} spyDailyMove={spyDailyMove} fg={fg} fgColor={fgColor} when={spyWhen}
                    updatedAt={updatedAt} saved={anySaved} loading={loading}
                    onRefresh={refreshNow} busy={refreshing}
                />
            </ErrorBoundary>
            {/* ↩ Back pill — after any jump, one tap back to where he was */}
            <ErrorBoundary resetKey={refreshTick}><BackPill /></ErrorBoundary>

            {/* FOOTER */}
            <footer className="dashboard-footer">
                <p>Jalal's Financial Dashboard v7.0 — Data from FRED, CNN, Polygon, Finnhub, CNBC, Nasdaq, Yahoo Finance, Frankfurter, Polymarket, the Ken French Data Library &amp; Google Sheets</p>
                {process.env.NEXT_PUBLIC_BUILD_TIME && (
                    <p style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: '4px' }}>
                        {/* Pinned to New York time: the server renders in UTC and the browser in local
                            time, so an unpinned format never matched and every load threw React #425
                            (hydration text mismatch) from Feb to Sep 2026. */}
                        Deployed: {new Date(process.env.NEXT_PUBLIC_BUILD_TIME).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'America/New_York' })} ET
                    </p>
                )}
            </footer>

            {/* SYSTEM ERROR LOGS */}
            {apiErrors.length > 0 && (
                <div style={{
                    margin: '0 auto 24px',
                    maxWidth: '1200px',
                    width: 'calc(100% - 48px)',
                    padding: '16px',
                    backgroundColor: 'rgba(239, 68, 68, 0.05)',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    borderRadius: '8px',
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: '0.8rem',
                    color: 'rgba(255, 255, 255, 0.8)'
                }}>
                    <div style={{ fontWeight: 600, color: '#ef4444', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        ⚠️ System Diagnostic Logs
                    </div>
                    {apiErrors.map((err, i) => (
                        <div key={i} style={{ marginBottom: '4px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#fca5a5' }}>
                            {err}
                        </div>
                    ))}
                </div>
            )}

            {/* SYSTEM STATUS BAR */}
            {systemStatus && (
                <div className="system-status-bar">
                    <div className="status-items">
                        <span className={`status-item ${systemStatus.spy?.hasErrors ? 'status-error' :
                            systemStatus.spy?.source?.includes('FRED') ? 'status-warn' : ''
                            }`}>
                            [SPY: {
                                systemStatus.spy?.source?.includes('yfinance') ? 'yfinance' :
                                    systemStatus.spy?.source?.includes('Polygon') ? 'Polygon' :
                                        systemStatus.spy?.source?.includes('Google Sheet') ? 'GSheet' :
                                            systemStatus.spy?.source?.includes('FRED') ? 'FRED Fallback' :
                                                systemStatus.spy?.source || 'OK'
                            }]
                        </span>
                        <span className={`status-item ${systemStatus.fred?.hasErrors ? 'status-error' : ''}`}>
                            [FRED: {systemStatus.fred?.messages?.[0]?.replace('Loaded ', '').replace(' series', '') || '18/18'}]
                        </span>
                        <span className={`status-item ${systemStatus.fg?.source?.includes('Stale') || systemStatus.fg?.source?.includes('Failed') ? 'status-error' :
                            (systemStatus.fg?.source?.includes('VIX') || systemStatus.fg?.source?.includes('Proxy')) ? 'status-warn' :
                                systemStatus.fg?.hasErrors ? 'status-warn' : ''
                            }`}>
                            [F&G: {
                                systemStatus.fg?.source?.includes('CNN') ? 'CNN' :
                                    systemStatus.fg?.source?.includes('RapidAPI') ? 'RapidAPI' :
                                        systemStatus.fg?.source?.includes('VIXCLS') ? 'FRED VIX' :
                                            systemStatus.fg?.source?.includes('VIX') ? 'VIX Proxy' :
                                                systemStatus.fg?.source?.includes('Stale') ? 'STALE' :
                                                    systemStatus.fg?.source?.includes('Failed') ? 'FAILED' :
                                                        systemStatus.fg?.hasErrors ? 'PARTIAL' : 'LIVE OK'
                            }]
                        </span>
                        <span className={`status-item ${(systemStatus.sheets?.source?.includes('Failed') || systemStatus.sheets?.source?.includes('Static')) ? 'status-error' :
                            systemStatus.sheets?.source?.includes('Stale') ? 'status-error' :
                                (systemStatus.sheets?.source?.includes('Cached') || systemStatus.sheets?.source?.includes('Alt') || systemStatus.sheets?.source?.includes('Proxy') || systemStatus.sheets?.source?.includes('FRED')) ? 'status-warn' :
                                    ''
                            }`}>
                            [SHEETS: {
                                (systemStatus.sheets?.source?.includes('Failed') || systemStatus.sheets?.source?.includes('Static')) ? 'FAILED' :
                                    systemStatus.sheets?.source?.includes('Stale') ? 'STALE' :
                                        systemStatus.sheets?.source?.includes('Cached') ? 'CACHE' :
                                            systemStatus.sheets?.source?.includes('Alt') ? 'ALT OK' :
                                                (systemStatus.sheets?.source?.includes('Proxy') || systemStatus.sheets?.source?.includes('FRED')) ? 'FRED Proxy' :
                                                    'LIVE OK'
                            }]
                        </span>
                        {systemStatus.extra && (
                            <span className={`status-item ${systemStatus.extra?.messages?.some(m => m.includes('unavailable'))
                                ? 'status-error'
                                : systemStatus.extra?.hasErrors
                                    ? 'status-warn'
                                    : ''
                                }`}>
                                [MKTS: {systemStatus.extra?.messages?.join(' | ') || 'LIVE OK'}]
                            </span>
                        )}
                    </div>
                </div>
            )}
        </div>
        </MarkProvider>
    );
}
