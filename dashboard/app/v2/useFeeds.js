'use client';
/**
 * v2's data layer: the SAME feeds, saved copies, refresh pace and "since last visit"
 * bookkeeping as v1 (app/page.js), lifted into one hook so the v2 page is only layout.
 * v1 keeps its own inline copy on purpose (his daily tool stays byte-for-byte as it was);
 * a behaviour fix in one belongs in the other too.
 */
import { useState, useEffect, useLayoutEffect, useRef, useMemo } from 'react';
import { getJson } from '../../lib/loadJson';
import { readSnap, writeSnap, savedLabel, purgeOldSnaps, isLiveAnswer } from '../../lib/snapshot';
import { readSeen, writeSeen, mergeSeen, pickSeen, SINCE_MIN_GAP_MS } from '../../lib/lastVisit';
import { collectLiveValues } from '../../components/MarkProvider';
import { refreshMs } from '../../lib/marketClock';

const useIsoLayoutEffect = typeof window !== 'undefined' ? useLayoutEffect : useEffect;

export const FEEDS = [
    { key: 'spy', path: '/api/spy' },
    { key: 'sheets', path: '/api/sheets' },
    { key: 'spyDailyMove', path: '/api/spy-daily-move' },
    { key: 'fg', path: '/api/fear-greed' },
    { key: 'fred', path: '/api/fred' },
    { key: 'extra', path: '/api/market-extra' },
    { key: 'history', path: '/api/history' },
    { key: 'jev', path: '/api/jev-pills' },
    { key: 'vol', path: '/api/vol' },
];

// While any of these is a saved copy, fresh-print marks stay off (as v1).
const MARK_INPUTS = ['history', 'fred', 'extra', 'sheets'];

export default function useFeeds() {
    const [data, setData] = useState({});
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [refreshTick, setRefreshTick] = useState(0);
    const [updatedAt, setUpdatedAt] = useState(null);
    const [systemStatus, setSystemStatus] = useState(null);
    const [apiErrors, setApiErrors] = useState([]);
    const [savedAt, setSavedAt] = useState({});
    const savedRef = useRef({});
    const [seenBase, setSeenBase] = useState({ rec: null, at: 0 });
    const readBase = () => setSeenBase({ rec: readSeen(), at: Date.now() });
    const [landedAt, setLandedAt] = useState({});
    const landedRef = useRef({});
    const hasLoadedRef = useRef(false);
    const lastFetchRef = useRef(0);
    const inFlightRef = useRef(false);
    const queuedRef = useRef(false);
    const [pending, setPending] = useState(() => Object.fromEntries(FEEDS.map((f) => [f.key, true])));
    const [bustKey, setBustKey] = useState(false);

    const put = (key, v) => setData((d) => ({ ...d, [key]: v }));
    const dropSaved = (key) => {
        if (!(key in savedRef.current)) return;
        const next = { ...savedRef.current };
        delete next[key];
        savedRef.current = next;
        setSavedAt(next);
    };

    // Last visit's numbers paint before the first frame; live ones replace them feed by feed.
    useIsoLayoutEffect(() => {
        readBase();
        purgeOldSnaps();
        const got = {};
        const snaps = {};
        for (const { key } of FEEDS) {
            const snap = readSnap(key);
            if (!snap) continue;
            snaps[key] = snap.data;
            got[key] = snap.savedAt;
        }
        if (Object.keys(snaps).length) setData((d) => ({ ...snaps, ...d }));
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
        const stamp = () => { if (live) setUpdatedAt(Date.now()); };
        try {
            const load = async ({ key, path }) => {
                const res = await getJson(path, { bust });
                got[key] = res;
                if (isLiveAnswer(key, res)) {
                    live++;
                    put(key, res);
                    dropSaved(key);
                    landedRef.current = { ...landedRef.current, [key]: Date.now() };
                    setLandedAt(landedRef.current);
                    setTimeout(() => writeSnap(key, res), 0);
                } else if (!(key in savedRef.current)) {
                    if (landedRef.current[key]) {
                        savedRef.current = { ...savedRef.current, [key]: landedRef.current[key] };
                        setSavedAt(savedRef.current);
                    } else if (res != null) {
                        put(key, res);
                    }
                }
                setPending((p) => (p[key] ? { ...p, [key]: false } : p));
            };
            // jev-pills reads the other routes over HTTP: ask after they landed (edge HITs).
            const loads = Object.fromEntries(FEEDS.filter((f) => f.key !== 'jev').map((f) => [f.key, load(f)]));
            const core = Promise.all(['spy', 'sheets', 'fg', 'fred'].map((k) => loads[k])).then(() => { setLoading(false); stamp(); });
            await Promise.all([...Object.values(loads), core, Promise.all(['spy', 'sheets', 'fg', 'fred', 'vol'].map((k) => loads[k]))
                .then(() => load(FEEDS.find((f) => f.key === 'jev')))]);
            hasLoadedRef.current = true;
            setSystemStatus({ spy: got.spy, fred: got.fred, fg: got.fg, sheets: got.sheets, extra: got.extra });
            const errors = [];
            for (const [k, label] of [['sheets', 'SHEETS'], ['spy', 'SPY'], ['fg', 'F&G'], ['fred', 'FRED'], ['extra', 'EXTRA']]) {
                if (got[k] == null) errors.push(`[${label}] no answer`);
                else if (got[k]?.error) errors.push(`[${label}] ${got[k].error}`);
            }
            setApiErrors(errors);
            stamp();
            setRefreshTick((t) => t + 1);
        } catch (e) {
            console.error('Dashboard fetch error:', e);
            setApiErrors((prev) => [...prev, `[NETWORK] ${e.toString()}`]);
        }
        setLoading(false);
        setRefreshing(false);
        inFlightRef.current = false;
        if (queuedRef.current) { queuedRef.current = false; setTimeout(() => fetchAll({ bust: true }), 0); }
    }
    const refreshNow = () => fetchAll({ bust: true });

    // Tap-to-open for the CSS tooltips (touch has no :hover); Esc or a tap elsewhere closes.
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

    // R refreshes (skips the edge cache), unless typing or Cmd/Ctrl+R.
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

    // Same pace as v1: 5 min while NYSE is open, 30 otherwise; never while hidden or offline.
    useEffect(() => {
        fetchAll();
        const interval = setInterval(() => {
            if (!document.hidden && navigator.onLine !== false && Date.now() - lastFetchRef.current > refreshMs() - 30e3) fetchAll();
        }, 60e3);
        const onVisible = () => {
            if (document.hidden) return;
            if (Date.now() - lastFetchRef.current >= SINCE_MIN_GAP_MS) readBase();
            if (navigator.onLine !== false && Date.now() - lastFetchRef.current > refreshMs()) fetchAll();
        };
        document.addEventListener('visibilitychange', onVisible);
        return () => {
            clearInterval(interval);
            document.removeEventListener('visibilitychange', onVisible);
        };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const { spy, fg, fred, extra, sheets, vol } = data;
    const saved = (...keys) => {
        const ts = keys.map((k) => savedAt[k]).filter(Number.isFinite);
        return ts.length ? savedLabel(Math.min(...ts)) : undefined;
    };
    const liveOnly = (key, v) => (!(key in savedAt) && landedAt[key] > seenBase.at ? v : null);
    const seenNow = useMemo(() => pickSeen(
        { spy: liveOnly('spy', spy), vol: liveOnly('vol', vol), fg: liveOnly('fg', fg), extra: liveOnly('extra', extra) },
        collectLiveValues(liveOnly('fred', fred), liveOnly('extra', extra), liveOnly('sheets', sheets)),
    ), [spy, vol, fg, extra, fred, sheets, savedAt, landedAt, seenBase.at]); // eslint-disable-line react-hooks/exhaustive-deps
    useEffect(() => {
        if (!updatedAt || !Object.values(seenNow.v).some((x) => x != null)) return;
        writeSeen(mergeSeen(readSeen(), seenNow, Date.now(), landedAt));
    }, [seenNow, updatedAt]); // eslint-disable-line react-hooks/exhaustive-deps

    return {
        data, loading, refreshing, refreshTick, updatedAt, systemStatus, apiErrors,
        pending, bustKey, savedAt, saved, seenBase, seenNow, refreshNow,
        marksOff: MARK_INPUTS.some((k) => k in savedAt),
        anySaved: saved(...Object.keys(savedAt)),
    };
}
