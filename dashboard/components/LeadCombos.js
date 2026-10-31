'use client';
/**
 * v2 "Lead combos": the pairs of dashboard stats that, past their lines together, led to a good next
 * 6 months (lib/signalStudy.js COMBOS). One row each: both legs as meters filling toward their line,
 * SPY's average 6 months after every past signal, and today's state. Tap a row for the proof
 * (components/SignalStudy.js). The baked files load after first paint; the 90-day sheet series
 * carries each leg up to today.
 */
import { useEffect, useMemo, useState } from 'react';
import SignalStudy from './SignalStudy';
import { useSeries } from './MarkProvider';
import { decodeLong } from '../lib/longHistory';
import { COMBOS, comboSeries, signalEpisodes, signalStudy, plusMonthsIso } from '../lib/signalStudy';

const KEYS = [...new Set(COMBOS.flatMap((c) => c.legs.map((l) => l.key)))];
const load = (key) => fetch(`/history/${key}.json`).then((r) => (r.ok ? r.json() : null)).then(decodeLong);
const DAY = 86400000;
const sheetPts = (s, key) => (s?.v?.[key] || []).flatMap((x, i) => (typeof x === 'number' && Number.isFinite(x)
    ? [{ date: new Date(Date.parse(`${s.from}T00:00:00Z`) + i * DAY).toISOString().slice(0, 10), value: x }] : []));
const tail = (pts, more) => [...pts, ...more.filter((p) => p.date > pts[pts.length - 1].date)];
const sign = (v) => (v > 0 ? '+' : v < 0 ? '−' : '');
/** A leg's reading: a change gets its sign and unit, a level just its number. */
const fmt = (v, l, dp = l.pct ? 1 : 2) => (l.days ? `${sign(v)}${Math.abs(v).toFixed(dp)}${l.pct ? '%' : ''}` : v.toFixed(1));
const lineOf = (l) => (l.days ? `${sign(l.line)}${Math.abs(l.line)}${l.pct ? '%' : ''}` : `${l.line}`);
const shortDate = (iso) => new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
const pct0 = (r) => `${sign(r)}${Math.abs(r * 100).toFixed(0)}%`;

export default function LeadCombos({ spy, onState }) {
    const series = useSeries();
    const [files, setFiles] = useState(null); // {key: points} | false when a file failed
    const [open, setOpen] = useState(null);

    useEffect(() => {
        let live = true;
        Promise.all([...KEYS, 'spy-daily'].map(load))
            .then((got) => { if (live) setFiles(got.every(Boolean) ? Object.fromEntries([...KEYS, 'spy-daily'].map((k, i) => [k, got[i]])) : false); })
            .catch(() => { if (live) setFiles(false); });
        return () => { live = false; };
    }, []);

    const rows = useMemo(() => {
        if (!files) return null;
        const today = new Date().toLocaleDateString('en-CA');
        const s = files['spy-daily'].map((p) => ({ date: p.date, price: p.value }));
        return COMBOS.map((c) => {
            const pts = comboSeries(c, ...c.legs.map((l) => tail(files[l.key], sheetPts(series, l.key))));
            const now = pts[pts.length - 1];
            const last = signalEpisodes(pts, 1).pop();
            const until = last && plusMonthsIso(last.date, 6);
            const { stats, base } = signalStudy(pts, s, 6, 1) || {};
            return { c, pts, now, until: until >= today ? until : null, stats, base };
        });
    }, [files, series]);

    // For the page's top line: each pair's state, and its leg furthest from its line.
    useEffect(() => {
        if (!onState) return;
        if (files === false) onState([]);
        if (rows) onState(rows.filter((r) => r.now).map(({ c, now, until }) => {
            const k = now.legs.reduce((m, v, i) => (v / c.legs[i].line < now.legs[m] / c.legs[m].line ? i : m), 0);
            return { jump: 'Lead combos', fire: until && `${c.name} bullish to ${shortDate(until)}`, share: now.value,
                gap: `${c.name} · ${c.legs[k].label} ${fmt(now.legs[k], c.legs[k])} → ${lineOf(c.legs[k])}` };
        }));
    }, [files, rows, onState]);

    const sel = rows?.find((r) => r.c.id === open);
    return (
        <div className="card v2-combos">
            <div className="v2-cb-head" aria-hidden="true"><span /><span>Leg → line</span><span>SPY 6M after</span><span>Now</span></div>
            {COMBOS.map((c, i) => {
                const r = rows?.[i], lit = r?.now ? r.now.legs.filter((v, k) => v / c.legs[k].line >= 1).length : 0;
                const state = !r?.now ? (files === null ? '…' : 'N/A') : r.until ? `Bullish to ${shortDate(r.until)}` : `${lit} of 2`;
                return (
                    <button key={c.id} type="button" className={`v2-cb-row${r?.until ? ' is-on' : ''}`} disabled={!r?.now}
                        aria-haspopup="dialog" onClick={() => setOpen(c.id)}>
                        <span className="v2-cb-name"><span className="tooltip-trigger" data-tooltip={`${c.tip} Tap for the proof.`}>{c.name}</span></span>
                        <span className="v2-cb-legs">
                            {c.legs.map((l, k) => {
                                const v = r?.now?.legs[k], f = Number.isFinite(v) ? Math.max(0, Math.min(1, v / l.line)) : 0;
                                return (
                                    <span key={l.key} className={`v2-cb-leg${f >= 1 ? ' is-lit' : ''}`}>
                                        <span className="v2-cb-leg-txt">{l.label} <b>{Number.isFinite(v) ? fmt(v, l) : '—'}</b> <i>→ {lineOf(l)}</i></span>
                                        <span className="v2-cb-track"><span style={{ width: `${f * 100}%` }} /></span>
                                    </span>
                                );
                            })}
                        </span>
                        <span className="v2-cb-rec" title={r?.stats && r.base ? `SPY 6 months after: avg of ${r.stats.n} times (any day ${pct0(r.base.avg)})` : undefined}>
                            {r?.stats?.n ? <><b className="tone-good">{pct0(r.stats.avg)}</b> <i>{r.stats.n}×</i></> : ''}
                        </span>
                        <span className={`v2-cb-now${r?.until ? ' tone-good' : lit ? ' is-half' : ''}`}>{state}</span>
                    </button>
                );
            })}
            {sel && (
                <SignalStudy id={`combo-${sel.c.id}`} src={sel.c.legs.map((l) => `/history/${l.key}.json`)} join={(...f) => comboSeries(sel.c, ...f)}
                    line={1} horizons={[1, 3, 6, 12]} start={6} title={sel.c.name}
                    sub={`Now ${sel.c.legs.map((l, k) => `${l.label} ${fmt(sel.now.legs[k], l)}`).join(' · ')}`}
                    label={(v, p) => (p?.legs ? sel.c.legs.map((l, k) => `${l.label} ${fmt(p.legs[k], l, 0)}`).join(' · ') : `${v.toFixed(2)}× the lines`)}
                    foot={`${sel.c.tip.replace(/\.$/, '')}, after 3 quiet months`}
                    sheet={sel.pts} spyLive={spy} onClose={() => setOpen(null)} />
            )}
        </div>
    );
}
