'use client';
/**
 * Today: one slim card under the /v2 top line that answers "do I need to do anything today?". A verdict, five
 * one-line cells (each taps to a plain-words sheet) and four links out. It reports; it never advises.
 *
 * It reads the Signals card's split and Centred states plus one small relay, /api/own: what Main held at the last close,
 * its nearest rule, the morning alarms and the race. Main comes ONLY from own.json, never the FrontRunner sheet (a
 * different algo: on 28 Nov the sheet said cash while Main held bonds 50 / cash 50). "Nothing to do" only when every source was read.
 */
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import useSheetLock, { useSheetFocus } from './useSheetLock';
import { getJson } from '../lib/loadJson';
import { publicOwn } from '../lib/own';
import { ownStale } from '../lib/servedFreshness';
import { jumpToCard } from './WhatMoved';

export const TODAY_LINKS = [
    ['NUTS', 'https://nuts-sooty.vercel.app'],
    ['Radar', 'https://nuts-radar.vercel.app'],
    ['Night Shift', 'https://nightshift-jalal.vercel.app'],
    ['Wayback', 'https://wayback-backtester.vercel.app'],
];
const day = (iso) => new Date(`${iso}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
const pts = (v) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}`;
const KIND = { stocks: 'Stocks 3x', bonds: 'Bonds', cash: 'Cash', fear: 'Fear fund', short: 'Short', other: 'Other' };
/** Main's holding as the cell says it: one kind = its label; more = "Bonds 50 · Cash 50", top 3 then "+N"; null = not read. */
export const holdsText = (h) => (!h ? null : h.length === 1 ? KIND[h[0].kind]
    : h.slice(0, 3).map((x) => `${KIND[x.kind]} ${x.pct}`).join(' · ') + (h.length > 3 ? ` +${h.length - 3}` : ''));

/** The own feed as the card reads it: undefined = loading, null = unread (missing, broken, v != 1 or over 30 h old). */
export const ownRead = (own, now = Date.now()) => (own === undefined ? undefined : own && !ownStale(own, now) ? own : null);

/**
 * The verdict: the things that want a look (amber), most important first, then what could not be read (grey).
 * sig = the Signals card's states ({ split, centred }; undefined = loading, null = unread).
 * -> null while loading with nothing to say yet, else { quiet, parts: [{ text, tone, cell }] }.
 */
export function todayVerdict(sig, own) {
    const s = sig || {}, look = [], unread = [];
    const loading = sig === undefined || own === undefined || (!!sig && (s.split === undefined || s.centred === undefined));
    if (s.split?.out) look.push({ text: `Split ${s.split.main} / ${100 - s.split.main}`, cell: 'split' });
    (own?.alarms || []).filter((a) => a.state === 'tripped').forEach((a) => look.push({ text: a.name, cell: 'alarms' }));
    if (own?.near?.gap === 0) look.push({ text: own.near.side === 'dip' ? 'Dip day' : 'Hot day', cell: 'near' });
    if (s.centred?.off > 0) look.push({ text: `${s.centred.off} off centre`, cell: 'centred' });
    if (s.split?.totalsOld) look.push({ text: 'Split totals old', cell: 'split' });
    if (own === null) unread.push({ text: 'Alarms unread', cell: 'alarms' });
    const n = own ? own.alarms.filter((a) => a.state === 'unread').length : 0;
    if (n) unread.push({ text: `${n} alarm${n > 1 ? 's' : ''} unread`, cell: 'alarms' });
    if (own && !own.holds) unread.push({ text: 'Holds unread', cell: 'holds' });
    if (own && !own.near) unread.push({ text: 'Nearest unread', cell: 'near' });
    if (own && !own.race) unread.push({ text: 'Race unread', cell: 'race' });
    if (sig === null || s.split === null) unread.push({ text: 'Split unread', cell: 'split' });
    if (sig === null || s.centred === null) unread.push({ text: 'Centred unread', cell: 'centred' });
    if (loading && !look.length && !unread.length) return null;
    const parts = [...look.map((p) => ({ ...p, tone: 'warn' })), ...unread.map((p) => ({ ...p, tone: 'unread' }))];
    return { quiet: !parts.length, parts };
}

/** Main's nearest rule as the cell says it: "11.1 pts · hot side", "on · dip side" at 0; null = not read. */
export const nearText = (n) => (n ? `${n.gap === 0 ? 'on' : `${n.gap.toFixed(1)} pts`} · ${n.side} side` : null);

function Sheet({ title, head, onClose, children }) {
    const sheetRef = useRef(null), backRef = useRef(null);
    useSheetLock(true);
    useSheetFocus(true, sheetRef);
    useEffect(() => {
        const k = (e) => { if (e.key === 'Escape') onClose(); };
        document.addEventListener('keydown', k);
        return () => document.removeEventListener('keydown', k);
    }, [onClose]);
    if (typeof document === 'undefined') return null;
    return createPortal(
        <div ref={backRef} className="v2-aa-back" role="dialog" aria-modal="true" aria-labelledby="v2-tc-title"
            onClick={(e) => { if (e.target === backRef.current) onClose(); }}>
            <div className="v2-aa v2-tc-sheet" ref={sheetRef} data-sheet-scroll="">
                <div className="v2-aa-head">
                    <div className="v2-aa-title"><h2 id="v2-tc-title">{title}</h2><p>{head}</p></div>
                    <button type="button" className="v2-aa-close sheet-x" onClick={onClose} aria-label="Close">×</button>
                </div>
                {children}
            </div>
        </div>,
        document.querySelector('.v2') || document.body,
    );
}

const Say = ({ lines }) => lines.filter(Boolean).map((l) => <p key={l} className="v2-aa-foot">{l}</p>);

export default function TodayCard({ sig, tick }) {
    const [raw, setRaw] = useState(undefined);
    useEffect(() => { getJson('/api/own').then((j) => setRaw((had) => publicOwn(j) || (had ?? null))); }, [tick]); // a failed refresh keeps what it had
    const own = ownRead(raw);
    const [open, setOpen] = useState(null);
    const close = () => setOpen(null);
    const v = todayVerdict(sig, own);
    const s = sig || {}, near = nearText(own?.near), held = holdsText(own?.holds);
    const tripped = own ? own.alarms.filter((a) => a.state === 'tripped').length : 0;
    const unreadN = own ? own.alarms.filter((a) => a.state === 'unread').length : 0;
    const ahead = own?.race ? own.race.rows.filter((r) => r.vs > 0).length : 0;
    const wait = (x) => (x === undefined ? '…' : 'N/A');

    const cells = [
        { k: 'holds', label: 'Holds', value: held || (own === undefined ? '…' : 'N/A') },
        { k: 'near', label: 'Nearest', value: near || (own === undefined ? '…' : 'N/A'), tone: own?.near?.gap === 0 ? 'warn' : '' },
        { k: 'split', label: 'Split', value: s.split ? `${s.split.main} / ${100 - s.split.main}` : wait(s.split), tone: s.split?.out ? 'warn' : '' },
        { k: 'alarms', label: 'Alarms', value: own ? (tripped ? `${tripped} tripped${unreadN ? ` · ${unreadN} unread` : ''}` : unreadN ? `${unreadN} unread` : `${own.alarms.length} quiet`) : own === undefined ? '…' : 'unread', tone: tripped ? 'warn' : '' },
        { k: 'race', label: 'Race', value: own?.race ? `day ${own.race.day} · ${ahead} of ${own.race.rows.length} ahead` : own === undefined ? '…' : 'N/A' },
    ];

    return (
        <div className={`v2-tc${v?.quiet ? ' is-quiet' : v?.parts.some((p) => p.tone === 'warn') ? ' is-warn' : ''}`} role="group" aria-label="Today">
            <span className="v2-tc-verdict">
                <i className="v2-today-dot" aria-hidden="true" />
                {!v ? <span className="v2-tc-v">…</span> : v.quiet ? <span className="v2-tc-v">Nothing to do</span>
                    : v.parts.map((p) => (
                        <button key={p.text} type="button" className={`v2-tc-v is-${p.tone}`} onClick={() => (p.cell === 'centred' ? jumpToCard('Indicators') : setOpen(p.cell))}>{p.text}</button>
                    ))}
            </span>
            <span className="v2-tc-cells">
                {cells.map((c) => (
                    <button key={c.k} type="button" className={`v2-tc-cell is-${c.k}${c.tone ? ` is-${c.tone}` : ''}`} onClick={() => setOpen(c.k)} aria-haspopup="dialog">
                        <em>{c.label}</em>{c.value}
                    </button>
                ))}
            </span>
            <span className="v2-tc-links">
                {TODAY_LINKS.map(([t, href]) => <a key={t} href={href} target="_blank" rel="noopener noreferrer">{t} ↗</a>)}
            </span>

            {open === 'holds' && (
                <Sheet title="Holds" head={own?.holds ? `As of ${day(own.asof)}` : 'Not read today'} onClose={close}>
                    {own?.holds?.map((x) => <div key={x.kind} className="v2-sh-row"><span>{KIND[x.kind]}</span><span>{x.pct}%</span></div>)}
                    <Say lines={['What Main held at the last close.', "From Composer's own numbers.", 'Tickers are on NUTS.']} />
                </Sheet>
            )}
            {open === 'near' && (
                <Sheet title="Nearest rule" head={near || 'Not read today'} onClose={close}>
                    <Say lines={["Main's closest rule, at the last close.", 'Dip side: a buy-the-dip rule.', 'Hot side: a step-aside rule.', 'Live readings are on NUTS.']} />
                </Sheet>
            )}
            {open === 'split' && (
                <Sheet title="Split" head={s.split ? `Main ${s.split.main} / ${s.split.other} ${100 - s.split.main}` : 'Not read today'} onClose={close}>
                    {s.split && (
                        <>
                            <div className="v2-sh-row"><span>Target</span><span>{s.split.target} / {100 - s.split.target}</span></div>
                            <div className="v2-sh-row"><span>Fine zone</span><span>Main {s.split.target - s.split.band} to {s.split.target + s.split.band}</span></div>
                            <div className="v2-sh-row"><span>As of</span><span>{day(s.split.asOf)}</span></div>
                        </>
                    )}
                    <Say lines={[
                        s.split ? 'An estimate: your last totals, rolled on.' : 'Missing or not the newest close.',
                        s.split?.out ? 'Out of the fine zone today.' : s.split ? 'Inside the fine zone.' : null,
                        s.split?.totalsOld ? 'Your totals are over a month old.' : null,
                    ]} />
                </Sheet>
            )}
            {open === 'alarms' && (
                <Sheet title="Alarms" head={own ? `${tripped ? `${tripped} tripped` : `${own.alarms.length - unreadN} quiet`}${unreadN ? ` · ${unreadN} unread` : ''} · as of ${day(own.asof)}` : 'Unread'} onClose={close}>
                    {own ? own.alarms.map((a) => (
                        <div key={a.id} className={`v2-sh-row v2-tc-row is-${a.state}`}><span>{a.name}</span><span><b>{a.state === 'ok' ? 'quiet' : a.state}</b>{a.note}</span></div>
                    )) : null}
                    <Say lines={own ? ['Checked each morning on what you own.'] : ['The morning check could not be read.', 'Missing, broken or over 30 h old.']} />
                </Sheet>
            )}
            {open === 'race' && (
                <Sheet title="Race" head={own?.race ? `Day ${own.race.day} since ${day(own.race.start)}` : 'Not read today'} onClose={close}>
                    {own?.race && (
                        <>
                            <div className="v2-sh-row v2-tc-row is-head"><span>v Main, points</span><span><b>return</b><b>fall</b></span></div>
                            {own.race.rows.map((r) => (
                                <div key={r.id} className="v2-sh-row v2-tc-row"><span>{r.name}</span><span><b>{pts(r.vs)}</b><b>{pts(r.fall)}</b></span></div>
                            ))}
                        </>
                    )}
                    <Say lines={['A watch, not evidence. Quarterly check.']} />
                </Sheet>
            )}
        </div>
    );
}
