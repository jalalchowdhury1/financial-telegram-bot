'use client';
/**
 * Today: one slim card under the /v2 top line that answers "do I need to do anything today?". A verdict, five
 * one-line cells (each taps to a plain-words sheet) and four links out. It reports; it never advises.
 *
 * It reads what the page already knows (the Signals card's FrontRunner, split and Centred states, the top line's
 * items) plus one small relay, /api/own (the morning alarms + race). "Nothing to do" only when every source was read.
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
const ticker = (fr) => (/^[A-Z]{2,5}\b/.exec(fr) || [])[0];

/** The own feed as the card reads it: undefined = loading, null = unread (missing, broken, v != 1 or over 30 h old). */
export const ownRead = (own, now = Date.now()) => (own === undefined ? undefined : own && !ownStale(own, now) ? own : null);

/**
 * The verdict: the things that want a look (amber), most important first, then what could not be read (grey).
 * sig = the Signals card's states ({ fr, frMode, split, centred }; undefined = loading, null = unread).
 * -> null while loading with nothing to say yet, else { quiet, parts: [{ text, tone, cell }] }.
 */
export function todayVerdict(sig, own) {
    const s = sig || {}, look = [], unread = [];
    const loading = sig === undefined || own === undefined || (!!sig && (s.split === undefined || s.fr === undefined || s.centred === undefined));
    if (s.split?.out) look.push({ text: `Split ${s.split.main} / ${100 - s.split.main}`, cell: 'split' });
    (own?.alarms || []).filter((a) => a.state === 'tripped').forEach((a) => look.push({ text: a.name, cell: 'alarms' }));
    if (s.fr && s.frMode !== 'Cash') look.push({ text: s.frMode === 'Dip buy' ? 'Dip day' : s.frMode, cell: 'holds' });
    if (s.centred?.off > 0) look.push({ text: `${s.centred.off} off centre`, cell: 'centred' });
    if (s.split?.totalsOld) look.push({ text: 'Split totals old', cell: 'split' });
    if (own === null) unread.push({ text: 'Alarms unread', cell: 'alarms' });
    else if (own?.alarms.some((a) => a.state === 'unread')) unread.push({ text: `${own.alarms.filter((a) => a.state === 'unread').length} alarm unread`, cell: 'alarms' });
    if (sig === null || s.split === null) unread.push({ text: 'Split unread', cell: 'split' });
    if (sig === null || s.fr === null) unread.push({ text: 'FrontRunner unread', cell: 'holds' });
    if (sig === null || s.centred === null) unread.push({ text: 'Centred unread', cell: 'centred' });
    if (loading && !look.length && !unread.length) return null;
    const parts = [...look.map((p) => ({ ...p, tone: 'warn' })), ...unread.map((p) => ({ ...p, tone: 'unread' }))];
    return { quiet: !parts.length && !loading, parts };
}

/** The top line's "closest": the read, not-firing signal nearest its line. */
export function nearest(items) {
    const known = (items || []).filter((s) => s && !s.fire && s.gap);
    return known.length ? known.reduce((a, b) => (b.share > a.share ? b : a)) : null;
}

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

export default function TodayCard({ items, sig, tick }) {
    const [raw, setRaw] = useState(undefined);
    useEffect(() => { getJson('/api/own').then((j) => setRaw((had) => publicOwn(j) || (had ?? null))); }, [tick]); // a failed refresh keeps what it had
    const own = ownRead(raw);
    const [open, setOpen] = useState(null);
    const close = () => setOpen(null);
    const v = todayVerdict(sig, own);
    const s = sig || {}, near = nearest(items);
    const tripped = own ? own.alarms.filter((a) => a.state === 'tripped').length : 0;
    const unreadN = own ? own.alarms.filter((a) => a.state === 'unread').length : 0;
    const ahead = own?.race ? own.race.rows.filter((r) => r.vs > 0).length : 0;
    const wait = (x) => (x === undefined ? '…' : 'N/A');

    const cells = [
        { k: 'holds', label: 'Holds', value: s.fr ? (ticker(s.fr) ? `${ticker(s.fr)} · ${s.frMode}` : s.frMode) : wait(s.fr), tone: s.fr && s.frMode !== 'Cash' ? 'warn' : '' },
        { k: 'near', label: 'Nearest', value: near ? near.gap : items ? 'N/A' : '…' },
        { k: 'split', label: 'Split', value: s.split ? `${s.split.main} / ${100 - s.split.main}` : wait(s.split), tone: s.split?.out ? 'warn' : '' },
        { k: 'alarms', label: 'Alarms', value: own ? (tripped ? `${tripped} tripped` : unreadN ? `${unreadN} unread` : `${own.alarms.length} quiet`) : own === undefined ? '…' : 'unread', tone: tripped ? 'warn' : '' },
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
                <Sheet title="Holds" head={s.fr ? s.frMode : 'Not read today'} onClose={close}>
                    {s.fr && s.fr.length <= 40 && <div className="v2-sh-row"><span>Today</span><span>{s.fr}</span></div>}
                    <Say lines={[
                        'What the FrontRunner model holds today.',
                        !s.fr ? 'The morning sheet could not be read.' : s.frMode === 'Cash' ? 'Cash: none of its rules is on.' : s.frMode === 'Dip buy' ? 'A dip buy is on: it holds the dip.' : `${s.frMode}: one of its rules is on.`,
                        'Its 15 RSIs are on the FrontRunner tile.',
                    ]} />
                </Sheet>
            )}
            {open === 'near' && (
                <Sheet title="Nearest rule" head={near ? near.gap : 'Not read today'} onClose={close}>
                    <Say lines={['The quiet signal closest to its line.', 'Nothing fires until it crosses.', 'Same reading as the top line.']} />
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
                <Sheet title="Alarms" head={own ? `${tripped ? `${tripped} tripped` : `${own.alarms.length - unreadN} quiet`} · as of ${day(own.asof)}` : 'Unread'} onClose={close}>
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
