'use client';
/**
 * The page-bottom source line: one readable row, "✓ feeds live · SPY yfinance · FRED 17/17 · …".
 * Green only when it is true: a feed that did not answer, answered its fallback body, or is
 * a saved copy is an amber/red chip ("⚠ SPY: no data", "🕐 FRED saved 10:42"), and tapping
 * it jumps to that feed's card. Healthy feeds stay plain text. Reads the bodies the page
 * already fetched; no fetch of its own.
 */
import { isDegraded } from '../lib/cdn';
import { isLiveAnswer } from '../lib/snapshot';
import { jumpToCard } from './WhatMoved';

const has = (s, ...words) => typeof s === 'string' && words.some((w) => s.includes(w));

// Each feed's label + level from its own _meta, the same mapping the footer always had.
const READ = {
    spy: (b, m) => ({
        text: m.stale ? 'stale' : has(m.source, 'yfinance') ? 'yfinance' : has(m.source, 'Polygon') ? 'Polygon'
            : has(m.source, 'Google Sheet') ? 'GSheet' : has(m.source, 'FRED') ? 'FRED fallback' : m.source || 'live',
        level: m.hasErrors && !m.stale ? 'error' : m.stale || has(m.source, 'FRED') || isDegraded(b) ? 'warn' : 'ok',
    }),
    fred: (b, m) => ({
        text: m.messages?.[0]?.replace('Loaded ', '').replace(' series', '') || 'live',
        level: m.hasErrors ? 'error' : isDegraded(b) ? 'warn' : 'ok',
    }),
    fg: (b, m) => ({
        // stale first: a cached copy's source still names CNN ("Stale cache (…) ← CNN")
        text: (m.stale && !m.proxy) ? 'stale' : has(m.source, 'CNN') ? 'CNN' : has(m.source, 'RapidAPI') ? 'RapidAPI'
            : has(m.source, 'VIXCLS') ? 'FRED VIX' : has(m.source, 'VIX') ? 'VIX proxy' : has(m.source, 'Stale') ? 'stale'
                : has(m.source, 'Failed') ? 'failed' : m.hasErrors ? 'partial' : 'live',
        level: (m.stale && !m.proxy) || has(m.source, 'Stale', 'Failed') ? 'error'
            : has(m.source, 'VIX', 'Proxy') || m.hasErrors ? 'warn' : 'ok',
    }),
    sheets: (b, m) => ({
        text: has(m.source, 'Failed', 'Static') ? 'failed' : has(m.source, 'Stale') ? 'stale' : has(m.source, 'Cached') ? 'cache'
            : has(m.source, 'Alt') ? 'alt OK' : has(m.source, 'Proxy', 'FRED') ? 'FRED proxy' : 'live',
        level: has(m.source, 'Failed', 'Static', 'Stale') ? 'error' : has(m.source, 'Cached', 'Alt', 'Proxy', 'FRED') ? 'warn' : 'ok',
    }),
    extra: (b, m) => ({
        text: m.messages?.map((x) => x.replace(/skipped (.+)$/i, (_, l) => `skipped ${l.split(',').length}`)).join(' · ') || 'live',
        level: m.messages?.some((x) => x.includes('unavailable')) ? 'error' : m.hasErrors ? 'warn' : 'ok',
    }),
};
const FEEDS = [
    { key: 'spy', name: 'SPY', jump: 'SPY overview' },
    { key: 'fred', name: 'FRED', jump: 'Economy' },
    { key: 'fg', name: 'F&G', jump: 'Fear & Greed' },
    { key: 'sheets', name: 'Sheets', jump: 'Indicators' },
    { key: 'extra', name: 'Markets', jump: 'Markets' },
];

/**
 * [{key, name, jump, text, level: 'ok'|'warn'|'error', icon}] for this cycle's bodies.
 * `saved` = feed key → "10:42" while that card shows a saved copy: never "live".
 */
export function statusFeeds(bodies = {}, saved = {}) {
    return FEEDS.map((f) => {
        const b = bodies[f.key];
        if (saved[f.key]) return { ...f, text: `saved ${saved[f.key]}`, level: 'warn', icon: '🕐' };
        if (!isLiveAnswer(f.key, b)) return { ...f, text: 'no data', level: 'error', icon: '⚠' };
        return { ...f, ...READ[f.key](b, b._meta && typeof b._meta === 'object' ? b._meta : {}), icon: '⚠' };
    });
}

// Problems lead the line (red, then amber): on a phone it scrolls sideways, and a down feed
// at the end sat past the screen edge while four green ones filled the view.
const RANK = { error: 0, warn: 1, ok: 2 };

export default function StatusLine({ bodies, saved }) {
    const feeds = statusFeeds(bodies, saved).sort((a, b) => RANK[a.level] - RANK[b.level]);
    const allOk = feeds.every((f) => f.level === 'ok');
    return (
        <div className="system-status-bar">
            <div className="status-items">
                {allOk && <span className="status-item status-ok">✓ feeds live</span>}
                {feeds.map((f) => (f.level === 'ok'
                    ? <span key={f.key} className="status-item">{f.name} {f.text}</span>
                    : (
                        <button key={f.key} type="button" className={`status-item status-chip status-${f.level}`} onClick={() => jumpToCard(f.jump)}>
                            {f.icon} {f.name}: {f.text}
                        </button>
                    )))}
            </div>
        </div>
    );
}
