'use client';

import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import useSheetLock, { useSheetFocus } from './useSheetLock';
import { PILL_EXPLAIN, INPUT_EXPLAIN, INPUT_NAMES, PILL_FEEDS } from './jevExplain';
import SplitMap from './SplitMap';
import NumberLine from './NumberLine';
import RECORD from '../lib/data/pillRecord.json';

const FRIENDLY = {
    'risk-on': 'Risk-on',
    'neutral': 'Neutral',
    'risk-off': 'Risk-off',
    'low': 'Low',
    'rising': 'Rising',
    'high': 'High',
    'broad': 'Broad',
    'narrow': 'Narrow',
    'rolling-over': 'Rolling over',
    'cheap': 'Cheap',
    'fair': 'Fair',
    'expensive': 'Expensive',
    'aligned': 'Aligned',
    'mild-divergence': 'Mild divergence',
    'major-divergence': 'Major divergence',
    'unknown': 'No data',
};

const SEVERITY = {
    'risk-on': 'badge-green', 'neutral': 'badge-yellow', 'risk-off': 'badge-red',
    'low': 'badge-green', 'rising': 'badge-yellow', 'high': 'badge-red',
    'broad': 'badge-green', 'narrow': 'badge-yellow', 'rolling-over': 'badge-red',
    'cheap': 'badge-green', 'fair': 'badge-yellow', 'expensive': 'badge-red',
    'unknown': 'badge-gray',
    'aligned': 'badge-green', 'mild-divergence': 'badge-yellow', 'major-divergence': 'badge-red',
};

function friendly(v) {
    return FRIENDLY[v] || (v ? v.charAt(0).toUpperCase() + v.slice(1) : v);
}

/**
 * Which sources does this pill read?
 */
const PILL_SOURCE_KEYS = {
    regime: ['spy', 'fg'],
    recession: ['fred'],
    breadth: ['breadth'],
    hedging: ['vol'],
    conflict: ['spy', 'fg', 'fred', 'breadth'],
};

const PILL_BACKUP_KEYS = { regime: ['hyOas20'], recession: ['sahm', 'claims', 'nfci'], conflict: ['t10y3m', 'hyOas20'] };

const PROVIDER_NAMES = {
    polygon: 'Polygon', cnbc: 'CNBC', 'cnbc-quote': 'CNBC', cboe: 'CBOE', fred: 'FRED', 'st. louis fed': 'FRED', yahoo: 'Yahoo',
    'finnhub spot': 'Finnhub', 'google sheet': 'Sheet', 'live horsemen': 'Horsemen',
};

/**
 * ['RSP:polygon · VIX:cboe+live', 'Nasdaq + CNBC (fallback)', 'Google Sheet (last-known-good …)']
 * → 'Polygon · CBOE · Nasdaq · CNBC · Sheet (saved)'
 */
function providers(vals) {
    const names = new Set();
    for (const val of vals) {
        for (const part of String(val).split(/ · | \+ /)) {
            let name = /^[A-Z0-9^.]+:/.test(part) ? part.slice(part.indexOf(':') + 1).replace('+live', '') : part;
            const saved = /last-known-good|cached/.test(name);
            name = name.replace(/\s*\(.*?\)/g, '').trim();
            if (name) names.add((PROVIDER_NAMES[name.toLowerCase()] || name) + (saved ? ' (saved)' : ''));
        }
    }
    return names.size ? [...names].join(' · ') : '—';
}

export default function JevPillModal({ pillKey, data, onClose }) {
    const backdropRef = useRef(null);
    const sheetRef = useRef(null);
    const pillOpen = !!data?.pills?.[pillKey];
    // the page underneath holds still while the sheet is open (and html gets .sheet-open);
    // focus moves to the ×, Tab stays in the sheet, and close hands focus back to the pill
    useSheetLock(pillOpen);
    useSheetFocus(pillOpen, sheetRef);

    // close on Escape
    useEffect(() => {
        const handler = (e) => {
            if (e.key === 'Escape') onClose();
        };
        document.addEventListener('keydown', handler);
        return () => document.removeEventListener('keydown', handler);
    }, [onClose]);

    const pill = data?.pills?.[pillKey];
    if (!pill) return null;

    const verdictBadge = SEVERITY[pill.verdict] || 'badge-blue';

    // ---- Header sub-line: who decided, Jev only when it differs, oldest data date ----
    const jevBlock = pill.jev;
    const pct = (p) => `${Math.round((p ?? 0) * 100)}%`;
    let decidedLine;
    if (pill.verdict === 'unknown') decidedLine = 'Too few numbers';
    else if (pill.by === 'jev') decidedLine = `Jev, ${pct(pill.p)} sure`;
    else if (data._meta?.jev === 'off') decidedLine = 'Rule · Jev off';
    else if (jevBlock?.verdict && jevBlock.verdict !== pill.verdict) decidedLine = `Rule · Jev: ${friendly(jevBlock.verdict)} ${pct(jevBlock.p)}`;
    else decidedLine = 'Rule';

    const dataAsOf = data._meta?.dataAsOf || {};
    const oldest = (PILL_FEEDS[pillKey] || []).map(k => dataAsOf[k]).filter(Boolean).sort()[0];
    if (oldest) decidedLine += ` · ${new Date(`${oldest}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })}`;

    const factors = data.factors?.[pillKey];
    const factorRows = factors?.rows || [];
    // only the regime score is news; the other summaries repeat the lit key row
    const factorSummary = pillKey === 'regime' ? factors?.summary || null : null;

    const sources = data._meta?.sources || {};
    const sourcesText = providers((PILL_SOURCE_KEYS[pillKey] || []).filter(k => sources[k] != null).map(k => sources[k]));
    // inputs that can fall back to a backup source (/api/jev-pills inputSources), per pill
    const backups = (PILL_BACKUP_KEYS[pillKey] || [])
        .map(k => [k, data._meta?.inputSources?.[k]])
        .filter(([, v]) => v && v !== 'fred-route' && v !== 'fred')
        .map(([k, v]) => `${k} ${v}`);

    const explain = PILL_EXPLAIN[pillKey];

    // Portalled to <body>: .card sets backdrop-filter, which traps a position:fixed
    // descendant inside the card (see the .mark-pop note in globals.css).
    if (typeof document === 'undefined') return null;

    return createPortal(
        <div
            ref={backdropRef}
            className="jev-modal-backdrop"
            onClick={(e) => {
                if (e.target === backdropRef.current) onClose();
            }}
            role="dialog"
            aria-modal="true"
            aria-labelledby="jev-modal-title"
        >
            <div className="jev-modal" ref={sheetRef} data-sheet-scroll="" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="jev-modal-header">
                    <div>
                        <h2 id="jev-modal-title" className="jev-modal-title">
                            <span className="jev-modal-label">{PILL_LABELS[pillKey] || pillKey}</span>
                            <span className={`badge ${verdictBadge}`}>{friendly(pill.verdict)}</span>
                        </h2>
                        <p className="jev-modal-sub">{decidedLine}</p>
                    </div>
                    <button
                        className="jev-modal-close sheet-x"
                        onClick={onClose}
                        aria-label="Close modal"
                    >
                        ×
                    </button>
                </div>

                {/* The question + what each verdict needs; today's verdict lit */}
                {explain && (
                    <section className="jev-modal-section">
                        <p className="jev-modal-ask">{explain.ask}</p>
                        {explain.weak && <p className="jev-modal-weak">⚠ {explain.weak}</p>}
                        <ul className="jev-key">
                            {Object.entries(explain.key).map(([v, rule]) => (
                                <li key={v} className={v === pill.verdict ? 'jev-key-on' : ''}>
                                    <span className={`badge ${SEVERITY[v]}`}>{friendly(v)}</span>
                                    <span>{rule}</span>
                                </li>
                            ))}
                        </ul>
                    </section>
                )}

                {/* Report card: what SPY did next after each colour (scripts/bake-pill-record.mjs) */}
                {explain && RECORD.pills[pillKey] && (() => {
                    const rec = RECORD.pills[pillKey];
                    const pc = (x) => `${x > 0 ? '+' : ''}${x.toFixed(1)}%`.replace('-', '−');
                    return (
                        <section className="jev-modal-section">
                            <p className="jev-record-title">Track record · SPY next 3 months</p>
                            <table className="jev-record">
                                <thead><tr><th></th><th>times</th><th>avg</th><th>worst dip</th></tr></thead>
                                <tbody>
                                    {Object.keys(explain.key).filter((v) => rec.verdicts[v]).map((v) => (
                                        <tr key={v} className={v === pill.verdict ? 'jev-key-on' : ''}>
                                            <td><span className={`badge ${SEVERITY[v]}`}>{friendly(v)}</span></td>
                                            <td>{rec.verdicts[v].runs}</td>
                                            <td>{pc(rec.verdicts[v].avg3m)}</td>
                                            <td>{pc(rec.verdicts[v].worst3m)}</td>
                                        </tr>
                                    ))}
                                    <tr className="jev-record-all"><td>All days</td><td></td><td>{pc(rec.all.avg3m)}</td><td>{pc(rec.all.worst3m)}</td></tr>
                                </tbody>
                            </table>
                            <p className="jev-record-note">{rec.note.replace(/-(\d)/g, '−$1')}</p>
                            <p className="jev-modal-foot">{rec.from.slice(0, 4)}–{rec.to.slice(0, 4)} · {rec.days.toLocaleString('en-US')} days</p>
                        </section>
                    );
                })()}

                {/* One line per number; tap a row for its meaning and rule */}
                <section className="jev-modal-section">
                    {factorSummary && <p className="jev-inputs-summary">{factorSummary}</p>}
                    {factorRows.length > 0 ? (
                        <div className="jev-inputs">
                            {factorRows.map((row, i) => {
                                const multi = String(row.value).includes(' · ');
                                // a −1 vote or a split is bad news; other effects share the verdict colours
                                const tone = SEVERITY[row.effect] || (row.effect === '−1' || row.effect === 'divergence' ? 'badge-red' : 'badge-green');
                                return (
                                    <details key={i} className={`jev-input ${row.hit ? 'jev-input-hit' : ''}`} data-tone={row.hit ? tone : undefined}>
                                        <summary>
                                            <span className="jev-input-label">{INPUT_NAMES[row.label] || row.label}</span>
                                            {!multi && <span className="jev-input-value">{row.value}</span>}
                                            {row.hit && (
                                                <span className={`jev-hit-mark ${tone}`}>{row.effect === 'divergence' ? 'splits' : friendly(row.effect)}</span>
                                            )}
                                        </summary>
                                        {row.plot ? (
                                            // conflict pair: one sentence, the map, the answer
                                            <div className="jev-input-more">
                                                <div>{INPUT_EXPLAIN[row.label]}</div>
                                                <SplitMap plot={row.plot} hit={row.hit} />
                                                <div className={row.hit ? 'sm-verdict sm-verdict-hit' : 'sm-verdict'}>
                                                    {row.plot.x == null || row.plot.y == null ? 'No data today' : row.hit ? 'Dot in a red box → they split' : 'Dot outside the red → they agree'}
                                                </div>
                                            </div>
                                        ) : row.line ? (
                                            // one number: one sentence, the ruler, what it does today
                                            <div className="jev-input-more">
                                                <div>{INPUT_EXPLAIN[row.label]}</div>
                                                <NumberLine line={row.line} />
                                                {row.line.note && <div>{row.line.note}</div>}
                                                <div className={`nl-verdict ${row.hit ? tone : ''}`}>
                                                    {row.line.v == null ? 'No data today' : `Today ${row.value} → ${row.hit ? (pillKey === 'regime' ? `adds ${row.effect}` : friendly(row.effect)) : pillKey === 'regime' ? 'adds 0' : pillKey === 'recession' ? 'no warning' : 'no effect'}`}
                                                </div>
                                            </div>
                                        ) : (
                                        <div className="jev-input-more">
                                            {/* the ticker/code label only; plain-word labels just repeat the name */}
                                            {INPUT_NAMES[row.label] && /[A-Z]{2,}|\d/.test(row.label) && <div className="jev-input-note">{row.label}</div>}
                                            {INPUT_EXPLAIN[row.label] && <div>{INPUT_EXPLAIN[row.label]}</div>}
                                            {multi && String(row.value).split(' · ').map((part, j) => <div key={`v${j}`} className="jev-input-note">{part}</div>)}
                                            {row.note && <div className="jev-input-note">{row.note}</div>}
                                            {String(row.test).split(' · ').map((part, j) => <div key={j}>{part}</div>)}
                                        </div>
                                        )}
                                    </details>
                                );
                            })}
                        </div>
                    ) : (
                        <p className="jev-modal-foot">Inputs unavailable</p>
                    )}
                </section>

                <p className="jev-modal-foot">{sourcesText}</p>
                {backups.length > 0 && <p className="jev-modal-foot">Backups: {backups.join(' · ')}</p>}
            </div>
        </div>,
        document.body
    );
}

const PILL_LABELS = {
    regime: 'Regime',
    recession: 'Recession',
    breadth: 'Breadth',
    hedging: 'Hedges',
    conflict: 'Conflict',
};