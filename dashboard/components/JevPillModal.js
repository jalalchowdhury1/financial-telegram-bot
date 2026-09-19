'use client';

import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { PILL_EXPLAIN, INPUT_EXPLAIN, FEED_NAMES, PILL_FEEDS } from './jevExplain';

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
};

const SEVERITY = {
    'risk-on': 'badge-green', 'neutral': 'badge-yellow', 'risk-off': 'badge-red',
    'low': 'badge-green', 'rising': 'badge-yellow', 'high': 'badge-red',
    'broad': 'badge-green', 'narrow': 'badge-yellow', 'rolling-over': 'badge-red',
    'cheap': 'badge-green', 'fair': 'badge-yellow', 'expensive': 'badge-red',
    'aligned': 'badge-green', 'mild-divergence': 'badge-yellow', 'major-divergence': 'badge-red',
};

function friendly(v) {
    return FRIENDLY[v] || (v ? v.charAt(0).toUpperCase() + v.slice(1) : v);
}

/**
 * Which sources does this pill read?
 */
const PILL_SOURCE_KEYS = {
    regime: ['spy', 'fg', 'breadth'],
    recession: ['fred'],
    breadth: ['breadth'],
    hedging: ['vol'],
    conflict: ['spy', 'fg', 'fred', 'breadth'],
};

/**
 * trim a source value to 60 chars with an ellipsis if longer
 */
function trimSource(val) {
    if (!val) return '—';
    return val.length > 60 ? val.slice(0, 60) + '…' : val;
}

export default function JevPillModal({ pillKey, data, onClose }) {
    const backdropRef = useRef(null);

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

    // ---- Header sub-line ----
    let decidedLine;
    if (pill.by === 'jev') {
        decidedLine = `Decided by Jev (confidence ${(pill.p ?? 0).toFixed(2)})`;
    } else {
        decidedLine = 'Decided by the rule';
    }

    // ---- Why section ----
    const conflictPairs = data.conflictPairs || [];

    // ---- Inputs section ----
    const factors = data.factors?.[pillKey];
    const factorRows = factors?.rows || [];
    const factorSummary = factors?.summary || null;

    // ---- Jev's view section ----
    const jevBlock = pill.jev;
    let jevMessage;
    let jevKey = 'answer';
    if (data.mode === 'rules') {
        jevMessage = 'Jev not consulted (rules-only mode)';
        jevKey = 'not-consulted';
    } else if (data._meta?.jev === 'off') {
        jevMessage = 'Jev off — no API key configured';
        jevKey = 'off';
    } else if (jevBlock && jevBlock.verdict) {
        if (pill.by === 'jev') {
            jevMessage = `Jev decided: ${friendly(jevBlock.verdict)} (${(jevBlock.p ?? 0).toFixed(2)})`;
        } else {
            // by === 'rule' — Jev answered but didn't override
            const match = jevBlock.verdict === pill.verdict;
            if (match) {
                jevMessage = `Jev said ${friendly(jevBlock.verdict)} (${(jevBlock.p ?? 0).toFixed(2)}) — agrees with the rule`;
            } else {
                jevMessage = `Jev said ${friendly(jevBlock.verdict)} (${(jevBlock.p ?? 0).toFixed(2)}) — below the 0.6 confidence floor, so the rule stands`;
            }
        }
    } else {
        jevMessage = 'Jev gave no answer this refresh';
        jevKey = 'no-answer';
    }

    // ---- Sources section ----
    const sourceKeys = PILL_SOURCE_KEYS[pillKey] || [];
    const sources = data._meta?.sources || {};
    const sourceEntries = sourceKeys
        .filter(k => sources[k] != null)
        .map(k => `${k}: ${trimSource(sources[k])}`);
    const sourcesText = sourceEntries.length > 0 ? sourceEntries.join(' · ') : '—';

    // "Data through" — the last date each feed this pill reads was updated
    const dataAsOf = data._meta?.dataAsOf || {};
    const throughText = (PILL_FEEDS[pillKey] || [])
        .filter(k => dataAsOf[k])
        .map(k => `${FEED_NAMES[k] || k} ${dataAsOf[k]}`)
        .join(' · ');

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
            <div className="jev-modal" onClick={(e) => e.stopPropagation()}>
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
                        className="jev-modal-close"
                        onClick={onClose}
                        aria-label="Close modal"
                    >
                        ×
                    </button>
                </div>

                {/* What this measures */}
                {explain && (
                    <section className="jev-modal-section">
                        <h3 className="jev-modal-section-label">What this measures</h3>
                        <p className="jev-modal-muted">{explain}</p>
                    </section>
                )}

                {/* Why — only the conflict pill has content the inputs list does not already show */}
                {pillKey === 'conflict' && conflictPairs.length > 0 && (
                    <section className="jev-modal-section">
                        <h3 className="jev-modal-section-label">Why</h3>
                        <div className="jev-modal-muted">
                            {conflictPairs.map((cp, i) => (
                                <p key={i} style={{ marginBottom: i < conflictPairs.length - 1 ? 6 : 0 }}>
                                    {cp.pair}: {cp.detail}
                                </p>
                            ))}
                        </div>
                    </section>
                )}

                {/* Inputs the rule checks */}
                <section className="jev-modal-section">
                    <h3 className="jev-modal-section-label">Inputs the rule checks</h3>
                    {factorSummary && (
                        <p className="jev-modal-muted jev-inputs-summary">{factorSummary}</p>
                    )}
                    {factorRows.length > 0 ? (
                        <div className="jev-inputs">
                            {factorRows.map((row, i) => (
                                <div key={i} className={`jev-input ${row.hit ? 'jev-input-hit' : ''}`}>
                                    <div className="jev-input-main">
                                        <div className="jev-input-label">{row.label}</div>
                                        {INPUT_EXPLAIN[row.label] && (
                                            <div className="jev-input-explain">{INPUT_EXPLAIN[row.label]}</div>
                                        )}
                                    </div>
                                    <div className="jev-input-side">
                                        <div className="jev-input-value">{row.value}</div>
                                        {row.note && <div className="jev-input-note">{row.note}</div>}
                                        <div className="jev-input-rule">{row.test}</div>
                                        <div className="jev-input-fired">
                                            {row.hit ? (
                                                <span className="jev-hit-mark">✓ fired {row.effect}</span>
                                            ) : (
                                                <span className="jev-no-hit">not fired</span>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="jev-modal-muted">Inputs unavailable</p>
                    )}
                </section>

                {/* Jev's view */}
                <section className="jev-modal-section">
                    <h3 className="jev-modal-section-label">Jev&rsquo;s view</h3>
                    <p className="jev-modal-muted" key={jevKey}>{jevMessage}</p>
                </section>

                {/* Sources */}
                <section className="jev-modal-section">
                    <h3 className="jev-modal-section-label">Sources</h3>
                    <p className="jev-modal-muted jev-modal-sources">{sourcesText}</p>
                    {throughText && (
                        <p className="jev-modal-muted jev-modal-sources">Data through: {throughText}</p>
                    )}
                </section>
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