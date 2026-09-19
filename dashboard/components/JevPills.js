'use client';

import { useState } from 'react';
import Skeleton from './Skeleton';
import JevPillModal from './JevPillModal';

const SEVERITY = {
    'risk-on':      { badge: 'badge-green',  order: 0 },
    'neutral':      { badge: 'badge-yellow', order: 1 },
    'risk-off':     { badge: 'badge-red',    order: 2 },
    'low':          { badge: 'badge-green',  order: 0 },
    'rising':       { badge: 'badge-yellow', order: 1 },
    'high':         { badge: 'badge-red',    order: 2 },
    'broad':        { badge: 'badge-green',  order: 0 },
    'narrow':       { badge: 'badge-yellow', order: 1 },
    'rolling-over': { badge: 'badge-red',    order: 2 },
    'cheap':        { badge: 'badge-green',  order: 0 },
    'fair':         { badge: 'badge-yellow', order: 1 },
    'expensive':    { badge: 'badge-red',    order: 2 },
    'aligned':      { badge: 'badge-green',  order: 0 },
    'mild-divergence': { badge: 'badge-yellow', order: 1 },
    'major-divergence': { badge: 'badge-red',    order: 2 },
};

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

function friendlyVerdict(v) {
    return FRIENDLY[v] || (v ? v.charAt(0).toUpperCase() + v.slice(1) : v);
}

function formatAsOf(asOf) {
    if (!asOf) return '—';
    try {
        const d = new Date(asOf);
        if (isNaN(d.getTime())) return `as of ${asOf}`;
        const time = d.toLocaleTimeString('en-US', {
            timeZone: 'America/New_York',
            hour: 'numeric',
            minute: '2-digit',
        });
        return `as of ${time} ET`;
    } catch {
        return `as of ${asOf}`;
    }
}

const PILL_ENTRIES = [
    { key: 'regime',    label: 'Regime' },
    { key: 'recession', label: 'Recession' },
    { key: 'breadth',   label: 'Breadth' },
    { key: 'hedging',   label: 'Hedges' },
    { key: 'conflict',  label: 'Conflict' },
];

export default function JevPills({ data, loading }) {
    const [open, setOpen] = useState(null);

    if (data?.enabled === false) return null;

    if (loading) {
        return (
            <div className="card">
                <div className="card-header">
                    <h2>🧪 Jev Regime Pills</h2>
                </div>
                <div className="jev-grid">
                    {Array.from({ length: 5 }).map((_, i) => (
                        <div key={i} className="jev-pill-skeleton">
                            <Skeleton type="text" count={2} />
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    if (!data) return null;

    const { pills, conflictPairs, since, asOf, mode, _meta } = data;

    // Build "Since yesterday" chip
    let sinceChip;
    if (!since || since.noBaseline) {
        sinceChip = (
            <span className="badge badge-blue" title="Since yesterday: no baseline">
                No baseline yet
            </span>
        );
    } else if (since.changed && since.changed.length > 0) {
        const changes = since.changed.map(c => `${c.pill} ${c.from} → ${c.to}`).join(', ');
        const directionLabel =
            since.direction === 'hardening' ? 'Hardening ↑' :
            since.direction === 'softening' ? 'Softening ↓' :
            'Mixed';
        const directionClass =
            since.direction === 'hardening' ? 'badge-red' :
            since.direction === 'softening' ? 'badge-green' :
            'badge-yellow';
        sinceChip = (
            <span className={`badge ${directionClass}`} title={`Since yesterday: ${changes}`}>
                {directionLabel}
            </span>
        );
    } else {
        sinceChip = (
            <span className="badge badge-green" title="Since yesterday: unchanged">
                Unchanged since yesterday
            </span>
        );
    }

    // Build footer
    const footerParts = [formatAsOf(asOf)];
    if (mode === 'rules') footerParts.push('· rules only');
    if (_meta?.jev === 'off') footerParts.push('· Jev off');

    return (
        <div className="card">
            <div className="card-header" style={{ flexWrap: 'wrap', gap: 8 }}>
                <h2>🧪 Jev Regime Pills</h2>
                {sinceChip}
            </div>
            <div className="jev-grid">
                {PILL_ENTRIES.map(({ key, label }) => {
                    const p = pills?.[key];
                    if (!p) return null;
                    const sv = SEVERITY[p.verdict] || { badge: 'badge-blue' };
                    const sourceTag =
                        p.by === 'jev' && Number.isFinite(p.p) ? `Jev ${p.p.toFixed(2)}` :
                        p.by === 'jev' ? 'Jev' :
                        'rule';

                    // Build title
                    let title;
                    if (key === 'conflict' && conflictPairs && conflictPairs.length > 0) {
                        title = conflictPairs.map(cp => cp.pair).join(', ');
                    } else {
                        title = p.reason;
                    }

                    return (
                        <button
                            key={key}
                            type="button"
                            className="jev-pill"
                            onClick={() => setOpen(key)}
                            title={title}
                        >
                            <span className="jev-pill-label">{label}</span>
                            <span className="jev-pill-right">
                                <span className={`badge ${sv.badge}`}>
                                    {friendlyVerdict(p.verdict)}
                                </span>
                                <span className="jev-pill-source">{sourceTag}</span>
                                <span className="jev-pill-chevron">›</span>
                            </span>
                        </button>
                    );
                })}
            </div>
            <div className="jev-footer">
                {footerParts.join(' ')}
            </div>

            {open && (
                <JevPillModal
                    pillKey={open}
                    data={data}
                    onClose={() => setOpen(null)}
                />
            )}
        </div>
    );
}