'use client';

import Skeleton from './Skeleton';

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

function Pill({ label, verdict, p, by, reason, conflictPairs }) {
    const sv = SEVERITY[verdict] || { badge: 'badge-blue' };
    const sourceTag = by === 'jev' && Number.isFinite(p) ? `Jev ${p.toFixed(2)}` : by === 'jev' ? 'Jev' : 'rule';

    // Build title: conflict pill lists the pairs; others show the reason
    let title;
    if (label === 'Conflict' && conflictPairs && conflictPairs.length > 0) {
        title = conflictPairs.map(cp => cp.pair).join(', ');
    } else {
        title = reason;
    }

    return (
        <div className="indicator-pill" title={title}>
            <div className="label">{label}</div>
            <div className="value" style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                <span className={`badge ${sv.badge}`}>{verdict}</span>
                <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                    {sourceTag}
                </span>
            </div>
        </div>
    );
}

export default function JevPills({ data, loading }) {
    if (data?.enabled === false) return null;

    if (loading) {
        return (
            <div className="card">
                <div className="card-header">
                    <h2>🧪 Jev Regime Pills</h2>
                </div>
                <div className="indicator-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
                    {Array.from({ length: 5 }).map((_, i) => (
                        <div key={i} className="indicator-pill">
                            <Skeleton type="text" count={2} />
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    if (!data) return null;

    const { pills, conflictPairs, since, asOf, mode } = data;

    // Build "Since yesterday" chip
    let sinceChip;
    if (!since || since.noBaseline) {
        sinceChip = <span className="badge badge-blue">Since yesterday: no baseline yet</span>;
    } else if (since.changed && since.changed.length > 0) {
        const changes = since.changed.map(c => `${c.pill} ${c.from} → ${c.to}`).join(', ');
        sinceChip = (
            <span className={`badge ${since.direction === 'hardening' ? 'badge-red' : since.direction === 'softening' ? 'badge-green' : 'badge-yellow'}`}>
                Since yesterday: {since.direction} ({changes})
            </span>
        );
    } else {
        sinceChip = <span className="badge badge-green">Since yesterday: none</span>;
    }

    const pillEntries = [
        { key: 'regime',    label: 'Regime' },
        { key: 'recession', label: 'Recession' },
        { key: 'breadth',   label: 'Breadth' },
        { key: 'hedging',   label: 'Hedges' },
        { key: 'conflict',  label: 'Conflict' },
    ];

    return (
        <div className="card">
            <div className="card-header">
                <h2>🧪 Jev Regime Pills</h2>
                {sinceChip}
            </div>
            <div className="indicator-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
                {pillEntries.map(({ key, label }) => {
                    const p = pills?.[key];
                    if (!p) return null;
                    return (
                        <Pill
                            key={key}
                            label={label}
                            verdict={p.verdict}
                            p={p.p}
                            by={p.by}
                            reason={p.reason}
                            conflictPairs={label === 'Conflict' ? conflictPairs : undefined}
                        />
                    );
                })}
            </div>
            <div style={{ marginTop: 12, fontSize: '0.72rem', color: 'var(--text-muted)', display: 'flex', gap: 16 }}>
                <span>as of {asOf || '—'}</span>
                {mode === 'rules' && <span>mode: rules</span>}
            </div>
        </div>
    );
}