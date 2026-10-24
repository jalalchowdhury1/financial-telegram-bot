'use client';
import ErrorBoundary from './ErrorBoundary';
import Skeleton from './Skeleton';
import Delta from './Delta';
import { useMark } from './MarkProvider';
import { freshnessNote } from '../lib/freshness';

const TOOLTIPS = {
    nfci: 'Chicago Fed money tightness. Below 0 = easy (good for stocks).',
    m2: 'Cash + deposits. Growing = tailwind for stocks.',
    retail: 'Retail sales, 3-month growth. Consumers drive 70% of GDP.',
    housing: 'New homes started. OK also needs a rising trend.',
    indpro: 'Factory output. Often turns before the economy.',
    jolts: 'Job openings (JOLTS). High = firms want workers.',
    durable: 'Big-ticket orders, ex transport. Tracks business investment.',
    savings: 'Share of income saved. A cushion for spending.'
};

const SUBTITLES = {
    nfci: 'How tight money is (below 0 = easy)',
    m2: 'Strong over 4% · OK over 2% a year',
    retail: 'Consumer spending strength',
    housing: 'Strong over 1,400K · OK over 1,300K',
    indpro: '6-month manufacturing trend',
    jolts: 'Strong over 7,000K · OK over 6,000K',
    durable: 'Business investment (3mo trend)',
    savings: 'Healthy 5%+ · OK 3.5%+'
};

const BENCHMARKS = {
    nfci: (status) => status === 'strong' ? '← Easy' : status === 'good' ? '← Easy' : '← Tight',
    m2: (status) => status === 'strong' ? '← Growing' : status === 'good' ? '← Growing' : '← Contracting',
    retail: (status) => status === 'strong' ? '← Growing' : status === 'good' ? '← Growing' : '← Declining',
    housing: (status) => status === 'strong' ? '← Strong' : status === 'good' ? '← OK' : '← Weak',
    indpro: (status) => status === 'strong' ? '← Expanding' : status === 'good' ? '← Expanding' : status === 'contracting' ? '← Contracting' : '← Contracting',
    jolts: (status) => status === 'strong' ? '← Strong' : status === 'good' ? '← OK' : '← Weak',
    durable: (status) => status === 'strong' ? '← Rising' : status === 'good' ? '← Rising' : '← Falling',
    savings: (status) => status === 'strong' ? '← Healthy' : status === 'good' ? '← OK' : '← Low'
};

/** How each checklist value is rendered. Shared by the tile and its "was" popover. */
function fmtValue(key, v) {
    if (typeof v !== 'number') return 'N/A';
    if (key === 'housing' || key === 'jolts') return `${v.toFixed(0)}K`;
    if (key === 'nfci') return v.toFixed(2);
    return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;
}

/**
 * One checklist tile. Extracted from the map so it can call `useMark` — hooks
 * cannot run inside a callback. The checklist keys are already the marks keys.
 */
function ChecklistItem({ itemKey, item }) {
    const mark = useMark(itemKey, item.value);
    const note = freshnessNote(item);
    const icon = note.tone === 'unavailable' ? '\u26AA'
        : note.tone === 'stale' ? '\u{1F550}'
        : item.bullish ? '\u2705' : '\u{1F534}';
    // A stale or unavailable number is not a fresh print — never mark it.
    const shown = note.tone === 'fresh' ? mark : null;
    return (
        <div className="checklist-item" style={{ padding: '12px 14px', alignItems: 'center' }}>
            <span className="checklist-icon">{icon}</span>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '3px' }}>
                <span
                    className="checklist-text tooltip-trigger"
                    style={{ flex: 'none', color: 'var(--text-primary)' }}
                    data-tooltip={`${TOOLTIPS[itemKey]}${note.suffix}`}
                >
                    {item.label}
                </span>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', lineHeight: 1.2, whiteSpace: 'nowrap' }}>
                    {SUBTITLES[itemKey]}
                </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '3px' }}>
                <Delta mark={shown} format={(v) => fmtValue(itemKey, v)} chartKey={itemKey} raw={item.value}
                    className={`checklist-value ${item.bullish ? 'stat-positive' : 'stat-negative'}`}>
                    <span style={{ fontSize: '0.95rem', ...(note.tone === 'stale' ? { color: 'var(--orange)' } : note.tone === 'unavailable' ? { color: 'var(--yellow)' } : {}) }}>
                        {fmtValue(itemKey, item.value)}
                    </span>
                </Delta>
                <span className="checklist-benchmark" style={{ opacity: 0.9 }}>
                    {BENCHMARKS[itemKey] ? BENCHMARKS[itemKey](item.status) : '\u2014'}
                </span>
            </div>
        </div>
    );
}

export default function BullChecklist({ fred, loading }) {
    return (
        <div className="card full-width" style={{ animationDelay: '0.6s' }}>
            <div className="card-header">
                <h2>📋 Bull Market Checklist</h2>
                {fred?.checklist && (() => {
                    const items = Object.values(fred.checklist);
                    const bullish = items.filter(i => i.bullish).length;
                    const pct = items.length ? (bullish / items.length) * 100 : 0;
                    return <span className={`badge ${pct >= 75 ? 'badge-green' : pct >= 50 ? 'badge-yellow' : 'badge-red'}`}>{bullish}/{items.length} ({pct.toFixed(0)}%)</span>;
                })()}
            </div>
            <ErrorBoundary>
                {loading || !fred || fred.error ? <Skeleton count={8} /> : (
                    <>
                        <div className="checklist-grid">
                            {Object.entries(fred.checklist).map(([key, item]) => (
                                <ChecklistItem key={key} itemKey={key} item={item} />
                            ))}
                        </div>
                        {(() => {
                            const items = Object.values(fred.checklist);
                            const bullish = items.filter(i => i.bullish).length;
                            const pct = items.length ? (bullish / items.length) * 100 : 0;
                            const regime = pct >= 75 ? '🟢 BULL MARKET' : pct >= 50 ? '🟡 MIXED' : '🔴 BEAR WARNING';
                            const bg = pct >= 75 ? 'var(--green-bg)' : pct >= 50 ? 'var(--yellow-bg)' : 'var(--red-bg)';
                            const color = pct >= 75 ? 'var(--green)' : pct >= 50 ? 'var(--yellow)' : 'var(--red)';
                            return (
                                <div className="checklist-score" style={{ background: bg, color }}>
                                    {regime} · {bullish}/{items.length} bullish
                                </div>
                            );
                        })()}
                    </>
                )}
            </ErrorBoundary>
        </div>
    );
}
