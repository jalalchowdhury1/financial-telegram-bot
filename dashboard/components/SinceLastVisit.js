'use client';
/**
 * 👋 One line on open: what changed since the numbers you last saw (lib/lastVisit.js).
 * Neutral colours on purpose — VIX up is not "good", SPY down is not an alarm here;
 * the cards below carry the meaning. ✕ hides it until the next visit.
 */
import { useMemo, useState } from 'react';
import { sinceLastVisit } from '../lib/lastVisit';

export default function SinceLastVisit({ base, live }) {
    const [hiddenFor, setHiddenFor] = useState(null);
    const diff = useMemo(() => {
        try { return sinceLastVisit(base, live, Date.now()); } catch { return null; }
    }, [base, live]);
    if (!diff || hiddenFor === diff.anchorAt) return null;
    return (
        <div className="since-line" role="note" aria-label={`Since you last looked, ${diff.since}`}>
            <span className="since-title">👋 Since {diff.since}</span>
            {diff.items.map((it) => (
                <span key={it.key} className="since-item"><b>{it.label}</b> {it.text}</span>
            ))}
            {diff.prints.length > 0 && (
                <span className="since-item since-prints" title="New or revised data prints since then">
                    🆕 {diff.prints.join(', ')}{diff.more ? ` +${diff.more} more` : ''}
                </span>
            )}
            <button type="button" className="since-close" aria-label="Hide until next visit" onClick={() => setHiddenFor(diff.anchorAt)}>✕</button>
        </div>
    );
}
