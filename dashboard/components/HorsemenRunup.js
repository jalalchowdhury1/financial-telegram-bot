'use client';

/**
 * Four Horsemen — deterioration bars.
 *
 * Replaces the four-line overlay, which normalised every series onto its own hidden
 * scale and so could only show shape. The question this card exists to answer is
 * "how close is this to looking like the run-up to a recession?", and the tell for
 * that is the 12-month CHANGE, not the level (see lib/horsemenRunup.js).
 *
 * Each row: where the horseman has moved in a year (●) against how far it had
 * typically moved by the start of past recessions (│). The yield curve is deliberately
 * a different row — its tell is the inversion a year or more earlier.
 */

import {
    changeOver, preRecessionRunups, runupMedian, horsemanStatus, lastInversion,
} from '../lib/horsemenRunup';

const STATUS_COLOUR = {
    improving: 'var(--green)',
    watch: 'var(--yellow)',
    'recession-like': 'var(--red)',
    inversion: 'var(--text-muted)',
    unknown: 'var(--text-muted)',
};

const ROWS = [
    { key: 'claims', label: 'Jobless claims', mode: 'pct', unit: '%', pick: (f) => f?.horsemen?.claims },
    { key: 'unemployment', label: 'Unemployment', mode: 'pp', unit: 'pp', pick: (f) => f?.horsemen?.unemployment },
    { key: 'spread', label: '10Y − 2Y curve', mode: 'pp', unit: 'pp', pick: (f) => f?.yieldCurve },
    { key: 'bankruptcies', label: 'Bankruptcies', mode: 'pct', unit: '%', pick: (f) => f?.horsemen?.bankruptcies },
];

const fmt = (v, unit) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(unit === 'pp' ? 1 : 0)}${unit}`);

/** Bar with 0 in the middle, today's move as a dot, the pre-recession median as a tick. */
function Bar({ change, median, colour }) {
    // Scale so the median sits at 80% of the half-width; a move past it still fits.
    const half = Math.max(Math.abs(median) * 1.25, Math.abs(change) * 1.1, 1e-9);
    const pos = (v) => 50 + 50 * Math.max(-1, Math.min(1, v / half));
    return (
        <div style={{ position: 'relative', height: 18, flex: 1, minWidth: 90 }}>
            <div style={{ position: 'absolute', top: 8, left: 0, right: 0, height: 2, background: 'rgba(148,163,184,0.18)', borderRadius: 2 }} />
            <div style={{ position: 'absolute', top: 3, left: '50%', width: 1, height: 12, background: 'rgba(148,163,184,0.35)' }} />
            <div title="typical move by the start of past recessions"
                style={{ position: 'absolute', top: 1, left: `${pos(median)}%`, width: 2, height: 16, background: 'var(--red)', opacity: 0.75, transform: 'translateX(-1px)' }} />
            <div data-testid="fh-dot"
                style={{ position: 'absolute', top: 4, left: `${pos(change)}%`, width: 10, height: 10, borderRadius: '50%', background: colour, transform: 'translateX(-5px)' }} />
        </div>
    );
}

function Row({ children, k, status }) {
    return (
        <div data-testid={`fh-row-${k}`} data-status={status}
            style={{ display: 'grid', gridTemplateColumns: 'minmax(96px, 1.1fr) minmax(0, 2fr) minmax(120px, 1.4fr)', gap: 10, alignItems: 'center', padding: '5px 0' }}>
            {children}
        </div>
    );
}

const Label = ({ children }) => (
    <span style={{ fontSize: '0.66rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--text-muted)' }}>{children}</span>
);

const Note = ({ colour, children }) => (
    <span style={{ fontSize: '0.62rem', fontFamily: "'JetBrains Mono', monospace", color: colour, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{children}</span>
);

export default function RunupBars({ fred, now = Date.now() }) {
    const recessions = fred?.recessions || [];
    const rows = ROWS.map((r) => {
        const m = r.pick(fred);
        const history = m?.history;
        const change = changeOver(history, now, r.mode);
        const runups = preRecessionRunups(history, recessions, r.mode);
        const median = runupMedian(runups);
        return { ...r, history, change, runups, median };
    });

    const deteriorating = rows.filter((r) => r.key !== 'spread'
        && ['watch', 'recession-like'].includes(horsemanStatus(r.change, r.median, true))).length;

    return (
        <div>
            {rows.map((r) => {
                if (r.key === 'spread') {
                    const inv = lastInversion(r.history, now);
                    const colour = inv?.currentlyInverted ? 'var(--red)' : STATUS_COLOUR.inversion;
                    return (
                        <Row key={r.key} k={r.key} status="inversion">
                            <Label>{r.label}</Label>
                            <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>
                                {inv ? (inv.currentlyInverted
                                    ? 'inverted now — the tell is firing'
                                    : `inverted ${inv.startYear}–${inv.endYear} · tell fired ${inv.monthsSince} months ago`)
                                    : 'never inverted in this history'}
                            </span>
                            <Note colour={colour}>
                                {r.pick(fred)?.current != null ? `${r.pick(fred).current >= 0 ? '+' : ''}${r.pick(fred).current.toFixed(2)}% now` : '—'}
                                {inv && !inv.currentlyInverted ? ' · re-steepened' : ''}
                            </Note>
                        </Row>
                    );
                }
                const status = horsemanStatus(r.change, r.median, true);
                const colour = STATUS_COLOUR[status];
                return (
                    <Row key={r.key} k={r.key} status={status}>
                        <Label>{r.label}</Label>
                        {r.median == null || r.change == null ? (
                            <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>not enough history</span>
                        ) : (
                            <Bar change={r.change} median={r.median} colour={colour} />
                        )}
                        <Note colour={colour}>
                            {fmt(r.change, r.unit)} vs 1y · recession {fmt(r.median, r.unit)} ({r.runups.length} recessions)
                        </Note>
                    </Row>
                );
            })}
            <div style={{ color: 'var(--text-muted)', fontSize: '0.6rem', marginTop: 6, opacity: 0.85 }}>
                Dot = how far it has moved in 12 months. Red tick = how far it had typically moved by the
                start of past recessions (median). Levels don&apos;t warn — in March 2020 claims sat near
                their calmest ever. {deteriorating} of 3 moving the wrong way.
            </div>
        </div>
    );
}
