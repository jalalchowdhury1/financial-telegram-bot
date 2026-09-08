'use client';
import { useEffect, useState } from 'react';
import ErrorBoundary from './ErrorBoundary';
import Skeleton from './Skeleton';
import { DIAL_ORDER, dialLabel } from '../lib/rubberBand';

/**
 * 🪢 Rubber Band Radar v1.1 — "is the dip-buying regime still alive?" in five dials, plus the
 * decision layer that turns a red into an action.
 *
 * Everything shown here is computed nightly on the Mac mini (scripts/rubber_band.py and
 * scripts/defensive_trigger.py) and relayed by /api/rubber-band; this component only draws it.
 * Tap (or double-click) any dial for its ELI5: what it measures, when it goes red/amber, today's
 * numbers and the backtest record. The rules (docs/rubber-band.md, v1.1 — 2026-09-08):
 *   1. slow     — last 30 oversold dips (RSI-10 < 32): did buying them beat an ordinary day?
 *                 Below zero on 45 of the last 60 trading days = STOP. 1971→: 8 alarms, none since 1996.
 *   2. fast     — same, last 20 dips. LOOK only: can nudge OK → WATCH, never fires the trigger.
 *   3. age      — years the 30 dips span. Trust gauge, never counts in the verdict.
 *   4. rip      — last 30 overbought days (RSI-10 > 79): rips still running on 45 of 60 = the 1970s.
 *   5. machines — each leg's Composer backtest vs its written line AND its own records (deepest
 *                 completed drawdown, longest underwater), plus the hedge-failure check on GLD/BTAL.
 *   Trigger: slow/rip red for 1 close, or machines red 5 closes in a row → GO DEFENSIVE (half the
 *   book to cash, hourly nag until "done"); 10 green closes with slow > +0.2% → RE-ENTER. Its state
 *   arrives as snapshot.defensive (stamped by the trigger after every evaluation).
 */

const COLOUR_VAR = { green: 'var(--green)', amber: 'var(--orange)', red: 'var(--red)', grey: 'var(--text-muted)' };
const COLOUR_BG = { green: 'var(--green-bg)', amber: 'var(--orange-bg)', red: 'var(--red-bg)', grey: 'rgba(148,163,184,0.12)' };
const COLOUR_WORD = { green: 'OK', amber: 'WATCH', red: 'STOP', grey: 'NO DATA' };
const BADGE = { green: 'badge-green', amber: 'badge-yellow', red: 'badge-red', grey: 'badge-blue' };

const signed = (v, d = 2) => (v == null || !Number.isFinite(v) ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(d)}%`);
const pct0 = (v) => (v == null || !Number.isFinite(v) ? '—' : `${Math.round(v * 100)}%`);
const num = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(d));
const ceil75 = (n) => Math.ceil(0.75 * (n || 0));

// Defaults mirror scripts/rubber_band.py SPEC v1.1; the snapshot's own `spec` wins when present.
const SPEC_DEFAULT = {
    dip: { rsi_below: 32, slow_n: 30, fast_n: 20, stop_of: 45, stop_window: 60 },
    rip: { rsi_above: 79, n: 30, hot_of: 45, hot_window: 60 },
    machines: { near_line_pts: 10, min_history_days: 250, record_amber_frac: 0.85, underwater_amber_frac: 0.75, fast_window_days: 20, book_drop_pct: -10 },
};
const RULES_DEFAULT = { fire_after: { slow: 1, rip: 1, machines: 5 }, reentry_closes: 10, reentry_edge_pct: 0.2 };

function contextOf(data) {
    const spec = data?.spec || {};
    return {
        version: spec.version || '1.0',
        dip: { ...SPEC_DEFAULT.dip, ...(spec.dip || {}) },
        rip: { ...SPEC_DEFAULT.rip, ...(spec.rip || {}) },
        machines: { ...SPEC_DEFAULT.machines, ...(spec.machines || {}) },
        rules: { ...RULES_DEFAULT, ...(data?.defensive?.rules || {}) },
    };
}

// ELI5 per dial: what it measures, when it turns, today's numbers, the honest record.
const EXPLAIN = {
    slow: (d, c) => {
        const window = d.window ?? c.dip.stop_window, stopAfter = d.stop_after ?? c.dip.stop_of;
        return {
            what: `Of the last ${d.n} times QQQ got oversold (Wilder RSI-10 under ${c.dip.rsi_below} — the machine's own dip trigger), did buying that close beat an ordinary day? Above zero = the rubber band still snaps back. This is the dial the trigger listens to.`,
            today: `${signed(d.excess_pct)} per dip vs an ordinary day (noise ±${num(d.se_pct, 2)}%) · dips paid ${pct0(d.hit)} of the time · evidence ${d.first_event ?? '—'} → ${d.last_event ?? '—'}.`,
            red: `Below zero on ${stopAfter} of the last ${window} trading days — today it is ${d.red_days ?? 0} of ${window}.`,
            amber: `Today's number is below zero, or ${ceil75(stopAfter)}+ of the last ${window} days were.`,
            record: `1971→ backtest: 8 alarms, all in the 1970s–early 1990s, zero false alarms, no defensive day since 1996. Blind spot: the 2001 and 2008 grinding bears never showed here — Machine health covers those.`,
        };
    },
    fast: (d, c) => {
        const window = d.window ?? c.dip.stop_window, stopAfter = d.stop_after ?? c.dip.stop_of;
        return {
            what: `The same test on only the last ${d.n} dips. It reacts months earlier in a real regime flip (led by 200+ days in the 1970s) but is noisy: every alarm since 1993 was false.`,
            today: `${signed(d.excess_pct)} per dip (noise ±${num(d.se_pct, 2)}%) · ${d.red_days ?? 0} of the last ${window} days below zero.`,
            red: `Same count (${stopAfter} of ${window} days below zero) — shown so you can watch it, but it never fires the trigger. It can only nudge the verdict from OK to WATCH.`,
            amber: `Today's number below zero. Read it as LOOK, not act.`,
            record: `Leads the slow dial by months when a flip is real; 12 false alarms since 1993, so it never gets a vote.`,
        };
    },
    age: (d, c) => ({
        what: `How many years the ${c.dip.slow_n} dips behind "Dip pays?" span. Fresh evidence means the verdict is about today's market; old evidence means the market has been too calm to test the band lately.`,
        today: `${num(d.years, 1)} years of evidence · ${d.events_last_12m ?? '—'} dips in the last 12 months.`,
        red: `Over ${d.red_years} years old — but this dial never counts in the verdict. It is a trust gauge, not a forecast.`,
        amber: `Over ${d.amber_years} years old.`,
        record: `Information only, in every version of the rules.`,
    }),
    rip: (d, c) => {
        const window = d.window ?? c.rip.hot_window, redAfter = d.red_after ?? c.rip.hot_of;
        return {
            what: `The other half of the machine: after the last ${d.n} overbought days (RSI-10 above ${c.rip.rsi_above}, the sell trigger), did the market fade the next day? Negative = rips fade = normal since 2000. Rips that keep running = the 1970s.`,
            today: `${signed(d.excess_pct)} next-day drift after a rip (noise ±${num(d.se_pct, 2)}%) · rips faded ${pct0(1 - (d.hit ?? 0))} of the time.`,
            red: `Rips kept running (above zero) on ${redAfter} of the last ${window} trading days — today ${d.hot_days ?? 0} of ${window}.`,
            amber: `Today's number above zero, or ${ceil75(redAfter)}+ of the last ${window} days were.`,
            record: `Never red since 2000. Fires the trigger on its own, like "Dip pays?".`,
        };
    },
    machines: (d, c) => {
        const m = c.machines;
        const lines = (d.legs || []).filter((l) => l.line_pct != null).map((l) => `${l.name} ${l.line_pct}%`).join(', ') || 'none written';
        return {
            what: `Are my own machines doing something they have never done? Each leg is its Composer backtest curve (not the account balance), checked every night against its written line and against its own history.`,
            today: `${d.hedge_check ? `Hedge check: ${d.hedge_check}. ` : ''}${d.reasons?.length ? d.reasons.join(' · ') : 'All legs inside their lines, nothing unprecedented.'}`,
            red: `Any leg through its written line (${lines}) · a drawdown deeper than that leg's worst completed one · underwater longer than it has ever been · or hedge failure: the book (C3 68 / m1 20 / hedges 12) down ${Math.abs(m.book_drop_pct)}%+ in ${m.fast_window_days} trading days while the GLD/BTAL hedges did not rise.`,
            amber: `Within ${m.near_line_pts} points of a line · ${Math.round(m.record_amber_frac * 100)}% of the record drawdown · ${Math.round(m.underwater_amber_frac * 100)}% of the record underwater stretch (at least 3 months).`,
            record: `Records only count once a leg has ${m.min_history_days}+ days of history before its peak (Composer era, 2013→ / 2016→). This is the dial that catches grinding bears like 2001 and 2008, so the trigger waits for ${c.rules.fire_after.machines} red closes in a row here. m1-vs-C3 lag is shown for information, never judged.`,
        };
    },
};

function headline(key, d) {
    if (d.colour === 'grey') return { big: '—', sub: d.reason || 'not enough data' };
    switch (key) {
        case 'slow':
        case 'fast':
            return { big: signed(d.excess_pct), sub: `${d.n} dips · hit ${pct0(d.hit)} · neg ${d.red_days ?? 0}/${d.window ?? 60}d` };
        case 'age':
            return { big: `${num(d.years, 1)}y`, sub: `${d.events_last_12m ?? '—'} dips last 12m · amber >${d.amber_years}y` };
        case 'rip':
            return { big: signed(d.excess_pct), sub: `${d.n} rips · fade ${pct0(1 - (d.hit ?? 0))} · hot ${d.hot_days ?? 0}/${d.window ?? 60}d` };
        case 'machines': {
            const worst = (d.legs || []).filter((l) => l.dd_pct != null && l.role !== 'hedge').sort((a, b) => a.dd_pct - b.dd_pct)[0];
            return { big: worst ? `${worst.name} ${signed(worst.dd_pct, 0)}` : '—', sub: d.reasons?.length ? d.reasons[0] : 'all legs inside their lines' };
        }
        default:
            return { big: '—', sub: '' };
    }
}

/** One dial. Tap toggles its explanation panel; double-click always opens it; Enter/Space too. */
function Dial({ k, d, active, onToggle, onOpen, tooltip }) {
    const { big, sub } = headline(k, d);
    const col = COLOUR_VAR[d.colour] || COLOUR_VAR.grey;
    return (
        <div
            data-testid="rb-dial"
            data-colour={d.colour}
            role="button"
            tabIndex={0}
            aria-expanded={active}
            aria-label={`${dialLabel(k)}: ${COLOUR_WORD[d.colour]}. Tap for the rule.`}
            className="tooltip-trigger"
            data-tooltip={tooltip}
            onClick={onToggle}
            onDoubleClick={(e) => { e.preventDefault(); onOpen(); }}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle(); } }}
            style={{ background: COLOUR_BG[d.colour] || COLOUR_BG.grey, border: `${active ? 2 : 1}px solid ${col}`, borderRadius: 10, padding: '10px 12px', minWidth: 0, cursor: 'pointer', userSelect: 'none' }}
        >
            <div data-testid={`rb-dial-${k}`} data-colour={d.colour} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{dialLabel(k)}</span>
                <span style={{ fontSize: '0.62rem', fontWeight: 700, color: col }}>● {COLOUR_WORD[d.colour]}{k === 'fast' && d.colour !== 'green' ? ' (look)' : ''}</span>
            </div>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: col, fontVariantNumeric: 'tabular-nums', marginTop: 4, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{big}</div>
            <div style={{ fontSize: '0.66rem', color: 'var(--text-muted)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{sub}</div>
            <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: 4, opacity: 0.75 }}>{active ? '▴ close' : '▾ tap for the rule'}</div>
        </div>
    );
}

function ExplainRow({ tag, text, colour }) {
    return (
        <>
            <span style={{ color: colour || 'var(--text-muted)', fontWeight: 700, whiteSpace: 'nowrap', fontSize: '0.7rem' }}>{tag}</span>
            <span style={{ lineHeight: 1.45 }}>{text}</span>
        </>
    );
}

/** The ELI5 panel for the open dial — one at a time, drawn under the dial row so phones stay readable. */
function ExplainPanel({ k, d, ctx, onClose }) {
    const e = EXPLAIN[k] ? EXPLAIN[k](d, ctx) : null;
    if (!e) return null;
    const col = COLOUR_VAR[d.colour] || COLOUR_VAR.grey;
    return (
        <div data-testid={`rb-explain-${k}`} role="region" aria-label={`${dialLabel(k)} explained`}
            style={{ marginTop: 8, padding: '10px 12px', borderRadius: 10, border: `1px solid ${col}`, background: 'rgba(148,163,184,0.06)', fontSize: '0.76rem', display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 10px', alignItems: 'start' }}>
            <span style={{ gridColumn: '1 / -1', fontWeight: 700, color: col }}>{dialLabel(k)} — {COLOUR_WORD[d.colour]}</span>
            <ExplainRow tag="What" text={e.what} />
            <ExplainRow tag="Today" text={e.today} colour={col} />
            <ExplainRow tag="🔴 Red when" text={e.red} colour="var(--red)" />
            <ExplainRow tag="🟠 Amber when" text={e.amber} colour="var(--orange)" />
            <ExplainRow tag="Record" text={e.record} />
            <span style={{ gridColumn: '1 / -1', textAlign: 'right' }}>
                <button type="button" onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.7rem', padding: 0 }}>close ▴</button>
            </span>
        </div>
    );
}

/** Slow (solid) + fast (dashed) excess lines over the published history, zero line drawn. */
function BandChart({ history }) {
    const pts = (history || []).filter((h) => h.slow != null);
    if (pts.length < 20) return null;
    const W = 720, H = 120, padL = 34, padR = 8, padT = 8, padB = 18;
    const vals = pts.flatMap((h) => [h.slow, h.fast ?? h.slow]).filter((v) => v != null);
    const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    const span = hi - lo || 1;
    const x = (i) => padL + (i / (pts.length - 1)) * (W - padL - padR);
    const y = (v) => padT + (1 - (v - lo) / span) * (H - padT - padB);
    const path = (key) => pts.map((h, i) => (h[key] == null ? null : `${i === 0 || pts[i - 1][key] == null ? 'M' : 'L'}${x(i).toFixed(1)},${y(h[key]).toFixed(1)}`)).filter(Boolean).join(' ');
    const last = pts[pts.length - 1];
    const ticks = [lo, 0, hi].filter((v, i, a) => a.indexOf(v) === i);
    const first = pts[0].d, mid = pts[Math.floor(pts.length / 2)].d;
    return (
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Slow and fast dip-payoff lines over the last three years, with the zero line" style={{ width: '100%', height: 'auto', display: 'block' }}>
            {ticks.map((v) => (
                <g key={v}>
                    <line x1={padL} x2={W - padR} y1={y(v)} y2={y(v)} stroke={v === 0 ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.08)'} strokeDasharray={v === 0 ? '' : '2 4'} />
                    <text x={padL - 4} y={y(v) + 3} fontSize="9" fill="var(--text-muted)" textAnchor="end">{signed(v, 1)}</text>
                </g>
            ))}
            <rect x={padL} y={y(0)} width={W - padL - padR} height={Math.max(0, y(lo) - y(0))} fill="rgba(239,68,68,0.06)" />
            <path d={path('fast')} fill="none" stroke="var(--text-muted)" strokeWidth="1.2" strokeDasharray="3 3" opacity="0.9" />
            <path d={path('slow')} fill="none" stroke={last.slow >= 0 ? 'var(--green)' : 'var(--red)'} strokeWidth="2" />
            <circle cx={x(pts.length - 1)} cy={y(last.slow)} r="3" fill={last.slow >= 0 ? 'var(--green)' : 'var(--red)'} />
            <text x={padL} y={H - 5} fontSize="9" fill="var(--text-muted)">{first}</text>
            <text x={(padL + W - padR) / 2} y={H - 5} fontSize="9" fill="var(--text-muted)" textAnchor="middle">{mid}</text>
            <text x={W - padR} y={H - 5} fontSize="9" fill="var(--text-muted)" textAnchor="end">{last.d}</text>
            <text x={W - padR} y={padT + 9} fontSize="9" fill="var(--text-muted)" textAnchor="end">— slow (30 dips)   - - fast (20 dips)   · below zero = dips lose</text>
        </svg>
    );
}

/** Machine legs: drawdown vs the written line AND vs each leg's own completed record (v1.1). */
function Legs({ m, spm }) {
    const legs = m.legs || [];
    if (!legs.length) return null;
    const cell = { textAlign: 'right', fontVariantNumeric: 'tabular-nums' };
    const head = { ...cell, color: 'var(--text-muted)' };
    const recordOk = (l) => l.worst_dd_prior_pct != null && (l.days_before_peak == null || l.days_before_peak >= spm.min_history_days);
    const ddColour = (l) => {
        if (l.dd_pct == null) return 'var(--text-muted)';
        if (l.line_pct != null && l.dd_pct <= l.line_pct) return 'var(--red)';
        if (recordOk(l) && l.dd_pct <= l.worst_dd_prior_pct) return 'var(--red)';
        if (l.line_pct != null && l.dd_pct <= l.line_pct + spm.near_line_pts) return 'var(--orange)';
        if (recordOk(l) && l.dd_pct <= l.worst_dd_prior_pct * spm.record_amber_frac) return 'var(--orange)';
        return 'inherit';
    };
    const underwater = (l) => (l.months_underwater == null ? '—' : `${l.months_underwater}${l.longest_underwater_prior_months != null ? ` / ${l.longest_underwater_prior_months}` : ''} mo`);
    return (
        <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr 0.8fr 1.2fr 1fr', gap: '3px 8px', fontSize: '0.72rem', marginTop: 10 }}>
            <span style={{ color: 'var(--text-muted)' }}>Leg</span>
            <span style={head}>Drawdown</span>
            <span style={head}>Line</span>
            <span style={head}>Under water now / record</span>
            <span style={head}>Record DD</span>
            {legs.map((l) => (
                <LegRow key={l.name} l={l} cell={cell} colour={ddColour(l)} underwater={underwater(l)} />
            ))}
            {m.hedge_check && (
                <span data-testid="rb-hedge-check" style={{ gridColumn: '1 / -1', color: 'var(--text-muted)', fontSize: '0.68rem' }}>
                    Hedge check: {m.hedge_check} — STOP if the book drops {Math.abs(spm.book_drop_pct)}%+ in {spm.fast_window_days} days while the hedges do not rise.
                </span>
            )}
            {m.lag_pair && (
                <span style={{ gridColumn: '1 / -1', color: 'var(--text-muted)', fontSize: '0.68rem' }}>
                    {m.lag_pair[0]} vs {m.lag_pair[1]} lag: {m.lag_months ?? '—'} month{m.lag_months === 1 ? '' : 's'}{m.lag_is_info_only === false ? ' (exit rule at 2)' : ' — information only, not a rule'}.
                </span>
            )}
        </div>
    );
}

function LegRow({ l, cell, colour, underwater }) {
    return (
        <>
            <span style={{ fontWeight: 600 }}>{l.name}{l.missing ? <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> (no curve)</span> : ''}</span>
            <span style={{ ...cell, color: colour, fontWeight: 600 }}>{signed(l.dd_pct, 1)}</span>
            <span style={cell}>{l.role === 'hedge' ? 'hedge' : l.line_pct == null ? 'none' : `${l.line_pct}%`}</span>
            <span style={cell}>{underwater}</span>
            <span style={{ ...cell, color: 'var(--text-muted)' }}>{signed(l.worst_dd_prior_pct ?? l.worst_dd_pct, 0)}</span>
        </>
    );
}

const MODE = {
    INVESTED: { badge: 'badge-green', word: 'INVESTED', eli5: 'All machines running. Nothing to do.' },
    PENDING_DEFENSIVE: { badge: 'badge-red', word: 'GO DEFENSIVE — waiting for you', eli5: 'The 📡 thread has the 2-minute steps: sell half of each machine to cash, then reply "done".' },
    DEFENSIVE: { badge: 'badge-yellow', word: 'DEFENSIVE — half in cash', eli5: 'Half the book sits in cash until the radar has been green enough closes in a row.' },
    PENDING_REENTRY: { badge: 'badge-blue', word: 'RE-ENTER — waiting for you', eli5: 'The 📡 thread has the steps: put the cash back into each machine, then reply "done".' },
    UNKNOWN: { badge: 'badge-blue', word: 'UNKNOWN', eli5: 'The trigger could not report its state — check the Mac mini log.' },
};

/** The decision layer: live mode + streak counters, and the rule in one breath on tap. */
function Trigger({ def, rules, open, onToggle }) {
    const mode = def ? (MODE[def.mode] || MODE.UNKNOWN) : null;
    const st = def?.streak || {};
    const fa = rules.fire_after;
    const rule = `Fires when "Dip pays?" or "Rips fade?" is red for ${fa.slow} close, or Machine health is red ${fa.machines} closes in a row → 📡 GO DEFENSIVE: sell half of each machine to cash, hourly nag until you reply "done". Back in after ${rules.reentry_closes} green closes in a row with "Dip pays?" above +${rules.reentry_edge_pct}% → 📡 RE-ENTER, nag until "done". If the alarm clears before you act, it stands down by itself; if the green run breaks before you re-enter, it holds.`;
    return (
        <div data-testid="rb-trigger" style={{ marginTop: 12, padding: '10px 12px', borderRadius: 10, border: '1px solid rgba(148,163,184,0.25)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>What happens on red</span>
                {mode ? <span className={`badge ${mode.badge}`} data-testid="rb-mode">{mode.word}</span> : <span className="badge badge-blue">state not published yet</span>}
            </div>
            <div style={{ fontSize: '0.76rem', marginTop: 6, lineHeight: 1.45 }}>
                {mode ? mode.eli5 : 'The trigger stamps its state into the next nightly snapshot.'}
                {def?.defensive_since && def.mode !== 'INVESTED' ? ` Defensive since ${def.defensive_since}.` : ''}
            </div>
            {def && (
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 4, fontVariantNumeric: 'tabular-nums' }}>
                    Red closes in a row — dips {st.slow ?? 0}/{fa.slow} · rips {st.rip ?? 0}/{fa.rip} · machines {st.machines ?? 0}/{fa.machines} · green run {def.green_streak ?? 0}/{rules.reentry_closes}{def.last_asof ? ` · counted to ${def.last_asof}` : ''}
                </div>
            )}
            <button type="button" onClick={onToggle} aria-expanded={open}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.68rem', padding: 0, marginTop: 6 }}>
                {open ? '▴ hide the rule' : '▾ the rule in one breath'}
            </button>
            {open && <div data-testid="rb-trigger-rule" style={{ fontSize: '0.76rem', marginTop: 6, lineHeight: 1.45 }}>{rule}</div>}
        </div>
    );
}

export default function RubberBandRadar() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [open, setOpen] = useState(null);          // dial key, 'trigger', or null — one panel at a time

    useEffect(() => {
        const load = async () => {
            try {
                const res = await fetch('/api/rubber-band');
                const json = await res.json();
                if (!json || !json.dials || !json.verdict) setError(json?._meta?.messages?.[0] || 'no data');
                else setData(json);
            } catch {
                setError('fetch failed');
            } finally {
                setLoading(false);
            }
        };
        load();
    }, []);

    const verdictColour = data?.verdict?.colour || 'grey';
    const stale = !!data?._meta?.stale;
    const ctx = contextOf(data);
    const toggle = (k) => setOpen((cur) => (cur === k ? null : k));

    return (
        <div className="card" style={{ gridColumn: '1 / -1', animationDelay: '0.5s' }}>
            <div className="card-header">
                <h2>🪢 Rubber Band Radar</h2>
                <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    {stale && <span className="badge badge-red">Stale</span>}
                    <span className={`badge ${BADGE[verdictColour]}`}>{loading ? '…' : COLOUR_WORD[verdictColour]}</span>
                </span>
            </div>
            <ErrorBoundary>
                {loading ? <Skeleton count={4} /> : error || !data ? (
                    <div className="error-message" style={{ color: 'var(--text-muted)' }}>
                        ⚠️ Rubber band snapshot unavailable{error ? ` (${error})` : ''}. The Mac mini publishes it after each close.
                    </div>
                ) : (
                    <>
                        <div style={{ fontSize: '0.86rem', color: COLOUR_VAR[verdictColour], fontWeight: 600, marginBottom: 12, lineHeight: 1.4 }}>
                            {data.verdict.text}
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8 }}>
                            {DIAL_ORDER.map((k) => (
                                <Dial key={k} k={k} d={data.dials[k]} active={open === k} onToggle={() => toggle(k)} onOpen={() => setOpen(k)}
                                    tooltip={EXPLAIN[k] ? EXPLAIN[k](data.dials[k], ctx).what : ''} />
                            ))}
                        </div>
                        {open && data.dials[open] && <ExplainPanel k={open} d={data.dials[open]} ctx={ctx} onClose={() => setOpen(null)} />}
                        <div style={{ marginTop: 12 }}>
                            <BandChart history={data.history} />
                        </div>
                        <Legs m={data.dials.machines} spm={ctx.machines} />
                        <Trigger def={data.defensive} rules={ctx.rules} open={open === 'trigger'} onToggle={() => toggle('trigger')} />
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.65rem', marginTop: 10, opacity: 0.8 }}>
                            As of {data.asOf} · rules v{ctx.version} · QQQ Wilder RSI-10 (dips &lt;32, rips &gt;79) · tap any dial for its rule{stale ? ` · STALE (${data._meta?.ageDays}d old)` : ''}
                        </div>
                    </>
                )}
            </ErrorBoundary>
        </div>
    );
}
