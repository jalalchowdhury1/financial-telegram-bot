'use client';
import { Fragment, useEffect, useId, useState } from 'react';
import ErrorBoundary from './ErrorBoundary';
import Skeleton from './Skeleton';
import { gutterFor, spreadLabels, yearTicks } from '../lib/chartAxis';
import useElementWidth from './useElementWidth';
import AxisLabels from './AxisLabels';

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
 *
 * Layout (10 Oct 2026 redesign — "make it intuitive"): one big answer up top (stay invested /
 * go defensive), the three TRIPWIRES as plain questions each with a "which side of zero" gauge
 * and a fuse (how close to the alarm), the two look-only dials as small chips, the payoff chart,
 * and the trigger as a 4-step rail with "you are here". Styles: `.rb-*` in app/globals.css.
 */

const COLOUR_VAR = { green: 'var(--green)', amber: 'var(--orange)', red: 'var(--red)', blue: 'var(--blue)', grey: 'var(--text-muted)' };
const COLOUR_BG = { green: 'var(--green-bg)', amber: 'var(--orange-bg)', red: 'var(--red-bg)', blue: 'var(--blue-bg)', grey: 'rgba(148,163,184,0.12)' };
const COLOUR_EDGE = { green: 'rgba(34,197,94,0.45)', amber: 'rgba(245,158,11,0.5)', red: 'rgba(239,68,68,0.55)', blue: 'rgba(59,130,246,0.5)', grey: 'rgba(148,163,184,0.3)' };
const COLOUR_WORD = { green: 'All clear', amber: 'Watch', red: 'Stop', grey: 'No data' };
const BADGE = { green: 'badge-green', amber: 'badge-yellow', red: 'badge-red', grey: 'badge-blue' };
const STATUS = { green: 'Safe', amber: 'Watch', red: 'Alarm', grey: 'No data' };

// The three dials that can fire the trigger, then the two that are only ever shown.
const TRIPWIRES = ['slow', 'rip', 'machines'];
const LOOK_ONLY = ['fast', 'age'];
const PLAIN = {
    slow: { q: 'Do dips bounce back?', hint: 'Buying QQQ after a sharp drop' },
    rip: { q: 'Do rallies cool off?', hint: 'Selling after QQQ runs hot' },
    machines: { q: 'Are the machines healthy?', hint: 'Each Composer backtest vs its limits' },
    fast: { q: 'Early warning', hint: 'Same dip test, last 20 dips only' },
    age: { q: 'Evidence age', hint: 'How long the last 30 dips took' },
};
const statusWord = (k, colour) => {
    if (k === 'age') return { green: 'Fresh', amber: 'Getting old', red: 'Old', grey: 'No data' }[colour];
    return STATUS[colour] || STATUS.grey;
};

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const DAY_FMT = { timeZone: 'UTC', weekday: 'short', month: 'short', day: 'numeric' };
const dayName = (iso) => {
    const d = new Date(`${iso}T12:00:00Z`);
    return Number.isNaN(d.getTime()) ? String(iso || '—') : d.toLocaleDateString('en-US', DAY_FMT);
};
/** The engine runs every weekday evening: the next run after `iso` skips Sat/Sun. */
const nextRun = (iso) => {
    const d = new Date(`${iso}T12:00:00Z`);
    if (Number.isNaN(d.getTime())) return null;
    do { d.setUTCDate(d.getUTCDate() + 1); } while (d.getUTCDay() === 0 || d.getUTCDay() === 6);
    return d.toLocaleDateString('en-US', DAY_FMT);
};

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

/** Inline custom properties that colour one block (`.rb-*` CSS reads them). */
const tone = (c) => ({ '--rb-c': COLOUR_VAR[c] || COLOUR_VAR.grey, '--rb-bg': COLOUR_BG[c] || COLOUR_BG.grey, '--rb-edge': COLOUR_EDGE[c] || COLOUR_EDGE.grey });

/** Tap toggles, double-click always opens, Enter/Space toggles — shared by every tappable box. */
const pressable = (onToggle, onOpen) => ({
    role: 'button',
    tabIndex: 0,
    onClick: onToggle,
    onDoubleClick: (e) => { e.preventDefault(); onOpen(); },
    onKeyDown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle(); } },
});

/** Big numbers get a real minus sign (a hyphen reads as a gap at display sizes). */
const minus = (s) => String(s).replace(/^-/, '−');

function Chip({ text }) {
    return <span className="rb-chip">{text}</span>;
}

/** Where today's number sits against zero: one side is the danger side. */
function Gauge({ value, scale, badSide, left, right }) {
    if (value == null || !Number.isFinite(value)) return null;
    const m = Math.max(scale, Math.abs(value) * 1.2);
    const pos = clamp(50 + (value / m) * 50, 3, 97);
    return (
        <div className="rb-gauge" aria-hidden="true">
            <div className={`rb-gauge-track ${badSide === 'below' ? 'bad-left' : 'bad-right'}`}>
                <span className="rb-gauge-zero" />
                <span className="rb-gauge-dot" style={{ left: `${pos}%` }} />
            </div>
            <div className="rb-gauge-ends"><span>◂ {left}</span><span>0</span><span>{right} ▸</span></div>
        </div>
    );
}

/** How much of the alarm's fuse has burnt: `count` of the `of` it takes to trip. */
function Fuse({ label, count, of }) {
    const known = count != null && Number.isFinite(count);
    const frac = known && of ? clamp(count / of, 0, 1) : 0;
    const c = !known ? 'grey' : frac >= 1 ? 'red' : frac >= 0.75 ? 'amber' : 'green';
    return (
        <div className="rb-fuse">
            <div className="rb-fuse-row">
                <span>{label}</span>
                <span className="rb-fuse-n" style={{ color: COLOUR_VAR[c] }}>{known ? count : '—'}<i> · alarm at {of}</i></span>
            </div>
            <div className="rb-bar"><span style={{ width: `${known && count > 0 ? Math.max(frac * 100, 3) : 0}%`, background: COLOUR_VAR[c] }} /></div>
        </div>
    );
}

// A leg's record only counts once it has enough history before its peak (as scripts/rubber_band.py).
const recordCounts = (l, spm) => l.worst_dd_prior_pct != null && l.worst_dd_prior_pct < 0 && (l.days_before_peak ?? 0) >= spm.min_history_days;

/** The limit that trips first: the written line or the leg's own worst completed drawdown. */
function legLimit(l, spm) {
    const limits = [];
    if (l.line_pct != null) limits.push({ v: l.line_pct, why: 'your line' });
    if (recordCounts(l, spm)) limits.push({ v: l.worst_dd_prior_pct, why: 'worst ever' });
    return limits.length ? limits.reduce((a, b) => (b.v > a.v ? b : a)) : null;
}

function legColour(l, spm) {
    if (l.dd_pct == null) return 'grey';
    const rec = recordCounts(l, spm);
    if (l.line_pct != null && l.dd_pct <= l.line_pct) return 'red';
    if (rec && l.dd_pct <= l.worst_dd_prior_pct) return 'red';
    if (l.line_pct != null && l.dd_pct <= l.line_pct + spm.near_line_pts) return 'amber';
    if (rec && l.dd_pct <= l.worst_dd_prior_pct * spm.record_amber_frac) return 'amber';
    return 'green';
}

const legUsed = (l, spm) => {
    const lim = legLimit(l, spm);
    return lim && l.dd_pct != null ? { lim, frac: clamp(l.dd_pct / lim.v, 0, 1) } : null;
};

/** One machine: today's drop as a bar that fills toward the limit that would trip it. */
function LegBar({ l, spm }) {
    const used = legUsed(l, spm);
    const c = legColour(l, spm);
    const uw = l.months_underwater == null ? null
        : l.longest_underwater_prior_months != null ? `${l.months_underwater} of ${l.longest_underwater_prior_months} mo under water`
            : `${l.months_underwater} mo under water`;
    const lim = used ? `trips at ${signed(used.lim.v, 0)} (${used.lim.why})` : 'no limit yet';
    return (
        <div className="rb-leg">
            <div className="rb-leg-top">
                <span className="rb-leg-name">{l.name}</span>
                {l.role === 'hedge' && <span className="rb-leg-tag">hedge</span>}
                {l.missing && <span className="rb-leg-tag">no curve</span>}
                <span className="rb-leg-dd" style={{ color: COLOUR_VAR[c] }}>{minus(signed(l.dd_pct, 1))}</span>
            </div>
            <div className="rb-bar"><span style={{ width: `${used ? Math.max(used.frac * 100, l.dd_pct < 0 ? 2 : 0) : 0}%`, background: COLOUR_VAR[c] }} /></div>
            <div className="rb-leg-sub">{lim}{uw ? ` · ${uw}` : ''}</div>
        </div>
    );
}

/** The middle of a tripwire box: today's number, what it means, the gauge + history or the legs. */
function TripBody({ k, d, ctx, history }) {
    if (d.colour === 'grey') return <div className="rb-big-row"><span className="rb-big">—</span><span className="rb-cap">{d.reason || 'not enough data'}</span></div>;
    if (k === 'slow') {
        return (
            <>
                <div className="rb-big-row">
                    <span className="rb-big">{minus(signed(d.excess_pct))}</span>
                    <span className="rb-cap">extra per dip vs a normal day · paid off {pct0(d.hit)} of the time</span>
                </div>
                <Gauge value={d.excess_pct} scale={1} badSide="below" left="dips lose" right="dips pay" />
                <HistoryChart history={history} main="slow" ghost="fast" badSide="below" up="dips pay" down="dips lose"
                    legend="— slow (30 dips) · - - early warning (20 dips)" aria="Slow and fast dip-payoff lines" />
            </>
        );
    }
    if (k === 'rip') {
        return (
            <>
                <div className="rb-big-row">
                    <span className="rb-big">{minus(signed(d.excess_pct))}</span>
                    <span className="rb-cap">next-day move after a hot run · below 0 = it cools off</span>
                </div>
                <Gauge value={d.excess_pct} scale={0.5} badSide="above" left="cools off" right="keeps running" />
                <HistoryChart history={history} main="rip" badSide="above" up="keeps running" down="cools off"
                    legend="— last 30 hot runs" aria="Next-day move after hot runs" />
            </>
        );
    }
    const legs = d.legs || [];
    const judged = legs.filter((l) => l.dd_pct != null);
    const inside = judged.filter((l) => legColour(l, ctx.machines) !== 'red').length;
    const closest = legs.map((l) => ({ l, used: legUsed(l, ctx.machines) })).filter((x) => x.used).sort((a, b) => b.used.frac - a.used.frac)[0];
    return (
        <>
            <div className="rb-big-row">
                <span className="rb-big">{judged.length ? `${inside} of ${judged.length}` : '—'}</span>
                <span className="rb-cap">
                    legs inside their limits{closest ? ` · closest: ${closest.l.name}, ${Math.round(closest.used.frac * 100)}% of the way to ${closest.used.lim.why === 'worst ever' ? 'its worst-ever drop' : 'its line'}` : ''}
                </span>
            </div>
            {d.reasons?.length > 0 && <ul className="rb-reasons">{d.reasons.map((r) => <li key={r}>{r}</li>)}</ul>}
            <div className="rb-legs">{legs.map((l) => <LegBar key={l.name} l={l} spm={ctx.machines} />)}</div>
            {d.hedge_check && <div data-testid="rb-hedge-check" className="rb-note">Hedge check: {d.hedge_check}</div>}
        </>
    );
}

function tripFuse(k, d, ctx, def) {
    if (k === 'slow') return { label: `Bad days in the last ${d.window ?? ctx.dip.stop_window}`, count: d.red_days ?? 0, of: d.stop_after ?? ctx.dip.stop_of };
    if (k === 'rip') return { label: `Hot days in the last ${d.window ?? ctx.rip.hot_window}`, count: d.hot_days ?? 0, of: d.red_after ?? ctx.rip.hot_of };
    return { label: 'Red days in a row', count: def ? (def.streak?.machines ?? 0) : null, of: ctx.rules.fire_after.machines };
}

/** A tripwire: one plain question, today's answer, and how much of its fuse has burnt. */
function TripCard({ k, d, ctx, def, history, active, onToggle, onOpen }) {
    const word = statusWord(k, d.colour);
    return (
        <div data-testid="rb-dial" data-colour={d.colour} aria-expanded={active} aria-label={`${PLAIN[k].q} ${word}. Tap for the rule.`}
            className={`rb-trip${active ? ' is-open' : ''}`} style={tone(d.colour)} {...pressable(onToggle, onOpen)}>
            <div data-testid={`rb-dial-${k}`} data-colour={d.colour} className="rb-trip-head">
                <span className="rb-q">{PLAIN[k].q}</span>
                <Chip text={word} />
            </div>
            <div className="rb-hint">{PLAIN[k].hint}</div>
            <TripBody k={k} d={d} ctx={ctx} history={history} />
            <Fuse {...tripFuse(k, d, ctx, def)} />
        </div>
    );
}

/** A look-only dial: one compact row, shown but never able to fire anything. */
function LookChip({ k, d, active, onToggle, onOpen }) {
    const grey = d.colour === 'grey';
    const big = grey ? '—' : k === 'age' ? `${num(d.years, 1)} yrs` : minus(signed(d.excess_pct));
    const sub = grey ? (d.reason || 'not enough data')
        : k === 'age' ? `${d.events_last_12m ?? '—'} dips in the last 12 months · old after ${d.amber_years ?? '—'} yrs`
            : `${PLAIN[k].hint} · look only`;
    const word = statusWord(k, d.colour);
    return (
        <div data-testid="rb-dial" data-colour={d.colour} aria-expanded={active} aria-label={`${PLAIN[k].q} ${word}. Tap for the rule.`}
            className={`rb-look${active ? ' is-open' : ''}`} style={tone(d.colour)} {...pressable(onToggle, onOpen)}>
            <div data-testid={`rb-dial-${k}`} data-colour={d.colour} className="rb-look-main">
                <span className="rb-look-q">{PLAIN[k].q}</span>
                <span className="rb-look-sub">{sub}</span>
            </div>
            <span className="rb-look-big">{big}</span>
            <Chip text={word} />
        </div>
    );
}

/** The ELI5 panel for the open box — one at a time, drawn under its row so phones stay readable. */
function ExplainPanel({ k, d, ctx, onClose }) {
    const e = EXPLAIN[k] ? EXPLAIN[k](d, ctx) : null;
    if (!e) return null;
    const rows = [
        ['What it asks', e.what],
        ['Today', e.today],
        ['Red when', e.red],
        ['Amber when', e.amber],
        ['Track record', e.record],
    ];
    if (k === 'machines' && d.lag_pair) {
        rows.push(['Also shown', `${d.lag_pair[0]} vs ${d.lag_pair[1]} lag: ${d.lag_months ?? '—'} month${d.lag_months === 1 ? '' : 's'}${d.lag_is_info_only === false ? ' (exit rule at 2)' : ' — information only, not a rule'}.`]);
    }
    return (
        <div data-testid={`rb-explain-${k}`} role="region" aria-label={`${PLAIN[k].q} explained`} className="rb-explain" style={tone(d.colour)}>
            <div className="rb-explain-head">
                <b>{PLAIN[k].q}</b>
                <Chip text={statusWord(k, d.colour)} />
                <button type="button" className="rb-close" onClick={onClose}>Close ✕</button>
            </div>
            <dl className="rb-explain-grid">
                {rows.map(([tag, text]) => (
                    <Fragment key={tag}><dt>{tag}</dt><dd>{text}</dd></Fragment>
                ))}
            </dl>
        </div>
    );
}

/**
 * A tripwire's history: its line is green on the safe side of zero and red on the danger side
 * (`badSide`), the optional `ghost` series dashed. Lives inside the tripwire box.
 */
function HistoryChart({ history, main, ghost = null, badSide, up, down, legend, aria }) {
    const gid = `rb${useId().replace(/[^a-zA-Z0-9]/g, '')}`;
    const [plotRef, plotPx] = useElementWidth();
    const pts = (history || []).filter((h) => h[main] != null);
    if (pts.length < 20) return null;
    // Sized in real px once measured (k = viewBox units per CSS px; 1 until then): 120px tall,
    // labels 10px HTML — on a phone a fixed drawing shrinks to ~4px text.
    const W = 720, padR = 8;
    const k = plotPx ? W / plotPx : 1;
    const H = Math.round(120 * k);
    const padT = Math.ceil(8 * k), padB = Math.ceil(18 * k);
    const vals = pts.flatMap((h) => [h[main], ghost ? h[ghost] : null]).filter((v) => v != null);
    let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    // Always leave a visible band on both sides of zero, so "above = pays, below = loses" reads at a glance.
    lo = Math.min(lo, -0.3 * (hi || 1));
    hi = Math.max(hi, 0.3 * Math.abs(lo));
    const span = hi - lo;
    const ticks = [lo, 0, hi];
    const padL = gutterFor(ticks.map((v) => signed(v, 1)), W, plotPx, 34);
    const x = (i) => padL + (i / (pts.length - 1)) * (W - padL - padR);
    const y = (v) => padT + (1 - (v - lo) / span) * (H - padT - padB);
    const line = (key) => pts.map((h, i) => (h[key] == null ? null : `${i === 0 || pts[i - 1][key] == null ? 'M' : 'L'}${x(i).toFixed(1)},${y(h[key]).toFixed(1)}`)).filter(Boolean).join(' ');
    const area = `M${x(0).toFixed(1)},${y(0).toFixed(1)} ${pts.map((h, i) => `L${x(i).toFixed(1)},${y(h[main]).toFixed(1)}`).join(' ')} L${x(pts.length - 1).toFixed(1)},${y(0).toFixed(1)} Z`;
    const zeroAt = ((y(0) - padT) / (H - padT - padB)).toFixed(4);
    const top = badSide === 'below' ? 'var(--green)' : 'var(--red)';
    const bottom = badSide === 'below' ? 'var(--red)' : 'var(--green)';
    const last = pts[pts.length - 1];
    const lastBad = badSide === 'below' ? last[main] < 0 : last[main] > 0;
    const years = Math.max(1, Math.round((Date.parse(last.d) - Date.parse(pts[0].d)) / (365.25 * 86400000)));
    const tickLabels = spreadLabels([0, hi, lo].map((v) => ({ v, pos: y(v) / k })), 12);
    const labels = [
        ...tickLabels.map(({ v }) => ({ x: padL - 4 * k, y: y(v), text: signed(v, 1), ax: 'end', ay: 'middle' })),
        ...yearTicks(pts.map((h) => h.d), x, { maxLabels: 8, minGap: 36 * k }).map((t) => ({ x: t.x, y: H, text: t.label, ax: 'start', ay: 'bottom' })),
        { x: padL + 6 * k, y: y(0) - 4 * k, text: `▲ ${up}`, ax: 'start', ay: 'bottom' },
        { x: padL + 6 * k, y: y(0) + 4 * k, text: `▼ ${down}`, ax: 'start', ay: 'top' },
    ];
    // the danger side of zero gets a faint red wash
    const dangerY = badSide === 'below' ? y(0) : y(hi);
    const dangerH = badSide === 'below' ? y(lo) - y(0) : y(0) - y(hi);
    return (
        <div className="rb-chart">
            <div className="rb-chart-head">
                <span className="rb-chart-title">Last {years} years</span>
                <span className="band-legend">{legend}</span>
            </div>
            <div className="chart-plot" ref={plotRef}>
                <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${aria} over the last ${years} years, with the zero line`} style={{ width: '100%', height: 'auto', display: 'block' }}>
                    <defs>
                        <linearGradient id={`${gid}l`} gradientUnits="userSpaceOnUse" x1="0" y1={padT} x2="0" y2={H - padB}>
                            <stop offset="0" stopColor={top} /><stop offset={zeroAt} stopColor={top} />
                            <stop offset={zeroAt} stopColor={bottom} /><stop offset="1" stopColor={bottom} />
                        </linearGradient>
                        <linearGradient id={`${gid}a`} gradientUnits="userSpaceOnUse" x1="0" y1={padT} x2="0" y2={H - padB}>
                            <stop offset="0" stopColor={top} stopOpacity="0.28" /><stop offset={zeroAt} stopColor={top} stopOpacity="0.04" />
                            <stop offset={zeroAt} stopColor={bottom} stopOpacity="0.04" /><stop offset="1" stopColor={bottom} stopOpacity="0.28" />
                        </linearGradient>
                    </defs>
                    <rect x={padL} y={dangerY} width={W - padL - padR} height={Math.max(0, dangerH)} fill="rgba(239,68,68,0.05)" />
                    {[lo, hi].map((v) => (
                        <line key={v} x1={padL} x2={W - padR} y1={y(v)} y2={y(v)} stroke="rgba(255,255,255,0.07)" strokeDasharray="2 4" vectorEffect="non-scaling-stroke" />
                    ))}
                    <path d={area} fill={`url(#${gid}a)`} />
                    <line x1={padL} x2={W - padR} y1={y(0)} y2={y(0)} stroke="rgba(255,255,255,0.4)" vectorEffect="non-scaling-stroke" />
                    {ghost && <path d={line(ghost)} fill="none" stroke="var(--text-secondary)" strokeWidth="1.2" strokeDasharray="3 3" opacity="0.6" vectorEffect="non-scaling-stroke" />}
                    <path d={line(main)} fill="none" stroke={`url(#${gid}l)`} strokeWidth="2.2" vectorEffect="non-scaling-stroke" />
                    <circle cx={x(pts.length - 1)} cy={y(last[main])} r={4 * k} fill={lastBad ? 'var(--red)' : 'var(--green)'} stroke="var(--bg-secondary)" strokeWidth={2 * k} />
                </svg>
                <AxisLabels w={W} h={H} labels={labels} />
            </div>
        </div>
    );
}

const MODE = {
    INVESTED: { badge: 'badge-green', word: 'INVESTED', colour: 'green', eli5: 'All machines running. Nothing to do.' },
    PENDING_DEFENSIVE: { badge: 'badge-red', word: 'GO DEFENSIVE — waiting for you', colour: 'red', eli5: 'The 📡 thread has the 2-minute steps: sell half of each machine to cash, then reply "done".' },
    DEFENSIVE: { badge: 'badge-yellow', word: 'DEFENSIVE — half in cash', colour: 'amber', eli5: 'Half the book sits in cash until the radar has been green enough closes in a row.' },
    PENDING_REENTRY: { badge: 'badge-blue', word: 'RE-ENTER — waiting for you', colour: 'blue', eli5: 'The 📡 thread has the steps: put the cash back into each machine, then reply "done".' },
    UNKNOWN: { badge: 'badge-blue', word: 'UNKNOWN', colour: 'grey', eli5: 'The trigger could not report its state — check the Mac mini log.' },
};

// The trigger's loop, in order. "You are here" follows snapshot.defensive.mode.
const STEPS = [
    { mode: 'INVESTED', title: 'Invested', body: () => 'All machines running. Nothing to do.' },
    { mode: 'PENDING_DEFENSIVE', title: 'A tripwire goes red', body: () => 'Sell half of each machine to cash. 📡 nags every hour until you reply "done".' },
    { mode: 'DEFENSIVE', title: 'Defensive', body: () => 'Half the book waits in cash. Nothing to do.' },
    { mode: 'PENDING_REENTRY', title: 'Back in', body: (r) => `${r.reentry_closes} green days in a row with dips paying over +${r.reentry_edge_pct}% → put the cash back, then step 1.` },
];

/** The big answer: what to do today, why, and when the radar last looked. */
function Hero({ data, stale }) {
    const def = data.defensive;
    const v = data.verdict.colour;
    const mode = def ? (MODE[def.mode] || MODE.UNKNOWN) : null;
    let h;
    if (def?.mode === 'PENDING_DEFENSIVE') h = { c: 'red', icon: '!', title: 'Go defensive — waiting for you' };
    else if (def?.mode === 'DEFENSIVE') h = { c: 'amber', icon: '‖', title: 'Defensive — half in cash' };
    else if (def?.mode === 'PENDING_REENTRY') h = { c: 'blue', icon: '↺', title: 'Time to go back in' };
    else if (v === 'red') h = { c: 'red', icon: '!', title: 'A tripwire is red' };
    else if (v === 'amber') h = { c: 'amber', icon: '!', title: 'Stay invested — keep an eye on it' };
    else if (v === 'grey') h = { c: 'grey', icon: '?', title: 'No reading today' };
    else h = { c: 'green', icon: '✓', title: 'Stay invested' };
    const act = def && def.mode !== 'INVESTED'
        ? `${mode.eli5}${def.defensive_since ? ` Defensive since ${dayName(def.defensive_since)}.` : ''}`
        : v === 'green' ? 'Nothing to do.' : null;
    const next = nextRun(data.asOf);
    return (
        <div className="rb-hero" style={tone(h.c)}>
            <div className="rb-hero-icon" aria-hidden="true">{h.icon}</div>
            <div className="rb-hero-main">
                <div className="rb-hero-title">{h.title}</div>
                <div className="rb-hero-text">{data.verdict.text}</div>
                {act && <div className="rb-hero-act">{act}</div>}
            </div>
            <div className="rb-hero-side">
                {def && <div className="rb-stat"><b>{def.green_streak ?? 0}</b><span>all-clear days in a row</span></div>}
                <div className="rb-stat">
                    <b>{dayName(data.asOf)}</b>
                    <span>{stale ? `STALE — ${data._meta?.ageDays ?? '?'} days old` : `last check${next ? ` · next ${next}` : ''}`}</span>
                </div>
            </div>
        </div>
    );
}

/** The decision layer as a 4-step loop with "you are here", the streak counters, and the rule on tap. */
function Trigger({ def, rules, open, onToggle }) {
    const mode = def ? (MODE[def.mode] || MODE.UNKNOWN) : null;
    const here = def ? STEPS.findIndex((s) => s.mode === def.mode) : -1;
    const st = def?.streak || {};
    const fa = rules.fire_after;
    const rule = `Fires when "Do dips bounce back?" or "Do rallies cool off?" is red for ${fa.slow} close, or "Are the machines healthy?" is red ${fa.machines} closes in a row → 📡 GO DEFENSIVE: sell half of each machine to cash, hourly nag until you reply "done". Back in after ${rules.reentry_closes} green closes in a row with dips paying above +${rules.reentry_edge_pct}% → 📡 RE-ENTER, nag until "done". If the alarm clears before you act, it stands down by itself; if the green run breaks before you re-enter, it holds.`;
    return (
        <div data-testid="rb-trigger" className="rb-rail-box">
            <div className="rb-label-row">
                <span className="rb-label">What happens if a tripwire goes red</span>
                {mode ? <span className={`badge ${mode.badge}`} data-testid="rb-mode">{mode.word}</span> : <span className="badge badge-blue">state not published yet</span>}
            </div>
            <ol className="rb-rail">
                {STEPS.map((s, i) => (
                    <li key={s.mode} className={`rb-step${i === here ? ' is-here' : ''}`} style={i === here ? tone(MODE[s.mode].colour) : undefined}>
                        <span className="rb-step-num">{i + 1}</span>
                        <div>
                            <b>{s.title}</b>{i === here && <span className="rb-here">you are here</span>}
                            <p>{s.body(rules)}</p>
                        </div>
                    </li>
                ))}
            </ol>
            {def && (
                <div className="rb-counters">
                    Red closes in a row: dips <b>{st.slow ?? 0}</b>/{fa.slow} · rallies <b>{st.rip ?? 0}</b>/{fa.rip} · machines <b>{st.machines ?? 0}</b>/{fa.machines} · green run <b>{def.green_streak ?? 0}</b>/{rules.reentry_closes}{def.last_asof ? ` · counted to ${dayName(def.last_asof)}` : ''}
                </div>
            )}
            <button type="button" className="rb-link" onClick={onToggle} aria-expanded={open}>
                {open ? '▴ hide the rule' : '▾ the rule in one breath'}
            </button>
            {open && <div data-testid="rb-trigger-rule" className="rb-rule">{rule}</div>}
        </div>
    );
}

export default function RubberBandRadar({ onVerdict = null } = {}) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [open, setOpen] = useState(null);          // dial key, 'trigger', or null — one panel at a time

    // 📡 Market Pulse shows this card's verdict as a chip: hand it the answer (null = failed).
    useEffect(() => {
        if (loading || !onVerdict) return;
        try { onVerdict(data); } catch { /* the pulse line never breaks this card */ }
    }, [loading, data, onVerdict]);

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
    const box = (k) => ({ k, d: data.dials[k], active: open === k, onToggle: () => toggle(k), onOpen: () => setOpen(k) });
    const panel = (keys) => (keys.includes(open) && data.dials[open]
        ? <ExplainPanel k={open} d={data.dials[open]} ctx={ctx} onClose={() => setOpen(null)} />
        : null);

    return (
        <div className="card rb" style={{ gridColumn: '1 / -1', animationDelay: '0.5s' }}>
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
                        <Hero data={data} stale={stale} />
                        <div className="rb-label-row">
                            <span className="rb-label">The 3 tripwires — any one red sends half the book to cash</span>
                            <span className="rb-label-hint">tap a box for its rule</span>
                        </div>
                        <div className="rb-trips">
                            {TRIPWIRES.map((k) => <TripCard key={k} {...box(k)} ctx={ctx} def={data.defensive} history={data.history} />)}
                        </div>
                        {panel(TRIPWIRES)}
                        <div className="rb-label-row">
                            <span className="rb-label">Also watching — these never trip anything</span>
                        </div>
                        <div className="rb-looks">
                            {LOOK_ONLY.map((k) => <LookChip key={k} {...box(k)} />)}
                        </div>
                        {panel(LOOK_ONLY)}
                        <Trigger def={data.defensive} rules={ctx.rules} open={open === 'trigger'} onToggle={() => toggle('trigger')} />
                        <div className="rb-foot">
                            Computed nightly on the Mac mini · rules v{ctx.version} · QQQ Wilder RSI-10: a dip = under {ctx.dip.rsi_below}, a hot run = over {ctx.rip.rsi_above}{stale ? ` · STALE (${data._meta?.ageDays}d old)` : ''}
                        </div>
                    </>
                )}
            </ErrorBoundary>
        </div>
    );
}
