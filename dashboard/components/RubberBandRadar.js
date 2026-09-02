'use client';
import { useEffect, useState } from 'react';
import ErrorBoundary from './ErrorBoundary';
import Skeleton from './Skeleton';
import { DIAL_ORDER, dialLabel } from '../lib/rubberBand';

/**
 * 🪢 Rubber Band Radar — "is the dip-buying regime still alive?" in five dials.
 *
 * Everything shown here is computed nightly on the Mac mini (scripts/rubber_band.py)
 * and relayed by /api/rubber-band; this component only draws it. The dials, in the
 * order the research ranked them (docs/rubber-band.md):
 *   1. slow  — last 30 oversold dips (RSI-10 < 32): did buying them beat an ordinary day?
 *              Below zero for 60 straight days = STOP. Fired 15× in 1972–91, never since 1993.
 *   2. fast  — same, last 20 dips. LOOK only: leads by months in a real flip, but every
 *              alarm since 1993 was false.
 *   3. age   — how many years the 30 dips span (fresh < 3.3y, stale > 4y).
 *   4. rip   — last 30 overbought days (RSI-10 > 79): rips that keep running for 60 days
 *              = the 1970s pattern.
 *   5. machines — each leg's backtest drawdown vs its written line, months underwater,
 *              m1-vs-C3 lag.
 */

const COLOUR_VAR = { green: 'var(--green)', amber: 'var(--orange)', red: 'var(--red)', grey: 'var(--text-muted)' };
const COLOUR_BG = { green: 'var(--green-bg)', amber: 'var(--orange-bg)', red: 'var(--red-bg)', grey: 'rgba(148,163,184,0.12)' };
const COLOUR_WORD = { green: 'OK', amber: 'WATCH', red: 'STOP', grey: 'NO DATA' };
const BADGE = { green: 'badge-green', amber: 'badge-yellow', red: 'badge-red', grey: 'badge-blue' };

const signed = (v, d = 2) => (v == null || !Number.isFinite(v) ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(d)}%`);
const pct0 = (v) => (v == null || !Number.isFinite(v) ? '—' : `${Math.round(v * 100)}%`);
const num = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(d));

// ELI5 line + the honest record, shown as a tooltip on each dial.
const EXPLAIN = {
    slow: (d) => `Take the last ${d.n} days QQQ was oversold (RSI-10 below 32 — the machine's own dip trigger) and ask: did buying the close beat an ordinary day? ${signed(d.excess_pct)} says yes by that much per day (noise ±${num(d.se_pct, 2)}%). Below zero for ${d.stop_after} straight days = STOP. Record: fired 15 times in 1972–91, zero false alarms since 1993. Blind spot: the 2001 and 2008 grinding bears stayed green.`,
    fast: (d) => `Same test on only the last ${d.n} dips. It reacts months earlier in a real regime flip (led by 200+ days in the 1970s) but is noisy: all 12 red spells since 1993 were false alarms. It never acts on its own — it says LOOK.`,
    age: (d) => `The 30 dips behind the slow dial span ${num(d.years, 1)} years. Fresh evidence (under ${d.amber_years}y) means the verdict is about today's market; over ${d.red_years}y means the market has been too calm to test the band lately. It is a trust gauge, not a forecast.`,
    rip: (d) => `The other half of the machine: after ${d.n} overbought days (RSI-10 above 79, the sell trigger) does the market fade the next day? Negative = rips fade = normal since 2000. If rips keep running for ${d.red_after} straight days, the 1970s pattern is back. Record: never red since 2000.`,
    machines: (d) => `Each leg's Composer backtest curve vs the written tripwire lines: through the line = STOP, within 10 points = WATCH, 9 months underwater = STOP, ${d.lag_pair ? `${d.lag_pair[0]} lagging ${d.lag_pair[1]}` : 'm1 lagging C3'} two full months = the exit rule. Backtest curves, not account values.`,
};

function headline(key, d) {
    if (d.colour === 'grey') return { big: '—', sub: d.reason || 'not enough data' };
    switch (key) {
        case 'slow':
        case 'fast':
            return { big: signed(d.excess_pct), sub: `${d.n} dips · hit ${pct0(d.hit)}${d.red_days ? ` · red ${d.red_days}/${d.stop_after}d` : ''}` };
        case 'age':
            return { big: `${num(d.years, 1)}y`, sub: `${d.events_last_12m ?? '—'} dips last 12m · amber >${d.amber_years}y` };
        case 'rip':
            return { big: signed(d.excess_pct), sub: `${d.n} rips · fade ${pct0(1 - (d.hit ?? 0))}${d.hot_days ? ` · hot ${d.hot_days}/${d.red_after}d` : ''}` };
        case 'machines': {
            const worst = (d.legs || []).filter((l) => l.dd_pct != null).sort((a, b) => a.dd_pct - b.dd_pct)[0];
            return { big: worst ? `${worst.name} ${signed(worst.dd_pct, 0)}` : '—', sub: d.reasons?.length ? d.reasons[0] : 'all legs inside their lines' };
        }
        default:
            return { big: '—', sub: '' };
    }
}

function Dial({ k, d }) {
    const { big, sub } = headline(k, d);
    const col = COLOUR_VAR[d.colour] || COLOUR_VAR.grey;
    return (
        <div
            data-testid="rb-dial"
            data-colour={d.colour}
            className="tooltip-trigger"
            data-tooltip={EXPLAIN[k] ? EXPLAIN[k](d) : ''}
            style={{ background: COLOUR_BG[d.colour] || COLOUR_BG.grey, border: `1px solid ${col}`, borderRadius: 10, padding: '10px 12px', minWidth: 0 }}
        >
            <div data-testid={`rb-dial-${k}`} data-colour={d.colour} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{dialLabel(k)}</span>
                <span style={{ fontSize: '0.62rem', fontWeight: 700, color: col }}>● {COLOUR_WORD[d.colour]}{k === 'fast' && d.colour !== 'green' ? ' (look)' : ''}</span>
            </div>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: col, fontVariantNumeric: 'tabular-nums', marginTop: 4, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{big}</div>
            <div style={{ fontSize: '0.66rem', color: 'var(--text-muted)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{sub}</div>
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
            <text x={W - padR} y={padT + 9} fontSize="9" fill="var(--text-muted)" textAnchor="end">— slow (30 dips)   - - fast (20 dips)</text>
        </svg>
    );
}

function Legs({ m }) {
    const legs = m.legs || [];
    if (!legs.length) return null;
    const cols = '1fr 1fr 1fr 1fr 1fr';
    const cell = { textAlign: 'right', fontVariantNumeric: 'tabular-nums' };
    const ddColour = (l) => {
        if (l.dd_pct == null) return 'var(--text-muted)';
        if (l.line_pct != null && l.dd_pct <= l.line_pct) return 'var(--red)';
        if (l.line_pct != null && l.dd_pct <= l.line_pct + 10) return 'var(--orange)';
        return 'inherit';
    };
    return (
        <div style={{ display: 'grid', gridTemplateColumns: cols, gap: '3px 8px', fontSize: '0.74rem', marginTop: 10 }}>
            <span style={{ color: 'var(--text-muted)' }}>Leg</span>
            <span style={{ ...cell, color: 'var(--text-muted)' }}>Drawdown</span>
            <span style={{ ...cell, color: 'var(--text-muted)' }}>Line</span>
            <span style={{ ...cell, color: 'var(--text-muted)' }}>Underwater</span>
            <span style={{ ...cell, color: 'var(--text-muted)' }}>Worst ever</span>
            {legs.map((l) => (
                <ContentsRow key={l.name} l={l} cell={cell} colour={ddColour(l)} />
            ))}
            {m.lag_pair && (
                <span style={{ gridColumn: '1 / -1', color: 'var(--text-muted)', fontSize: '0.68rem' }}>
                    {m.lag_pair[0]} lagging {m.lag_pair[1]}: {m.lag_months ?? '—'} month{m.lag_months === 1 ? '' : 's'} running (exit rule at 2).
                </span>
            )}
        </div>
    );
}

function ContentsRow({ l, cell, colour }) {
    return (
        <>
            <span style={{ fontWeight: 600 }}>{l.name}{l.missing ? <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> (no curve)</span> : ''}</span>
            <span style={{ ...cell, color: colour, fontWeight: 600 }}>{signed(l.dd_pct, 1)}</span>
            <span style={cell}>{l.line_pct == null ? 'none' : `${l.line_pct}%`}</span>
            <span style={cell}>{l.months_underwater == null ? '—' : `${l.months_underwater} mo`}</span>
            <span style={{ ...cell, color: 'var(--text-muted)' }}>{signed(l.worst_dd_pct, 0)}</span>
        </>
    );
}

export default function RubberBandRadar() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

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
                            {DIAL_ORDER.map((k) => <Dial key={k} k={k} d={data.dials[k]} />)}
                        </div>
                        <div style={{ marginTop: 12 }}>
                            <BandChart history={data.history} />
                        </div>
                        <Legs m={data.dials.machines} />
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.65rem', marginTop: 10, opacity: 0.8 }}>
                            As of {data.asOf} · QQQ Wilder RSI-10 (dips &lt;32, rips &gt;79){stale ? ` · STALE (${data._meta?.ageDays}d old)` : ''}
                        </div>
                    </>
                )}
            </ErrorBoundary>
        </div>
    );
}
