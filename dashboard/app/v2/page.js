'use client';
/**
 * v2 — the same live feeds as v1 (useFeeds.js copies v1's loading rules), laid out
 * top-down by importance: the market now (SPY, Fear & Greed, the read), his trade
 * signals, then Market → Risk → Economy → World → Crowd. v1 at / is untouched.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import ErrorBoundary from '../../components/ErrorBoundary';
import Skeleton from '../../components/Skeleton';
import SpyChart from '../../components/SpyChart';
import MiniChart from '../../components/MiniChart';
import JevPills from '../../components/JevPills';
import EconomicIndicatorGrid from '../../components/EconomicIndicatorGrid';
import FourHorsemen from '../../components/FourHorsemen';
import BullChecklist from '../../components/BullChecklist';
import ExtraMarketsGrid from '../../components/ExtraMarketsGrid';
import PolymarketTable from '../../components/PolymarketTable';
import VolMetricsTable from '../../components/VolMetricsTable';
import RubberBandRadar from '../../components/RubberBandRadar';
import FactorRow from '../../components/FactorRow';
import Delta from '../../components/Delta';
import MarkChip from '../../components/MarkChip';
import MarksSheet from '../../components/MarksSheet';
import { MarkProvider, useMark, collectLiveValues } from '../../components/MarkProvider';
import BackPill from '../../components/BackPill';
import WhatMoved, { jumpToCard } from '../../components/WhatMoved';
import StatusLine from '../../components/StatusLine';
import SinceLastVisit from '../../components/SinceLastVisit';
import MarketClock from '../../components/MarketClock';
import { UpdatedAgo, OfflineBanner, PullToRefresh } from '../../components/PhonePolish';
import { staleNote, aaiiCaption } from '../../components/CustomIndicatorBar';
import { isLiveAnswer } from '../../lib/snapshot';
import { freshnessNote, formatAsOf } from '../../lib/freshness';
import { fgHistoryCells } from '../../lib/fgHistory';
import { staleOf } from '../../lib/glance';
import { movedWhen } from '../../lib/whatMoved';
import { pulseVerdicts } from '../../lib/pulseVerdicts';
import useFeeds from './useFeeds';

const SECTIONS = [
    ['now', 'Now'], ['signals', 'Signals'], ['market', 'Market'], ['risk', 'Risk'],
    ['economy', 'Economy'], ['world', 'World'], ['crowd', 'Crowd'],
];

const FG_ZONES = [
    { to: 25, tone: 'bad' }, { to: 45, tone: 'caution' }, { to: 55, tone: 'flat' }, { to: 75, tone: 'good' }, { to: 100, tone: 'good2' },
];
const RSI_ZONES = [{ to: 30, tone: 'good' }, { to: 70, tone: 'flat' }, { to: 100, tone: 'bad' }];
const fgTone = (s) => (s < 25 ? 'bad' : s < 45 ? 'caution' : s < 55 ? 'flat' : 'good');
const signed = (v, d = 2) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`;
const usd = (v) => `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

/** A card body as its own component, so the card's ErrorBoundary catches a throw in it. */
function Body({ children }) { return children(); }

function Unavailable({ what, onRetry }) {
    return <button type="button" className="card-unavailable" onClick={onRetry}>{what} unavailable · tap ↻</button>;
}

/** 0–100 strip with tinted zones and one marker: F&G and RSI read the same way. */
function Scale({ value, zones, ticks }) {
    let from = 0;
    return (
        <div className="v2-scale" role="img" aria-label={`${Math.round(value)} of 100`}>
            <div className="v2-scale-track">
                {zones.map((z) => {
                    const el = <span key={z.to} className={`v2-zone tone-${z.tone}`} style={{ left: `${from}%`, width: `${z.to - from}%` }} />;
                    from = z.to;
                    return el;
                })}
                <span className="v2-scale-mark" style={{ left: `${Math.max(0, Math.min(100, value))}%` }} />
            </div>
            <div className="v2-scale-ticks">{ticks.map((t) => <span key={t} style={{ left: `${t}%` }}>{t}</span>)}</div>
        </div>
    );
}

/** Last ~3 months of SPY closes: shape only, the full chart lives under Market. */
function Spark({ history }) {
    const pts = (history || []).slice(-63).map((p) => p.price).filter(Number.isFinite);
    if (pts.length < 2) return null;
    const lo = Math.min(...pts), hi = Math.max(...pts), w = 320, h = 64;
    const xy = pts.map((p, i) => [(i / (pts.length - 1)) * w, h - 4 - ((p - lo) / (hi - lo || 1)) * (h - 8)]);
    const line = xy.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
    const up = pts[pts.length - 1] >= pts[0];
    return (
        <svg className={`v2-spark ${up ? 'is-up' : 'is-down'}`} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
            <defs>
                <linearGradient id="v2SparkFill" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="currentColor" stopOpacity="0.18" />
                    <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
                </linearGradient>
            </defs>
            <path d={`${line}L${w},${h}L0,${h}Z`} fill="url(#v2SparkFill)" />
            <path d={line} fill="none" stroke="currentColor" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        </svg>
    );
}

function HeroValue({ markKey, raw, stale, format, className, children }) {
    const mark = useMark(markKey, raw);
    return <Delta mark={stale ? null : mark} format={format} chartKey={markKey} raw={raw} className={className}>{children}</Delta>;
}

/** Section: plain title on a hairline, then its content. `id` is the sticky nav's anchor. */
function Section({ id, title, note, children }) {
    return (
        <section id={id} className="v2-section" aria-labelledby={`${id}-h`}>
            <header className="v2-section-head">
                <h2 id={`${id}-h`}>{title}</h2>
                {note && <span className="v2-section-note">{note}</span>}
            </header>
            {children}
        </section>
    );
}

/**
 * A reused v1 card in a v2 slot: v2 hides the card's own emoji title (CSS) and draws
 * its own, so every panel reads the same. `jump` keeps v1's data-jump label (chips jump here).
 */
function Slot({ jump, title, tip, span = 12, saved, children }) {
    return (
        <div className={`v2-slot span-${span}`} data-jump={jump} data-cached={saved}>
            <div className="v2-slot-title">{tip ? <span className="tooltip-trigger" data-tooltip={tip}>{title}</span> : title}</div>
            {children}
        </div>
    );
}

/** Yield curve / profit margin / EPS: one number, one long chart. */
function FredTile({ fred, metric, title, jump, tip, markKey, fmt, tone, chart, refreshTick }) {
    const m = fred?.[metric];
    return (
        <div className="card v2-fred" data-jump={jump}>
            <div className="v2-card-head">
                <span className="tooltip-trigger" data-tooltip={`${tip}${freshnessNote({ value: m?.current, asOf: m?.asOf, stale: m?.stale }).suffix}`}>{title}</span>
            </div>
            <ErrorBoundary resetKey={refreshTick}>
                <Body>{() => !fred ? <Skeleton count={2} /> : (fred.error || m?.current == null) ? (
                    <div className="v2-fred-na">Unavailable · source busy, try again shortly</div>
                ) : (
                    <>
                        <div className="v2-fred-value">
                            <HeroValue markKey={markKey} raw={m.current} stale={m.stale} format={fmt}
                                className={`v2-num-lg tone-${m.stale ? 'caution' : tone(m.current)}`}>
                                {m.stale ? '🕐 ' : ''}{fmt(m.current)}
                            </HeroValue>
                            {m.stale && <span className="v2-muted">Last data {formatAsOf(m.asOf)}</span>}
                        </div>
                        <MiniChart history={m.history} recessions={fred.recessions || []} {...chart} />
                    </>
                )}</Body>
            </ErrorBoundary>
        </div>
    );
}

function Signals({ sheets, loading }) {
    const aaiiMark = useMark('aaiiDiff', parseFloat(sheets?.AAIIDiff));
    const na = loading ? '…' : 'N/A';
    const aaii = parseFloat(sheets?.AAIIDiff);
    const vixCur = parseFloat(sheets?.VIX?.current);
    const vix3m = parseFloat(sheets?.VIX?.threeMonth);
    const until = new Date();
    until.setMonth(until.getMonth() + 6);
    const tiles = [
        {
            key: 'nsb', label: 'NotSoBoring', tip: 'Watches QQQ and TMF for 6–7% drops; then adds gold + USD.',
            alert: !loading && sheets?.NotSoBoring && sheets.NotSoBoring !== 'ON',
            value: sheets?.NotSoBoring || na, sub: staleNote(sheets, ['NotSoBoring']) || <span className="v2-muted">ON = no drop trigger</span>,
        },
        {
            key: 'fr', label: 'FrontRunner', tip: 'VIX hedges when RSI tops 79; buys deep tech dips. Tap for Catalyst Radar.',
            href: 'https://nuts-radar.vercel.app',
            alert: !loading && sheets?.FrontRunner && !sheets.FrontRunner.startsWith('BIL'),
            value: sheets?.FrontRunner || na, sub: staleNote(sheets, ['FrontRunner']) || <span className="v2-muted">Catalyst Radar ↗</span>,
        },
        {
            key: 'aaii', label: 'AAII bear − bull', tip: 'Bears minus bulls. Over 20% has meant a bullish next 6 months.',
            alert: Number.isFinite(aaii) && aaii > 20,
            value: Number.isFinite(aaii) ? (
                <Delta mark={aaiiMark} chartKey="aaiiDiff" raw={aaii} format={(v) => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`}>
                    <span className={aaii > 20 ? 'tone-good' : ''}>{sheets.AAIIDiff}</span>
                </Delta>
            ) : na,
            sub: Number.isFinite(aaii) ? (
                <span className={aaii > 20 ? 'tone-good' : 'v2-muted'}>
                    {aaii > 20 ? 'Bullish' : 'Neutral'} to {until.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
                </span>
            ) : null,
            foot: aaiiCaption(sheets?.AAII, sheets?.AAIIDiff),
        },
        {
            key: 'vix', label: 'VIX · 3-month', tip: 'VIX above 3-month VIX: panic. Often before a rebound.',
            alert: Number.isFinite(vixCur) && vixCur > vix3m,
            value: Number.isFinite(vixCur) ? (
                <span className="v2-pair">
                    <Delta chartKey="vixCurrent" raw={vixCur}>{sheets.VIX.current}</Delta>
                    <span className="v2-pair-sep">/</span>
                    <Delta chartKey="vix3m" raw={vix3m}>{sheets.VIX.threeMonth}</Delta>
                </span>
            ) : na,
            sub: staleNote(sheets, ['vixCurrent', 'vixThreeMonth', 'vixFearGreed'])
                || (sheets?.VIX?.fearGreed ? <span className="v2-muted">{sheets.VIX.fearGreed}</span> : null),
        },
    ];
    return (
        <div className="v2-signals">
            {tiles.map((t) => {
                const inner = (
                    <>
                        <span className="v2-signal-label"><span className="tooltip-trigger" data-tooltip={t.tip}>{t.label}</span></span>
                        <span className="v2-signal-value">{t.value}</span>
                        <span className="v2-signal-sub">{t.sub}</span>
                        {t.foot && <span className="v2-signal-foot">{t.foot}</span>}
                    </>
                );
                const cls = `v2-signal${t.alert ? ' is-alert' : ''}`;
                return t.href
                    ? <a key={t.key} className={`${cls} is-link`} href={t.href} target="_blank" rel="noopener noreferrer">{inner}</a>
                    : <div key={t.key} className={cls}>{inner}</div>;
            })}
        </div>
    );
}

/** The five verdicts v1 shows as Market Pulse chips, as a readable row: verdict + why. */
function Readout({ fred, vol, rubberBand, saved, waiting }) {
    const chips = useMemo(() => pulseVerdicts({ fred, vol, rubberBand, saved }), [fred, vol, rubberBand, saved?.fred, saved?.vol]); // eslint-disable-line react-hooks/exhaustive-deps
    if (!chips.length) return waiting ? <div className="v2-readout is-waiting"><Skeleton count={1} /></div> : null;
    return (
        <div className="v2-readout" aria-label="The read: verdicts from the cards below">
            {chips.map((c) => (
                <button key={c.key} type="button" className={`v2-read tone-${c.tone}`} onClick={() => jumpToCard(c.jump)}
                    title={`${c.why}${c.old ? ` · ${c.old.note}` : ''} · tap for the card`}>
                    <span className="v2-read-text">{c.old?.kind === 'stale' ? '🕐 ' : ''}{c.text.replace(/\s*✓$/, '')}</span>
                    <span className="v2-read-why">{c.why}</span>
                </button>
            ))}
        </div>
    );
}

/** Sticky section tabs; the one in view is lit (IntersectionObserver, no scroll listener). */
function SectionNav({ glance }) {
    const [active, setActive] = useState('now');
    useEffect(() => {
        if (typeof IntersectionObserver === 'undefined') return undefined;
        const els = SECTIONS.map(([id]) => document.getElementById(id)).filter(Boolean);
        const io = new IntersectionObserver((entries) => {
            const hit = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
            if (hit) setActive(hit.target.id);
        }, { rootMargin: '-120px 0px -60% 0px' });
        els.forEach((el) => io.observe(el));
        return () => io.disconnect();
    }, []);
    const go = (id) => {
        const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
        document.getElementById(id)?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
    };
    return (
        <nav className="v2-tabs" aria-label="Sections">
            <div className="v2-tabs-row">
                {SECTIONS.map(([id, label]) => (
                    <button key={id} type="button" className={`v2-tab${active === id ? ' is-active' : ''}`} onClick={() => go(id)}>{label}</button>
                ))}
            </div>
            {glance}
        </nav>
    );
}

export default function DashboardV2() {
    const f = useFeeds();
    const { spy, spyDailyMove, fg, fred, extra, sheets, history, jev, vol } = f.data;
    const { pending, saved, refreshTick, refreshNow } = f;
    const [rubberBand, setRubberBand] = useState(undefined);
    const heroRef = useRef(null);
    const [heroGone, setHeroGone] = useState(false);
    useEffect(() => {
        if (!heroRef.current || typeof IntersectionObserver === 'undefined') return undefined;
        const io = new IntersectionObserver(([e]) => setHeroGone(!e.isIntersecting && e.boundingClientRect.top < 0));
        io.observe(heroRef.current);
        return () => io.disconnect();
    }, []);

    const spyOk = isLiveAnswer('spy', spy);
    const fgOk = isLiveAnswer('fg', fg);
    const moveOk = isLiveAnswer('spyDailyMove', spyDailyMove);
    const spyStale = spyOk ? staleOf(spy)?.label : undefined;
    const spyWhen = useMemo(() => { try { return movedWhen({ spy, vol }) || 'today'; } catch { return 'today'; } }, [spy, vol]);
    const dayPct = moveOk ? parseFloat(spyDailyMove.value) : spyOk && spy.dailyChange ? spy.dailyChange.pct : null;
    const dayText = moveOk ? spyDailyMove.value.replace(/^-/, '−') : dayPct != null ? `${signed(dayPct)}%` : null;
    const marks = collectLiveValues(fred, extra, sheets);

    const glance = heroGone && spyOk ? (
        <button type="button" className="v2-glance" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} aria-label="Back to top">
            <span>SPY <b>{usd(spy.current)}</b></span>
            {dayPct != null && <span className={dayPct >= 0 ? 'tone-good' : 'tone-bad'}>{dayText}</span>}
            {fgOk && <span>F&G <b className={`tone-${fgTone(fg.score)}`}>{Math.round(fg.score)}</b></span>}
        </button>
    ) : null;

    return (
        <MarkProvider history={f.marksOff ? null : history} series={history?.series || null}>
            <OfflineBanner onBack={refreshNow} since={f.anySaved} />
            <PullToRefresh onRefresh={refreshNow} busy={f.refreshing} />
            <div className="v2">
                <header className="v2-top">
                    <div className="v2-top-row">
                        <a className="v2-brand" href="/v2">Markets</a>
                        <div className="v2-top-clock"><ErrorBoundary><MarketClock /></ErrorBoundary></div>
                        <div className="v2-top-right">
                            <span className={`v2-updated${f.anySaved && (f.loading || !f.updatedAt) ? ' is-saved' : ''}`}>
                                {f.loading ? (f.anySaved ? `Saved ${f.anySaved} · loading` : 'Loading') : !f.updatedAt ? 'No live data yet' : <UpdatedAgo at={f.updatedAt} />}
                            </span>
                            <ErrorBoundary resetKey={refreshTick}><MarkChip values={marks} /></ErrorBoundary>
                            <button className="v2-refresh" onClick={refreshNow} disabled={f.refreshing} title="Refresh (R)" aria-label="Refresh all data">
                                <svg className={f.refreshing ? 'spinning' : ''} width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" />
                                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                                </svg>
                            </button>
                        </div>
                    </div>
                    <SectionNav glance={glance} />
                </header>

                <main className="v2-main">
                    {/* NOW — the price, the mood, the read */}
                    <section id="now" className="v2-now" aria-label="The market now">
                        <div className="v2-phone-clock"><ErrorBoundary><MarketClock /></ErrorBoundary></div>
                        <div className="v2-hero" ref={heroRef}>
                            <div className="v2-hero-spy" data-jump="SPY overview" data-cached={saved('spy', 'spyDailyMove')}>
                                <ErrorBoundary resetKey={refreshTick}>
                                    <Body>{() => !spyOk ? (pending.spy ? <Skeleton count={4} /> : <Unavailable what="SPY" onRetry={refreshNow} />) : (
                                        <>
                                            <div className="v2-hero-label">
                                                <span>S&P 500 · SPY</span>
                                                <span className="v2-hero-aside">3 months</span>
                                                {spyStale && <span className="v2-flag" data-testid="spy-provenance">Stale · {/ close$/.test(spyStale) ? spyStale : `cached ${spyStale}`}</span>}
                                            </div>
                                            <div className="v2-hero-price-row">
                                                <span className="v2-hero-price">{usd(spy.current)}</span>
                                                {dayPct != null && (
                                                    <span className={`v2-move ${dayPct >= 0 ? 'is-up' : 'is-down'}`}>
                                                        {dayPct >= 0 ? '▲' : '▼'} {dayText.replace(/^[+−-]/, '')} <span className="v2-move-when">{spyWhen}</span>
                                                    </span>
                                                )}
                                            </div>
                                            <Spark history={spy.chartHistory} />
                                            <dl className="v2-facts">
                                                <div><dt>200-day avg</dt><dd>{usd(spy.ma200.value)} <span className={spy.ma200.pct >= 0 ? 'tone-good' : 'tone-bad'}>{signed(spy.ma200.pct)}%</span></dd></div>
                                                <div><dt>52-week high</dt><dd>{spy.week52High.pct >= 0 ? <span className="tone-good">At the high</span> : <>{usd(spy.week52High.value)} <span className={spy.week52High.pct >= -1 ? 'tone-good' : 'tone-caution'}>{signed(spy.week52High.pct)}%</span></>}</dd></div>
                                                <div><dt>3-year return</dt><dd className={spy.return3y == null ? '' : spy.return3y >= 0 ? 'tone-good' : 'tone-bad'}>{spy.return3y == null ? 'N/A' : `${signed(spy.return3y, 1)}%`}</dd></div>
                                            </dl>
                                        </>
                                    )}</Body>
                                </ErrorBoundary>
                            </div>

                            <div className="v2-hero-side">
                                <div className="v2-gauge-block" data-jump="Fear & Greed" data-cached={saved('fg')}>
                                    <ErrorBoundary resetKey={refreshTick}>
                                        <Body>{() => !fgOk ? (pending.fg ? <Skeleton type="gauge" /> : <Unavailable what="F&G" onRetry={refreshNow} />) : (
                                            <>
                                                <div className="v2-hero-label">
                                                    <span>Fear & Greed</span>
                                                    {(fg._meta?.proxy || fg._meta?.stale) && (
                                                        <span className="v2-flag" data-testid="fg-provenance">{fg._meta?.proxy ? 'VIX proxy, not CNN' : `Stale · ${String(fg._meta?.lastGoodAt || '').slice(0, 10)}`}</span>
                                                    )}
                                                </div>
                                                <div className="v2-gauge-head">
                                                    <HeroValue markKey="cnnFearGreed" raw={fg._meta?.proxy ? null : fg.score} className={`v2-num-xl tone-${fgTone(fg.score)}`}>{Math.round(fg.score)}</HeroValue>
                                                    <span className={`v2-gauge-word tone-${fgTone(fg.score)}`}>{fg.rating}</span>
                                                </div>
                                                <Scale value={fg.score} zones={FG_ZONES} ticks={[0, 25, 50, 75, 100]} />
                                                <div className="v2-history">
                                                    {fgHistoryCells(fg).map(({ label, val, diff }) => (
                                                        <div key={label}>
                                                            <span className="v2-history-label">{label}</span>
                                                            <span className="v2-history-val">{val ?? '-'}{diff != null && diff !== 0 && <small className={diff > 0 ? 'tone-good' : 'tone-bad'}>{diff > 0 ? '▲' : '▼'}{Math.abs(diff)}</small>}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </>
                                        )}</Body>
                                    </ErrorBoundary>
                                </div>
                                {spyOk && (
                                    <div className="v2-gauge-block is-rsi">
                                        <div className="v2-hero-label">
                                            <span className="tooltip-trigger" data-tooltip="9-day RSI of SPY. Under 30 oversold, over 70 overbought.">RSI 9-day</span>
                                            <span className={`v2-rsi-word tone-${spy.rsi > 70 ? 'bad' : spy.rsi < 30 ? 'good' : 'flat'}`}>{spy.rsi > 70 ? 'Overbought' : spy.rsi < 30 ? 'Oversold' : 'Neutral'}</span>
                                            <span className="v2-rsi-num">{spy.rsi.toFixed(1)}</span>
                                        </div>
                                        <Scale value={spy.rsi} zones={RSI_ZONES} ticks={[0, 30, 70, 100]} />
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="saved-wrap" data-cached={saved('fred', 'vol')} style={{ display: 'contents', '--saved': saved('fred', 'vol') ? JSON.stringify(saved('fred', 'vol')) : undefined }}>
                            <ErrorBoundary resetKey={refreshTick}>
                                <Readout fred={fred} vol={vol} rubberBand={rubberBand} saved={{ fred: saved('fred'), vol: saved('vol') }} waiting={pending.fred || pending.vol} />
                            </ErrorBoundary>
                        </div>
                        <div className="v2-strips">
                            <ErrorBoundary resetKey={refreshTick}>
                                <WhatMoved spy={spy} fg={fg} extra={extra} vol={vol} history={history}
                                    saved={saved('spy', 'fg', 'extra', 'vol', 'history')}
                                    waiting={['spy', 'fg', 'extra', 'vol', 'history'].some((k) => pending[k])} />
                            </ErrorBoundary>
                            <ErrorBoundary resetKey={refreshTick}><SinceLastVisit base={f.seenBase.rec} live={f.seenNow} /></ErrorBoundary>
                        </div>
                    </section>

                    <Section id="signals" title="Signals">
                        <div data-jump="Indicators" data-cached={saved('sheets')}>
                            <ErrorBoundary resetKey={refreshTick}><Signals sheets={sheets} loading={pending.sheets && !sheets} /></ErrorBoundary>
                        </div>
                        <div className="v2-grid">
                            <ErrorBoundary resetKey={refreshTick}>
                                {jev && <Slot title="Regime" span={12} saved={saved('jev')}><JevPills data={jev} /></Slot>}
                            </ErrorBoundary>
                        </div>
                    </Section>

                    <Section id="market" title="Market">
                        <div className="v2-grid">
                            <Slot jump="SPY chart" title="SPY with 50 and 200-day averages" span={7} saved={saved('spy')}>
                                <div className="card">
                                    <div className="card-header" />
                                    <ErrorBoundary resetKey={refreshTick}>
                                        <Body>{() => !spyOk || !spy.chartHistory ? (pending.spy ? <Skeleton count={4} /> : <Unavailable what="SPY chart" onRetry={refreshNow} />) : (
                                            <SpyChart chartHistory={spy.chartHistory} recessions={fred?.recessions || []} current={spy.current} />
                                        )}</Body>
                                    </ErrorBoundary>
                                </div>
                            </Slot>
                            <Slot jump="Volatility" title="Volatility" span={5}>
                                <ErrorBoundary resetKey={refreshTick}><VolMetricsTable data={vol} saved={saved('vol')} waiting={pending.vol} /></ErrorBoundary>
                            </Slot>
                            <div className="v2-slot span-12 is-bare">
                                <ErrorBoundary resetKey={refreshTick}><FactorRow refreshKey={refreshTick} bust={f.bustKey} /></ErrorBoundary>
                            </div>
                        </div>
                    </Section>

                    <Section id="risk" title="Risk">
                        <div className="v2-grid">
                            <Slot jump="Rubber band" title="Rubber band" tip="Is dip-buying still paying? Nightly check of three tripwires." span={12}>
                                <ErrorBoundary resetKey={refreshTick}><RubberBandRadar onVerdict={setRubberBand} /></ErrorBoundary>
                            </Slot>
                            <Slot jump="Recession watch" title="Recession watch" tip="Four recession tells: each one's 1-year move vs its usual move before past recessions." span={12} saved={saved('fred')}>
                                <ErrorBoundary resetKey={refreshTick}><FourHorsemen fred={fred} loading={pending.fred && !fred} /></ErrorBoundary>
                            </Slot>
                        </div>
                    </Section>

                    <Section id="economy" title="Economy">
                        <div className="v2-grid">
                            <Slot jump="Bull checklist" title="Bull market checklist" span={7} saved={saved('fred')}>
                                <ErrorBoundary resetKey={refreshTick}><BullChecklist fred={fred} loading={pending.fred && !fred} /></ErrorBoundary>
                            </Slot>
                            <Slot jump="Economy" title="Economic indicators" span={5} saved={saved('fred')}>
                                <ErrorBoundary resetKey={refreshTick}><EconomicIndicatorGrid fred={fred} loading={pending.fred && !fred} statusColor={statusColor} /></ErrorBoundary>
                            </Slot>
                            <div className="v2-trio span-12" data-cached={saved('fred')}>
                                <FredTile fred={fred} metric="yieldCurve" title="Yield curve" jump="Yield curve" markKey="yieldCurve" refreshTick={refreshTick}
                                    tip="10-year minus 2-year. Below 0 = classic recession warning."
                                    fmt={(v) => `${signed(v)}%`} tone={(v) => (v >= 0 ? 'good' : 'bad')}
                                    chart={{ color: '#9fb4c7', gradientId: 'v2Yield', showZero: true }} />
                                <FredTile fred={fred} metric="profitMargin" title="Profit margin" jump="Profit margin" markKey="profitMargin" refreshTick={refreshTick}
                                    tip="Corporate profits ÷ GDP. High = pricing power."
                                    fmt={(v) => `${v.toFixed(2)}%`} tone={() => 'flat'}
                                    chart={{ color: '#9fb4c7', gradientId: 'v2Profit' }} />
                                <FredTile fred={fred} metric="spEps" title="S&P 500 EPS" jump="S&P 500 EPS" markKey={undefined} refreshTick={refreshTick}
                                    tip="S&P 500 earnings per share, last 12 months. History in today's dollars."
                                    fmt={(v) => `$${v.toFixed(2)}`} tone={() => 'flat'}
                                    chart={{ color: '#9fb4c7', gradientId: 'v2Eps', cadence: 'monthly' }} />
                            </div>
                        </div>
                    </Section>

                    <Section id="world" title="World">
                        <div className="v2-grid">
                            <Slot jump="Markets" title="Rates, currencies and commodities" span={12} saved={saved('extra')}>
                                <ErrorBoundary resetKey={refreshTick}><ExtraMarketsGrid data={extra} loading={pending.extra && !extra} /></ErrorBoundary>
                            </Slot>
                        </div>
                    </Section>

                    <Section id="crowd" title="Crowd" note="Polymarket, no sports">
                        <div className="v2-grid">
                            <Slot jump="Polymarket" title="What the crowd is betting on" span={12}>
                                <ErrorBoundary resetKey={refreshTick}><PolymarketTable refreshKey={refreshTick} bust={f.bustKey} /></ErrorBoundary>
                            </Slot>
                        </div>
                    </Section>
                </main>

                <footer className="v2-foot">
                    {f.systemStatus && (
                        <StatusLine bodies={f.systemStatus}
                            saved={{ spy: saved('spy'), fred: saved('fred'), fg: saved('fg'), sheets: saved('sheets'), extra: saved('extra') }} />
                    )}
                    {f.apiErrors.length > 0 && (
                        <div className="v2-errors" role="status">{f.apiErrors.map((e, i) => <div key={i}>{e}</div>)}</div>
                    )}
                    <div className="v2-foot-links">
                        <a href="/">Classic view</a>
                        <a href="https://docs.google.com/spreadsheets/d/1lA-_yjLMc3qDTt9sogSPQrCohNULIk5wwJYfb5wIHfc/edit?gid=0#gid=0" target="_blank" rel="noopener noreferrer">History sheet</a>
                    </div>
                </footer>

                <ErrorBoundary resetKey={refreshTick}><MarksSheet values={marks} /></ErrorBoundary>
                <ErrorBoundary resetKey={refreshTick}><BackPill /></ErrorBoundary>
            </div>
        </MarkProvider>
    );
}

// The same word → colour map v1 hands the economic indicator grid.
function statusColor(s) {
    return ['safe', 'healthy', 'strong', 'tight', 'easy', 'rising'].includes(s) ? 'stat-positive'
        : ['danger', 'weak', 'stressed', 'restrictive', 'falling'].includes(s) ? 'stat-negative' : 'stat-neutral';
}
