'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Skeleton from './Skeleton';
import MarketModal from './MarketModal';
import Spark from './OddsSpark';
import { getJson } from '../lib/loadJson';

/**
 * 📊 Market Sentiment — three slices of polymarket.com (sports removed from all three):
 *   📈 Macro    = its Macro dashboard (recession, Fed, GDP…): odds, 30-day change, sparkline
 *   🔥 Trending = its front-page picks: the two likeliest outcomes of each
 *   ⚡ Breaking = its Breaking News: the biggest 24h moves, with a 24h sparkline
 * Tap anything for the details sheet and a link to that market on Polymarket.
 * Data: /api/polymarket (Lambda first, direct Polymarket fallback, then last-good).
 * Re-fetches on the page's refresh cycle (`refreshKey`; a manual refresh passes `bust`).
 */
const MIN_REFETCH_MS = 60e3;
const SHOW = { trending: 6, breaking: 6 };

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

/** 0.62 -> "62%"; tiny and near-certain odds read like Polymarket's ("<1%", ">99%"). */
export function pct(odds) {
  if (!finite(odds)) return '—';
  const p = odds * 100;
  if (p > 0 && p < 1) return '<1%';
  if (p > 99 && p < 100) return '>99%';
  return `${Math.round(p)}%`;
}

/** A change in odds (fraction) -> "▲3" / "▼0.5" points, or null when under `minPts`. */
export function pts(change, minPts = 1) {
  if (!finite(change)) return null;
  const p = change * 100;
  const a = Math.abs(p);
  if (a < minPts || a < 0.05) return null;
  const text = a >= 10 ? String(Math.round(a)) : String(Math.round(a * 10) / 10);
  return { up: p > 0, text: `${p > 0 ? '▲' : '▼'}${text}` };
}

/** Dollar volume, short: $1.2B, $180M, $21.4M, $850k, $900. */
export function money(v) {
  if (!finite(v) || v <= 0) return '$0';
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e8) return `$${Math.round(v / 1e6)}M`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `$${Math.round(v / 1e3)}k`;
  return `$${Math.round(v)}`;
}

const clamp01 = (v) => (finite(v) ? Math.min(1, Math.max(0, v)) : 0);

/**
 * The payload the card draws. Also reads the OLD shape ({bets}) — a last-good copy saved
 * before this card changed can still be served for a while after the deploy.
 */
export function normalize(data) {
  const d = data && typeof data === 'object' ? data : {};
  const arr = (x) => (Array.isArray(x) ? x.filter((r) => r && typeof r === 'object') : []);
  let trending = arr(d.trending);
  if (!trending.length && arr(d.bets).length) {
    trending = arr(d.bets).map((b) => ({
      title: b.name || b.question || '', slug: b.eventSlug || null, topicEmoji: b.topicEmoji || '',
      volume: b.volume, outcomes: finite(b.odds) ? [{ label: 'Yes', odds: b.odds, change: null }] : [], nOutcomes: 1,
    }));
  }
  return {
    trending: trending.filter((t) => t.title && arr(t.outcomes).length),
    breaking: arr(d.breaking).filter((m) => m.question && finite(m.odds)),
    macro: arr(d.macro).filter((t) => t.title && finite(t.odds)),
    sources: d.sources && typeof d.sources === 'object' ? d.sources : {},
    stale: !!d._meta?.stale,
    timestamp: d.timestamp || null,
  };
}

function Delta({ change, minPts, title }) {
  const d = pts(change, minPts);
  return <span className={`pm-delta${d ? (d.up ? ' up' : ' down') : ''}`} title={d ? title : undefined}>{d ? d.text : ''}</span>;
}

function when(iso) {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return 'earlier';
  return new Date(t).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

// What the details sheet shows for each kind of row.
const trendBet = (t) => ({
  name: t.title, slug: t.slug, volume: t.volume, odds: t.outcomes[0]?.odds,
  outcomes: t.outcomes.length > 1 ? t.outcomes : null, endDate: t.endDate,
});
const breakBet = (m) => ({
  name: m.question, slug: m.slug, volume: m.volume, odds: m.odds, spark: m.spark, change: m.change, window: '24 hours',
});
const macroBet = (t) => ({
  name: t.title, slug: t.slug, volume: t.volume, odds: t.odds, spark: t.spark, change: t.change, window: '30 days',
  label: t.label && t.label !== 'Yes' ? t.label : null,
});

export default function PolymarketTable({ refreshKey = null, bust = false }) {
  const [board, setBoard] = useState(null);
  const [status, setStatus] = useState('loading'); // loading | ready | empty | error
  const [selected, setSelected] = useState(null);
  const [all, setAll] = useState({ trending: false, breaking: false });
  const close = useCallback(() => setSelected(null), []);
  const lastFetch = useRef(0);
  // A mounted flag instead of a per-effect `alive`: under StrictMode (dev) the effect
  // re-runs inside MIN_REFETCH_MS and a per-effect flag would drop the only response.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    // The floor stops the first-load double fetch (mount, then the page's first tick);
    // a manual refresh (`bust`) always goes through.
    if (!bust && Date.now() - lastFetch.current < MIN_REFETCH_MS) return undefined;
    lastFetch.current = Date.now();
    getJson('/api/polymarket', { bust }).then((data) => {
      if (!mounted.current) return;
      const b = data ? normalize(data) : null;
      if (b && (b.trending.length || b.breaking.length || b.macro.length)) {
        setBoard(b); setStatus('ready');
      } else {
        // Keep what we had if a refresh comes back empty.
        setStatus((s) => (s === 'ready' ? 'ready' : data ? 'empty' : 'error'));
      }
    });
    return undefined;
  }, [refreshKey]); // `bust` changes together with each tick

  if (status === 'loading') {
    return <div className="card"><Skeleton count={8} /></div>;
  }
  if (status !== 'ready' || !board) {
    return (
      <div className="card">
        <div className="error-message" style={status === 'error' ? { color: 'var(--red)' } : undefined}>
          {status === 'error'
            ? '⚠️ Polymarket data unavailable right now. Try refreshing the page.'
            : 'No betting markets available.'}
        </div>
      </div>
    );
  }

  const { trending, breaking, macro, sources } = board;
  const shown = {
    trending: all.trending ? trending : trending.slice(0, SHOW.trending),
    breaking: all.breaking ? breaking : breaking.slice(0, SHOW.breaking),
  };
  const more = (key, list) => list.length > SHOW[key] && (
    <button type="button" className="pm-more" aria-expanded={all[key]}
      onClick={() => setAll((a) => ({ ...a, [key]: !a[key] }))}>
      {all[key] ? 'Show fewer' : `Show all ${list.length}`}
    </button>
  );

  return (
    <section aria-label="Polymarket market sentiment">
      <div className="card pm-card" style={{ animationDelay: '0.8s' }}>
        <div className="card-header">
          <h2>📊 Market Sentiment</h2>
          <span className="badge badge-blue">What the crowd&apos;s betting on</span>
        </div>

        {board.stale && (
          <div className="pm-note pm-stale">🕐 Saved copy from {when(board.timestamp)}. Live sources didn&apos;t answer.</div>
        )}

        {macro.length > 0 && (
          <div className="pm-section">
            <div className="pm-head"><h3>📈 Macro</h3><span>30-day trend</span></div>
            <div className="pm-macro">
              {macro.map((t) => (
                <button type="button" key={t.slug || t.title} className="pm-tile" onClick={() => setSelected(macroBet(t))}>
                  <span className="pm-tile-title">{t.title}</span>
                  {t.label && t.label !== 'Yes' && <span className="pm-tile-label">{t.label}</span>}
                  <span className="pm-tile-row">
                    <span className="pm-tile-odds">{pct(t.odds)}</span>
                    <Delta change={t.change} minPts={0.5} title="Change over 30 days, in points" />
                  </span>
                  <Spark points={t.spark} className="pm-spark pm-tile-spark" label="Odds over the last 30 days" />
                </button>
              ))}
            </div>
          </div>
        )}
        {!macro.length && sources.macro == null && (
          <div className="pm-note">📈 Macro odds unavailable right now.</div>
        )}

        <div className="pm-cols">
          <div className="pm-section">
            <div className="pm-head"><h3>🔥 Trending</h3><span>Polymarket&apos;s front page</span></div>
            {trending.length ? (
              <ul className="pm-list">
                {shown.trending.map((t) => {
                  const binary = t.outcomes.length === 1;
                  return (
                    <li key={t.slug || t.title}>
                      <button type="button" className="pm-row pm-trend" onClick={() => setSelected(trendBet(t))}>
                        <span className="pm-trend-head">
                          <span className="pm-title">{t.topicEmoji ? `${t.topicEmoji} ` : ''}{t.title}</span>
                          <span className="pm-vol" title="Total traded">{money(t.volume)}</span>
                        </span>
                        {t.outcomes.slice(0, 2).map((o, i) => (
                          <span key={`${o.label}-${i}`} className={`pm-out${i ? ' pm-out-2' : ''}`}>
                            <span className="pm-out-label">{binary ? 'Chance' : o.label}</span>
                            <span className="pm-bar"><span style={{ width: `${clamp01(o.odds) * 100}%` }} /></span>
                            <span className="pm-out-odds">{pct(o.odds)}</span>
                            <Delta change={o.change} minPts={1} title="Change in the last 24 hours, in points" />
                          </span>
                        ))}
                      </button>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <div className="pm-note">Front-page picks unavailable right now.</div>
            )}
            {more('trending', trending)}
          </div>

          <div className="pm-section">
            <div className="pm-head"><h3>⚡ Breaking</h3><span>Biggest moves, last 24 h</span></div>
            {breaking.length ? (
              <ul className="pm-list">
                {shown.breaking.map((m) => (
                  <li key={`${m.slug}-${m.question}`}>
                    <button type="button" className="pm-row pm-break" onClick={() => setSelected(breakBet(m))}>
                      <span className="pm-title pm-q">{m.topicEmoji ? `${m.topicEmoji} ` : ''}{m.question}</span>
                      <Spark points={m.spark} className="pm-spark pm-break-spark" label="Odds over the last 24 hours" />
                      <span className="pm-break-num">
                        <span className="pm-break-odds">{pct(m.odds)}</span>
                        <Delta change={m.change} minPts={0} title="Change in the last 24 hours, in points" />
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="pm-note">
                {sources.breaking == null
                  ? 'Breaking moves unavailable right now.'
                  : 'Quiet day: no market moved 5+ points in the last 24 hours.'}
              </div>
            )}
            {more('breaking', breaking)}
          </div>
        </div>

        <div className="pm-foot">Live from Polymarket · sports left out · tap any market for details</div>
      </div>

      <MarketModal bet={selected} isOpen={!!selected} onClose={close} />
    </section>
  );
}
