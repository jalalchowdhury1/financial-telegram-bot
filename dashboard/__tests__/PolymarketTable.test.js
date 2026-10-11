import React from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';
import PolymarketTable, { pct, pts, money, normalize } from '../components/PolymarketTable';

const answer = (body, status = 200) => ({ status, json: async () => body });
const outcome = (label, odds, change = null) => ({ label, odds, change });
const trend = (i, extra = {}) => ({
  title: `Trending market ${i}`, slug: `trend-${i}`, topic: 'Politics', topicEmoji: '🏛️',
  volume: 1e6 * (i + 1), volume24h: 1e5, outcomes: [outcome('Leader', 0.6, 0.02), outcome('Runner-up', 0.3)], nOutcomes: 2, ...extra,
});
const mover = (i, extra = {}) => ({
  question: `Breaking question ${i}?`, slug: `break-${i}`, odds: 0.4, change: -0.1 - i / 100,
  volume: 1e5, spark: [0.5, 0.45, 0.4], topic: 'Geopolitics', topicEmoji: '🌍', ...extra,
});
const BOARD = {
  macro: [
    { title: 'US recession by end of 2026?', slug: 'us-recession-by-end-of-2026', label: 'Yes', odds: 0.065, change: -0.02, volume: 9e6, spark: [0.085, 0.07, 0.065] },
    { title: 'Fed decision in October?', slug: 'fed-october', label: 'No change', odds: 0.84, change: 0.05, volume: 3e7, spark: [] },
  ],
  trending: [
    trend(0, { title: 'Balance of Power: 2026 Midterms', slug: 'balance-of-power-2026-midterms', volume: 21_400_000,
      outcomes: [outcome('Democrats Sweep', 0.62, 0.012), outcome('R Senate, D House', 0.3, -0.02), outcome('Republicans Sweep', 0.08)], nOutcomes: 3 }),
    trend(1, { title: 'Modi out by December 31, 2026?', slug: 'modi-out', outcomes: [outcome('Yes', 0.07, 0.0115)], nOutcomes: 1 }),
    ...Array.from({ length: 6 }, (_, i) => trend(i + 2)),
  ],
  breaking: [
    mover(0, { question: 'Will Putin meet Lukashenko in Turkmenistan?', slug: 'putin-turkmenistan', odds: 0.015, change: -0.54, spark: [0.555, 0.3, 0.015] }),
    mover(1, { question: 'MrBeast video views?', slug: 'mrbeast', odds: 0.665, change: 0.35, spark: [0.3, 0.5, 0.665] }),
  ],
  sources: { trending: 'featured', breaking: 'biggest-movers', macro: 'macro-tags' },
  source: 'Polymarket API',
  timestamp: '2026-10-11T16:00:00Z',   // Oct 11 in every US time zone and UTC
  _meta: { hasErrors: false, messages: [] },
};

beforeEach(() => { global.fetch = jest.fn(); });

async function renderBoard(body = BOARD, props = {}) {
  fetch.mockResolvedValue(answer(body));
  const view = render(<PolymarketTable {...props} />);
  await screen.findByText('📊 Market Sentiment');
  return view;
}

describe('PolymarketTable', () => {
  test('shows loading skeletons first', () => {
    fetch.mockReturnValueOnce(new Promise(() => {}));
    render(<PolymarketTable />);
    expect(document.querySelectorAll('.skeleton').length).toBeGreaterThan(0);
  });

  test('draws the three slices: Macro, Trending, Breaking', async () => {
    await renderBoard();
    expect(screen.getByText('📈 Macro')).toBeInTheDocument();
    expect(screen.getByText('🔥 Trending')).toBeInTheDocument();
    expect(screen.getByText('⚡ Breaking')).toBeInTheDocument();
    // macro tile: odds, 30-day change in points, sparkline; a multi-outcome leader is named
    const rec = screen.getByText('US recession by end of 2026?').closest('button');
    expect(within(rec).getByText('7%')).toBeInTheDocument();
    expect(within(rec).getByText('▼2')).toBeInTheDocument();
    expect(rec.querySelector('svg polyline')).not.toBeNull();
    const fed = screen.getByText('Fed decision in October?').closest('button');
    expect(within(fed).getByText('No change')).toBeInTheDocument();
    expect(fed.querySelector('svg')).toBeNull();          // no history: no chart, number still shown
    // trending: the two likeliest outcomes, volume, 24h change ≥ 1 point
    const bop = screen.getByText(/Balance of Power: 2026 Midterms/).closest('button');
    expect(within(bop).getByText('Democrats Sweep')).toBeInTheDocument();
    expect(within(bop).getByText('R Senate, D House')).toBeInTheDocument();
    expect(within(bop).queryByText('Republicans Sweep')).toBeNull();
    expect(within(bop).getByText('$21.4M')).toBeInTheDocument();
    expect(within(bop).getByText('▲1.2')).toBeInTheDocument();
    // a single-market event reads as a chance
    const modi = screen.getByText(/Modi out by December 31/).closest('button');
    expect(within(modi).getByText('Chance')).toBeInTheDocument();
    // breaking: odds, 24h move, sparkline coloured by direction
    const putin = screen.getByText(/Will Putin meet Lukashenko/).closest('button');
    expect(within(putin).getByText('2%')).toBeInTheDocument();
    expect(within(putin).getByText('▼54')).toHaveClass('down');
    expect(putin.querySelector('svg').getAttribute('data-trend')).toBe('down');
    const beast = screen.getByText(/MrBeast video views/).closest('button');
    expect(within(beast).getByText('▲35')).toHaveClass('up');
  });

  test('long lists show 6 rows until "Show all" is tapped', async () => {
    await renderBoard();
    const trendingList = () => screen.getByText('🔥 Trending').closest('.pm-section').querySelectorAll('li');
    expect(trendingList()).toHaveLength(6);
    fireEvent.click(screen.getByRole('button', { name: 'Show all 8' }));
    expect(trendingList()).toHaveLength(8);
    fireEvent.click(screen.getByRole('button', { name: 'Show fewer' }));
    expect(trendingList()).toHaveLength(6);
    expect(screen.queryByRole('button', { name: /Show all 2/ })).toBeNull();   // breaking has only 2
  });

  test('tapping a breaking row opens its details with a direct Polymarket link', async () => {
    await renderBoard();
    fireEvent.click(screen.getByText(/Will Putin meet Lukashenko/).closest('button'));
    const sheet = screen.getByRole('dialog');
    expect(within(sheet).getByText('Market Details')).toBeInTheDocument();
    expect(within(sheet).getByText('Last 24 hours')).toBeInTheDocument();
    expect(within(sheet).getByText(/56% → 2%/)).toBeInTheDocument();
    expect(within(sheet).getByRole('link')).toHaveAttribute('href', 'https://polymarket.com/event/putin-turkmenistan');
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  test('tapping a trending row lists every outcome', async () => {
    await renderBoard();
    fireEvent.click(screen.getByText(/Balance of Power: 2026 Midterms/).closest('button'));
    const sheet = screen.getByRole('dialog');
    expect(within(sheet).getByText('Outcomes')).toBeInTheDocument();
    expect(within(sheet).getByText('Republicans Sweep')).toBeInTheDocument();
    expect(within(sheet).queryByText('Probability')).toBeNull();
    expect(within(sheet).getByRole('link')).toHaveAttribute('href', 'https://polymarket.com/event/balance-of-power-2026-midterms');
  });

  test('an old {bets} copy (saved before this redesign) still draws as Trending', async () => {
    await renderBoard({ bets: [{ name: 'Bitcoin $200k by 2026?', odds: 0.62, volume: 1_800_000, topicEmoji: '🪙', eventSlug: 'btc-200k' }], source: 'cache' });
    const row = screen.getByText(/Bitcoin \$200k by 2026\?/).closest('button');
    expect(within(row).getByText('Chance')).toBeInTheDocument();
    expect(within(row).getByText('62%')).toBeInTheDocument();
    expect(screen.queryByText('📈 Macro')).toBeNull();
    // an old copy has no `sources`: unknown, so never "Quiet day"
    expect(screen.getByText('Breaking moves unavailable right now.')).toBeInTheDocument();
    expect(screen.queryByText(/Quiet day/)).toBeNull();
  });

  test('an empty Breaking list says why: quiet day vs. feed down', async () => {
    const view = await renderBoard({ ...BOARD, breaking: [] });
    expect(screen.getByText(/Quiet day/)).toBeInTheDocument();
    view.unmount();
    await renderBoard({ ...BOARD, breaking: [], sources: { ...BOARD.sources, breaking: null } });
    expect(screen.getByText('Breaking moves unavailable right now.')).toBeInTheDocument();
  });

  test('a saved copy is labelled as one', async () => {
    await renderBoard({ ...BOARD, _meta: { stale: true, hasErrors: true, messages: [] } });
    expect(screen.getByText(/Saved copy from Oct 11/)).toBeInTheDocument();
  });

  test('sparklines never draw NaN, whatever the history holds', async () => {
    await renderBoard({ ...BOARD, macro: [{ ...BOARD.macro[0], spark: [0.1, null, 'x', NaN, Infinity, 0.2] }],
      breaking: [mover(0, { spark: [0.4] })] });
    for (const pl of document.querySelectorAll('polyline')) expect(pl.getAttribute('points')).not.toMatch(/NaN|Infinity/);
    expect(screen.getByText(/Breaking question 0\?/).closest('button').querySelector('svg')).toBeNull();   // 1 point: no chart
  });

  test('nothing usable: says so', async () => {
    fetch.mockResolvedValue(answer({ trending: [], breaking: [], macro: [], timestamp: 'x' }));
    render(<PolymarketTable />);
    expect(await screen.findByText('No betting markets available.')).toBeInTheDocument();
  });

  test('no answer at all: says the data is unavailable', async () => {
    fetch.mockRejectedValue(new Error('network down'));
    render(<PolymarketTable />);
    expect(await screen.findByText(/Polymarket data unavailable/, {}, { timeout: 4000 })).toBeInTheDocument();
  });

  test('re-fetches on the page refresh tick; a manual refresh skips the edge cache', async () => {
    const { rerender } = await renderBoard(BOARD, { refreshKey: 0 });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe('/api/polymarket');
    fetch.mockResolvedValue(answer({ ...BOARD, breaking: [mover(5, { question: 'A fresher move?' })] }));
    rerender(<PolymarketTable refreshKey={1} bust />);
    expect(await screen.findByText(/A fresher move\?/)).toBeInTheDocument();   // the new answer replaced the old
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls[1][0]).toMatch(/^\/api\/polymarket\?_t=\d+$/);
  });
});

describe('formatting helpers', () => {
  test('pct reads like Polymarket', () => {
    expect([pct(0.62), pct(0.004), pct(0), pct(0.996), pct(1), pct(NaN), pct(null)])
      .toEqual(['62%', '<1%', '0%', '>99%', '100%', '—', '—']);
  });
  test('pts shows a move in points, hiding noise', () => {
    expect(pts(-0.54)).toEqual({ up: false, text: '▼54' });
    expect(pts(0.012)).toEqual({ up: true, text: '▲1.2' });
    expect(pts(0.009)).toBeNull();
    expect(pts(0.009, 0.5)).toEqual({ up: true, text: '▲0.9' });
    expect(pts(null)).toBeNull();
  });
  test('money is short', () => {
    expect([money(715_300_000), money(180_000_000), money(21_354_286), money(153_000), money(900), money(0), money(2.5e9)])
      .toEqual(['$715M', '$180M', '$21.4M', '$153k', '$900', '$0', '$2.5B']);
  });
  test('normalize drops rows it cannot draw', () => {
    const b = normalize({ trending: [{ title: '', outcomes: [] }, null, trend(0)], breaking: [{ question: 'q', odds: 'x' }], macro: 'nope' });
    expect(b.trending).toHaveLength(1);
    expect(b.breaking).toEqual([]);
    expect(b.macro).toEqual([]);
  });
});
