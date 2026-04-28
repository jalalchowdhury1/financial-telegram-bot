import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import PolymarketTable from '../components/PolymarketTable';

// Mock fetch globally
global.fetch = jest.fn();

describe('PolymarketTable', () => {
  afterEach(() => {
    fetch.mockClear();
  });

  test('renders loading state initially', () => {
    fetch.mockReturnValueOnce(
      new Promise(() => {}) // Never resolves
    );

    render(<PolymarketTable />);
    // Skeleton component renders with skeleton-text classes
    const skeletons = document.querySelectorAll('.skeleton');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  test('renders table with fetched data', async () => {
    const mockData = {
      bets: [
        { name: 'Trump re-election', odds: 0.65, volume: 2500000 },
        { name: 'Bitcoin $100k', odds: 0.72, volume: 1800000 }
      ],
      timestamp: '2026-04-27T20:15:00Z',
      error: null
    };

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockData
    });

    render(<PolymarketTable />);

    await waitFor(() => {
      expect(screen.getByText('Trump re-election')).toBeInTheDocument();
      expect(screen.getByText('Bitcoin $100k')).toBeInTheDocument();
      expect(screen.getByText('0.65')).toBeInTheDocument();
      expect(screen.getByText('1,800,000')).toBeInTheDocument();
    });
  });

  test('displays error state when API fails', async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      json: async () => ({ error: 'Polymarket API unavailable', bets: [] })
    });

    render(<PolymarketTable />);

    await waitFor(() => {
      expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
    });
  });

  test('displays empty state when no bets available', async () => {
    const mockData = {
      bets: [],
      timestamp: '2026-04-27T20:15:00Z',
      error: null
    };

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockData
    });

    render(<PolymarketTable />);

    await waitFor(() => {
      expect(screen.getByText(/no data/i)).toBeInTheDocument();
    });
  });

  test('handles network timeout gracefully', async () => {
    fetch.mockRejectedValueOnce(new Error('Network timeout'));

    render(<PolymarketTable />);

    await waitFor(() => {
      expect(screen.getByText(/failed to fetch/i)).toBeInTheDocument();
    });
  });
});
