import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
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
      // Check for the percentage values (odds * 100)
      expect(screen.getByText('65%')).toBeInTheDocument();
      expect(screen.getByText('72%')).toBeInTheDocument();
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
      expect(screen.getByText(/no betting markets available/i)).toBeInTheDocument();
    });
  });

  test('handles network timeout gracefully', async () => {
    fetch.mockRejectedValueOnce(new Error('Network timeout'));

    render(<PolymarketTable />);

    await waitFor(() => {
      expect(screen.getByText(/failed to fetch/i)).toBeInTheDocument();
    });
  });

  test('opens modal when a row is clicked', async () => {
    const mockData = {
      bets: [
        {
          name: 'Trump re-election',
          odds: 0.65,
          volume: 2500000,
          question: 'Trump re-election',
          probability: 0.65,
          slug: 'trump-reelection'
        }
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
    });

    // Click the row
    const row = screen.getByText('Trump re-election').closest('div[style*="cursor: pointer"]');
    fireEvent.click(row);

    // Modal should display the market details
    await waitFor(() => {
      expect(screen.getByText('Market Details')).toBeInTheDocument();
      expect(screen.getByText('Probability')).toBeInTheDocument();
      expect(screen.getByText('Trading Volume')).toBeInTheDocument();
    });
  });

  test('modal displays correct bet data from clicked row', async () => {
    const mockData = {
      bets: [
        {
          name: 'Bitcoin $100k',
          odds: 0.72,
          volume: 1800000,
          question: 'Bitcoin $100k',
          probability: 0.72,
          slug: 'bitcoin-100k'
        }
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
      expect(screen.getByText('Bitcoin $100k')).toBeInTheDocument();
    });

    // Click the row
    const row = screen.getByText('Bitcoin $100k').closest('div[style*="cursor: pointer"]');
    fireEvent.click(row);

    // Modal should display the correct data - check for modal-specific elements
    await waitFor(() => {
      expect(screen.getByText('Market Details')).toBeInTheDocument();
      expect(screen.getByText('Probability')).toBeInTheDocument();
      expect(screen.getByText('$1,800,000')).toBeInTheDocument();
    });
  });

  test('closes modal when close button is clicked', async () => {
    const mockData = {
      bets: [
        {
          name: 'Trump re-election',
          odds: 0.65,
          volume: 2500000,
          question: 'Trump re-election',
          probability: 0.65,
          slug: 'trump-reelection'
        }
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
    });

    // Click the row to open modal
    const row = screen.getByText('Trump re-election').closest('div[style*="cursor: pointer"]');
    fireEvent.click(row);

    await waitFor(() => {
      expect(screen.getByText('Market Details')).toBeInTheDocument();
    });

    // Click close button
    const closeButton = screen.getAllByLabelText('Close modal')[0];
    fireEvent.click(closeButton);

    // Modal should be closed
    await waitFor(() => {
      expect(screen.queryByText('Market Details')).not.toBeInTheDocument();
    });
  });

  test('closes modal when backdrop is clicked', async () => {
    const mockData = {
      bets: [
        {
          name: 'Trump re-election',
          odds: 0.65,
          volume: 2500000,
          question: 'Trump re-election',
          probability: 0.65,
          slug: 'trump-reelection'
        }
      ],
      timestamp: '2026-04-27T20:15:00Z',
      error: null
    };

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockData
    });

    const { container } = render(<PolymarketTable />);

    await waitFor(() => {
      expect(screen.getByText('Trump re-election')).toBeInTheDocument();
    });

    // Click the row to open modal
    const row = screen.getByText('Trump re-election').closest('div[style*="cursor: pointer"]');
    fireEvent.click(row);

    await waitFor(() => {
      expect(screen.getByText('Market Details')).toBeInTheDocument();
    });

    // Click backdrop
    const fixedDivs = container.querySelectorAll('div[style*="position: fixed"]');
    const backdrop = fixedDivs[0];
    fireEvent.click(backdrop);

    // Modal should be closed
    await waitFor(() => {
      expect(screen.queryByText('Market Details')).not.toBeInTheDocument();
    });
  });

  test('updates modal content when clicking different row while modal is open', async () => {
    const mockData = {
      bets: [
        {
          name: 'Trump re-election',
          odds: 0.65,
          volume: 2500000,
          question: 'Trump re-election',
          probability: 0.65,
          slug: 'trump-reelection'
        },
        {
          name: 'Bitcoin $100k',
          odds: 0.72,
          volume: 1800000,
          question: 'Bitcoin $100k',
          probability: 0.72,
          slug: 'bitcoin-100k'
        }
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
    });

    // Click first row
    const row1 = screen.getByText('Trump re-election').closest('div[style*="cursor: pointer"]');
    fireEvent.click(row1);

    await waitFor(() => {
      // Check that modal header and market details appear
      expect(screen.getByText('Market Details')).toBeInTheDocument();
      expect(screen.getByText('Probability')).toBeInTheDocument();
      expect(screen.getByText('Trading Volume')).toBeInTheDocument();
    });

    // Click second row
    const row2 = screen.getByText('Bitcoin $100k').closest('div[style*="cursor: pointer"]');
    fireEvent.click(row2);

    // Modal should now display Bitcoin data (verify volume is from second bet)
    await waitFor(() => {
      expect(screen.getByText('$1,800,000')).toBeInTheDocument();
    });
  });

  test('row has hover effect indicating it is clickable', async () => {
    const mockData = {
      bets: [
        { name: 'Trump re-election', odds: 0.65, volume: 2500000 }
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
    });

    const row = screen.getByText('Trump re-election').closest('div[style*="cursor: pointer"]');
    expect(row).toHaveStyle('cursor: pointer');

    // Simulate hover
    fireEvent.mouseEnter(row);
    expect(row.style.background).toBe('rgba(255, 255, 255, 0.05)');

    // Simulate mouse leave
    fireEvent.mouseLeave(row);
    expect(row.style.background).toBe('transparent');
  });
});
