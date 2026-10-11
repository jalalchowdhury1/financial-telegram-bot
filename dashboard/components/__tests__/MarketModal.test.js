import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import MarketModal from '../MarketModal';

describe('MarketModal Component', () => {
  const mockBet = {
    name: 'Will Espanyol qualify for the League Phase of the 2026-27 UEFA Europa League?',
    odds: 0.47,
    volume: 99990,
    slug: 'will-espanyol-qualify-europa-league'
  };

  const mockOnClose = jest.fn();

  beforeEach(() => {
    mockOnClose.mockClear();
  });

  test('renders nothing when isOpen is false', () => {
    const { container } = render(
      <MarketModal bet={mockBet} isOpen={false} onClose={mockOnClose} />
    );
    expect(container.firstChild).toBeNull();
  });

  test('renders nothing when bet is null', () => {
    const { container } = render(
      <MarketModal bet={null} isOpen={true} onClose={mockOnClose} />
    );
    expect(container.firstChild).toBeNull();
  });

  test('renders modal when isOpen is true and bet exists', () => {
    render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    expect(screen.getByText('Market Details')).toBeInTheDocument();
    expect(screen.getByText(mockBet.name)).toBeInTheDocument();
  });

  test('displays full market question without truncation', () => {
    render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    const question = screen.getByText(mockBet.name);
    expect(question).toBeInTheDocument();
    expect(question.textContent).toBe(mockBet.name);
  });

  test('displays probability percentage correctly', () => {
    render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    expect(screen.getByText('47%')).toBeInTheDocument();
  });

  test('displays formatted trading volume', () => {
    render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    expect(screen.getByText('$99,990')).toBeInTheDocument();
  });

  test('formats volume with commas for large numbers', () => {
    const betWithLargeVolume = {
      ...mockBet,
      volume: 1234567
    };

    render(
      <MarketModal bet={betWithLargeVolume} isOpen={true} onClose={mockOnClose} />
    );

    expect(screen.getByText('$1,234,567')).toBeInTheDocument();
  });

  test('link opens that market on Polymarket (its event page)', () => {
    render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    const link = screen.getByRole('link');
    // The event slug is the one polymarket.com itself links to: /event/<event slug>.
    expect(link).toHaveAttribute('href', 'https://polymarket.com/event/will-espanyol-qualify-europa-league');
  });

  test('a row without a slug links to the Polymarket homepage', () => {
    render(
      <MarketModal bet={{ ...mockBet, slug: undefined }} isOpen={true} onClose={mockOnClose} />
    );

    expect(screen.getByRole('link')).toHaveAttribute('href', 'https://polymarket.com');
  });

  test('link opens in new tab with security attributes', () => {
    render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });

  test('close button calls onClose when clicked', () => {
    render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    const closeButton = screen.getAllByLabelText('Close modal')[0];
    fireEvent.click(closeButton);

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  test('backdrop click calls onClose', () => {
    const { container } = render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    // Get all fixed positioned divs and find the backdrop (the one before the modal)
    const fixedDivs = container.querySelectorAll('div[style*="position: fixed"]');
    const backdrop = fixedDivs[0]; // Backdrop is rendered first

    expect(backdrop).toBeTruthy();
    fireEvent.click(backdrop);

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  test('modal content click does not trigger onClose', () => {
    const { container } = render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    // Get the modal card content
    const modalContent = screen.getByText('Market Details');
    fireEvent.click(modalContent);

    expect(mockOnClose).not.toHaveBeenCalled();
  });

  test('probability bar width matches probability value', () => {
    const { container } = render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    // The bar should have width of 47%
    const bars = container.querySelectorAll('div[style*="width:"]');
    const probabilityBar = Array.from(bars).find(bar =>
      bar.style.width === '47%' && bar.style.height === '100%'
    );

    expect(probabilityBar).toBeInTheDocument();
  });

  test('displays correct color for probability bar based on value', () => {
    const { container: containerMid } = render(
      <MarketModal bet={{ ...mockBet, odds: 0.5 }} isOpen={true} onClose={mockOnClose} />
    );

    // 0.5 probability should use yellow color
    expect(screen.getByText('50%')).toHaveStyle('color: var(--yellow)');
  });

  test('handles edge case: 0 probability', () => {
    render(
      <MarketModal bet={{ ...mockBet, odds: 0 }} isOpen={true} onClose={mockOnClose} />
    );

    expect(screen.getByText('0%')).toBeInTheDocument();
  });

  test('handles edge case: 1.0 (100%) probability', () => {
    render(
      <MarketModal bet={{ ...mockBet, odds: 1.0 }} isOpen={true} onClose={mockOnClose} />
    );

    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  test('handles edge case: 0 volume', () => {
    render(
      <MarketModal bet={{ ...mockBet, volume: 0 }} isOpen={true} onClose={mockOnClose} />
    );

    expect(screen.getByText('$0')).toBeInTheDocument();
  });

  test('close button has aria-label for accessibility', () => {
    render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    const closeButtons = screen.getAllByLabelText('Close modal');
    expect(closeButtons.length).toBeGreaterThan(0);
  });

  test('modal is semantically structured', () => {
    const { container } = render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    // Check for header with close button
    expect(screen.getByText('Market Details')).toBeInTheDocument();

    // Check for question section
    expect(screen.getByText(mockBet.name)).toBeInTheDocument();

    // Check for probability label
    expect(screen.getByText('Probability')).toBeInTheDocument();

    // Check for volume label
    expect(screen.getByText('Trading Volume')).toBeInTheDocument();
  });

  test('renders with different probability values and correct colors', () => {
    const testCases = [
      { odds: 0.15, expectedColor: 'var(--red)' },
      { odds: 0.35, expectedColor: '#f97316' },
      { odds: 0.55, expectedColor: 'var(--yellow)' },
      { odds: 0.75, expectedColor: '#22c55e' },
      { odds: 0.95, expectedColor: 'var(--green)' }
    ];

    testCases.forEach(({ odds, expectedColor }) => {
      const { unmount } = render(
        <MarketModal bet={{ ...mockBet, odds }} isOpen={true} onClose={mockOnClose} />
      );

      const percentText = screen.getByText(`${(odds * 100).toFixed(0)}%`);
      expect(percentText).toHaveStyle(`color: ${expectedColor}`);

      unmount();
    });
  });
  describe('QoL: behaves like a real sheet', () => {
    afterEach(() => {
      document.documentElement.className = '';
      document.documentElement.style.overflow = '';
      document.body.style.overflow = '';
    });

    test('is announced as a modal dialog named by its title', () => {
      render(<MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />);
      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-modal', 'true');
      expect(dialog).toHaveAccessibleName('Market Details');
    });

    test('Escape closes it; other keys do not', () => {
      render(<MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />);
      fireEvent.keyDown(document, { key: 'Enter' });
      expect(mockOnClose).not.toHaveBeenCalled();
      fireEvent.keyDown(document, { key: 'Escape' });
      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    test('a closed modal does not listen for Escape', () => {
      render(<MarketModal bet={mockBet} isOpen={false} onClose={mockOnClose} />);
      fireEvent.keyDown(document, { key: 'Escape' });
      expect(mockOnClose).not.toHaveBeenCalled();
    });

    test('holds the page still while open and lets go on close and on unmount', () => {
      const html = document.documentElement;
      const { rerender, unmount } = render(<MarketModal bet={mockBet} isOpen={false} onClose={mockOnClose} />);
      expect(html.classList.contains('sheet-open')).toBe(false);
      rerender(<MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />);
      expect(html.classList.contains('sheet-open')).toBe(true);
      expect(html.style.overflow).toBe('hidden');
      rerender(<MarketModal bet={mockBet} isOpen={false} onClose={mockOnClose} />);
      expect(html.classList.contains('sheet-open')).toBe(false);
      expect(html.style.overflow).toBe('');
      rerender(<MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />);
      unmount();
      expect(html.classList.contains('sheet-open')).toBe(false);
      expect(html.style.overflow).toBe('');
    });

    test('the sheet scrolls itself and its close button carries the 44px hit class', () => {
      render(<MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />);
      expect(document.querySelector('[data-sheet-scroll]')).not.toBeNull();
      expect(screen.getAllByLabelText('Close modal').some((b) => b.tagName === 'BUTTON' && b.classList.contains('sheet-x'))).toBe(true);
    });
  });
});

describe('MarketModal: Market Sentiment extras', () => {
  const onClose = jest.fn();

  test('a multi-outcome event lists every outcome instead of one probability', () => {
    const { container } = render(
      <MarketModal
        bet={{
          name: 'Balance of Power: 2026 Midterms', slug: 'balance-of-power-2026-midterms', volume: 21_400_000, odds: 0.62,
          outcomes: [{ label: 'Democrats Sweep', odds: 0.62 }, { label: 'R Senate, D House', odds: 0.3 }, { label: 'Republicans Sweep', odds: 0.004 }],
        }}
        isOpen={true}
        onClose={onClose}
      />
    );
    expect(screen.getByText('Outcomes')).toBeInTheDocument();
    expect(screen.queryByText('Probability')).toBeNull();
    expect(container.querySelectorAll('.pm-sheet-out')).toHaveLength(3);
    expect(screen.getByText('Republicans Sweep')).toBeInTheDocument();
    expect(screen.getByText('<1%')).toBeInTheDocument();      // tiny odds read like Polymarket's
  });

  test('a single outcome keeps the probability bar', () => {
    render(<MarketModal bet={{ name: 'Modi out?', odds: 0.004, outcomes: [{ label: 'Yes', odds: 0.004 }] }} isOpen={true} onClose={onClose} />);
    expect(screen.getByText('Probability')).toBeInTheDocument();
    expect(screen.getByText('<1%')).toBeInTheDocument();
    expect(screen.queryByText('Outcomes')).toBeNull();
  });

  test('a breaking move shows its 24-hour chart, start → now and the move in points', () => {
    const { container } = render(
      <MarketModal
        bet={{ name: 'Will Putin meet Lukashenko in Turkmenistan?', odds: 0.015, spark: [0.555, 0.3, 0.015], change: -0.54, window: '24 hours' }}
        isOpen={true}
        onClose={onClose}
      />
    );
    expect(screen.getByText('Last 24 hours')).toBeInTheDocument();
    expect(screen.getByText(/56% → 2%/)).toBeInTheDocument();
    expect(screen.getByText('▼54 pts')).toBeInTheDocument();
    expect(container.querySelector('.pm-sheet-spark').getAttribute('data-trend')).toBe('down');
  });

  test('a macro tile names its outcome over 30 days', () => {
    render(
      <MarketModal
        bet={{ name: 'Fed decision in October?', label: 'No change', odds: 0.996, spark: [0.79, 0.84, 0.996], change: 0.206, window: '30 days' }}
        isOpen={true}
        onClose={onClose}
      />
    );
    expect(screen.getByText('No change: last 30 days')).toBeInTheDocument();
    expect(screen.getByText(/79% → >99%/)).toBeInTheDocument();
    expect(screen.getByText('▲20.6 pts')).toBeInTheDocument();
  });

  test('no chart without two real points', () => {
    const { container } = render(
      <MarketModal bet={{ name: 'Thin history', odds: 0.4, spark: [0.4, NaN, null], change: 0.1, window: '24 hours' }} isOpen={true} onClose={onClose} />
    );
    expect(screen.queryByText('Last 24 hours')).toBeNull();
    expect(container.querySelector('svg')).toBeNull();
  });
});
