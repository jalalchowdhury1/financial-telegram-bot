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

  test('link has correct href to Polymarket', () => {
    render(
      <MarketModal bet={mockBet} isOpen={true} onClose={mockOnClose} />
    );

    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', `https://polymarket.com/market/${mockBet.slug}`);
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
});
