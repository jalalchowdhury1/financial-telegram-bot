/**
 * MarketModal Usage Example
 *
 * This file demonstrates how to integrate MarketModal into PolymarketTable
 * and manage the modal state in a parent component.
 */

import { useState } from 'react';
import PolymarketTable from './PolymarketTable';
import MarketModal from './MarketModal';

/**
 * Example: Integration with PolymarketTable
 *
 * Add this hook to PolymarketTable or a parent component that wraps it:
 */
export function PolymarketTableWithModal() {
  const [selectedBet, setSelectedBet] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleRowClick = (bet) => {
    setSelectedBet(bet);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    // Optional: Clear selectedBet after modal closes for cleanup
    setTimeout(() => setSelectedBet(null), 300);
  };

  return (
    <>
      {/* PolymarketTable with click handlers */}
      {/*
        In PolymarketTable.js, wrap each bet row with:
        onClick={() => handleRowClick(bet)}
        style={{ cursor: 'pointer' }}
      */}

      {/* Modal overlay */}
      <MarketModal
        bet={selectedBet}
        isOpen={isModalOpen}
        onClose={handleCloseModal}
      />
    </>
  );
}

/**
 * Expected bet data structure:
 *
 * {
 *   name: string (short name for table display, e.g., "Will Espanyol qualify...")
 *   question: string (full question text for modal, e.g., "Will Espanyol qualify for the...")
 *   probability: number (decimal 0-1, e.g., 0.47)
 *   odds: number (same as probability, for backwards compatibility)
 *   volume: number (trading volume, e.g., 99990)
 *   slug: string (URL slug, e.g., "will-espanyol-qualify-europa-league")
 * }
 */

/**
 * Integration Steps:
 *
 * 1. Import MarketModal in PolymarketTable or parent:
 *    import MarketModal from './MarketModal';
 *
 * 2. Add state management:
 *    const [selectedBet, setSelectedBet] = useState(null);
 *    const [isModalOpen, setIsModalOpen] = useState(false);
 *
 * 3. Add click handler to bet rows:
 *    onClick={() => {
 *      setSelectedBet(bet);
 *      setIsModalOpen(true);
 *    }}
 *    style={{ cursor: 'pointer' }}
 *
 * 4. Render the modal after table:
 *    <MarketModal
 *      bet={selectedBet}
 *      isOpen={isModalOpen}
 *      onClose={() => setIsModalOpen(false)}
 *    />
 *
 * 5. Add visual feedback on hover:
 *    onMouseEnter={(e) => {
 *      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.02)';
 *    }}
 *    onMouseLeave={(e) => {
 *      e.currentTarget.style.background = 'transparent';
 *    }}
 */

/**
 * CSS Classes (optional, from globals.css):
 *
 * The component uses inline styles but also respects CSS variables:
 * - --text-primary: Main text color
 * - --text-secondary: Secondary text color
 * - --text-accent: Accent text color (for links)
 * - --red: Red indicator color
 * - --yellow: Yellow indicator color
 * - --green: Green indicator color
 * - --bg-card: Card background
 *
 * Color thresholds for probability bar:
 * - < 0.2: Red (unlikely)
 * - 0.2-0.4: Orange (low probability)
 * - 0.4-0.6: Yellow (medium probability)
 * - 0.6-0.8: Green (likely)
 * - > 0.8: Bright green (very likely)
 */

/**
 * Accessibility Features:
 *
 * - aria-label on close button and backdrop
 * - Semantic HTML structure
 * - Keyboard accessible (can use Escape key when implemented)
 * - Screen reader friendly labels
 * - Color contrast compliant
 * - Links open in new tab with rel="noopener noreferrer"
 */

/**
 * Responsive Design:
 *
 * - 90% width on small screens
 * - Max width 520px on larger screens
 * - Centered on viewport with transform
 * - Padding adjusts for touch devices
 * - Backdrop blur for visual separation
 */
