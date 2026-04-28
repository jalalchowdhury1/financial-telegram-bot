# Design: Full Market Text Modal for PolymarketTable

**Date:** April 28, 2026
**Feature:** Click market to view full text in modal dialog
**Status:** Approved for implementation

## Problem

Market questions in the PolymarketTable are truncated at 42 characters with "..." (e.g., "Will Espanyol qualify for the League Phase..."). Users need to see the full market text to make informed decisions.

## Solution

Add a modal dialog that displays full market details when a user clicks any row:
- Full market question (no truncation)
- Probability percentage (color-coded bar + percentage)
- Trading volume (formatted)
- Link to view on Polymarket.com (opens in new tab)

Modal closes via:
- Close button (X, top right)
- Clicking outside the modal (backdrop)

## Architecture

### Components

**MarketModal.js** (new)
- Presentational component, receives props: `bet`, `isOpen`, `onClose`
- Displays full market details in glassmorphism card
- Renders close button and Polymarket link button
- Centered overlay with dark backdrop (rgba background, pointer-events: none on backdrop)

**PolymarketTable.js** (enhanced)
- Add state: `selectedBet`, `isModalOpen`
- Add click handler to market rows → sets selectedBet + isModalOpen
- Add close handler → clears selectedBet + isModalOpen
- Render `<MarketModal>` at bottom of component
- Import modal component

### Data Flow

1. User clicks market row
2. Row click handler: `setSelectedBet(bet)` + `setIsModalOpen(true)`
3. Modal renders with bet data
4. User clicks close button or outside → `setIsModalOpen(false)`
5. Modal unmounts

### Polymarket URL Generation

Market URLs follow pattern: `https://polymarket.com/market/<slug>`

The `slug` field is already in the API response from Polymarket. Button: `"View on Polymarket"` → link to market page.

## Styling

Modal matches dashboard glassmorphism:
- Card container with backdrop blur effect
- Dark background: `rgba(30, 41, 59, 0.8)`
- Border: `1px solid rgba(255, 255, 255, 0.1)`
- Centered on viewport
- Responsive padding (larger on desktop, smaller on mobile)
- Close button styled as text (cursor: pointer, hover effect)

Reuse `getOddsColor()` function for probability bar (consistent with table).

## UI Layout

```
┌─────────────────────────────────────────┐
│ Market Details                        X │
├─────────────────────────────────────────┤
│                                         │
│ Will Espanyol qualify for the League   │
│ Phase of the 2026-27 UEFA Europa       │
│ League?                                 │
│                                         │
│ Probability: [████░░░░░░░░] 47%        │
│                                         │
│ Volume: $99,990                         │
│                                         │
│         [View on Polymarket →]          │
│                                         │
└─────────────────────────────────────────┘
```

## Implementation Checklist

- [ ] Create MarketModal.js component
- [ ] Add modal state to PolymarketTable
- [ ] Add row click handler
- [ ] Add modal close handler (button + backdrop)
- [ ] Generate Polymarket URL correctly
- [ ] Test modal open/close
- [ ] Test responsive design on mobile
- [ ] Verify link opens in new tab

## Files Modified/Created

- **Create:** `dashboard/components/MarketModal.js`
- **Modify:** `dashboard/components/PolymarketTable.js`

## Testing

- Click each market row → modal opens with correct data
- Close button closes modal
- Clicking outside modal closes it
- "View on Polymarket" link works and opens in new tab
- Modal responsive on mobile (viewport < 768px)
