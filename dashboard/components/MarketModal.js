'use client';

/**
 * MarketModal Component - Displays full market details in a modal dialog
 *
 * Features:
 * - Glassmorphism card styling matching dashboard design system
 * - Displays full market question without truncation
 * - Color-coded probability bar with dynamic width
 * - Formatted trading volume display
 * - Link to that market on Polymarket (its event page when the row carries a slug)
 * - Optional extras the Market Sentiment rows pass: every outcome's odds (`outcomes`),
 *   the leading outcome (`label`) and an odds chart (`spark` over `window`, with `change`)
 * - Dismissible via close button, backdrop click or Esc
 * - A modal dialog (role="dialog", aria-modal) that holds the page still while open
 *   (useSheetLock), takes focus to its × and hands it back on close (useSheetFocus),
 *   with a 44px tap area on its × (.sheet-x)
 */
import { useEffect, useRef } from 'react';
import useSheetLock, { useSheetFocus } from './useSheetLock';
import OddsSpark from './OddsSpark';

const finite = (v) => typeof v === 'number' && Number.isFinite(v);
const sectionLabel = {
  display: 'block',
  fontSize: '0.85rem',
  fontWeight: 600,
  color: 'var(--text-secondary)',
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  marginBottom: '10px'
};

export default function MarketModal({ bet, isOpen, onClose }) {
  const open = !!(isOpen && bet);
  const dialogRef = useRef(null);
  useSheetLock(open);
  useSheetFocus(open, dialogRef); // focus moves to the ×, stays in the sheet, then goes back

  // Esc closes it, like the Jev sheet
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  // Early return if modal is closed
  if (!open) {
    return null;
  }

  // Get odds color based on probability value
  const getOddsColor = (odds) => {
    if (odds < 0.2) return 'var(--red)';
    if (odds < 0.4) return '#f97316';
    if (odds < 0.6) return 'var(--yellow)';
    if (odds < 0.8) return '#22c55e';
    return 'var(--green)';
  };

  // Format volume with dollar sign and commas
  const formatVolume = (volume) => {
    return `$${(volume || 0).toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
  };

  // The event page Polymarket itself links to (polymarket.com/event/<event slug>);
  // the homepage when a row has no slug (e.g. an old saved copy).
  const polymarketUrl = bet.slug
    ? `https://polymarket.com/event/${encodeURIComponent(bet.slug)}`
    : 'https://polymarket.com';
  const outcomes = Array.isArray(bet.outcomes) ? bet.outcomes.filter((o) => o && finite(o.odds)) : [];
  const spark = Array.isArray(bet.spark) ? bet.spark.filter(finite) : [];
  const fmtPct = (v) => (v > 0 && v < 0.01 ? '<1%' : v > 0.99 && v < 1 ? '>99%' : `${Math.round(v * 100)}%`);

  // Use 'odds' from API (not 'probability')
  const odds = bet.odds || 0;
  const barColor = getOddsColor(odds);

  return (
    <>
      {/* Backdrop overlay and centering container */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(30, 41, 59, 0.8)',
          backdropFilter: 'blur(4px)',
          zIndex: 9998,
          animation: 'fadeIn 0.2s ease forwards',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px',
          overscrollBehavior: 'contain'
        }}
        onClick={onClose}
        aria-label="Close modal"
      >
        {/* Modal card - prevent click from propagating to backdrop */}
        <div
          ref={dialogRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby="market-modal-title"
          data-sheet-scroll=""
          style={{
            width: '100%',
            maxWidth: '520px',
            maxHeight: '90vh',
            overflowY: 'auto',
            overscrollBehavior: 'contain',
            zIndex: 9999,
            animation: 'fadeInUp 0.3s ease forwards',
            pointerEvents: 'auto'
          }}
          onClick={(e) => e.stopPropagation()}
        >
        {/* Glass card */}
        <div
          style={{
            background: 'rgba(17, 24, 39, 0.9)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '16px',
            padding: '28px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.04)'
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header with close button */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              marginBottom: '24px',
              paddingBottom: '16px',
              borderBottom: '1px solid rgba(255, 255, 255, 0.08)'
            }}
          >
            <h2
              id="market-modal-title"
              style={{
                fontSize: '1.1rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                margin: 0
              }}
            >
              Market Details
            </h2>

            {/* Close button */}
            <button
              className="sheet-x"
              onClick={onClose}
              aria-label="Close modal"
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '6px',
                width: '32px',
                height: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                color: 'var(--text-secondary)',
                fontSize: '18px',
                transition: 'all 0.2s ease',
                padding: 0,
                fontFamily: 'inherit'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
                e.currentTarget.style.color = 'var(--text-primary)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                e.currentTarget.style.color = 'var(--text-secondary)';
              }}
            >
              ×
            </button>
          </div>

          {/* Question section */}
          <div style={{ marginBottom: '28px' }}>
            <p
              style={{
                fontSize: '1rem',
                lineHeight: '1.6',
                color: 'var(--text-primary)',
                fontWeight: 500,
                margin: 0
              }}
            >
              {bet.name || bet.question}
            </p>
          </div>

          {/* Every outcome (multi-outcome events), likeliest first */}
          {outcomes.length > 1 && (
            <div style={{ marginBottom: '28px' }}>
              <span style={sectionLabel}>Outcomes</span>
              {outcomes.map((o, i) => (
                <div key={`${o.label}-${i}`} className="pm-sheet-out">
                  <span className="pm-sheet-out-label">{o.label}</span>
                  <span className="pm-bar"><span style={{ width: `${Math.min(1, Math.max(0, o.odds)) * 100}%` }} /></span>
                  <span className="pm-sheet-out-odds">{fmtPct(o.odds)}</span>
                </div>
              ))}
            </div>
          )}

          {/* Probability section */}
          {outcomes.length < 2 && (
            <div style={{ marginBottom: '28px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '10px'
                }}
              >
                <label
                  style={{
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    color: 'var(--text-secondary)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    margin: 0
                  }}
                >
                  Probability
                </label>
                <span
                  style={{
                    fontSize: '0.9rem',
                    fontWeight: 700,
                    fontFamily: "'JetBrains Mono', monospace",
                    color: barColor
                  }}
                >
                  {fmtPct(odds)}
                </span>
              </div>

              {/* Probability bar */}
              <div
                style={{
                  width: '100%',
                  height: '6px',
                  background: 'rgba(255, 255, 255, 0.08)',
                  borderRadius: '3px',
                  overflow: 'hidden'
                }}
              >
                <div
                  style={{
                    width: `${odds * 100}%`,
                    height: '100%',
                    background: barColor,
                    boxShadow: `0 0 6px ${barColor}50`,
                    borderRadius: '3px',
                    transition: 'width 0.3s ease'
                  }}
                />
              </div>
            </div>

          )}

          {/* Odds over time (Breaking: 24 hours, Macro: 30 days) */}
          {spark.length >= 2 && (
            <div style={{ marginBottom: '28px' }}>
              <span style={sectionLabel}>{bet.label ? `${bet.label}: last ${bet.window || 'days'}` : `Last ${bet.window || 'days'}`}</span>
              <OddsSpark points={spark} className="pm-sheet-spark" label={`Odds over the last ${bet.window || 'days'}`} />
              <p className="pm-sheet-move">
                {fmtPct(spark[0])} → {fmtPct(finite(odds) ? odds : spark[spark.length - 1])}
                {finite(bet.change) && Math.abs(bet.change) >= 0.0005 && (
                  <span style={{ color: bet.change > 0 ? 'var(--green)' : 'var(--red)', marginLeft: '8px' }}>
                    {bet.change > 0 ? '▲' : '▼'}{Math.abs(Math.round(bet.change * 1000) / 10)} pts
                  </span>
                )}
              </p>
            </div>
          )}

          {/* Volume section */}
          <div style={{ marginBottom: '28px' }}>
            <label
              style={{
                display: 'block',
                fontSize: '0.85rem',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                marginBottom: '8px'
              }}
            >
              Trading Volume
            </label>
            <p
              style={{
                fontSize: '1.1rem',
                fontWeight: 700,
                fontFamily: "'JetBrains Mono', monospace",
                color: 'var(--text-primary)',
                margin: 0
              }}
            >
              {formatVolume(bet.volume)}
            </p>
          </div>

          {/* Link button */}
          <a
            href={polymarketUrl}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              padding: '12px 18px',
              background: 'rgba(99, 102, 241, 0.15)',
              border: '1px solid rgba(129, 140, 248, 0.3)',
              borderRadius: '8px',
              color: 'var(--text-accent)',
              fontSize: '0.9rem',
              fontWeight: 600,
              textDecoration: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              fontFamily: 'inherit'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(99, 102, 241, 0.25)';
              e.currentTarget.style.borderColor = 'rgba(129, 140, 248, 0.5)';
              e.currentTarget.style.boxShadow = '0 0 12px rgba(129, 140, 248, 0.2)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(99, 102, 241, 0.15)';
              e.currentTarget.style.borderColor = 'rgba(129, 140, 248, 0.3)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            View on Polymarket
            <span>→</span>
          </a>
        </div>
        </div>
      </div>
    </>
  );
}
