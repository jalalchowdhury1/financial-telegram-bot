'use client';

/**
 * MarketModal Component - Displays full market details in a modal dialog
 *
 * Features:
 * - Glassmorphism card styling matching dashboard design system
 * - Displays full market question without truncation
 * - Color-coded probability bar with dynamic width
 * - Formatted trading volume display
 * - Link to Polymarket.com
 * - Dismissible via close button or backdrop click
 * - Responsive and accessible
 */

export default function MarketModal({ bet, isOpen, onClose }) {
  // Early return if modal is closed
  if (!isOpen || !bet) {
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

  // Link to Polymarket homepage (specific market URLs may not work reliably)
  const polymarketUrl = 'https://polymarket.com';

  // Use 'odds' from API (not 'probability')
  const odds = bet.odds || 0;
  const oddsPercent = (odds * 100).toFixed(0);
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
          padding: '20px'
        }}
        onClick={onClose}
        aria-label="Close modal"
      >
        {/* Modal card - prevent click from propagating to backdrop */}
        <div
          style={{
            width: '100%',
            maxWidth: '520px',
            maxHeight: '90vh',
            overflowY: 'auto',
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

          {/* Probability section */}
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
                {oddsPercent}%
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
