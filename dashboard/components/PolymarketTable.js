'use client';

import { useEffect, useState } from 'react';
import Skeleton from './Skeleton';
import MarketModal from './MarketModal';

/**
 * PolymarketTable component - Displays trending Polymarket bets in a premium dark theme
 *
 * Features:
 * - Matches dashboard design system (glassmorphism, dark theme)
 * - Color-coded odds visualization (probability bars)
 * - Monospace font for numeric values
 * - Animated entrance and hover states
 * - Loading skeletons and error handling
 */
export default function PolymarketTable() {
  const [bets, setBets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedBet, setSelectedBet] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    const fetchPolymarketData = async () => {
      try {
        const response = await fetch('/api/polymarket');
        const data = await response.json();

        if (!response.ok || data.error) {
          setError(data.error || 'Failed to fetch Polymarket data');
          setBets([]);
        } else {
          setBets(data.bets || []);
          setError(null);
        }
      } catch (err) {
        setError('Failed to fetch Polymarket data');
        setBets([]);
      } finally {
        setLoading(false);
      }
    };

    fetchPolymarketData();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <Skeleton count={8} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="error-message" style={{ color: 'var(--red)' }}>
          ⚠️ Data unavailable — {error}. Try refreshing the page.
        </div>
      </div>
    );
  }

  if (bets.length === 0) {
    return (
      <div className="card">
        <div className="error-message">No betting markets available.</div>
      </div>
    );
  }

  // Color gradient for odds visualization
  const getOddsColor = (odds) => {
    if (odds < 0.2) return 'var(--red)';
    if (odds < 0.4) return '#f97316';
    if (odds < 0.6) return 'var(--yellow)';
    if (odds < 0.8) return '#22c55e';
    return 'var(--green)';
  };

  // Handle row click - open modal with selected bet
  const handleRowClick = (bet) => {
    setSelectedBet(bet);
    setIsModalOpen(true);
  };

  // Handle modal close
  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  return (
    <section aria-label="Polymarket Trending Bets">
      <div className="card" style={{ animationDelay: '0.8s' }}>
        <div className="card-header">
          <h2>📊 Polymarket Trending Bets</h2>
          <span className="badge badge-blue">Real-time · Top 10 Markets</span>
        </div>

        {/* Compact mobile-friendly list - matches ExtraMarketsGrid density */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {bets.slice(0, 8).map((bet, idx) => (
            <div
              key={bet.name}
              onClick={() => handleRowClick(bet)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 0',
                borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                transition: 'background 0.2s ease',
                animationDelay: `${0.8 + idx * 0.04}s`,
                animation: 'fadeInUp 0.6s ease forwards',
                opacity: 0,
                cursor: 'pointer',
                fontSize: '0.8rem'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
              }}
            >
              {/* Left: Name + Bar */}
              <div style={{ flex: 1, minWidth: 0, marginRight: '8px' }}>
                {/* Name */}
                <div style={{
                  color: 'var(--text-secondary)',
                  fontWeight: 500,
                  fontSize: '0.75rem',
                  marginBottom: '2px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}>
                  {bet.name.length > 45 ? bet.name.substring(0, 42) + '...' : bet.name}
                </div>

                {/* Odds Bar */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}>
                  <div style={{
                    flex: 1,
                    height: '2px',
                    background: 'rgba(255, 255, 255, 0.08)',
                    borderRadius: '1px',
                    overflow: 'hidden',
                    minWidth: '25px'
                  }}>
                    <div style={{
                      width: `${bet.odds * 100}%`,
                      height: '100%',
                      background: getOddsColor(bet.odds),
                      boxShadow: `0 0 4px ${getOddsColor(bet.odds)}40`
                    }} />
                  </div>
                  <span style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontWeight: 700,
                    fontSize: '0.68rem',
                    color: getOddsColor(bet.odds),
                    whiteSpace: 'nowrap',
                    minWidth: '22px',
                    textAlign: 'right'
                  }}>
                    {(bet.odds * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Right: Volume */}
              <div style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontWeight: 600,
                fontSize: '0.7rem',
                color: 'var(--text-primary)',
                textAlign: 'right',
                whiteSpace: 'nowrap',
                minWidth: '55px'
              }}>
                {bet.volume > 1000 ? (bet.volume / 1000).toFixed(0) + 'k' : bet.volume.toFixed(0)}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div style={{
          marginTop: '16px',
          paddingTop: '12px',
          borderTop: '1px solid rgba(255, 255, 255, 0.04)',
          fontSize: '0.65rem',
          color: 'var(--text-muted)',
          textAlign: 'center'
        }}>
          Data refreshes every 5 minutes · Powered by Polymarket API
        </div>
      </div>

      {/* Market Modal */}
      <MarketModal
        bet={selectedBet}
        isOpen={isModalOpen}
        onClose={handleCloseModal}
      />
    </section>
  );
}
