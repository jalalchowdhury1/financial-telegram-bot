'use client';

import { useEffect, useState } from 'react';
import Skeleton from './Skeleton';

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
      <div className="full-width">
        <div className="card">
          <Skeleton count={8} />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="full-width">
        <div className="card">
          <div className="error-message" style={{ color: 'var(--red)' }}>
            ⚠️ Data unavailable — {error}. Try refreshing the page.
          </div>
        </div>
      </div>
    );
  }

  if (bets.length === 0) {
    return (
      <div className="full-width">
        <div className="card">
          <div className="error-message">No betting markets available.</div>
        </div>
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

  return (
    <section aria-label="Polymarket Trending Bets" className="full-width">
      <div className="card" style={{ animationDelay: '0.8s' }}>
        <div className="card-header">
          <h2>📊 Polymarket Trending Bets</h2>
          <span className="badge badge-blue">Real-time · Top 10 Markets</span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '0.9rem',
          }}>
            <thead>
              <tr style={{
                borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
                marginBottom: '8px'
              }}>
                <th scope="col" style={{
                  textAlign: 'left',
                  padding: '12px 0',
                  fontWeight: 700,
                  fontSize: '0.72rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  color: 'var(--text-muted)',
                  fontFamily: 'Inter'
                }}>
                  Market
                </th>
                <th scope="col" style={{
                  textAlign: 'center',
                  padding: '12px 0',
                  fontWeight: 700,
                  fontSize: '0.72rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  color: 'var(--text-muted)',
                  fontFamily: 'Inter'
                }}>
                  Odds
                </th>
                <th scope="col" style={{
                  textAlign: 'right',
                  padding: '12px 0',
                  fontWeight: 700,
                  fontSize: '0.72rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  color: 'var(--text-muted)',
                  fontFamily: 'Inter'
                }}>
                  Volume
                </th>
              </tr>
            </thead>
            <tbody>
              {bets.map((bet, idx) => (
                <tr
                  key={bet.name}
                  style={{
                    borderBottom: '1px solid rgba(255, 255, 255, 0.03)',
                    transition: 'background 0.2s ease',
                    animationDelay: `${0.8 + idx * 0.05}s`,
                    animation: 'fadeInUp 0.6s ease forwards',
                    opacity: 0
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.02)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                  }}
                >
                  {/* Bet Name Column */}
                  <td style={{
                    padding: '14px 0',
                    color: 'var(--text-secondary)',
                    fontWeight: 500,
                    fontSize: '0.9rem',
                    maxWidth: '400px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}>
                    {bet.name}
                  </td>

                  {/* Odds Column with Visualization */}
                  <td style={{
                    padding: '14px 0',
                    textAlign: 'center',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                    minHeight: '50px'
                  }}>
                    {/* Odds Bar */}
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}>
                      {/* Probability bar background */}
                      <div style={{
                        width: '40px',
                        height: '4px',
                        background: 'rgba(255, 255, 255, 0.08)',
                        borderRadius: '2px',
                        overflow: 'hidden',
                        position: 'relative'
                      }}>
                        {/* Filled portion */}
                        <div style={{
                          width: `${bet.odds * 100}%`,
                          height: '100%',
                          background: getOddsColor(bet.odds),
                          borderRadius: '2px',
                          transition: 'width 0.3s ease',
                          boxShadow: `0 0 8px ${getOddsColor(bet.odds)}40`
                        }} />
                      </div>
                      {/* Odds percentage */}
                      <span style={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontWeight: 700,
                        fontSize: '0.9rem',
                        color: getOddsColor(bet.odds),
                        minWidth: '35px',
                        textAlign: 'right'
                      }}>
                        {(bet.odds * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>

                  {/* Volume Column */}
                  <td style={{
                    padding: '14px 0',
                    textAlign: 'right',
                    fontFamily: "'JetBrains Mono', monospace",
                    fontWeight: 600,
                    fontSize: '0.85rem',
                    color: 'var(--text-primary)'
                  }}>
                    {bet.volume.toLocaleString(undefined, {
                      minimumFractionDigits: 0,
                      maximumFractionDigits: 2
                    })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
    </section>
  );
}
