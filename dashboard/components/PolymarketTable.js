'use client';

import { useEffect, useState } from 'react';
import Skeleton from './Skeleton';

/**
 * PolymarketTable component - Displays trending Polymarket bets in a table format
 *
 * Note: Uses bet.name as the React key, assuming bet names are unique within the market context.
 * If bet names are not guaranteed unique, replace with a unique bet ID property.
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
      <div className="p-4">
        <Skeleton count={5} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-red-600">
        Data unavailable — {error}. Try refreshing the page.
      </div>
    );
  }

  if (bets.length === 0) {
    return <div className="p-4 text-gray-600">No data available</div>;
  }

  return (
    <section aria-label="Polymarket Trending Bets">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4">Polymarket Trending Bets</h2>
        <table className="w-full">
          <thead>
            <tr className="border-b">
              <th scope="col" className="text-left py-2">Bet Name</th>
              <th scope="col" className="text-center py-2">Current Odds</th>
              <th scope="col" className="text-right py-2">Volume</th>
            </tr>
          </thead>
          <tbody>
            {bets.map((bet) => (
              <tr key={bet.name} className="border-b hover:bg-gray-50">
                <td className="py-3">{bet.name}</td>
                <td className="text-center">{bet.odds}</td>
                <td className="text-right">{bet.volume.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
