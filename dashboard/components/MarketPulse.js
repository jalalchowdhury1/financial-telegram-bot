'use client';

export default function MarketPulse({ spy, fg, fred, loading, fgColor }) {
    if (loading || !spy || !fg || spy.error || fg.error) return null;

    return (
        <div className="market-pulse">
            <span className="pulse-label">📡 Market Pulse</span>
            <span className="pulse-items">
                <span className={spy.dailyChange?.pct >= 0 ? 'stat-positive' : 'stat-negative'}>
                    SPY {spy.dailyChange?.pct >= 0 ? '▲' : '▼'}{Math.abs(spy.dailyChange?.pct || 0).toFixed(2)}%
                </span>
                <span className="pulse-sep">·</span>
                <span style={{ color: fgColor(fg.score) }}>
                    F&G {Math.round(fg.score)} {fg.rating}
                </span>
                <span className="pulse-sep">·</span>
                <span className={spy.rsi > 70 ? 'stat-negative' : spy.rsi < 30 ? 'stat-positive' : ''}>
                    RSI {spy.rsi.toFixed(0)}
                </span>
                {fred?.checklist && (() => {
                    const bullishItems = Object.values(fred.checklist).filter(i => i.bullish).length;
                    const totalItems = Object.values(fred.checklist).length;
                    return (
                        <>
                            <span className="pulse-sep">·</span>
                            <span style={{ color: 'var(--green)' }}>Bull {bullishItems}/{totalItems}</span>
                        </>
                    );
                })()}
            </span>
        </div>
    );
}
