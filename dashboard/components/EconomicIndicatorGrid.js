'use client';
import ErrorBoundary from './ErrorBoundary';
import Skeleton from './Skeleton';

export default function EconomicIndicatorGrid({ fred, loading, statusColor }) {
    return (
        <div className="card" style={{ animationDelay: '0.5s' }}>
            <div className="card-header">
                <h2>📊 Economic Indicators</h2>
            </div>
            <ErrorBoundary>
                {loading || !fred || fred.error ? <Skeleton count={6} /> : (
                    <>
                        {[
                            { icon: fred.indicators.sahmRule?.status === 'safe' ? '✅' : '🔴', label: 'Sahm Rule', tooltip: "Recession indicator: triggers if 3mo average unemployment rises 0.5% above its 12mo low.", value: fred.indicators.sahmRule?.value?.toFixed(2) ?? 'N/A', status: fred.indicators.sahmRule?.status, benchmark: '< 0.50' },
                            { icon: '🛒', label: 'Consumer Sentiment', tooltip: "University of Michigan survey assessing consumer confidence.", value: fred.indicators.sentiment?.value?.toFixed(1) ?? 'N/A', status: fred.indicators.sentiment?.status, benchmark: '> 80 strong' },
                            { icon: '📋', label: 'Initial Claims (4wk)', tooltip: "4-week moving average of initial jobless claims. A critical real-time labor market gauge.", value: fred.indicators.claims?.value ? `${fred.indicators.claims.value.toFixed(0)}K` : 'N/A', status: fred.indicators.claims?.status, benchmark: '< 250K healthy' },
                            { icon: '🏦', label: 'BBB Credit Spread', tooltip: "The premium corporations pay over Treasuries to borrow. Widening indicates market stress.", value: fred.indicators.creditSpread?.value ? `${fred.indicators.creditSpread.value.toFixed(2)}%` : 'N/A', status: fred.indicators.creditSpread?.status, benchmark: '< 1.5% tight' },
                            { icon: '💵', label: 'Real Yields (10Y TIPS)', tooltip: "10-Year Treasury Inflation-Indexed Security. Shows the true inflation-adjusted cost of capital.", value: fred.indicators.realYields?.value ? `${fred.indicators.realYields.value.toFixed(2)}%` : 'N/A', status: fred.indicators.realYields?.status, benchmark: '< 0% easy' },
                            { icon: '📊', label: 'Leading Economic Index', tooltip: "The Conference Board LEI tracks 10 forward-looking economic components.", value: fred.indicators.lei?.change ? `${fred.indicators.lei.change >= 0 ? '+' : ''}${fred.indicators.lei.change.toFixed(2)}%` : 'N/A', status: fred.indicators.lei?.status, benchmark: '> 0% rising' },
                            { icon: '💎', label: 'Market Valuation', tooltip: "Current S&P 500 P/E Ratio. A measure of how expensive the market is historically.", value: fred.peRatio ? `P/E ~${fred.peRatio.toFixed(1)}` : 'P/E N/A', status: fred.peRatio > 25 ? 'restrictive' : 'neutral', benchmark: 'Fair at ~20' },
                        ].map(ind => (
                            <div className="stat-row" key={ind.label}>
                                <span className="stat-label">
                                    {ind.icon} <span className="tooltip-trigger" data-tooltip={ind.tooltip}>{ind.label}</span>
                                </span>
                                <span className="stat-right">
                                    <span className={`stat-value ${statusColor(ind.status)}`}>{ind.value}</span>
                                    <span className="stat-benchmark">{ind.benchmark}</span>
                                </span>
                            </div>
                        ))}
                    </>
                )}
            </ErrorBoundary>
        </div>
    );
}
