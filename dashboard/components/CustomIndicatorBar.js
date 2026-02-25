'use client';

export default function CustomIndicatorBar({ sheets, loading }) {
    return (
        <div className="indicator-bar">
            <div className={`indicator-pill${!loading && sheets?.NotSoBoring && sheets.NotSoBoring !== 'ON' ? ' pill-alert' : ''}`}>
                <div className="label">
                    <span className="tooltip-trigger" data-tooltip="Crash Detector: Monitors Tech (QQQ) and Bonds (TMF) for drops >6-7%. Defensive shift adds Gold and USD to dilute risk.">
                        <span className="emoji">🛡️</span>NotSoBoring
                    </span>
                </div>
                <div className="value">{loading ? '...' : (sheets?.NotSoBoring || 'N/A')}</div>
            </div>
            <div className={`indicator-pill${!loading && sheets?.FrontRunner && !sheets.FrontRunner.startsWith('BIL') ? ' pill-alert' : ''}`}>
                <div className="label">
                    <span className="tooltip-trigger" data-tooltip="A contrarian strategy that rotates into Volatility (VIX) hedges when markets overheat (RSI > 79) and buys oversold Tech/Leveraged ETFs during deep dips.">
                        <span className="emoji">🔑</span>FrontRunner
                    </span>
                </div>
                <div className="value">{loading ? '...' : (sheets?.FrontRunner || 'N/A')}</div>
            </div>
            <div className={`indicator-pill${!loading && sheets?.AAIIDiff && parseFloat(sheets.AAIIDiff) > 20 ? ' pill-alert' : ''}`}>
                <div className="label"><span className="emoji">🔸</span>AAII Diff</div>
                {!loading && sheets?.AAIIDiff ? (() => {
                    const val = parseFloat(sheets.AAIIDiff);
                    const isBullish = val > 20;
                    const targetDate = new Date();
                    targetDate.setMonth(targetDate.getMonth() + 6);
                    const dateStr = targetDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
                    return (
                        <>
                            <div className="value" style={{ color: isBullish ? 'var(--green)' : 'var(--text-primary)' }}>
                                {sheets.AAIIDiff}
                            </div>
                            <div className="pill-detail" style={{ fontSize: '0.68rem', marginTop: '4px', color: isBullish ? 'var(--green)' : 'var(--text-muted)', fontWeight: 600 }}>
                                {isBullish ? '🟢' : '⚪'} {isBullish ? 'Bullish' : 'Neutral'} outlook → {dateStr}
                            </div>
                            <div className="pill-detail" style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                                Threshold: &gt;20% = bullish 6mo forward
                            </div>
                        </>
                    );
                })() : <div className="value">{loading ? '...' : 'N/A'}</div>}
            </div>
            <div className={`indicator-pill${!loading && sheets?.VIX?.current && parseFloat(sheets.VIX.current) > parseFloat(sheets.VIX.threeMonth) ? ' pill-alert' : ''}`}>
                <div className="label">
                    <span className="tooltip-trigger" data-tooltip="Oversold Signal: Short-term panic (Current VIX) exceeds medium-term expectations (3M VIX). Often precedes a market recovery.">
                        <span className="emoji">🎢</span>VIX (Current | 3M)
                    </span>
                </div>
                <div className="value">
                    {loading ? '...' : (sheets?.VIX?.current
                        ? `${sheets.VIX.current} | ${sheets.VIX.threeMonth} | ${sheets.VIX.fearGreed}`
                        : 'N/A')}
                </div>
            </div>
        </div>
    );
}
