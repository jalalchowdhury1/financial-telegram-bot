'use client';
import ErrorBoundary from './ErrorBoundary';
import Skeleton from './Skeleton';

function Sparkline({ data, color }) {
    if (!data || data.length < 2) return null;
    const values = data.map(d => d.value ?? d.price).filter(v => v != null && !isNaN(v));
    if (values.length < 2) return null;
    const min = Math.min(...values), max = Math.max(...values);
    const range = max - min || 1;
    const w = 56, h = 22;
    const pts = values.map((v, i) =>
        `${((i / (values.length - 1)) * w).toFixed(1)},${(h - ((v - min) / range) * h).toFixed(1)}`
    ).join(' ');
    return (
        <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ overflow: 'visible', flexShrink: 0 }}>
            <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
        </svg>
    );
}

function MarketRow({ item }) {
    if (!item?.data) return null;
    const d = item.data;
    const val = d.current;
    const pct = d.dailyChange?.pct ?? 0;
    const hasChange = d.history?.length > 1;
    const isPos = pct >= 0;
    const color = !hasChange ? 'var(--text-muted)' : isPos ? 'var(--green)' : 'var(--red)';
    const sign = isPos ? '+' : '';
    const displayVal = item.format
        ? item.format(val)
        : `${item.prefix ?? ''}${val != null ? val.toFixed(item.decimals ?? 2) : 'N/A'}${item.suffix ?? ''}`;

    return (
        <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.05)'
        }}>
            <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', marginBottom: '1px' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', fontFamily: "'JetBrains Mono', monospace" }}>
                        {item.ticker}
                    </span>
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {item.name}
                    </span>
                </div>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: "'JetBrains Mono', monospace" }}>
                    {displayVal}
                </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '4px', paddingLeft: '10px', flexShrink: 0 }}>
                {hasChange ? (
                    <span style={{ fontSize: '0.68rem', fontWeight: 600, color, fontFamily: "'JetBrains Mono', monospace" }}>
                        {sign}{pct.toFixed(2)}%
                    </span>
                ) : (
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>live</span>
                )}
                <Sparkline data={d.history} color={color} />
            </div>
        </div>
    );
}

function Column({ title, items }) {
    return (
        <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{
                fontSize: '0.62rem', fontWeight: 700, letterSpacing: '0.09em',
                textTransform: 'uppercase', color: 'var(--text-muted)',
                paddingBottom: '8px', marginBottom: '2px',
                borderBottom: '1px solid rgba(255,255,255,0.08)'
            }}>
                {title}
            </div>
            {items.filter(Boolean).map((item, i) => <MarketRow key={i} item={item} />)}
        </div>
    );
}

export default function ExtraMarketsGrid({ data, loading }) {
    if (loading) return (
        <div style={{ gridColumn: '1 / -1' }}>
            <div className="card"><Skeleton count={7} /></div>
        </div>
    );
    if (!data || data.error) return null;

    const { fx, commodities, rates, realEstate } = data;

    // ---- LEFT: Real Estate + Rates (7 items) ----
    const left = [
        realEstate?.rentIndex ? {
            ticker: 'ZRI', name: 'US Median Monthly Rent',
            data: realEstate.rentIndex, prefix: '$', suffix: '', decimals: 0
        } : null,
        realEstate?.mortgagePayment ? {
            ticker: 'MTGPMT', name: 'Estimated Monthly Mortgage',
            data: realEstate.mortgagePayment, prefix: '$', suffix: '', decimals: 0
        } : null,
        rates?.mortgageRate ? {
            ticker: 'MORT30', name: '30-Year Fixed Mortgage Rate',
            data: rates.mortgageRate, prefix: '', suffix: '%', decimals: 2
        } : null,
        rates?.tnx ? {
            ticker: 'TNX', name: '10-Year Treasury Yield',
            data: rates.tnx, prefix: '', suffix: '%', decimals: 2
        } : null,
        rates?.t2y ? {
            ticker: 'T2Y', name: '2-Year Treasury Yield',
            data: rates.t2y, prefix: '', suffix: '%', decimals: 2
        } : null,
        fx?.dxy ? {
            ticker: 'DXY', name: 'US Dollar Index',
            data: fx.dxy, prefix: '', suffix: '', decimals: 2
        } : null,
        commodities?.cl ? {
            ticker: 'CL', name: 'Crude Oil WTI ($/bbl)',
            data: commodities.cl, prefix: '$', suffix: '', decimals: 2
        } : null,
        realEstate?.atnhpi ? {
            ticker: 'ATNHPI', name: 'US House Price Index',
            data: realEstate.atnhpi, prefix: '', suffix: '', decimals: 2
        } : null,
    ].filter(Boolean);

    // ---- RIGHT: FX + Commodities + Crypto (8 items) ----
    const right = [
        fx?.usdbdt ? {
            ticker: 'USD/BDT', name: 'US Dollar to Bangladeshi Taka',
            data: fx.usdbdt, prefix: '', suffix: '', decimals: 2
        } : null,
        fx?.usdcad ? {
            ticker: 'USD/CAD', name: 'US Dollar to Canadian Dollar',
            data: fx.usdcad, prefix: '', suffix: '', decimals: 4
        } : null,
        commodities?.gc ? {
            ticker: 'GOLD', name: 'Gold Spot (USD/oz)',
            data: commodities.gc,
            format: v => `$${v ? v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : 'N/A'}`
        } : null,
        commodities?.btc ? {
            ticker: 'BTC', name: 'Bitcoin / US Dollar',
            data: commodities.btc,
            format: v => `$${v ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : 'N/A'}`
        } : null,
        fx?.cadbdt ? {
            ticker: 'CAD/BDT', name: 'Canadian Dollar to Bangladeshi Taka',
            data: fx.cadbdt, prefix: '', suffix: '', decimals: 2
        } : null,
        fx?.usdinr ? {
            ticker: 'USD/INR', name: 'US Dollar to Indian Rupee',
            data: fx.usdinr, prefix: '', suffix: '', decimals: 2
        } : null,
        fx?.inrbdt ? {
            ticker: 'INR/BDT', name: 'Indian Rupee to Bangladeshi Taka',
            data: fx.inrbdt, prefix: '', suffix: '', decimals: 4
        } : null,
        fx?.cadinr ? {
            ticker: 'CAD/INR', name: 'Canadian Dollar to Indian Rupee',
            data: fx.cadinr, prefix: '', suffix: '', decimals: 2
        } : null,
    ].filter(Boolean);

    return (
        <div style={{ gridColumn: '1 / -1' }}>
            <div className="card" style={{ animationDelay: '0.6s' }}>
                <div className="card-header" style={{ marginBottom: '16px' }}>
                    <h2>🌐 Global Markets</h2>
                    <span className="badge badge-blue">FX · Commodities · Real Estate · Rates</span>
                </div>
                <ErrorBoundary>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 48px' }}>
                        <Column title="🏠 Real Estate & Rates" items={left} />
                        <Column title="💱 FX, Commodities & Crypto" items={right} />
                    </div>
                </ErrorBoundary>
            </div>
        </div>
    );
}
