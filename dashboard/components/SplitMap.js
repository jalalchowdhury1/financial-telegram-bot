/**
 * Conflict pair picture: two numbers as one dot on a map; the red boxes are where
 * the pair "splits" (the rule's own thresholds, from pillFactors' row.plot).
 * plot = { x, y, xName, yName, xr: [lo, hi], yr: [lo, hi], zones: [[x0, x1, y0, y1]] }
 * (a null zone edge runs to the map edge).
 */
const W = 320, H = 170, L = 34, R = 10, T = 22, B = 34;

const fmt = (v) => (Math.abs(v) < 10 && v % 1 ? v.toFixed(1) : String(Math.round(v))).replace('-', '−');

export default function SplitMap({ plot, hit }) {
    const { x, y, xName, yName, zones } = plot;
    const has = x != null && y != null;
    // stretch the map so today's dot always sits inside it
    const fit = ([lo, hi], v) => {
        const pad = (hi - lo) * 0.08;
        return has ? [Math.min(lo, v - pad), Math.max(hi, v + pad)] : [lo, hi];
    };
    const xr = fit(plot.xr, x), yr = fit(plot.yr, y);
    const sx = (v) => L + ((v - xr[0]) / (xr[1] - xr[0])) * (W - L - R);
    const sy = (v) => H - B - ((v - yr[0]) / (yr[1] - yr[0])) * (H - T - B);
    const xTicks = [...new Set([xr[0], xr[1], ...(xr[0] < 0 && xr[1] > 0 ? [0] : []), ...zones.flatMap(z => [z[0], z[1]]).filter(v => v != null)])];
    const yTicks = [...new Set([yr[0], yr[1], ...(yr[0] < 0 && yr[1] > 0 ? [0] : []), ...zones.flatMap(z => [z[2], z[3]]).filter(v => v != null)])];

    return (
        <svg className="split-map" viewBox={`0 0 ${W} ${H}`} role="img"
            aria-label={`${xName} ${has ? fmt(x) : 'n/a'}, ${yName} ${has ? fmt(y) : 'n/a'}: ${hit ? 'in the split zone' : 'outside the split zone'}`}>
            <rect className="sm-bg" x={L} y={T} width={W - L - R} height={H - T - B} />
            {zones.map(([x0, x1, y0, y1], i) => {
                const a = sx(x0 ?? xr[0]), b = sx(x1 ?? xr[1]), c = sy(y1 ?? yr[1]), d = sy(y0 ?? yr[0]);
                return (
                    <g key={i}>
                        <rect className="sm-zone" x={a} y={c} width={b - a} height={d - c} />
                        <text className="sm-zone-lbl" x={(a + b) / 2} y={(c + d) / 2 + 3} textAnchor="middle">split</text>
                    </g>
                );
            })}
            {xr[0] < 0 && xr[1] > 0 && <line className="sm-zero" x1={sx(0)} x2={sx(0)} y1={T} y2={H - B} />}
            {yr[0] < 0 && yr[1] > 0 && <line className="sm-zero" x1={L} x2={W - R} y1={sy(0)} y2={sy(0)} />}
            {xTicks.map(v => <text key={`x${v}`} className="sm-tick" x={sx(v)} y={H - B + 11} textAnchor={v === xr[0] ? 'start' : 'middle'}>{fmt(v)}</text>)}
            {yTicks.map(v => <text key={`y${v}`} className="sm-tick" x={L - 4} y={sy(v) + 3} textAnchor="end">{fmt(v)}</text>)}
            <text className="sm-axis" x={W - R} y={H - 4} textAnchor="end">{xName}{has ? ` · today ${fmt(x)}` : ''} →</text>
            <text className="sm-axis" x={L} y={T - 8}>↑ {yName}{has ? ` · today ${fmt(y)}` : ''}</text>
            {has && (
                <g className={hit ? 'sm-dot sm-dot-hit' : 'sm-dot'}>
                    <circle cx={sx(x)} cy={sy(y)} r="5" />
                    <text x={sx(x) + (sx(x) > W - 80 ? -9 : 9)} y={sy(y) + 4} textAnchor={sx(x) > W - 80 ? 'end' : 'start'}>today</text>
                </g>
            )}
        </svg>
    );
}
