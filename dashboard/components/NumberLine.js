/**
 * One pill input as a ruler: coloured bands = what the rule does in each range
 * (its own thresholds, from pillFactors' row.line), dot = today.
 * line = { v, r: [lo, hi], zones: [[from, to, tone, label]], note? } (null edge = ruler end).
 */
const W = 320, H = 64, L = 8, R = 8, Y = 24, BH = 20;

const fmt = (v) => (Math.abs(v) < 10 && v % 1 ? v.toFixed(1) : String(Math.round(v))).replace('-', '−');

export default function NumberLine({ line }) {
    const { v, zones } = line;
    const has = v != null;
    const [lo0, hi0] = line.r, pad = (hi0 - lo0) * 0.06;
    // stretch the ruler so today's dot always sits on it
    const lo = has ? Math.min(lo0, v - pad) : lo0, hi = has ? Math.max(hi0, v + pad) : hi0;
    const sx = (x) => L + ((x - lo) / (hi - lo)) * (W - L - R);
    const ticks = [...new Set(zones.flatMap(z => [z[0], z[1]]).filter(x => x != null))];
    const at = has ? zones.find(([a, b]) => (a == null || v >= a) && (b == null || v < b)) : null;

    return (
        <svg className="num-line" viewBox={`0 0 ${W} ${H}`} role="img"
            aria-label={has ? `Today ${fmt(v)}: ${at ? at[3] : ''}` : 'No data today'}>
            {zones.map(([a, b, tone, label], i) => {
                const x0 = sx(a ?? lo), x1 = sx(b ?? hi), half = label.length * 3.3, mid = (x0 + x1) / 2;
                // the label steps aside when today's marker would cross it
                let lx = mid;
                if (has && Math.abs(sx(v) - mid) < half + 6) {
                    lx = sx(v) < mid ? sx(v) + half + 6 : sx(v) - half - 6;
                    lx = Math.min(Math.max(lx, x0 + half + 2), x1 - half - 2);
                }
                return (
                    <g key={i} className={`nl-zone nl-${tone}`}>
                        <rect x={x0} y={Y} width={x1 - x0} height={BH} rx="3" />
                        <text x={lx} y={Y + BH / 2 + 4} textAnchor="middle">{label}</text>
                    </g>
                );
            })}
            {ticks.map(t => <text key={t} className="sm-tick" x={sx(t)} y={Y + BH + 13} textAnchor="middle">{fmt(t)}</text>)}
            {has && (
                <g className={`nl-dot nl-${at ? at[2] : 'gray'}`}>
                    <line x1={sx(v)} x2={sx(v)} y1={Y - 6} y2={Y + BH + 2} />
                    <text x={Math.min(Math.max(sx(v), 30), W - 30)} y={Y - 9} textAnchor="middle">today {(Number.isInteger(v) || Math.abs(v) >= 10 ? fmt(v) : v.toFixed(2)).replace('-', '−')}</text>
                </g>
            )}
        </svg>
    );
}
