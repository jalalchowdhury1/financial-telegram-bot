/**
 * OddsSpark — a tiny line chart of a market's odds over time (Polymarket card + sheet).
 * Green when the odds ended higher, red when lower, grey when flat. Draws nothing
 * without two real numbers, so a missing or broken history can never paint NaN paths.
 */
const finite = (v) => typeof v === 'number' && Number.isFinite(v);

export default function OddsSpark({ points, className = 'pm-spark', label }) {
  const p = (Array.isArray(points) ? points : []).filter(finite);
  if (p.length < 2) return null;
  const W = 100, H = 24, PAD = 2;
  const lo = Math.min(...p), hi = Math.max(...p);
  const y = (v) => (hi === lo ? H / 2 : PAD + (1 - (v - lo) / (hi - lo)) * (H - PAD * 2));
  const xy = p.map((v, i) => `${((i / (p.length - 1)) * W).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
  const move = p[p.length - 1] - p[0];
  const trend = Math.abs(move) < 0.005 ? 'flat' : move > 0 ? 'up' : 'down';   // < half a point = flat
  const stroke = { up: 'var(--green)', down: 'var(--red)', flat: 'var(--text-muted)' }[trend];
  return (
    <svg className={className} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
      role="img" aria-label={label} data-trend={trend}>
      <polyline points={xy} fill="none" stroke={stroke} strokeWidth="1.5"
        vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}
