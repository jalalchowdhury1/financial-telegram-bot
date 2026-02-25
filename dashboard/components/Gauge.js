export default function Gauge({ score, segments, size = 240, labels }) {
    const cx = size / 2;
    const cy = size / 2 + 4;
    const outerR = size / 2 - 18;
    const thickness = outerR * 0.22;
    const midR = outerR - thickness / 2;     // arc centerline
    const clamp = Math.min(Math.max(score, 0), 100);
    const needleAngle = Math.PI - (clamp / 100) * Math.PI;
    const viewH = size / 2 + 16;

    // Helper: arc path from angle a1 to a2 at radius r
    function arcPath(a1Deg, a2Deg, r) {
        const a1 = (Math.PI / 180) * a1Deg;
        const a2 = (Math.PI / 180) * a2Deg;
        const x1 = cx + r * Math.cos(a1);
        const y1 = cy - r * Math.sin(a1);
        const x2 = cx + r * Math.cos(a2);
        const y2 = cy - r * Math.sin(a2);
        const sweep = a2Deg - a1Deg > 180 ? 1 : 0;
        return `M${x1},${y1} A${r},${r} 0 ${sweep} 0 ${x2},${y2}`;
    }

    return (
        <svg viewBox={`0 0 ${size} ${viewH}`} style={{ width: '100%', maxWidth: `${size}px`, display: 'block', margin: '0 auto' }}>
            {/* Background track */}
            <path
                d={arcPath(0, 180, midR)}
                fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={thickness}
                strokeLinecap="butt"
            />
            {/* Colored segments */}
            {segments.map(({ start, end, color }, i) => {
                // Convert 0-100 score range to 0-180 degrees
                const startAngle = (start / 100) * 180;
                const endAngle = (end / 100) * 180;
                return (
                    <path
                        key={i}
                        d={arcPath(startAngle, endAngle, midR)}
                        fill="none" stroke={color} strokeWidth={thickness}
                        strokeLinecap="butt"
                    />
                );
            })}
            {/* Tick labels */}
            {labels && labels.map((val, i) => {
                const theta = Math.PI - (val / 100) * Math.PI;
                return (
                    <text
                        key={i}
                        x={cx + (outerR + 10) * Math.cos(theta)}
                        y={cy - (outerR + 10) * Math.sin(theta)}
                        fill="rgba(255,255,255,0.35)" fontSize="9" fontWeight="600"
                        textAnchor="middle" dominantBaseline="middle"
                    >{val}</text>
                );
            })}
            {/* Needle */}
            <line
                x1={cx} y1={cy}
                x2={cx + (outerR - 4) * Math.cos(needleAngle)}
                y2={cy - (outerR - 4) * Math.sin(needleAngle)}
                stroke="#f1f5f9" strokeWidth="2.5" strokeLinecap="round"
            />
            <circle cx={cx} cy={cy} r="5" fill="#f1f5f9" />
            <circle cx={cx} cy={cy} r="2" fill="var(--bg-primary, #0a0e17)" />
        </svg>
    );
}
