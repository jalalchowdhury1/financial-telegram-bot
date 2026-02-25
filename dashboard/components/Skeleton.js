export default function Skeleton({ type = 'text', count = 3 }) {
    if (type === 'gauge') return <div className="skeleton skeleton-gauge" />;
    return (
        <div>
            {Array.from({ length: count }).map((_, i) => (
                <div key={i} className={`skeleton skeleton-text ${i === count - 1 ? 'short' : i % 2 ? 'medium' : ''}`} />
            ))}
        </div>
    );
}
