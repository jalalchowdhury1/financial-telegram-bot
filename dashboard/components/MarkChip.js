'use client';
/**
 * The one place the marks announce themselves globally: a small chip beside the
 * "Updated …" badge. It renders ONLY when something is actually lit, so a quiet
 * day — 52% of them — adds nothing to the page at all.
 *
 * Tapping it (or the glance bar's ● N) opens the ● sheet (MarksSheet.js): every lit number,
 * biggest change first, each row a jump to that number.
 */
import { useMarkCounts } from './MarkProvider';

// The chip and the glance bar's dot open the ONE sheet page.js mounts.
const openers = new Set();
export function openMarksSheet() { openers.forEach((fn) => fn()); }
export function onOpenMarks(fn) { openers.add(fn); return () => openers.delete(fn); }

/** "3 new prints · 1 outsized move" — shared by the chip and the glance bar. */
export function markSummary({ print, move }) {
    const parts = [];
    if (print) parts.push(`${print} new print${print === 1 ? '' : 's'}`);
    if (move) parts.push(`${move} outsized move${move === 1 ? '' : 's'}`);
    return parts.join(' · ');
}

export default function MarkChip({ values }) {
    const counts = useMarkCounts(values);
    if (!counts.total) return null;

    return (
        <button type="button" className="mark-chip" onClick={openMarksSheet}
            title="See every number that changed">
            <span className="mark-chip-dot" />
            {markSummary(counts)}
        </button>
    );
}
