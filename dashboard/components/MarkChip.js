'use client';
/**
 * The one place the marks announce themselves globally: a small chip beside the
 * "Updated …" badge. It renders ONLY when something is actually lit, so a quiet
 * day — 52% of them — adds nothing to the page at all.
 *
 * Clicking it walks through the marked numbers on the page, one per click.
 */
import { useMarkCounts } from './MarkProvider';

// One shared cursor: the header chip and the glance bar's dot walk the SAME list, so
// tapping either one always goes to the next mark, never back to one already seen.
let cursor = 0;
export function jumpToNextMark() {
    const marks = document.querySelectorAll('[data-mark]');
    if (!marks.length) return null;
    const el = marks[cursor % marks.length];
    cursor += 1;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return el;
}
export function resetMarkCursor() { cursor = 0; }

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
        <button type="button" className="mark-chip" onClick={jumpToNextMark}
            title="Jump to the next number that changed">
            <span className="mark-chip-dot" />
            {markSummary(counts)}
        </button>
    );
}
