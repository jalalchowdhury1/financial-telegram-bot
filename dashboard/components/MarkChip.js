'use client';
/**
 * The one place the marks announce themselves globally: a small chip beside the
 * "Updated …" badge. It renders ONLY when something is actually lit, so a quiet
 * day — 52% of them — adds nothing to the page at all.
 *
 * Clicking it walks through the marked numbers on the page, one per click.
 */
import { useRef } from 'react';
import { useMarkCounts } from './MarkProvider';

export default function MarkChip({ values }) {
    const { print, move, total } = useMarkCounts(values);
    const idx = useRef(0);

    if (!total) return null;

    const parts = [];
    if (print) parts.push(`${print} new print${print === 1 ? '' : 's'}`);
    if (move) parts.push(`${move} outsized move${move === 1 ? '' : 's'}`);

    const jump = () => {
        const marks = document.querySelectorAll('[data-mark]');
        if (!marks.length) return;
        const el = marks[idx.current % marks.length];
        idx.current += 1;
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };

    return (
        <button type="button" className="mark-chip" onClick={jump}
            title="Jump to the next number that changed">
            <span className="mark-chip-dot" />
            {parts.join(' · ')}
        </button>
    );
}
