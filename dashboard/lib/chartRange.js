/**
 * 📈 Chart range chips for the tap-a-number popover: 1M · 3M · 6M · 1Y · 5Y · MAX.
 *
 * /api/history `series` carries every day the sheet has (capped at 2 years), and a stat with
 * baked long history (lib/longHistory.js) reaches back decades; the popover slices the
 * joined line here, on the device. The pick is remembered (one choice for every chart on the
 * page, kept per device in localStorage) and 3M is the default, the old fixed 90-day chart.
 *
 * A stat shows only the chips that would show something different (`rangesFor`): a stat with
 * 7 months of history gets 1M · 3M · 6M · MAX, and a remembered 5Y falls back to its MAX.
 * The eyebrow never lies: when a stat has less history than the chip asks for, it says
 * "since Mar 12" instead of "180 days".
 */
export const RANGES = [
    { id: '1M', days: 30, label: '30 days' },
    { id: '3M', days: 90, label: '90 days' },
    { id: '6M', days: 180, label: '180 days' },
    { id: '1Y', days: 365, label: '1 year' },
    { id: '5Y', days: 1826, label: '5 years' },
    { id: 'MAX', days: Infinity },
];
export const DEFAULT_RANGE = '3M';
const KEY = 'ftb:chartRange';
const LEGACY = { ALL: 'MAX' }; // the chip's name before 2026-10-10
const isRange = (id) => RANGES.some((r) => r.id === id);

let current = null;

export function getRange() {
    if (current) return current;
    let saved = null;
    try { saved = window.localStorage.getItem(KEY); } catch { /* private mode / SSR */ }
    saved = LEGACY[saved] || saved;
    current = isRange(saved) ? saved : DEFAULT_RANGE;
    return current;
}

export function setRange(id) {
    if (!isRange(id)) return;
    current = id;
    try { window.localStorage.setItem(KEY, id); } catch { /* still works for this visit */ }
}

/** Test hook: forget the in-memory pick. */
export function resetRange() { current = null; }

const DAY = 86400000;
const t = (iso) => Date.parse(`${iso}T00:00:00Z`);
const minus = (iso, n) => new Date(t(iso) - n * DAY).toISOString().slice(0, 10);

/** The points inside the chip's window, ending at the newest point. */
export function sliceRange(points, id) {
    const pts = points || [];
    const r = RANGES.find((x) => x.id === id) || RANGES.find((x) => x.id === DEFAULT_RANGE);
    if (!pts.length || r.days === Infinity) return pts;
    const from = minus(pts[pts.length - 1].date, r.days - 1);
    return pts.filter((p) => p.date >= from);
}

function fmtSince(iso, lastIso) {
    const d = new Date(`${iso}T00:00:00`);
    if (Number.isNaN(d.getTime())) return iso;
    const long = t(lastIso) - t(iso) > 300 * DAY; // past ~10 months, "Oct 9" alone is ambiguous
    return d.toLocaleDateString('en-US', long ? { month: 'short', day: 'numeric', year: 'numeric' } : { month: 'short', day: 'numeric' });
}

/**
 * The eyebrow words for a sliced chart: "30 days" … "5 years", or "since Mar 12". A window
 * counts as full when its first point sits within a week of where the window starts — or,
 * from 1Y up, within 40 days (old history can be weekly or monthly).
 */
export function rangeLabel(sliced, id) {
    if (!sliced?.length) return '';
    const r = RANGES.find((x) => x.id === id);
    const first = sliced[0].date, last = sliced[sliced.length - 1].date;
    const slack = r && r.days >= 365 ? 40 : 7;
    if (r && r.days !== Infinity && t(first) - t(minus(last, r.days - 1)) <= slack * DAY) return r.label;
    return `since ${fmtSince(first, last)}`;
}

/** The chips worth showing for a stat whose history spans `span` days: each window shorter
 *  than the history, then MAX. */
export function rangesFor(span) {
    return RANGES.filter((r) => r.days === Infinity || r.days < span);
}

/** The remembered pick if this stat offers it, else MAX (the closest to a longer pick). */
export function pickFor(id, available) {
    return available.some((r) => r.id === id) ? id : 'MAX';
}
