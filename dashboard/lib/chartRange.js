/**
 * 📈 Chart range chips for the tap-a-number popover: 1M · 3M · 6M · ALL.
 *
 * /api/history `series` carries every day the sheet has (capped at 2 years); the popover
 * slices it here, on the device — switching chips never fetches. The pick is remembered
 * (one choice for every chart on the page, kept per device in localStorage) and 3M is the
 * default, which is exactly the old fixed 90-day chart.
 *
 * The eyebrow never lies: when a metric has less history than the chip asks for, it says
 * "since Mar 12" instead of "180 days".
 */
export const RANGES = [
    { id: '1M', days: 30 },
    { id: '3M', days: 90 },
    { id: '6M', days: 180 },
    { id: 'ALL', days: Infinity },
];
export const DEFAULT_RANGE = '3M';
const KEY = 'ftb:chartRange';
const isRange = (id) => RANGES.some((r) => r.id === id);

let current = null;

export function getRange() {
    if (current) return current;
    let saved = null;
    try { saved = window.localStorage.getItem(KEY); } catch { /* private mode / SSR */ }
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

/** The eyebrow words for a sliced chart: "30 days", "90 days", "180 days" or "since Mar 12". */
export function rangeLabel(sliced, id) {
    if (!sliced?.length) return '';
    const r = RANGES.find((x) => x.id === id);
    const first = sliced[0].date, last = sliced[sliced.length - 1].date;
    // the window is "full" when its first point sits within a week of where the window starts
    if (r && r.days !== Infinity && t(first) - t(minus(last, r.days - 1)) <= 7 * DAY) return `${r.days} days`;
    return `since ${fmtSince(first, last)}`;
}
