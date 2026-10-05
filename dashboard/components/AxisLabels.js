/**
 * AxisLabels — chart tick labels as small HTML spans laid over an SVG.
 *
 * The charts draw with preserveAspectRatio="none" (480×180 squeezed into ~336×172 on a
 * phone), so SVG text came out ~4.5px and squashed sideways. These spans sit at the same
 * viewBox point by % (axisPos), so they stay a sharp 10px whatever size the chart is.
 * pointer-events:none (CSS) — taps go straight through to the SVG's tap-to-read.
 *
 * labels: [{ x, y, text, ax: 'start'|'middle'|'end', ay: 'top'|'middle'|'bottom' }]
 *   ax/ay say which edge of the label sits on (x, y).
 */
import { axisPos } from '../lib/chartAxis';

const SHIFT_X = { start: '0', middle: '-50%', end: '-100%' };
const SHIFT_Y = { top: '0', middle: '-50%', bottom: '-100%' };

export default function AxisLabels({ w, h, labels }) {
    if (!labels || !labels.length) return null;
    return (
        <div className="axis-layer" aria-hidden="true">
            {labels.map((l, i) => (
                <span key={`${l.text}-${i}`} className="axis-lbl" style={{
                    ...axisPos(l.x, l.y, w, h),
                    transform: `translate(${SHIFT_X[l.ax] || '-50%'}, ${SHIFT_Y[l.ay] || '-50%'})`,
                }}>{l.text}</span>
            ))}
        </div>
    );
}
