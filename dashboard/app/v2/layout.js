import { Geist } from 'next/font/google';
import './v2.css';

// Geist for words and hero numbers; tables keep JetBrains Mono (already loaded by the
// root layout, and every reused v1 card is set in it), so v2 adds one font, not two.
const sans = Geist({ subsets: ['latin'], variable: '--v2-sans', display: 'swap' });

export const metadata = {
    title: 'Markets · v2',
    description: "Jalal's market dashboard, second edition: same live feeds, calmer layout.",
};

export default function V2Layout({ children }) {
    return <div className={sans.variable}>{children}</div>;
}
