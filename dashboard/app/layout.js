import './globals.css';

export const metadata = {
    title: 'Financial Dashboard — Live Market Intelligence',
    description: 'Premium live financial dashboard with SPY analytics, Fear & Greed Index, economic indicators, and AI-powered market assessment.',
};

// The page paints with system fonts at once (display=swap) and switches when these
// arrive. As <link>s in <head> the browser finds them immediately — they used to be
// an @import inside globals.css, found only after that file downloaded. The
// preconnects open both font hosts in parallel with everything else.
const FONTS_CSS = 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap';

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <head>
                <link rel="preconnect" href="https://fonts.googleapis.com" />
                <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
                <link rel="stylesheet" href={FONTS_CSS} />
            </head>
            <body>{children}</body>
        </html>
    );
}
