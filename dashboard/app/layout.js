import './globals.css';

export const metadata = {
    title: 'Financial Dashboard — Live Market Intelligence',
    description: 'Premium live financial dashboard with SPY analytics, Fear & Greed Index, economic indicators, and AI-powered market assessment.',
};

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    );
}
