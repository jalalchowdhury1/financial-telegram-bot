/** @type {import('next').NextConfig} */
const nextConfig = {
    env: {
        // Inlined at build time (Vercel builds on every deploy), so the footer's
        // "Deployed:" line shows when the site actually shipped — it used to
        // render new Date() and claim it deployed the moment you loaded the page.
        NEXT_PUBLIC_BUILD_TIME: new Date().toISOString(),
    },
};

module.exports = nextConfig;
