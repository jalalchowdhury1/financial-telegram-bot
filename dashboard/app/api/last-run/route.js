import fs from 'fs';

export const dynamic = 'force-dynamic';

const CACHE_FILE = '/tmp/last-run-cache.json';
const REPO = 'jalalchowdhury1/financial-telegram-bot';
const WORKFLOW = 'daily_report.yml';

function saveCache(lastRun) {
    try { fs.writeFileSync(CACHE_FILE, JSON.stringify({ lastRun, cachedAt: new Date().toISOString() })); } catch {}
}

function loadCache() {
    try {
        if (fs.existsSync(CACHE_FILE)) return JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
    } catch {}
    return null;
}

async function ghFetch(url) {
    const headers = { 'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28' };
    if (process.env.GITHUB_TOKEN) headers['Authorization'] = `Bearer ${process.env.GITHUB_TOKEN}`;
    const res = await fetch(url, { headers, next: { revalidate: 0 } });
    if (!res.ok) throw new Error(`GitHub returned ${res.status}`);
    return res.json();
}

export async function GET() {
    const messages = [];

    // Layer 1: GitHub Actions — successful runs of this specific workflow
    try {
        const data = await ghFetch(`https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/runs?status=success&per_page=1`);
        if (data?.workflow_runs?.length > 0) {
            const lastRun = data.workflow_runs[0].updated_at;
            saveCache(lastRun);
            return Response.json({ lastRun, _meta: { source: 'GitHub Actions (success)', messages } });
        }
        throw new Error('No successful runs found');
    } catch (e) { messages.push(`Layer 1 failed: ${e.message}`); }

    // Layer 2: GitHub Actions — any run of this workflow (completed, regardless of status)
    try {
        const data = await ghFetch(`https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/runs?per_page=1`);
        if (data?.workflow_runs?.length > 0) {
            const lastRun = data.workflow_runs[0].updated_at;
            saveCache(lastRun);
            return Response.json({ lastRun, _meta: { source: 'GitHub Actions (any status)', messages } });
        }
        throw new Error('No runs found');
    } catch (e) { messages.push(`Layer 2 failed: ${e.message}`); }

    // Layer 3: GitHub Actions — all workflows in the repo (pick most recent completed)
    try {
        const data = await ghFetch(`https://api.github.com/repos/${REPO}/actions/runs?per_page=5`);
        const runs = data?.workflow_runs;
        if (runs?.length > 0) {
            const lastRun = runs[0].updated_at;
            saveCache(lastRun);
            return Response.json({ lastRun, _meta: { source: 'GitHub Actions (all workflows)', messages } });
        }
        throw new Error('No workflow runs found');
    } catch (e) { messages.push(`Layer 3 failed: ${e.message}`); }

    // Layer 4: GitHub repo commits — last commit time as proxy
    try {
        const data = await ghFetch(`https://api.github.com/repos/${REPO}/commits?per_page=1`);
        if (data?.length > 0) {
            const lastRun = data[0].commit?.committer?.date || data[0].commit?.author?.date;
            if (lastRun) {
                saveCache(lastRun);
                return Response.json({ lastRun, _meta: { source: 'GitHub Commits (proxy)', messages } });
            }
        }
        throw new Error('No commit data');
    } catch (e) { messages.push(`Layer 4 failed: ${e.message}`); }

    // Layer 5: /tmp cache + graceful unknown (never returns 500)
    const cached = loadCache();
    return Response.json({
        lastRun: cached?.lastRun || null,
        _meta: {
            source: cached ? `Stale Cache (from ${cached.cachedAt})` : 'Unknown',
            hasErrors: true,
            messages
        }
    });
}
