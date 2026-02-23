// /api/last-run - Fetch the time the telegram bot was last run via github actions
export const dynamic = 'force-dynamic';

export async function GET() {
    try {
        const url = 'https://api.github.com/repos/jalalchowdhury1/financial-telegram-bot/actions/workflows/daily_report.yml/runs?status=success&per_page=1';
        const res = await fetch(url, { next: { revalidate: 0 } });

        if (!res.ok) throw new Error(`GitHub API returned ${res.status}`);

        const data = await res.json();

        let lastRunTime = null;
        if (data && data.workflow_runs && data.workflow_runs.length > 0) {
            lastRunTime = data.workflow_runs[0].updated_at;
        }

        return Response.json({
            lastRun: lastRunTime
        });
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
