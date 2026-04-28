export const dynamic = 'force-dynamic';

export async function GET() {
    const lambdaUrl = process.env.LAMBDA_URL;
    if (!lambdaUrl) return Response.json({ error: 'LAMBDA_URL not configured' }, { status: 500 });
    const res = await fetch(`${lambdaUrl}/api/polymarket`, { cache: 'no-store' });
    return Response.json(await res.json(), { status: res.status });
}
