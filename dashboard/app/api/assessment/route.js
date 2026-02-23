// /api/assessment - Generate AI market assessment
export async function POST(request) {
    try {
        const data = await request.json();

        const groqKey = process.env.GROQ_API_KEY;
        const openaiKey = process.env.OPENAI_API_KEY;

        if (!groqKey && !openaiKey) {
            return Response.json({ assessment: generateRuleBased(data) });
        }

        const yieldStatus = data.yieldCurve < 0 ? 'Inverted (recession signal)' : 'Positive (no inversion)';
        const sahmStatus = data.sahmRule >= 0.5 ? 'RECESSION SIGNAL' : 'Safe';
        const fearStatus = data.fearGreed < 25 ? 'Extreme Fear' : data.fearGreed < 45 ? 'Fear' : data.fearGreed < 55 ? 'Neutral' : data.fearGreed < 75 ? 'Greed' : 'Extreme Greed';

        const prompt = `You are a senior quantitative macroeconomic analyst preparing an executive summary for institutional investors.
Analyze the following comprehensive economic indicators to provide a nuanced, in-depth macroeconomic assessment.

📊 COMPLETE MARKET DATA:

**MACRO & SENTIMENT:**
- Yield Curve (10Y-2Y): ${data.yieldCurve?.toFixed(2)}% → ${yieldStatus}
- Fear & Greed Index: ${data.fearGreed?.toFixed(1)}/100 → ${fearStatus}

**LABOR & RECESSION SIGNALING:**
- Sahm Rule Recession Indicator: ${data.sahmRule?.toFixed(2)} → ${sahmStatus}
- Initial Jobless Claims (4wk avg): ${data.claims?.toFixed(0)}K

**CONSUMER HEALTH & FORWARD PATH:**
- Consumer Sentiment Index: ${data.sentiment?.toFixed(1)}/100
- Leading Economic Index: ${data.leiChange?.toFixed(2)}% change

**CREDIT & MONETARY TIGHTNESS:**
- BBB Credit Spread: ${data.creditSpread?.toFixed(2)}%
- Real Yields (10Y TIPS): ${data.realYields?.toFixed(2)}%

TASK: Synthesize an intelligent, highly nuanced assessment of the current structural regime. Limit your response to 4-6 sentences.

STYLE & SUBSTANCE REQUIREMENTS:
- Read as an advanced, institutional macro narrative. Retain a clear and readable tone.
- Synthesize conflicting data intelligently.
- Conclude with a definitive macro verdict specifying the exact regime and your stance (e.g., BULLISH, BEARISH, CAUTIOUS).
- Use appropriate emojis to format your response cleanly (📈, 🔴, ⚠️, 🎯, 💪) but don't overdo it.
- Mention the most critical data values.

Output the assessment now:`;

        let assessment = '';

        if (groqKey) {
            const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${groqKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'llama-3.3-70b-versatile',
                    messages: [{ role: 'user', content: prompt }],
                    temperature: 0.7,
                    max_tokens: 350
                })
            });
            const result = await res.json();
            if (!res.ok) {
                console.error('Groq API Error:', result);
                assessment = `⚠️ Groq API Error: ${result.error?.message || res.statusText}`;
            } else {
                assessment = result.choices?.[0]?.message?.content?.trim() || 'Assessment unavailable';
            }
        } else if (openaiKey) {
            const res = await fetch('https://api.openai.com/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${openaiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'gpt-4o',
                    messages: [{ role: 'user', content: prompt }],
                    temperature: 0.7,
                    max_tokens: 350
                })
            });
            const result = await res.json();
            if (!res.ok) {
                console.error('OpenAI API Error:', result);
                assessment = `⚠️ OpenAI API Error: ${result.error?.message || res.statusText}`;
            } else {
                assessment = result.choices?.[0]?.message?.content?.trim() || 'Assessment unavailable';
            }
        }

        return Response.json({ assessment });
    } catch (error) {
        return Response.json({ assessment: `⚠️ AI assessment unavailable: ${error.message}` });
    }
}

function generateRuleBased(data) {
    const signals = [];
    if (data.yieldCurve > 0) signals.push('📈 Yield curve positive');
    else signals.push('🔴 Yield curve inverted');
    if (data.sahmRule < 0.5) signals.push('✅ Sahm Rule safe');
    else signals.push('🔴 Sahm Rule triggered');
    if (data.fearGreed > 50) signals.push('📈 Market sentiment positive');
    else signals.push('⚠️ Market fear detected');
    return signals.join(' | ') + '\n\n💡 Rule-based assessment (no AI key configured)';
}
