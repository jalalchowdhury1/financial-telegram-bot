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

        let assessment = null;
        let lastError = 'No API keys configured.';
        let usedModel = '';

        const configs = [];

        // Priority 1: OpenAI Models (GPT-4o)
        if (process.env.OPENROUTER_API_KEY) {
            configs.push({ name: 'OpenRouter GPT-4o', url: 'https://openrouter.ai/api/v1/chat/completions', model: 'openai/gpt-4o-2024-11-20', key: process.env.OPENROUTER_API_KEY });
        }
        if (process.env.OPENAI_API_KEY) {
            configs.push({ name: 'OpenAI GPT-4o', url: 'https://api.openai.com/v1/chat/completions', model: 'gpt-4o', key: process.env.OPENAI_API_KEY });
        }

        // Priority 2: Llama Models (Llama 3.3 70B)
        if (process.env.GROQ_API_KEY) {
            configs.push({ name: 'Groq Llama 3.3 70B', url: 'https://api.groq.com/openai/v1/chat/completions', model: 'llama-3.3-70b-versatile', key: process.env.GROQ_API_KEY });
        }
        if (process.env.OPENROUTER_API_KEY) {
            configs.push({ name: 'OpenRouter Llama 3.3', url: 'https://openrouter.ai/api/v1/chat/completions', model: 'meta-llama/llama-3.3-70b-instruct', key: process.env.OPENROUTER_API_KEY });
        }

        // Priority 3: K2 Models (Moonshot Kimi K2)
        if (process.env.OPENROUTER_API_KEY) {
            configs.push({ name: 'OpenRouter K2', url: 'https://openrouter.ai/api/v1/chat/completions', model: 'moonshotai/kimi-k2-instruct-0905', key: process.env.OPENROUTER_API_KEY });
        }
        if (process.env.GROQ_API_KEY) {
            configs.push({ name: 'Groq K2', url: 'https://api.groq.com/openai/v1/chat/completions', model: 'kimi-k2-instruct', key: process.env.GROQ_API_KEY });
        }
        if (process.env.MOONSHOT_API_KEY) {
            configs.push({ name: 'Moonshot K2', url: 'https://api.moonshot.cn/v1/chat/completions', model: 'moonshot-v1-8k', key: process.env.MOONSHOT_API_KEY });
        }

        // Safety Fallback: Groq's smaller Llama model if the 70B is rate-limited and nothing else is available
        if (process.env.GROQ_API_KEY) {
            configs.push({ name: 'Groq Llama 3.1 8B (Fallback)', url: 'https://api.groq.com/openai/v1/chat/completions', model: 'llama-3.1-8b-instant', key: process.env.GROQ_API_KEY });
        }

        if (configs.length === 0) {
            return Response.json({ assessment: generateRuleBased(data) });
        }

        for (const conf of configs) {
            try {
                const headers = {
                    'Authorization': `Bearer ${conf.key}`,
                    'Content-Type': 'application/json'
                };

                if (conf.url.includes('openrouter')) {
                    headers['HTTP-Referer'] = 'https://financial-dashboard.local';
                    headers['X-Title'] = 'Financial Telegram Dashboard';
                }

                const res = await fetch(conf.url, {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify({
                        model: conf.model,
                        messages: [{ role: 'user', content: prompt }],
                        temperature: 0.7,
                        max_tokens: 350
                    })
                });

                const result = await res.json();

                if (!res.ok) {
                    console.warn(`[ASSESSMENT FALLBACK] ${conf.name} failed:`, result.error?.message || res.statusText);
                    lastError = `[${conf.name}] ${result.error?.message || res.statusText}`;
                    continue; // Model failed (rate limit, etc). Automatically step to the next one!
                }

                assessment = result.choices?.[0]?.message?.content?.trim();

                if (assessment) {
                    usedModel = conf.name;
                    console.log(`[ASSESSMENT SUCCESS] Successfully generated with ${conf.name}`);
                    break; // SUCCESS! Stop cascading.
                }
            } catch (networkError) {
                console.warn(`[ASSESSMENT FALLBACK] ${conf.name} network fetch failed:`, networkError.message);
                lastError = `[${conf.name} Network] ${networkError.message}`;
            }
        }

        if (!assessment) {
            return Response.json({ assessment: `⚠️ All AI models exhausted or rate-limited.\n\nFinal Error:\n${lastError}` }, { status: 500 });
        }

        // Append a subtle tag to let the user know which model successfully serviced the request
        return Response.json({ assessment: assessment + `\n\n*(Provider: ${usedModel})*` });
    } catch (error) {
        return Response.json({ assessment: `⚠️ AI assessment unavailable: ${error.message}` }, { status: 500 });
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
