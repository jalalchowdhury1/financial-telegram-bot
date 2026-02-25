"""
Market assessment logic for the financial-telegram-bot.
Handles AI-powered assessments via LLMs and rule-based fallback logic.
"""

import os
import requests

def generate_ai_assessment(data):
    """
    Generate an advanced AI-powered macroeconomic assessment using available LLM providers.
    """
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        groq_api_key = os.getenv('GROQ_API_KEY')
        
        provider = None
        if openai_api_key: provider = 'openai'
        elif gemini_api_key: provider = 'gemini'
        elif groq_api_key: provider = 'groq'
            
        if not provider:
            print("  ⚠ No AI API keys found, using rule-based assessment...")
            return generate_rule_based_assessment(data)

        print(f"Generating advanced AI market assessment using {provider.upper()}...")

        yield_status = "Inverted (recession signal)" if data['yield_curve'] < 0 else "Positive (no inversion)"
        sahm_status = "RECESSION SIGNAL" if data['sahm_rule'] >= 0.5 else "Safe"
        sentiment_status = "Weak" if data['consumer_sentiment'] < 60 else "Healthy" if data['consumer_sentiment'] < 80 else "Strong"
        claims_status = "Healthy" if data['initial_claims'] < 250 else "Elevated" if data['initial_claims'] < 350 else "Stressed"
        credit_status = "Tight (good)" if data['credit_spread'] < 1.5 else "Normal" if data['credit_spread'] < 2.5 else "Stressed"
        yields_status = "Easy" if data['real_yields'] < 0 else "Neutral" if data['real_yields'] < 2.0 else "Restrictive"
        lei_status = "Rising" if data['lei_change'] > 0 else "Falling"
        fear_status = "Extreme Fear" if data['fear_greed'] < 25 else "Fear" if data['fear_greed'] < 45 else "Neutral" if data['fear_greed'] < 55 else "Greed" if data['fear_greed'] < 75 else "Extreme Greed"

        prompt = f"""You are a senior quantitative macroeconomic analyst preparing an executive summary for institutional investors.
Analyze the following comprehensive economic indicators to provide a nuanced, in-depth macroeconomic assessment.

📊 COMPLETE MARKET DATA:
**MACRO & SENTIMENT:**
- Yield Curve (10Y-2Y): {data['yield_curve']:.2f}% → {yield_status}
- Corporate Profit Margins: {data['profit_margin']:.2f}%
- Fear & Greed Index: {data['fear_greed']:.1f}/100 → {fear_status}

**LABOR & RECESSION SIGNALING:**
- Sahm Rule Recession Indicator: {data['sahm_rule']:.2f} → {sahm_status}
- Initial Jobless Claims (4wk avg): {data['initial_claims']:.0f}K → {claims_status}

**CONSUMER HEALTH & FORWARD PATH:**
- Consumer Sentiment Index: {data['consumer_sentiment']:.1f}/100 → {sentiment_status}
- Leading Economic Index: {data['lei_change']:+.2f}% change → {lei_status}

**CREDIT & MONETARY TIGHTNESS:**
- BBB Credit Spread: {data['credit_spread']:.2f}% → {credit_status}
- Real Yields (10Y TIPS): {data['real_yields']:.2f}% → {yields_status}

TASK: Synthesize an intelligent, highly nuanced assessment. Limit to 4-6 sentences. Conclude with BULLISH, BEARISH, or CAUTIOUS verdict."""

        if provider == 'openai':
            headers = {'Authorization': f'Bearer {openai_api_key}', 'Content-Type': 'application/json'}
            payload = {'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.7, 'max_tokens': 350}
            r = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=30)
            assessment = r.json()['choices'][0]['message']['content'].strip()
        elif provider == 'gemini':
            headers = {'Content-Type': 'application/json'}
            payload = {'contents': [{'parts': [{'text': prompt}]}], 'generationConfig': {'temperature': 0.7, 'maxOutputTokens': 350}}
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={gemini_api_key}"
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            assessment = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        elif provider == 'groq':
            headers = {'Authorization': f'Bearer {groq_api_key}', 'Content-Type': 'application/json'}
            payload = {'model': 'llama-3.3-70b-versatile', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.7, 'max_tokens': 350}
            r = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload, timeout=30)
            assessment = r.json()['choices'][0]['message']['content'].strip()

        print(f"✓ Advanced AI assessment generated via {provider.upper()}")
        return assessment

    except Exception as e:
        print(f"  ⚠ AI assessment failed ({e}), falling back to rule-based...")
        return generate_rule_based_assessment(data)

def generate_rule_based_assessment(data):
    """Generate simple rule-based assessment as fallback"""
    assessment = "🤖 *MARKET ASSESSMENT*\n\n"
    risk_signals = 0
    if data['sahm_rule'] >= 0.3: risk_signals += 1
    if data['fear_greed'] < 45: risk_signals += 1
    if data['consumer_sentiment'] < 60: risk_signals += 1
    if data['lei_change'] < 0: risk_signals += 1

    if risk_signals >= 3: assessment += "⚠️ *Elevated Risk:* Multiple indicators showing weakness. "
    elif risk_signals >= 2: assessment += "🟡 *Cautious:* Mixed signals. "
    else: assessment += "🟢 *Stable:* Markets showing resilience. "

    if data['yield_curve'] > 0: assessment += "Yield curve positive. "
    else: assessment += "Yield curve inverted. "

    if data['sahm_rule'] >= 0.5: assessment += "⚠️ Sahm Rule triggered. "

    assessment += f"\n\n📍 *Watch:* Sentiment at {data['consumer_sentiment']:.0f}, Fear & Greed at {data['fear_greed']:.0f}."
    return assessment
