"""
Market assessment logic for the financial-telegram-bot.
Handles AI-powered assessments via LLMs and rule-based fallback logic.
"""

import os
import requests
from typing import Dict, Any, Optional
from bot.config import URLS

def generate_ai_assessment(data: Dict[str, float]) -> str:
    """
    Generate an advanced AI-powered macroeconomic assessment using available LLM providers.
    """
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        groq_api_key = os.getenv('GROQ_API_KEY')
        
        provider: Optional[str] = None
        if openai_api_key: provider = 'openai'
        elif gemini_api_key: provider = 'gemini'
        elif groq_api_key: provider = 'groq'
            
        if not provider:
            print("  ⚠ No AI API keys found, using rule-based assessment...")
            return generate_rule_based_assessment(data)

        print(f"Generating advanced AI market assessment using {provider.upper()}...")
        
        # Mapping statuses
        y_val = data.get('yield_curve', 0.0)
        yield_status = "Inverted (recession signal)" if y_val < 0 else "Positive (no inversion)"
        
        sahm_val = data.get('sahm_rule', 0.0)
        sahm_status = "RECESSION SIGNAL" if sahm_val >= 0.5 else "Safe"
        
        sent_val = data.get('consumer_sentiment', 0.0)
        sentiment_status = "Weak" if sent_val < 60 else "Healthy" if sent_val < 80 else "Strong"
        
        claims_val = data.get('initial_claims', 0.0)
        claims_status = "Healthy" if claims_val < 250 else "Elevated" if claims_val < 350 else "Stressed"
        
        credit_val = data.get('credit_spread', 0.0)
        credit_status = "Tight (good)" if credit_val < 1.5 else "Normal" if credit_val < 2.5 else "Stressed"
        
        real_val = data.get('real_yields', 0.0)
        yields_status = "Easy" if real_val < 0 else "Neutral" if real_val < 2.0 else "Restrictive"
        
        lei_val = data.get('lei_change', 0.0)
        lei_status = "Rising" if lei_val > 0 else "Falling"
        
        fg_val = data.get('fear_greed', 50.0)
        fear_status = "Extreme Fear" if fg_val < 25 else "Fear" if fg_val < 45 else "Neutral" if fg_val < 55 else "Greed" if fg_val < 75 else "Extreme Greed"

        prompt = f"""You are a senior quantitative macroeconomic analyst preparing an executive summary for institutional investors.
Analyze the following comprehensive economic indicators to provide a nuanced, in-depth macroeconomic assessment.

📊 COMPLETE MARKET DATA:
**MACRO & SENTIMENT:**
- Yield Curve (10Y-2Y): {y_val:.2f}% → {yield_status}
- Corporate Profit Margins: {data.get('profit_margin', 0.0):.2f}%
- Fear & Greed Index: {fg_val:.1f}/100 → {fear_status}

**LABOR & RECESSION SIGNALING:**
- Sahm Rule Recession Indicator: {sahm_val:.2f} → {sahm_status}
- Initial Jobless Claims (4wk avg): {claims_val:.0f}K → {claims_status}

**CONSUMER HEALTH & FORWARD PATH:**
- Consumer Sentiment Index: {sent_val:.1f}/100 → {sentiment_status}
- Leading Economic Index: {lei_val:+.2f}% change → {lei_status}

**CREDIT & MONETARY TIGHTNESS:**
- BBB Credit Spread: {credit_val:.2f}% → {credit_status}
- Real Yields (10Y TIPS): {real_val:.2f}% → {yields_status}

TASK: Synthesize an intelligent, highly nuanced assessment. Limit to 4-6 sentences. Conclude with BULLISH, BEARISH, or CAUTIOUS verdict."""

        if provider == 'openai' and openai_api_key:
            headers = {'Authorization': f'Bearer {openai_api_key}', 'Content-Type': 'application/json'}
            payload = {'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.7, 'max_tokens': 350}
            r = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=30)
            assessment = r.json()['choices'][0]['message']['content'].strip()
        elif provider == 'gemini' and gemini_api_key:
            headers = {'Content-Type': 'application/json'}
            payload = {'contents': [{'parts': [{'text': prompt}]}], 'generationConfig': {'temperature': 0.7, 'maxOutputTokens': 350}}
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={gemini_api_key}"
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            assessment = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        elif provider == 'groq' and groq_api_key:
            headers = {'Authorization': f'Bearer {groq_api_key}', 'Content-Type': 'application/json'}
            payload = {'model': 'llama-3.3-70b-versatile', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.7, 'max_tokens': 350}
            r = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload, timeout=30)
            assessment = r.json()['choices'][0]['message']['content'].strip()
        else:
            assessment = generate_rule_based_assessment(data)

        print(f"✓ Advanced AI assessment generated via {provider.upper() if provider else 'NONE'}")
        return assessment

    except Exception as e:
        print(f"  ⚠ AI assessment failed ({e}), falling back to rule-based...")
        return generate_rule_based_assessment(data)

def generate_rule_based_assessment(data: Dict[str, float]) -> str:
    """Generate simple rule-based assessment as fallback"""
    assessment = "🤖 *MARKET ASSESSMENT*\n\n"
    risk_signals = 0
    if data.get('sahm_rule', 0) >= 0.3: risk_signals += 1
    if data.get('fear_greed', 50) < 45: risk_signals += 1
    if data.get('consumer_sentiment', 70) < 60: risk_signals += 1
    if data.get('lei_change', 0) < 0: risk_signals += 1

    if risk_signals >= 3: assessment += "⚠️ *Elevated Risk:* Multiple indicators showing weakness. "
    elif risk_signals >= 2: assessment += "🟡 *Cautious:* Mixed signals. "
    else: assessment += "🟢 *Stable:* Markets showing resilience. "

    if data.get('yield_curve', 0) > 0: assessment += "Yield curve positive. "
    else: assessment += "Yield curve inverted. "

    if data.get('sahm_rule', 0) >= 0.5: assessment += "⚠️ Sahm Rule triggered. "

    assessment += f"\n\n📍 *Watch:* Sentiment at {data.get('consumer_sentiment', 0):.0f}, Fear & Greed at {data.get('fear_greed', 0):.0f}."
    return assessment
