"""
AWS Lambda Handler for the Financial Telegram Bot.

Triggered daily at 4:15 AM EST by an EventBridge rule.
Also handles HTTP GET requests from the Next.js dashboard via Function URL.

CloudWatch Logs: all print() calls are automatically captured.
"""

import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any

try:
    import pytz
except ImportError:
    pytz = None

def _clean_nans(obj: Any) -> Any:
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nans(v) for v in obj]
    return obj

from bot.fetchers import (
    fetch_google_sheet_indicators,
    fetch_spy_with_fallback,
    fetch_spy_daily_move,
    fetch_market_extra,
    fetch_polymarket_trending,
)
from bot.utils import load_environment_variables, send_to_telegram
from bot.config import TIMEZONE

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Read-only dashboard endpoints — no auth required (public market data)
_PUBLIC_GET_PATHS = {'/api/spy', '/api/spy-daily-move', '/api/market-extra', '/api/polymarket'}


def _cors_headers() -> Dict[str, str]:
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,x-bot-secret',
        'Access-Control-Allow-Methods': 'GET,OPTIONS',
        'Content-Type': 'application/json',
    }


def _ok(body: Any) -> Dict[str, Any]:
    cleaned_body = _clean_nans(body)
    return {'isBase64Encoded': False, 'statusCode': 200, 'headers': _cors_headers(), 'body': json.dumps(cleaned_body)}


def _err(status: int, message: str) -> Dict[str, Any]:
    return {'isBase64Encoded': False, 'statusCode': status, 'headers': _cors_headers(), 'body': json.dumps({'error': message})}


def handle_http_api(event: Dict[str, Any], env_vars: Dict[str, str]) -> Dict[str, Any]:
    """Handle HTTP GET requests from the Next.js dashboard via Lambda Function URL."""
    method = event.get('requestContext', {}).get('http', {}).get('method', 'GET')
    path = event.get('rawPath', '/')

    if method == 'OPTIONS':
        return {'statusCode': 200, 'headers': _cors_headers(), 'body': ''}

    # Auth check — skipped for public read-only dashboard endpoints
    if path not in _PUBLIC_GET_PATHS:
        headers = {k.lower(): v for k, v in event.get('headers', {}).items()}
        bot_secret = ''.join(deque(env_vars.get('TELEGRAM_TOKEN', ''), maxlen=10))
        if headers.get('x-bot-secret') != bot_secret:
            return _err(401, 'Unauthorized')

    fred_api_key = env_vars.get('FRED_API_KEY', '')
    polygon_api_key = env_vars.get('POLYGON_API_KEY', '') or None
    finnhub_api_key = env_vars.get('FINNHUB_API_KEY', '') or None

    try:
        if path == '/api/spy':
            logger.info('Dashboard: GET /api/spy')
            data = fetch_spy_with_fallback(fred_api_key=fred_api_key,
                                           polygon_api_key=polygon_api_key,
                                           finnhub_api_key=finnhub_api_key)
            return _ok(data)

        elif path == '/api/spy-daily-move':
            logger.info('Dashboard: GET /api/spy-daily-move')
            data = fetch_spy_daily_move()
            return _ok(data)

        elif path == '/api/market-extra':
            logger.info('Dashboard: GET /api/market-extra')
            data = fetch_market_extra(fred_api_key=fred_api_key,
                                      polygon_api_key=polygon_api_key,
                                      finnhub_api_key=finnhub_api_key)
            return _ok(data)

        elif path == '/api/polymarket':
            logger.info('Dashboard: GET /api/polymarket')
            bets = fetch_polymarket_trending()
            return _ok({
                'bets': bets,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'error': None if bets else 'Polymarket API unavailable'
            })

        else:
            return _err(404, f'Not Found: {path}')

    except Exception as e:
        logger.error(f'HTTP API error on {path}: {e}')
        return _err(500, str(e))


def handle_eventbridge(env_vars: Dict[str, str], run_time: str) -> Dict[str, Any]:
    """Handle the daily Telegram report scheduled by EventBridge."""
    telegram_token   = env_vars['TELEGRAM_TOKEN']
    telegram_chat_id = env_vars['TELEGRAM_CHAT_ID']
    fred_api_key     = env_vars.get('FRED_API_KEY', '')

    report_sections: list = []
    errors: list = []

    # 1. Google Sheets custom indicators
    logger.info('Step 1/2 – Fetching Google Sheet indicators...')
    try:
        gs_text = fetch_google_sheet_indicators()
        if gs_text:
            report_sections.append(gs_text)
            logger.info('✓ Google Sheets indicators fetched.')
        else:
            logger.warning('⚠ Google Sheets returned empty result.')
    except Exception as e:
        err = f'Google Sheets fetch failed: {e}'
        logger.error(err)
        errors.append(err)

    # 2. SPY snapshot
    logger.info('Step 2/2 – Fetching SPY data...')
    try:
        spy = fetch_spy_with_fallback(fred_api_key=fred_api_key)
        price      = spy.get('current', 0)
        ma200_pct  = spy.get('ma200', {}).get('pct', 0)
        high52_pct = spy.get('week52High', {}).get('pct', 0)
        rsi        = spy.get('rsi', 0)
        ret3y      = spy.get('return3y', 0)
        source     = spy.get('_meta', {}).get('source', 'unknown')

        daily = spy.get('dailyChange', {})
        chg_pct = daily.get('pct', 0)

        spy_text = (
            f'📈 *SPY Snapshot* _({source})_\n'
            f'Price : ${price:,.2f} ({chg_pct:+.2f}%)\n'
            f'vs MA200 : {ma200_pct:+.2f}% | 52W High : {high52_pct:+.2f}%\n'
            f'RSI(9) : {rsi:.1f} | 3Y Return : {ret3y:+.1f}%'
        )
        report_sections.append(spy_text)
        logger.info(f'✓ SPY fetched via {source}.')
    except Exception as e:
        err = f'SPY fetch failed: {e}'
        logger.error(err)
        errors.append(err)

    if not report_sections:
        msg = 'No data could be fetched — all sources failed.'
        logger.error(msg)
        return {'statusCode': 500, 'body': json.dumps(msg)}

    separator = '\n\n' + '─' * 30 + '\n\n'
    full_report = separator.join(report_sections)

    logger.info('Sending report to Telegram...')
    success = send_to_telegram(telegram_token, telegram_chat_id, caption=full_report)

    if success:
        summary = f'Report sent at {run_time}. Sections: {len(report_sections)}. Errors: {len(errors)}.'
        logger.info(f'✓ {summary}')
        return {'statusCode': 200, 'body': json.dumps(summary)}
    else:
        msg = 'Report assembled but Telegram delivery failed.'
        logger.error(msg)
        return {'statusCode': 500, 'body': json.dumps(msg)}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main entry point invoked by EventBridge schedule or Lambda Function URL."""
    if pytz:
        tz = pytz.timezone(TIMEZONE)
        run_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')
    else:
        run_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    logger.info('=' * 60)
    logger.info(f'Lambda triggered at {run_time}')

    is_http = ('rawPath' in event
               or ('requestContext' in event and 'http' in event.get('requestContext', {})))

    logger.info(f'Trigger: {"HTTP " + event.get("rawPath", "") if is_http else "EventBridge"}')
    logger.info('=' * 60)

    if is_http:
        # Dashboard routes only need market data API keys — no Telegram credentials required
        fred_api_key = os.environ.get('FRED_API_KEY', '')
        if not fred_api_key:
            return {'statusCode': 500, 'headers': _cors_headers(),
                    'body': json.dumps({'error': 'FRED_API_KEY not configured'})}
        return handle_http_api(event, {
            'FRED_API_KEY': fred_api_key,
            'POLYGON_API_KEY': os.environ.get('POLYGON_API_KEY', ''),
            'FINNHUB_API_KEY': os.environ.get('FINNHUB_API_KEY', ''),
        })
    else:
        try:
            env_vars = load_environment_variables()
        except SystemExit as e:
            msg = f'FATAL: Missing required environment variables: {e}'
            logger.error(msg)
            return {'statusCode': 500, 'body': json.dumps(msg)}
        return handle_eventbridge(env_vars, run_time)
