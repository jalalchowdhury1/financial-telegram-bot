"""
Utility functions for the financial-telegram-bot.
Handles environment variables and Telegram communication.
"""

import os
import sys
import requests
from typing import Dict, Optional


def report_marker(success: bool, sections: int = 0, errors: int = 0, reason: str = "") -> str:
    """A single stable, greppable line so CloudWatch/CI can detect a real delivery.

    Emitted by BOTH the Lambda (handle_eventbridge) and the runner (bot/main.py) only
    after Telegram has actually accepted the message. The health-check looks for these.
    """
    if success:
        return f"REPORT_DELIVERED ok=true sections={sections} errors={errors}"
    return f"REPORT_FAILED ok=false reason={reason or 'unknown'} sections={sections}"


TELEGRAM_MAX_CHARS = 4096


def _split_message(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list:
    """Split text into <=limit chunks, preferring newline boundaries.

    A single line longer than the limit is hard-split so nothing is ever dropped.
    """
    if len(text) <= limit:
        return [text]
    chunks: list = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = line if not current else current + "\n" + line
        if len(candidate) > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _post_telegram_text(token: str, chat_id: str, text: str) -> bool:
    """Send one chunk. On a 400 (usually unbalanced Markdown) retry as plain text."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url, data={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}, timeout=30
        )
        if resp.status_code == 400:
            resp = requests.post(url, data={'chat_id': chat_id, 'text': text}, timeout=30)
        return resp.status_code == 200
    except Exception as e:
        print(f"ERROR: Telegram send failed: {e}")
        return False


def load_environment_variables() -> Dict[str, str]:
    """Load and validate required environment variables"""
    config = {
        'FRED_API_KEY': os.getenv('FRED_API_KEY'),
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID')
    }

    missing_vars = [var for var, value in config.items() if not value]

    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    # We can safely cast because we checked for missing vars
    return {k: str(v) for k, v in config.items()}

def send_to_telegram(token: str, chat_id: str, image_path: Optional[str] = None, caption: str = "") -> bool:
    """Send message or image to Telegram chat"""
    if image_path:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'Markdown'}
                response = requests.post(url, files=files, data=data, timeout=30)
                response.raise_for_status()
            print(f"✓ Sent image to Telegram: {image_path}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to send image to Telegram: {e}")
            return False
    else:
        chunks = _split_message(caption)
        all_ok = True
        for chunk in chunks:
            if not _post_telegram_text(token, chat_id, chunk):
                all_ok = False
        if all_ok:
            print(f"✓ Sent text to Telegram ({len(chunks)} chunk(s))")
        else:
            print("ERROR: One or more Telegram chunks failed to send")
        return all_ok
