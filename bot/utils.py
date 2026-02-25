"""
Utility functions for the financial-telegram-bot.
Handles environment variables and Telegram communication.
"""

import os
import sys
import requests

def load_environment_variables():
    """Load and validate required environment variables"""
    required_vars = {
        'FRED_API_KEY': os.getenv('FRED_API_KEY'),
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID')
    }

    missing_vars = [var for var, value in required_vars.items() if not value]

    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    return required_vars

def send_to_telegram(token, chat_id, image_path=None, caption=""):
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
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            data = {'chat_id': chat_id, 'text': caption, 'parse_mode': 'Markdown'}
            response = requests.post(url, data=data, timeout=30)
            response.raise_for_status()
            print("✓ Sent text to Telegram")
            return True
        except Exception as e:
            print(f"ERROR: Failed to send text to Telegram: {e}")
            return False
