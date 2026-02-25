#!/usr/bin/env python3
"""
Environment & API Status Check
Quickly verifies if all required API keys and components are functional.
Designed for AI maintenance and rapid debugging.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def check_env():
    print("📋 Checking Environment Variables...")
    required = [
        'FRED_API_KEY', 'TELEGRAM_TOKEN', 'TELEGRAM_CHAT_ID'
    ]
    all_ok = True
    for key in required:
        val = os.getenv(key)
        if val:
            print(f"  ✅ {key}: Set")
        else:
            print(f"  ❌ {key}: MISSING")
            all_ok = False
    return all_ok

def check_git():
    print("\n📦 Checking Repository Status...")
    try:
        import subprocess
        remote = subprocess.check_output(['git', 'remote', '-v']).decode()
        print(f"  ✅ Remote URLs found:\n{remote.strip()}")
    except Exception:
        print("  ❌ Could not determine git remote status.")

def main():
    print("🚀 Financial Bot & Dashboard - System Health Check\n")
    env_ok = check_env()
    check_git()
    
    print("\n" + "="*40)
    if env_ok:
        print("🎉 System looks ready for development!")
    else:
        print("⚠️  Action required: Please fill in the missing .env keys.")
    print("="*40)

if __name__ == "__main__":
    main()
