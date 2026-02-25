"""
Proxy entrypoint for the financial-telegram-bot.
Delegates execution to the modular 'bot' package.
"""
from bot.main import main

if __name__ == "__main__":
    main()
