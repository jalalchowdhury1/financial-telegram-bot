"""
Centralized configuration for the financial-telegram-bot.
Consolidates FRED Series IDs, URLs, and constant parameters.
"""

# FRED Series Identifiers
FRED_SERIES = {
    'YIELD_10Y2Y': 'T10Y2Y',
    'UNEMPLOYMENT': 'UNRATE',
    'SENTIMENT': 'UMCSENT',
    'CLAIMS': 'ICSA',
    'CREDIT_SPREAD': 'BAMLC0A4CBBB',
    'REAL_YIELDS': 'DFII10',
    'LEI': 'USSLIND',
    'NFCI': 'NFCI',
    'M2_MONEY': 'M2SL',
    'RETAIL_SALES': 'RSXFS',
    'HOUSING_STARTS': 'HOUST',
    'INDUSTRIAL_PROD': 'INDPRO',
    'JOLTS': 'JTSJOL',
    'DURABLE_GOODS': 'DGORDER',
    'SAVINGS_RATE': 'PSAVERT',
    'CORP_PROFITS': 'A053RC1Q027SBEA',
    'GDP': 'GDP',
    'RECESSIONS': 'USREC',
    'SP500': 'SP500',
}

# External Resource URLs
URLS = {
    'NOT_SO_BORING': "https://docs.google.com/spreadsheets/d/10Y8Jus8_fMwH9H69vWh7thSzl2hH34Ri3BRbDw_GEgw/export?format=csv&gid=0",
    'FRONT_RUNNER': "https://docs.google.com/spreadsheets/d/1vdlPNlT6gRpzMHuQUT7olqUNb455CQM3ab4wPuCE5R0/export?format=csv&gid=1668420064",
    'AAII': "https://docs.google.com/spreadsheets/d/1zQQ2am1yhzTwY7nx8xPak4Q0WoNMwxWj7Ekr-fDEIF4/export?format=csv&gid=0",
    'VIX': "https://docs.google.com/spreadsheets/d/1vdlPNlT6gRpzMHuQUT7olqUNb455CQM3ab4wPuCE5R0/export?format=csv&gid=790638481",
    'FEAR_GREED': "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
}

# Technical Indicators Parameters
RSI_PERIOD = 9
MA_200_PERIOD = 200
RETURN_3Y_DAYS = 756
WEEK_52_DAYS = 252

# Chart Appearance
CHART_STYLE = "whitegrid"
DEFAULT_FIGSIZE = (12, 6)
TABLE_COLOR_PRIMARY = '#4472C4'
TABLE_COLOR_SECONDARY = '#E7E6E6'
REGIME_COLORS = {
    'BULL': '#90EE90',
    'CAUTIOUS': '#FFD700',
    'BEAR': '#FFB6C1'
}

# Scheduling (for bot/main.py)
TIMEZONE = "America/New_York"
REPORT_TIME = {
    'hour': 4,
    'minute': 15
}
