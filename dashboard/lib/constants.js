/**
 * Centralized constants for the Financial Dashboard.
 * Includes FRED Series IDs, External API URLs, and configuration.
 */

export const FRED_SERIES = {
    YIELD_CURVE: 'T10Y2Y',
    UNEMPLOYMENT: 'UNRATE',
    SENTIMENT: 'UMCSENT',
    CLAIMS: 'ICSA',
    CREDIT_SPREAD: 'BAMLC0A4CBBB',
    REAL_YIELDS: 'DFII10',
    LEI: 'USSLIND',
    NFCI: 'NFCI',
    M2_MONEY: 'M2SL',
    RETAIL_SALES: 'RSXFS',
    HOUSING_STARTS: 'HOUST',
    INDUSTRIAL_PROD: 'INDPRO',
    JOLTS: 'JTSJOL',
    DURABLE_GOODS: 'DGORDER',
    SAVINGS_RATE: 'PSAVERT',
    CORP_PROFITS: 'A053RC1Q027SBEA',
    GDP: 'GDP',
    RECESSIONS: 'USREC',
    MORTGAGE30: 'MORTGAGE30US',
    MEDIAN_HOME_PRICE: 'MSPUS',
    RENT_INDEX: 'CUUR0000SEHA'
};

export const EXTERNAL_URLS = {
    FRED_BASE: 'https://api.stlouisfed.org/fred/series/observations',
    STOOQ_SPY: 'https://stooq.com/q/d/l/?s=spy.us&i=d',
    YAHOO_SPY: 'https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5y&interval=1d',
    MULTPL_PE: 'https://www.multpl.com/s-p-500-pe-ratio',
    YAHOO_PE: 'https://finance.yahoo.com/quote/SPY/key-statistics',
    CNN_FEAR_GREED: 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata',
    RAPIDAPI_FEAR_GREED: 'https://fear-and-greed-index.p.rapidapi.com/v1/fgi',
    YAHOO_VIX: 'https://query1.finance.yahoo.com/v8/finance/chart/^VIX?range=1mo&interval=1d'
};

export const GOOGLE_SHEETS = {
    NOT_SO_BORING: 'https://docs.google.com/spreadsheets/d/10Y8Jus8_fMwH9H69vWh7thSzl2hH34Ri3BRbDw_GEgw/export?format=csv&gid=0',
    FRONT_RUNNER: 'https://docs.google.com/spreadsheets/d/1vdlPNlT6gRpzMHuQUT7olqUNb455CQM3ab4wPuCE5R0/export?format=csv&gid=1668420064',
    AAII: 'https://docs.google.com/spreadsheets/d/1zQQ2am1yhzTwY7nx8xPak4Q0WoNMwxWj7Ekr-fDEIF4/export?format=csv&gid=0',
    VIX: 'https://docs.google.com/spreadsheets/d/1vdlPNlT6gRpzMHuQUT7olqUNb455CQM3ab4wPuCE5R0/export?format=csv&gid=790638481',
    MARKET_BACKUP: 'https://docs.google.com/spreadsheets/d/1dexvcTRuwHFh8DuFyBdkgP0J5FofLkQrV9YltQRHE3Q/export?format=csv&gid=1165823281'
};

export const YAHOO_TICKERS = {
    SPY: 'SPY',
    TNX_10Y: '^TNX', // Need to divide by 10 for yield
    DXY: 'DX-Y.NYB',
    CRUDE_OIL: 'CL=F',
    USD_CAD: 'CAD=X',
    USD_INR: 'INR=X',
    USD_BDT: 'BDT=X',
    GOLD: 'GC=F',
    BTC: 'BTC-USD'
};

export const DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*'
};
