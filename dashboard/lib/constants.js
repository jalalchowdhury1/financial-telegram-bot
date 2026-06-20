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
    RENT_INDEX: 'CUUR0000SEHA',
    ATNHPI: 'ATNHPIUS39300Q'
};

/**
 * Maximum acceptable age (in days) of the latest observation for each FRED
 * series, based on how often the series actually updates. If the newest data
 * point is older than this, the metric is treated as STALE and shown as N/A
 * (never a misleadingly old number). Deadlines are generous on monthly/quarterly
 * series because the government itself reports those weeks late.
 */
// NOTE: FRED data points are dated the START of their period (a monthly series'
// newest point is the 1st of the month) and are reported weeks late, so a
// perfectly current monthly series can legitimately be ~2 months old by date,
// and JOLTS (a ~2-month lag) ~3 months. Deadlines are therefore generous: they
// exist only to catch a genuinely DEAD feed (e.g. USSLIND, frozen since 2020),
// not to flag normal reporting lag.
export const FRED_FRESHNESS = {
    // Daily (markets) — allow long holiday weekends
    T10Y2Y: 7,
    DFII10: 7,
    BAMLC0A4CBBB: 7,
    // Weekly
    ICSA: 14,
    NFCI: 14,
    // Monthly (dated 1st of month + multi-week reporting lag)
    UNRATE: 80,
    // UMCSENT's free FRED series is delayed ONE MONTH at the source's request, so
    // its newest point legitimately ages to ~86 days right before the next print
    // (month M reading lands on FRED ~the 26th of month M+2). 95 covers that worst
    // case + slack for a late release, while still catching a truly dead feed.
    UMCSENT: 95,
    M2SL: 80,
    RSXFS: 80,
    HOUST: 80,
    INDPRO: 80,
    DGORDER: 80,
    PSAVERT: 80,
    USREC: 80,
    PE10: 80,
    // Monthly but heavily lagged
    JTSJOL: 110,   // JOLTS publishes ~2 months behind
    // Quarterly (GDP/corp profits — newest quarter is months old by date)
    A053RC1Q027SBEA: 200,
    GDP: 200,
};

export const EXTERNAL_URLS = {
    FRED_BASE: 'https://api.stlouisfed.org/fred/series/observations',
    STOOQ_SPY: 'https://stooq.com/q/d/l/?s=spy.us&i=d',
    YAHOO_SPY: 'https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5y&interval=1d',
    NASDAQ_SPY: 'https://api.nasdaq.com/api/quote/SPY/historical',
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
    MARKET_BACKUP: 'https://docs.google.com/spreadsheets/d/1dexvcTRuwHFh8DuFyBdkgP0J5FofLkQrV9YltQRHE3Q/export?format=csv&gid=1165823281',
    SPY_INDICATORS: 'https://docs.google.com/spreadsheets/d/1FPxydetBtxFIm-qxrF5BR-sMZAUnbdA09LPbSu5lUCs/export?format=csv&gid=941079229',
    SPY_DAILY_MOVE: 'https://docs.google.com/spreadsheets/d/1T99550TEo19JB6I3aKnRRGXAblB8mWNBsM-48jrDGe4/export?format=csv&gid=0'
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
