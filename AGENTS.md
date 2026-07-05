# AGENTS.md — Financial Telegram Bot & Dashboard

> **This is the single source of truth for anyone (human or AI) touching this repo.**
> Read it fully before changing code, deploying, or "fixing" anything. It absorbs and
> replaces the old scattered LLM-facing docs — `.cursorrules`, the
> `docs/superpowers/specs/*` design docs, and the `docs/superpowers/plans/*`
> implementation plans — which were point-in-time and have been **deleted** (their durable
> facts live here now). `README.md` (human/GitHub landing page) and
> `.github/self-improve-prompt.md` (the live agent prompt, see §7) are kept. If something
> here is wrong, fix *this* file.

This project is the owner's **most important** tool. Production stability matters more
than speed. When in doubt, prefer a graceful fallback over a hard change, and verify a
deploy actually worked before claiming success.

Repo slug: **`jalalchowdhury1/financial-telegram-bot`** (public). Stack: Python 3.11
(Lambda + bot package) + JavaScript/Next.js 13.5 (dashboard). AWS account `463256610967`,
region `us-east-1`.

---

## 1. What this is

One AWS **Lambda** (`financial-telegram-report`, `us-east-1`, python3.11, 512 MB, 300 s
timeout, account `463256610967`) that does **two** unrelated jobs from the same
`lambda_handler.py` (it dispatches on the event shape — `rawPath`/`requestContext.http`
present ⇒ HTTP, else EventBridge):

1. **Daily Telegram report** — an EventBridge schedule invokes the Lambda (~**09:15 UTC**,
   i.e. ~4:15 AM New York; the live rule is hand-managed — see the config-drift note in §2);
   it assembles a market brief (Google-Sheet indicators via `bot/fetchers.py:
   fetch_google_sheet_indicators` + a SPY snapshot via `fetch_spy_with_fallback`) and sends
   it to Telegram (`bot/utils.py:send_to_telegram`). Entry: `handle_eventbridge`.
   *(There is also an independent **runner** report path — `.github/workflows/daily_report.yml`
   runs `python -m bot.main report` (→ `bot/main.run_report()`) on the GitHub runner at
   **09:45 UTC**, 30 min after the Lambda. It is a backstop: a guard step queries CloudWatch
   for a `REPORT_DELIVERED` marker in the last 24 h and **skips** if the Lambda already sent,
   so there's no double report. Manual / repository_dispatch / auto-remediation runs always
   send. See §7.)*
2. **Dashboard data API** — serves read-only market-data endpoints (`/api/spy`,
   `/api/spy-daily-move`, `/api/market-extra`, `/api/polymarket`) consumed by the
   Next.js dashboard. Entry: `handle_http_api`. These four paths are in `_PUBLIC_GET_PATHS`
   (no auth). Any other path requires an `x-bot-secret` header equal to the last 10 chars of
   `TELEGRAM_TOKEN` (and currently there are no such paths wired up).

The **dashboard** is a Next.js app in `/dashboard`, deployed on **Vercel**
(auto-deploys on push to `main`): <https://financial-telegram-bot-beryl.vercel.app/>.

### Architecture you MUST get right (this trips people up)

```
Browser ─▶ Vercel (Next.js /dashboard) ─▶ /api/* route ─┬─▶ API Gateway ─▶ Lambda   (primary)
                                                         └─▶ direct public APIs       (fallback)
```

- The dashboard reaches the Lambda through an **HTTP API Gateway**
  (`q1zp14pxal` / `14wb2ubzh3`, `$default` stage, base like
  `https://q1zp14pxal.execute-api.us-east-1.amazonaws.com`). The dashboard's
  `process.env.LAMBDA_URL` points at that gateway base.
- **The Lambda *Function URL* (`…lambda-url…on.aws`) is NOT used. It returns 403 (blocked
  at the account/org level) and is effectively dead. Do not point the dashboard at it,
  and don't waste time "fixing" its auth.**
- Every dashboard `/api/*` route calls the Lambda first, then falls back to direct public
  sources, so the dashboard keeps working even if the Lambda is down. Only `/api/spy`,
  `/api/spy-daily-move`, `/api/market-extra`, `/api/polymarket` actually call the Lambda;
  `/api/fred`, `/api/sheets`, `/api/fear-greed`, `/api/assessment`, `/api/last-run` are
  dashboard-only (no Lambda hop).

---

## 2. Deploying the Lambda (read before any backend change)

**You normally don't deploy by hand — just push to `main`.** GitHub Actions
(`.github/workflows/deploy-lambda.yml`) deploys automatically when a commit changes
`lambda_handler.py`, `bot/**`, `aws/requirements-lambda.txt`, or the deploy workflow
itself. You can also run it manually: Actions ▸ **Deploy to AWS Lambda** ▸ Run workflow.

The deploy: builds the zip (deps from `aws/requirements-lambda.txt` + `lambda_handler.py`
+ `bot/*.py`) → uploads to S3 (`financial-telegram-bot-deployments`, key
`lambda-deployments/deployment_<TS>.zip`; the package is >50 MB so direct upload won't
work) → `update-function-code` from S3 (serialized via a `concurrency` group +
`function-updated` waits + retried up to 5× on `ResourceConflictException`) → waits →
**smoke-tests the live API Gateway** (discovers all HTTP APIs, hits `GET /api/spy` on each;
the gateway(s) that return 200 must return valid JSON with no bare `NaN`/`Infinity` or the
deploy fails). Runs with `secrets.AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY` (the scoped IAM
user **`github-deploy-bot`**, not root).

### ⚠️ Config drift — the deploy only updates *code*
The deploy runs `update-function-code` ONLY. **`aws/template.yaml` (the SAM template) is
never applied.** Lambda env vars, memory/timeout, the EventBridge schedule, and the
Function URL are all hand-managed in the AWS console and can drift from the template.
**Editing `template.yaml` does NOT change the live function** — change those in the
console (or extend the workflow) and update the template to match. Concrete known drift:
`template.yaml` declares the schedule as `cron(15 8 * * ? *)` (08:15 UTC) but the live
EventBridge rule fires ~**09:15 UTC** (the workflows are written around 09:15). Trust the
code/workflow comments over the template for the live schedule.

### Hard rules for the deployment package
- **Never build Lambda deps on a Mac.** Native wheels (numpy/pandas) built locally crash
  on Lambda's Linux with `Runtime.ImportModuleError`. When you change
  `aws/requirements-lambda.txt`, the CI build already compiles for the right platform:
  ```bash
  pip install --platform manylinux2014_x86_64 --target=package \
      --implementation cp --python-version 3.11 --only-binary=:all: \
      -r aws/requirements-lambda.txt
  ```
  Add new backend deps to `aws/requirements-lambda.txt` (separate from local
  `requirements.txt`). Keep `bot/` **lite** — no heavy plotting/bulky libs; deploy size
  and cold start depend on it. (`aws/requirements-lambda-minimal.txt` is a pure-Python
  subset kept for reference; CI uses `requirements-lambda.txt`, which adds
  pandas/pandas-datareader/yfinance.)
- If you ever build locally, clean up `package/` and `*.zip` so they don't pollute git
  (both are gitignored). A small committed `aws/deployment.zip` exists as a historical
  artifact and is **not** what CI ships — ignore it.

### Three Lambda gotchas that have caused real outages
1. **yfinance must cache to `/tmp`.** Lambda's filesystem is read-only except `/tmp`.
   Without a writable TZ/cookie cache, every Yahoo call is a fresh cookieless scrape and
   Yahoo **429-rate-limits / IP-bans** the function. `bot/fetchers.py:_fetch_yfinance`
   already does `os.makedirs('/tmp/yfinance')` + `yf.set_tz_cache_location('/tmp/yfinance')`
   before any yfinance call — keep it that way.
2. **Sanitize NaN/Infinity before `json.dumps`.** Python emits bare `NaN`/`Infinity`
   tokens that AWS API Gateway (payload format 2.0) and JS parsers reject — a 200 locally
   becomes a **500 `Internal Server Error`** at the gateway. `lambda_handler.py:_clean_nans`
   recursively converts them to `null`; all dashboard responses go out through `_ok()`,
   which runs it. Never bypass `_ok()`.
3. **Polygon free tier is a day delayed.** When SPY falls back to Polygon, the spot price
   is yesterday's close. `fetch_spy_with_fallback` overrides the stale `current` with a
   live **Finnhub** quote (`_fetch_finnhub_quote`) — don't remove that override.

### Wiring a new API Gateway to the Lambda
If a newly attached gateway returns an instant **500 with no CloudWatch logs**, the
Lambda's resource policy is blocking it. Grant invoke:
```bash
aws lambda add-permission --function-name financial-telegram-report \
  --statement-id AllowMyAPIGatewayXYZ --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:us-east-1:463256610967:<API_ID>/*/*"
```

### SPY waterfall (Lambda `fetch_spy_with_fallback`)
yfinance (full history) → Polygon (full history) → Google Sheet `SPY_INDICATORS`
(pre-computed values, plus `SPY_DAILY_MOVE` for the 3Y return) → Stooq CSV → FRED `SP500`.
Whichever wins, the result is normalized to the `/api/spy` shape and **chart history +
MA50/MA200 are computed from FRED `SP500`** when only pre-computed indicators are available.
Finnhub spot overrides the latest price (gotcha #3). `_meta.source` records the winning tier.

---

## 3. The dashboard (`/dashboard`)

Deploys to Vercel automatically on push to `main`. Tests: from `dashboard/`,
`npm test` (Jest) and `npm run build` must pass before merge; `npm run test:watch` for
watch mode; `--testPathPattern <Name>` filters. Next.js 13.5.6, React 18 (App Router).

### Environment / secrets (Vercel, Production + Preview)
The repo is **public** — keys NEVER go in code; they live in **Vercel env vars**:
- `LAMBDA_URL` — the API Gateway base (NOT the Function URL — see §1).
- `FRED_API_KEY` — used by `/api/fred`, `/api/sheets` (VIX/sentiment proxy),
  `/api/fear-greed` (VIXCLS), and the copper/gold legs.
- `POLYGON_KEY` — **note the name** (not `POLYGON_API_KEY`; the Lambda's *own* env var IS
  `POLYGON_API_KEY`, but the **dashboard** reads `process.env.POLYGON_KEY`). Used by spy,
  spy-daily-move, market-extra, and the gold leg of copper/gold.
- `FINNHUB_KEY` — real-time SPY spot for `/api/spy` + `/api/spy-daily-move`.
- `RAPIDAPI_KEY` — optional, Layer 2 of `/api/fear-greed`. (A RapidAPI key was once
  committed here; don't repeat it — see §4 open items.)
- `GROQ_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `MOONSHOT_API_KEY` — optional
  LLM providers for `/api/assessment` (cascade in that priority; rule-based fallback if none).
- `GITHUB_TOKEN` — optional, raises the GitHub API rate limit for `/api/last-run`.

### Conventions (enforce on every change)
- **Never hardcode secrets** — see the env list above; the repo is public.
- **Never-throw routes.** Every cached `/api/*` GET (`spy`, `spy-daily-move`,
  `market-extra`, `polymarket`, `fred`) wraps its *entire* body in
  `lib/store.js: serve(key, produce, opts)`, which cascades **live → fallbacks →
  durable last-known-good (in `/tmp`, optionally Redis via `REDIS_URL`) → safe default** and
  always returns **HTTP 200 with valid JSON**. Define any `liveKey`/`faults`/`apiKey` vars
  *before* `serve()` so nothing can throw outside the try/catch (that would 500 with an empty
  body and break the guarantee). *(The legacy routes `sheets`, `fear-greed`, `last-run` use
  their own hand-rolled try/catch layer cascades + `/tmp` cache instead of `serve()`;
  `assessment` is POST-only and returns its own error JSON. Same never-blank goal, older
  pattern — match `serve()` for any new route.)*
- **Extract** new features into standalone components in `dashboard/components/`; add
  `'use client'` to any component with client-side state/interactivity.
- **External links** must use `target="_blank" rel="noopener noreferrer"`.
- Probability-bar colors (MarketModal `getOddsColor`): `<0.2` red, `0.2–0.4` orange,
  `0.4–0.6` yellow, `0.6–0.8` green, `≥0.8` bright green. The field is **`bet.odds`** (a
  decimal 0–1, rendered as a %) — not `bet.probability`. The modal links to the bare
  `https://polymarket.com` homepage; per-market deep links were deliberately avoided as
  unreliable (the API doesn't surface a usable slug).

### Polymarket "Market Sentiment" board (`/api/polymarket` + `PolymarketTable.js`)
Both the Lambda fetcher (`bot/fetchers.py:fetch_polymarket_trending`, **primary**) and the
dashboard JS fallback (`dashboard/app/api/polymarket/route.js:fallbackPoly`) implement the
**same curation** and must stay in sync. They turn the raw Gamma feed (a wall of 1%/100%
longshots) into a curated board of "what the crowd is betting on":
1. Fetch a broad pool — paginate the public Gamma REST API (`gamma-api.polymarket.com/markets`,
   **no key**), `active=true&closed=false`, `order=volume1wk` (recent interest, less churny
   than 24h), `limit=100`, offsets 0→400 (Gamma caps `limit` at 100).
2. **Group by event** (`events[0].ticker`/`slug`) so multi-candidate races collapse to one
   "Event: favorite" row; drop markets resolving in <1 day.
3. Keep **binary Yes/No** markets only; standalone markets must have odds in **[0.08, 0.92]**
   and volume ≥ ~$25k; event favorites use [0.05, 0.85] and event-summed volume ≥ ~$25k.
4. Filter **sports/esports** (a big keyword set + any `sport` tag); tag a **topic** (Crypto 🪙
   / Geopolitics 🌍 / Politics 🏛️ / Tech 🤖 / Economy 📉 / else World 🌐), cap **2 per topic**.
5. Rank by volume, de-dupe by event, and **cap longshots** (<30% odds, non-event) to ≈half
   the slots in a first pass (so the board is a spread, not a wall) — a second pass fills any
   remaining slots without the cap. Take **top 8**.
Per-bet contract: `{ name, odds (0–1), volume ($), change (oneMonthPriceChange, ±frac|null),
topic, topicEmoji, endDate (ISO|null), eventSlug }`. Backward-compatible: `name/odds/volume`
preserved so `MarketModal` keeps working. The fetcher returns `[]` on any failure (never raises).
Frontend row: topic emoji + question + colored **%** bar + **▲▼** 30-day momentum (shown only
when |change| ≥ 0.02) + volume in $M/$k + a muted "resolves in Nd". Out of scope (v2):
multi-candidate "favorites" lists; a fresher momentum window via the CLOB price-history endpoint.

### `market-extra` + the status footer (where a "red" bug lived)
- `/api/market-extra` is Lambda-primary, then fills only the metrics the Lambda returned
  `null` from **direct sources** (Polygon for FX/gold/BTC; FRED for oil/rates; ER-API →
  Frankfurter → Fawaz for USD FX rates; Coinbase → CoinGecko → Kraken for BTC; computed
  cross-rates for INR/BDT/CAD pairs). It **never overwrites a good Lambda value**, and **must
  reconcile `_meta` after backfilling** (drop the stale "unavailable: N" note, recompute
  `hasErrors` + the count from what's *still* missing). If you skip this, the page-bottom
  status footer shows red even when every metric was filled.
- The footer (`dashboard/app/page.js`, `.system-status-bar`) colors each source from its
  route's `_meta`/`source` string: red if `hasErrors`/Failed/Static/Stale, yellow for
  degraded sources (Stooq/FRED proxies/VIX proxy/cached), green otherwise. **Green must mean
  the data is actually healthy** — don't hardcode a status; derive it. (Note: `buildSpy` in
  `/api/spy` returns `hasErrors:false` on a *full validated build* — a degraded source is
  conveyed by the source label only; hard-coding `hasErrors:true` once made SPY perpetually red.)

### FRED route specifics (`/api/fred`) — subtle, don't regress
- The route uses `export const fetchCache = 'default-cache'` and stays dynamic by reading
  the request, so Next's Data Cache via per-fetch `revalidate` works (don't switch it to
  `force-dynamic` — that would force `no-store` and disable the caching below). FRED
  upstream + the P/E scrapes are cached **30 min** (`REVALIDATE_SECONDS = 1800`). It fetches
  17 FRED series in batches of 4 with a 150 ms stagger, plus a layered P/E scrape
  (multpl → Yahoo key-statistics → FRED CAPE `PE10`).
- FRED's free API ~120 req/min + shared-IP throttle → random `429`s. Retry `429` with backoff
  `[400, 900, 1800] ms`; **do not cache a final failure** (a thrown final 429 is not cached,
  so the next load retries). Mask any `api_key=…` in `_meta.messages` (`maskKey`).
- FRED dates observations at the *start* of the period and publishes weeks late, so fresh
  series legitimately look old. Per-metric freshness deadlines (`FRED_FRESHNESS` in
  `lib/constants.js`, days): daily≈7, weekly≈14, monthly≈80, JOLTS≈110, quarterly≈200.
  **Late-month monthlies are 95** (`UMCSENT`, `M2SL`, `DGORDER`, `PSAVERT`) — their free FRED
  series only print ~the 26th of the following month, so the newest point ages to ~85d before
  the next print; 80 false-alarmed them N/A for ~a week each month. Don't drop these to 80.
- **Stale ≠ N/A (graceful staleness).** A value past its deadline is NO LONGER nulled: it keeps
  showing as the **last-known value in orange with a 🕐 clock** ("As of <date> (stale)").
  `value` goes `null` (→ true N/A, yellow) ONLY when the fetch returns nothing
  (`unavailable:true`); `stale:true` means "had data, too old". `withFreshness` also returns
  `staleDays` (whole days past deadline) and `freshnessNote` returns `tone`
  (`fresh|stale|unavailable`). The health check (`scripts/health_check.py:check_indicators_na`)
  warns only when a metric is `unavailable` OR `staleDays > 3` (genuinely overdue) — normal
  reporting lag never alarms — and it sweeps BOTH `indicators` and `checklist`. A genuinely
  discontinued series keeps showing old orange data and warns daily until replaced or added to
  `KNOWN_DISCONTINUED`. `isGood` still keeps the payload as long as **≥1 series loaded**
  (`loadedCount > 0`); a total 0/17 outage falls through to last-known-good.
- **Three-tier FRED fallback (never N/A):** live FRED → `/tmp` last-known-good (`serve()`, 7d) →
  **Google-Sheet last-resort** (`lib/sheetLkg.js`, via `serve()`'s `lastResort` opt) → error. The
  last tier reads the `dashboard_lkg` helper tab of the financial-dashboard-history sheet
  (`EXTERNAL_URLS.SHEET_LKG`, public `export?format=csv&gid=` — NOT gviz, which merges the header
  row) only when the first two are BOTH gone (total outage on a cold instance). It's a
  self-describing `key,value` tab the **scraper** writes each run (values + asOf only, no chart
  history, N/A metrics omitted); the reader reconstructs an `/api/fred`-shaped payload with empty
  history, every metric `stale:true staleDays:4`, and `_meta.stale/hasErrors:true` so the health
  check still alerts. `fetchSheetLkg` never throws (returns null → falls to the error default).
  `?_fail=sheetlkg` disables it in fault tests. The helper tab is written by
  [[project-financial-dashboard-history]]'s `scraper.py` (`build_lkg_pairs`/`write_helper_tab`).
- **Copper/Gold ratio** (the `indicators.copperGold` tile) replaced the old `LEI`/`USSLIND`
  series, which FRED **discontinued/froze in 2020**. It is a leading growth/rates gauge
  (~1.4 = copper $/lb ÷ gold $/oz ×1000). The tile shows the **level + its ~1-month and
  ~3-month change** (the trend is what matters, not the level). Because this route runs on
  **Vercel datacenter IPs where Yahoo + Stooq are blocked/JS-walled** (the old 2-source
  version went permanently N/A), each leg cascades through several independent,
  datacenter-reachable sources, all normalized to the same unit so the ratio stays
  consistent regardless of which answered (see `fetchCopperGold` + `copperSources`/
  `goldSources` in the FRED route, the pure cascade in `lib/copperGold.js`
  (`resolveLeg`/`buildCopperGold`), and the parsers `cnbcQuotes`/`cnbcHistory`/`goldApiSpot`/
  `polygonDaily`/`fredObservations`/`yahooChart` in `lib/sources.js`):
  - **Copper $/lb**: CNBC `@HG.1` (keyless, daily history) → FRED `PCOPPUSDM` (key; ÷2204.6226
    from $/tonne; monthly) → gold-api.com `HG` (keyless spot) → Yahoo `HG=F` (self-heal).
  - **Gold $/oz**: CNBC `@GC.1` (keyless, daily history) → Polygon `C:XAUUSD` (key) → FRED
    `GOLDPMGBD228NLBM` (key; daily) → gold-api.com `XAU` (keyless spot) → Yahoo `GC=F`.
  - The gold leg's Polygon source reads `process.env.POLYGON_KEY` (passed through from GET).
  CNBC + FRED give history → the 1mo/3mo delta; spot-only sources still give the level. A
  genuine N/A here (all sources down) is a real signal the daily health-check flags.
  **Verified live on production 2026-06-06** (curling the public dashboard): the default
  path resolves `copper:cnbc · gold:cnbc` → ratio ~1.44, ▲+9.9%/1mo, ▲+25.8%/3mo — so **CNBC
  is reachable from Vercel's datacenter** (no reorder needed). Fault-injection on prod also
  confirmed the fallbacks from the datacenter: `?_fail=cg_cnbc` → `copper:fred · gold:polygon`
  (full delta still computed), and `?_fail=cg_cnbc,cg_fred,cg_polygon` → `copper:goldapi ·
  gold:goldapi` (keyless spot, ratio shown without a delta).
- **S&P 500 EPS** (the `spEps` top-level key + the "🧾 S&P 500 EPS" card next to Profit
  Margin) shows trailing-12-month as-reported index EPS — the E in P/E. Three independent
  sources (owner's rule: 3+ so there is always a backup), resolved by the pure cascade in
  `lib/spEps.js` (`resolveSpEps`, fetched lazily in order, fresh level preferred, stale
  level served with `stale:true` → orange 🕐 rather than N/A, first history wins):
  - **multpl** `/s-p-500-earnings/table/by-month` (level + monthly history to 1871;
    **inflation-adjusted to current dollars**; as-reported TTM lags ~2-3 quarters, so its
    freshness window is 400d) →
  - **derived** = FRED `SP500` close ÷ the route's live TTM P/E (spot-only, daily-fresh;
    **refuses CAPE** — dividing by a 10-yr smoothed P/E wouldn't give TTM EPS; that's why
    the P/E block tracks `peSource`) →
  - **datahub** GitHub-raw mirror of Shiller's dataset (`Real Earnings` column, same units
    as multpl; its earnings run years behind → in practice the graceful-staleness fallback
    that keeps the CHART alive when multpl is down).
  History is the full MONTHLY series from 1947 (`toMonthlyHistory`, ~945 points), rendered
  with `<MiniChart cadence="monthly">` — an explicit mode (added for this card) whose tab
  row is 1Y/3Y/5Y/10Y/20Y/30Y/ALL with 12 points/yr math. Other MiniChart users keep the
  auto-detect (quarterly <500 points, else daily) — don't feed monthly data through
  auto-detect, it would misread it as daily/quarterly. The block is guarded like copper/gold: a failure appends a `_meta`
  message but never sets `hasErrors` and can't break the FRED payload. Note the deliberate
  contrast with `resolveLeg`: copper/gold REJECTS stale sources; EPS SERVES them marked
  stale, because an old real earnings number beats an N/A. (`spEps` is not in the
  `dashboard_lkg` sheet tab — in total-outage last-resort mode the card shows N/A.)
- **Volatility metrics** (`/api/vol` + `VolMetricsTable.js`, added 2026-07-05) — IV, IV
  rank (1y), IV percentile (1y), 21-day realized vol, and VRP (IV − RV) for **SPY, QQQ,
  TQQQ, SQQQ, UVXY**, rendered as the "🌡️ Volatility Metrics" card after SPY Historical.
  Dashboard-only, `serve('vol', …)`-wrapped, 30-min cached. **IV is an index PROXY, not
  chain-derived** (Yahoo chains are unreachable from Vercel; same method as the owner's
  hedgelab tool): SPY→VIX, QQQ→VXN, TQQQ/SQQQ→**3×VXN** (leverage scales IV ~linearly),
  UVXY→VVIX. Rank/percentile are computed on the UNSCALED index series (a constant
  multiplier changes neither), the displayed IV level is scaled — keep it that way. Pure
  math + payload builder in `lib/vol.js` (`parseCboeCsv` handles BOTH CBOE schemas:
  `DATE,OPEN,HIGH,LOW,CLOSE` for VIX/VXN and two-column `DATE,VVIX`). Sources:
  - **Indices** (3-4 tiers each, hardened 2026-07-05): CBOE CDN daily-history CSVs
    (`cdn.cboe.com/api/global/us_indices/daily_prices/<NAME>_History.csv`, keyless, full
    history, verified reachable) → **CNBC `.VIX`/`.VXN`/`.VVIX`** daily bars (keyless; the
    `3M` range actually returns ~2y — enough for the 1y window; verified live, values match
    CBOE) → FRED `VIXCLS`/`VXNCLS` (key; **VVIX has NO FRED series**) → Yahoo
    `^VIX`/`^VXN`/`^VVIX` (blocked from Vercel today; self-heal tier like copper/gold).
  - **ETF closes** (for RV21, 3 tiers): CNBC harmony `3M` daily bars (keyless) → Polygon
    daily aggs (`POLYGON_KEY`; free-tier day delay is immaterial for a 21-day window) →
    Yahoo chart (self-heal tier).
  Per-leg failures null the affected cells, never the payload. Fault gates are **per
  SOURCE** (tripping one disables it everywhere, like `cg_*`): **`vol_cboe` / `vol_cnbc` /
  `vol_fred` / `vol_polygon` / `vol_yahoo`**. `_meta.source` lists the winning source per
  series (e.g. `VIX:cboe · SPY:cnbc`). The endpoint is in the health-check's GET sweep.
  UI thresholds (hedgelab convention): percentile/rank ≤10 green (cheap), ≥70 orange,
  ≥90 red (panic); negative VRP orange (realized above implied = stress).
- The `?_fail=` fault-injection harness (`lib/faults.js`) is intentionally **kept in
  production** — it only degrades the caller's own response and never writes caches.
  **Fault names:** `lambda`, `polygon`, `finnhub`, `yahoo`, `gamma`, `coinbase`, `coingecko`,
  `kraken`, `erapi`, `frankfurter`, `fawaz` (per-source gates in the relevant routes);
  `fred` (forces an invalid FRED key → all FRED *series* fail, exercising last-known-good;
  note this does **not** disable the copper/gold FRED leg — that's `cg_fred`); `lastgood`
  (in `serve()`: also skip reading last-known-good, to reach the safe default); and the
  per-source copper/gold gates **`cg_cnbc` / `cg_fred` / `cg_polygon` / `cg_goldapi` /
  `cg_yahoo`** (each disables that source in **both** legs). E.g. `?_fail=cg_cnbc` forces both
  legs past CNBC; `?_fail=cg_cnbc,cg_fred,cg_polygon` forces them down to the keyless gold-api
  spot tier. The served object's `tried` field shows the per-leg cascade trail (e.g.
  `cnbc:off → polygon:off → fred:ok`). The S&P EPS cascade has the same per-source gates:
  **`eps_multpl` / `eps_derived` / `eps_datahub`** (e.g. `?_fail=eps_multpl` → derived serves
  the fresh level and datahub the chart; `spEps.tried` shows the trail).

---

## 4. Known issues / open items (updated 2026-06-08)

**✅ Resolved 2026-06-01** (kept here so the history is legible):
- `bot/config.py` `URLS` now has `SPY_DAILY_MOVE`, `SPY_INDICATORS`, `STOOQ_SPY` (mirrored
  from `dashboard/lib/constants.js`) — the Lambda SPY fallback tiers no longer `KeyError`.
  *Note:* Stooq's `q/d/l` download endpoint now gates behind an apikey, so `STOOQ_SPY` is a
  graceful **dead fallback** (the yfinance/Polygon/Finnhub tiers cover SPY).
- Silent daily-report failure fixed: `bot/main.py` `run_report()` returns `False` on empty
  content / failed send and `__main__` `sys.exit(1)` → green == sent, retry harness fires.
- Telegram delivery hardened: `send_to_telegram` retries as plain text on a Markdown 400 and
  chunks >4096 chars (`_post_telegram_text` / `_split_message`).
- The discontinued LEI tile was replaced by the multi-sourced **Copper/Gold ratio** (§3).

**Still open (needs the owner):**
- **Secrets:** the leaked RapidAPI key (in git history) must still be **rotated**; the old
  root AWS access keys should be deactivated/deleted (CI uses the scoped `github-deploy-bot`).
- **Delivery monitoring:** grant `github-deploy-bot` `logs:FilterLogEvents` on
  `/aws/lambda/financial-telegram-report` so the health-check can confirm delivery from
  CloudWatch. Until then `report_delivered_today` safely degrades to a `warn` ("could not
  confirm") — never a false `critical`.

**Stale-config note (verify before touching, don't trust blindly):**
- `bot/config.py` `FRED_SERIES` still lists `'LEI': 'USSLIND'`, and the prober's
  `KNOWN_DISCONTINUED` allowlist still includes `lei`. The dashboard FRED route no longer
  serves LEI (replaced by copper/gold), but these references linger. Remove `lei` from
  `KNOWN_DISCONTINUED` only once nothing surfaces an `lei` indicator.

## 5. Never-break checklist (before you commit)
1. Did you sanitize pandas/math floats (no bare `NaN`) before any API response (`_ok()` /
   `serve()`)?
2. Do yfinance calls still cache to `/tmp`?
3. Are Lambda deps built for `manylinux2014_x86_64` (i.e., changes go through CI, not a
   local zip)?
4. Does SPY still override stale Polygon spot with live Finnhub?
5. Is every new/changed cached `/api/*` route still never-throw (whole body inside
   `serve()`), returning 200 + valid JSON, with no hardcoded secrets?
6. Did you keep `bot/` lite and put new UI in `dashboard/components/`?
7. If you changed Polymarket curation, did you mirror it in **both** `bot/fetchers.py` and
   `dashboard/app/api/polymarket/route.js`?
8. For backend changes: did the **Deploy to AWS Lambda** run go green (incl. the API
   Gateway smoke test)? For dashboard changes: does `npm test` + `npm run build` pass? For
   any change: does `ci.yml` (pytest + jest + build) pass — that's the merge gate.

## 6. Map / quick reference

**Backend (Python — Lambda + runner + bot package)**
- `lambda_handler.py` — Lambda entry; `_clean_nans`, `_ok`/`_err`, `handle_http_api`
  (dashboard GET routes), `handle_eventbridge` (daily report). Dispatches HTTP vs schedule.
- `bot/fetchers.py` — the data waterfalls: `fetch_google_sheet_indicators` (Telegram report
  body), `fetch_spy_with_fallback` (yfinance→Polygon→Sheet→Stooq→FRED, Finnhub spot override),
  `fetch_spy_daily_move`, `fetch_market_extra` (FX/commodities/rates/real-estate), and
  `fetch_polymarket_trending` (the curated sentiment board). `fetch_spy_stats`/`calculate_rsi`
  are legacy Stooq helpers.
- `bot/config.py` — `URLS` (Google-Sheet/data source URLs, mirror `dashboard/lib/constants.js`)
  + `FRED_SERIES` IDs + `TIMEZONE`/`REPORT_TIME` (for the `bot/main.py` scheduler).
- `bot/utils.py` — env loading (`load_environment_variables`, requires
  `FRED_API_KEY`/`TELEGRAM_TOKEN`/`TELEGRAM_CHAT_ID`), `send_to_telegram` (Markdown→plain
  retry + >4096 chunking), `report_marker` (the `REPORT_DELIVERED`/`REPORT_FAILED` CloudWatch line).
- `bot/main.py` — local/runner entry: `python -m bot.main report` → `run_report()`
  (Google-Sheet text only; SPY summary commented out). Bare `python -m bot.main` runs a
  Flask health server + APScheduler + a Telegram polling bot (`/report`, `/start`) — the
  long-running mode used by Render (`render.yaml`, `startCommand: python -m bot.main`).
- `bot/assessment.py` — rule-based + multi-LLM macro assessment (used by the bot path; the
  dashboard has its own `/api/assessment`). Not on the Lambda's daily path.
- `aws/template.yaml` — SAM template (reference only; **not** applied by CI — see §2).
- `aws/requirements-lambda.txt` — Lambda deps (CI). `aws/requirements-lambda-minimal.txt` —
  pure-Python subset (reference). `requirements.txt` — local/runner. `requirements-dev.txt`
  — adds `pytest`. `runtime.txt` — `python-3.11.10`.

**Dashboard (`/dashboard`, Next.js)**
- `dashboard/app/api/*/route.js` — per-metric routes:
  - Lambda-primary + never-throw via `serve()`: `spy`, `spy-daily-move`, `market-extra`,
    `polymarket`.
  - Dashboard-only: `fred` (also `serve()`-wrapped; FRED series + P/E + copper/gold),
    `sheets` (Google-Sheet brief, layered cache), `fear-greed` (CNN→RapidAPI→Yahoo VIX→FRED
    VIXCLS→cache), `last-run` (GitHub Actions status of `daily_report.yml`),
    `assessment` (**POST-only** LLM macro summary).
- `dashboard/lib/{store,sources,fetcher,faults,freshness,finance,copperGold,spEps,constants}.js` —
  the never-throw store + last-known-good, direct data sources, fetch/proxy helpers, fault
  injection, FRED freshness, math (RSI/dailyChange/copperGoldRatio), copper/gold cascade,
  S&P 500 EPS parsers + cascade, and shared constants (FRED IDs, freshness deadlines, URLs).
- `dashboard/app/page.js` — dashboard page + the `.system-status-bar` footer.
- `dashboard/components/*.js` — UI (MarketModal, PolymarketTable, SpyChart, Gauge,
  EconomicIndicatorGrid, BullChecklist, ExtraMarketsGrid, MarketPulse, MiniChart,
  CustomIndicatorBar, Skeleton, ErrorBoundary, MarketModal.example).
- `dashboard/{jest.config.js,jest.setup.js,next.config.js,package.json}` — build/test config.

**Self-healing / ops (scripts + workflows)**
- `scripts/health_check.py` — the daily self-check prober (see §7); pure check functions +
  HTTP probes + `--notify`/`--summary`.
- `scripts/collect_health_context.py` — distills recent health reports into the weekly
  agent's digest (`health-digest.md`).
- `tests/` — pytest: `test_polymarket_fetcher`, `test_run_report`, `test_utils`,
  `test_health_check`, `test_collect_health_context`. Dashboard Jest tests live under
  `dashboard/**/__tests__/`.
- `.github/workflows/` — `deploy-lambda.yml` (auto-deploys the Lambda on `bot/**` /
  `lambda_handler.py` changes), `daily_report.yml` (09:45 UTC runner-based report backstop +
  CloudWatch skip-guard + self-retry harness), `ci.yml` (pytest + jest + build on every PR —
  the merge gate), `health-check.yml` (14:00 UTC daily self-check + Telegram alert),
  `self-improve.yml` (weekly Wed 13:00 UTC agent that opens PRs — see §7), and
  `diagnose-lambda.yml` (manual on-demand: dumps the Lambda's recent CloudWatch invocations +
  outcomes to answer "why didn't the report send?"; read-only).
- `.github/self-improve-prompt.md` — the guardrailed prompt the weekly agent runs (kept; it
  is `cat`-ed by `self-improve.yml`).
- `.env.example` / `setup_env.sh` — local env scaffolding (human-facing).
- `README.md` — human/GitHub landing page (kept).

---

## 7. The self-healing health-check — and how to read its alerts

The repo monitors itself. **Daily** (`health-check.yml`, **14:00 UTC**) a pure-code prober
(`scripts/health_check.py`) checks the LIVE system and, on any warn/critical, sends the owner
a Telegram alert **written to be pasted straight into a Claude session** (a plain-English
"💬" line per finding + an ELI10 BOTTOM LINE). On a **manual** (`workflow_dispatch`) run it
always posts — a clean ✅ green check (`--summary`) or the alert; on a **scheduled** run it is
silent when green and alerts only on warn/critical (`--notify`). **Weekly**
(`self-improve.yml`, Wed 13:00 UTC) a headless Claude agent reads the recent reports + this
file and opens PRs the owner approves (it never self-merges; branch protection enforces it).

### How detection works (deterministic, $0)
`scripts/health_check.py` runs all checks, each yielding a finding
`{id, severity (ok|warn|critical), title, detail, remediation, evidence}`; `overall` = worst
severity. It is **never-throw** — a broken probe becomes a finding, not a crash. Checks:
- **`endpoint_<name>`** — HTTP-probes the live Vercel routes (`spy`, `spy-daily-move`,
  `market-extra`, `polymarket`, `fred`, `sheets`, `fear-greed`; **`assessment` is excluded**
  because it's POST-only). 200 + valid JSON + no bare `NaN`/`Infinity` + `_meta.hasErrors!=true`.
  It **warms up** the stack once, then retries each probe with backoff
  (`PROBE_ATTEMPTS=3`, `PROBE_BACKOFF=(3,10,20)`, `PROBE_TIMEOUT=45`), and **retries a
  self-reported-degraded 200** too — so a cold start / slow first load / momentary blip is
  never mistaken for an outage.
- **`indicators_na`** — inspects `/api/fred`'s `indicators`; an unexpected `null`/`unavailable`
  is `warn`; a metric on the `KNOWN_DISCONTINUED` allowlist (currently `lei`) is expected and
  never alarmed.
- **`report_delivered_today`** — from delivery evidence gathered by the workflow (CloudWatch
  `REPORT_DELIVERED`/`REPORT_FAILED` markers in the last 24 h + whether `daily_report.yml`
  succeeded today). `REPORT_FAILED` or no marker (and CloudWatch readable) → `critical` +
  `remediation: auto:redispatch_daily_report`; marker present or a GHA success → `ok`;
  CloudWatch unreadable with no corroboration → `warn` ("could not confirm"), never a false
  critical.
- **`known_issue_config_urls`**, **`secret_leak`** (gitleaks), **`ci_health`** (latest run
  per **active** workflow) — see the table below.

### When the owner pastes you a health alert, act on it like this
Each finding has an `id`. Map id → meaning → fix:

| Finding `id` | What it means | How to act |
|---|---|---|
| `report_delivered_today` | Couldn't confirm today's Telegram report went out | `cloudwatch_readable:false` → the `logs:FilterLogEvents` IAM grant is missing (owner action), **not a real outage**. A `REPORT_FAILED` marker → a real send failure: read the Lambda's CloudWatch logs; the daily run may already have re-dispatched `daily_report.yml`. |
| `endpoint_<name>` | `/api/<name>` returned non-200, invalid JSON, a bare `NaN`, or `_meta.hasErrors` | Hit the live URL, read `_meta.messages`; fix per §3 (never-throw, sanitize NaN). Slow first loads + transient degradation are already retried, so a flagged endpoint is genuinely failing. |
| `indicators_na` | A dashboard indicator is N/A **unexpectedly** | `detail` names the metric; repair/extend its fallback. Known-discontinued metrics are allowlisted (`KNOWN_DISCONTINUED`) and never alarmed. |
| `known_issue_config_urls` | `bot/config.py` `URLS` missing a required key | Add the key — the URLs live in `dashboard/lib/constants.js`. |
| `secret_leak` | gitleaks found a credential in the repo | **Rotate it immediately**, then remove the literal (env vars only). |
| `ci_health` | An **active** workflow's **latest** run failed | Open that run, read the failure, fix + PR. Historical/fixed failures and deleted workflows are already excluded. |

### Conventions
- Severity `ok` < `warn` < `critical`; a scheduled alert lists **only non-ok** findings. A
  `remediation: auto:redispatch_daily_report` tag means the daily workflow already retried.
- **History = artifacts, not commits.** Each daily run uploads `health-report.json` (90-day
  retention); `health-check.yml` runs with `contents: read` and does **not** push to `main`
  (branch protection blocks bot pushes). The weekly agent (`self-improve.yml`) downloads the
  last ~10 health-check run artifacts into `health/history/` and runs
  `scripts/collect_health_context.py` to build `health-digest.md`.
- Delivery uses a **24h rolling CloudWatch window** for the Lambda's
  `REPORT_DELIVERED`/`REPORT_FAILED` markers (`bot/utils.report_marker`), cross-checked with
  the `daily_report.yml` run status.

### The weekly self-improve agent (`self-improve.yml`)
- Trigger: cron `0 13 * * 3` (Wed 13:00 UTC) + `workflow_dispatch` (owner's override).
  `timeout-minutes: 30`. Permissions: `contents: write` (feature branch only — NOT main),
  `pull-requests: write`, `actions: read` (to download health artifacts).
- It installs the **Claude CLI** (`npm i -g @anthropic-ai/claude-code`) and runs
  `claude -p "<prompt + this week's digest>" --model claude-opus-4-8 --max-turns 15
  --allowedTools "Read,Edit,Write,Grep,Glob" --permission-mode acceptEdits` using the
  `ANTHROPIC_API_KEY` secret. **The agent only edits files** — it has no shell/git/test tools.
  *(This differs from the old Phase-2 plan, which proposed `anthropics/claude-code-action@v1`;
  the live workflow uses the CLI + a separate PR step instead — trust the workflow.)*
- A subsequent workflow step does the git work: if the agent changed nothing → Telegram
  "nothing to fix this week"; otherwise it runs `pytest` for a signal, creates branch
  `self-improve/run-<run_id>`, commits `[skip ci]`, pushes, opens a PR via `gh pr create`,
  and posts a Telegram summary with the PR link. It **never merges**; the owner reviews + merges.
- The agent's hard rules (`.github/self-improve-prompt.md`): be conservative (a clean no-op is
  success), one small single-concern change, cross-check this AGENTS.md first, and **never
  touch** secrets/keys, `aws/template.yaml`, live AWS config, `.env`, or `.github/workflows/`.
- **To fix anything here:** branch → fix + test → PR (CI gates it) → owner merges. Never push
  to `main`. Roll back via a PR's **Revert** button or a `known-good-*` git tag.
