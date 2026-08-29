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
console (or extend the workflow) and update the template to match. Live schedule (verified
against the actual rule 2026-08-06): EventBridge rule `daily-financial-report-trigger`,
`cron(15 8 * * ? *)` = **08:15 UTC**, matching the template. (This section used to claim
the live rule fired ~09:15 UTC — that was wrong. The downstream workflow crons at 09:45
and 14:00 UTC still work fine with an 08:15 send; they only assume "before 09:45".)

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

### Four Lambda gotchas that have caused real outages
0. **Recreating the Lambda wipes its resource policy — re-grant EventBridge, not just
   API Gateway.** The function was recreated 2026-06-01; the API Gateway invoke grants
   were restored (see "Wiring a new API Gateway" below) but the EventBridge grant was
   forgotten. Result: the schedule rule fired every morning and got AccessDenied at the
   Lambda — **1 silent `FailedInvocations`/day for 2 months** (the GHA 09:45 backstop
   masked it; fixed 2026-08-06). EventBridge invoke failures produce NO CloudWatch logs
   on the Lambda — check the `AWS/Events` `FailedInvocations` metric for the rule. Re-grant:
   ```bash
   aws lambda add-permission --function-name financial-telegram-report \
     --statement-id AllowEventBridgeDailyReport --action lambda:InvokeFunction \
     --principal events.amazonaws.com \
     --source-arn "arn:aws:events:us-east-1:463256610967:rule/daily-financial-report-trigger"
   ```
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
- `FRED_API_KEY` — used by `/api/fred`, `/api/sheets` (Layer 4's VIX/sentiment proxy, AND
  now the primary source for the VIX pill's fear/greed TAG — see below), `/api/fear-greed`
  (VIXCLS), and the copper/gold legs.
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
- **Every new GET route handler must "touch the request" first** —
  `request.headers.get('user-agent');` as the opening line (see fred/market-extra/vol).
  Without it Next statically prerenders the handler at BUILD time: the payload freezes
  and query params (incl. `?_fail=`) are silently ignored on production. `faultsFrom`
  alone does NOT make a route dynamic (its try/catch swallows Next's DynamicServerError
  probe — this bit /api/vol on 2026-07-05). Verify in `npm run build` output: the route
  must be listed as `λ` (server), never `○` (static).
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

### VIX pill fear/greed tag (`/api/sheets` + `lib/vixFearGreed.js`)
The VIX pill in `CustomIndicatorBar.js` shows a `current | threeMonth | fearGreed` triple
(e.g. "14.43 | 17.48 | GREED13"). `current`/`threeMonth` still come straight from the
Google Sheet (`GOOGLE_SHEETS.VIX`, gid `790638481`, cells A2/B2). `fearGreed` (the
"GREED13"-style tag) has been written into that sheet's **C2** once a day by a separate
repo, **vix-fear-greed**, which the owner is retiring — the computation (added
2026-08-28) is now folded into the dashboard itself so the pill keeps working once that
repo is deleted (it may still exist and still be writing C2 today; treat it as gone).
- **Formula** (`lib/vixFearGreed.js: fearGreedTag`, matches the retired repo's
  `fear_greed.py` exactly): `sma50` = 50-day rolling mean of VIX daily closes, `latest` =
  most recent close, `pct_diff = (latest - sma50) / sma50`, `score =
  round(abs(pct_diff)*100)` capped at 99 and zero-padded to 2 digits, tag = `FEAR<score>`
  if `pct_diff > 0`, `GREED<score>` if `pct_diff < 0`, else `NEUTRAL00`. Only the trailing
  50 valid closes matter (a rolling mean's last value depends on nothing earlier), so
  `computeVixFearGreedTag` fetches FRED `VIXCLS`'s newest ~280 observations (plenty of
  buffer for the odd missing `.` print) rather than a literal 1-year window.
- **Source cascade** (`resolveVixFearGreedTag`): FRED-computed (primary) → the sheet's C2
  value already parsed by whichever of the 5 sheets-cascade layers won (transition
  fallback, for as long as the retired repo's last-written value is still sitting in the
  sheet) → `'N/A'`. **Which source won is always named in `_meta.messages`** — the
  fallback never silently looks identical to a healthy computed reading (§7's governing
  principle: a fallback that satisfies the caller unnoticed is a false negative). Verify
  with `?_fail=vix_fred` (forces the FRED path to fail, exercising the sheet fallback) on
  prod after any change here.
- **Don't confuse this with `/api/fear-greed`** — that route computes CNN's unrelated
  0–100 "EXTREME FEAR…EXTREME GREED" index (a different gauge, feeding a different
  component) and shares no code with this tag despite the similar name.
- Once vix-fear-greed is fully deleted and C2 stops being refreshed, the sheet fallback
  tier becomes permanently stale — harmless to leave as a dead tier (it degrades to
  `'N/A'` once the cell goes empty) rather than ripping it out, unless it gets confusing.

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

### ✦ Fresh Print Marks (`/api/history` + `Delta.js` + `lib/marks.js`)

Marks a number whose change is NEWS, so the owner can scan the page and see what moved.
Design doc: `docs/superpowers/specs/2026-08-25-fresh-print-marks-design.md`.

**The rule is CADENCE, not diff.** Measured over 167 days of the history sheet, the columns
split cleanly: 17 change <0.2×/day (they only move when an agency publishes, so a change IS
the news → `●` print mark), 5 change 0.35–0.67×/day (only a 2σ day is news → `⌃`/`⌄` move
mark), the rest change daily (never marked). The measured rate sits beside every entry in
`SHEET_METRICS` as its justification. **Do not derive the class at runtime** — a series
frozen by a dead upstream looks "slow" and would then fire loudly on recovery.
Result: **52% of days nothing lights at all.** That silence is the feature.

**Two clocks, and they must not be confused.** `/api/history` reads the sheet's DAILY
SNAPSHOT and returns only what history can know (`baseline`, `heldFrom`, `runs`, `sigma`).
`markFor()` compares that against the **live** value. Deciding the mark inside the digest is
wrong twice over: the sheet is a 10am/10pm snapshot, so a print landing at 8:30am would go
unmarked for hours, and before today's row exists the comparison degenerates to a row
against itself.

**Five traps, all of which bit during implementation:**
1. **Precision.** The scraper writes rounded values (`-0.56`); `/api/fred` returns full
   precision (`-0.559`). Comparing them directly lit SIX metrics every single day, all
   false. Compare at the sheet's stored precision (`decimalsOf` → `compareDp`, floored at
   2 dp). Regression-tested with the real observed pairs.
2. **Timezone.** `todayET()` is pinned to `America/New_York`. The route runs on Vercel in
   UTC while the sheet's dates come from an ET cron — a naive `new Date()` picks the wrong
   baseline every evening between 8pm and midnight.
3. **Stacking context.** The popover MUST be portalled to `document.body` with
   `position: fixed`. `.card` sets `backdrop-filter`, which creates a stacking context; an
   absolutely-positioned popover is trapped inside it and the next card paints over it. A
   test asserts the popover is not a descendant of its trigger.
4. **Chip vs. page.** The cards refuse to mark a stale/unavailable value; `collectLiveValues`
   applies the SAME `fresh()` filter. Without it the header chip claimed "4 new prints"
   above a page with no marks on it (seen on a Sheet-LKG load).
5. **Unit + sentinel changes in the history.** 2026-03-18 switched units
   (`212000 → 212`); rows before 2026-05-08 use a bare `0` as a missing sentinel. Both are
   rejected (`isUnitJump`, `parseValue`) or they light false marks.

**NOT markable, deliberately:** `spEps`, `horsemen.unemployment/bankruptcies/claims`.
`/api/fred` ships full `history[]` for them, but that is a list of OBSERVATIONS, not daily
snapshots — its newest point IS the current value, so a mark could never fire, and an
observation is dated to its period and published weeks later so its date says nothing about
when the print arrived. Marking them needs four new far-right scraper columns.

**Failure mode is invisible.** If `/api/history` fails, the digest is empty, no marks render,
and every number reads exactly as it does today. A wrong mark is far worse than a missing one,
so `isGood` rejects an empty digest rather than letting it claim "nothing changed".

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
  **Corporate profits (`A053RC1Q027SBEA`, the Profit Margin card) is 250, not 200** — same
  class of mis-tuning. FRED dates a quarter at its START and BEA publishes corporate profits
  with the GDP **2nd estimate ~2 months after the quarter ENDS**, so Q2 2026 (dated
  `2026-04-01`) only prints ~2026-08-27 and then stays the newest point until Q3 prints
  ~2026-11-25 — ageing to **238 days while being completely current**. At 200 the card was
  wrongly orange 🕐 for ~5 weeks of every quarter (observed live 2026-08-06: newest point
  `2026-01-01`, 217 days old). 250 = the 238-day worst case + slack, and still catches a feed
  that misses a whole print. This mattered more once the card entered the N/A sweep below:
  a too-tight deadline stopped being a cosmetic 🕐 and became a daily false `warn`.
- **Stale ≠ N/A (graceful staleness).** A value past its deadline is NO LONGER nulled: it keeps
  showing as the **last-known value in orange with a 🕐 clock** ("As of <date> (stale)").
  `value` goes `null` (→ true N/A, yellow) ONLY when the fetch returns nothing
  (`unavailable:true`); `stale:true` means "had data, too old". `withFreshness` also returns
  `staleDays` (whole days past deadline) and `freshnessNote` returns `tone`
  (`fresh|stale|unavailable`). The health check (`scripts/health_check.py:check_indicators_na`)
  warns only when a metric is `unavailable` OR `staleDays > 3` (genuinely overdue) — normal
  reporting lag never alarms — and it sweeps `indicators`, `checklist`, the `horsemen`
  block AND the **top-level cards `yieldCurve` / `profitMargin` / `spEps`**
  (`fred_metrics_for_na_check` maps `current`→`value` for both groups, so e.g. a dead
  uscourts bankruptcies feed eventually warns as `horsemen_bankruptcies`). The top-level
  three were OUTSIDE the sweep until 2026-08-06, which mattered most for **`yieldCurve`,
  because it IS the 10Y−2Y horseman** (the card reuses `fred.yieldCurve` rather than
  duplicating it): if FRED froze `T10Y2Y` and the repair cascade adopted nothing, `failed`
  stayed empty, `hasErrors` stayed false and `endpoint_fred` stayed green **forever**, while
  claims and unemployment — the two lines drawn right beside it — would have warned via
  `staleDays > 3`. **When building these cards, SPREAD the whole `withFreshness` result
  (`{...yc, current: yc.value}`) — never cherry-pick.** `buildResponse` used to hand-pick
  `{current, asOf, stale, date, history}` and DROP `staleDays`/`unavailable`, which is
  exactly what made them invisible to the sweep; the same applies to the repaired spread in
  `repairHorsemen` and to `reconstructFred` in `lib/sheetLkg.js`. `resolveSpEps` now also
  returns `staleDays` for the same reason (graceful staleness is deliberate there, but it
  must still be VISIBLE — multpl and derived both dying, leaving datahub to serve a level
  years behind, is a real break, not reporting lag). `peRatio` is deliberately NOT in the
  sweep — `check_pe_source` reports the more specific CAPE-for-TTM substitution. A genuinely
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
  - **Live intraday overrides (added 2026-07-15)**: one keyless CNBC quote call
    (`.VIX`/`.VXN`/`.VVIX`, 5-min revalidate, gated by `vol_cnbc`) feeds
    `buildVolMetrics` a live "current" level that replaces the last EOD close ONLY
    when finite, > 0, and its date is a well-formed `YYYY-MM-DD` strictly newer than
    the last EOD point — rank/%ile still use the EOD 1y window (UNSCALED), the live
    quote never enters the RV 21d closes (CNBC daily bars verified EOD-only at the
    2026-07-15 open — no partial today-bar), VRP = live IV − EOD RV. Rows gain `live`, payload
    gains `live_at` (full ISO or null — a date-only quote timestamp is withheld so
    the UI can't misparse it as UTC midnight), sources show `VIX:cboe+live`, and the
    card footnote shows a green dot + ET time ("As of 2026-07-15, 1:42 PM ET ·
    intraday"). Off-hours / quote failure ⇒ identical to pre-2026-07-15 EOD behavior
    (never-throw kept). Values verified against CBOE's own delayed quotes 2026-07-15.
  Per-leg failures null the affected cells, never the payload. Fault gates are **per
  SOURCE** (tripping one disables it everywhere, like `cg_*`): **`vol_cboe` / `vol_cnbc` /
  `vol_fred` / `vol_polygon` / `vol_yahoo`**. `_meta.source` lists the winning source per
  series (e.g. `VIX:cboe · SPY:cnbc`). The endpoint is in the health-check's GET sweep.
  - **Staleness gate + honest `hasErrors` (added 2026-08-06).** Both cascades now go
    through `lib/vol.js: resolveVolSeries`, which is the SAME contract as copper/gold's
    `resolveLeg` — skip fault-injected sources, reject empty ones, and **reject any series
    whose newest point is older than `VOL_FRESHNESS_DAYS` (7)**, falling through to the next
    tier with a `tried` trail (`cboe:stale(2026-05-01) → cnbc:ok`). Before this the cascade
    accepted any array with `length > 0`, so a frozen CBOE CSV or a CNBC endpoint replaying
    an old window would win tier 1 forever and the table would present **months-old vol as
    today's number**. Everything here is a DAILY series, so 7 days = a long holiday weekend
    (same reasoning as `FRED_FRESHNESS.T10Y2Y`); Polygon's tier gets `freshnessDays: 10`
    because its free tier is a day behind BY DESIGN (§2 gotcha #3). A served value is
    therefore never stale — null is the only way a cell can be wrong.
    `_meta.hasErrors` is now `volIncompleteTickers(tickers).length > 0` — **any ticker
    missing `iv` OR `rv21`**. The old test was `hasErrors: !anyData`, which only tripped
    when EVERY cell of EVERY row was null: a permanently dead **VVIX** cascade (VVIX has
    **no FRED tier**, so it is the most fragile of the three indices) nulled UVXY's entire
    row while the endpoint still reported itself green. `/api/vol` gets no equivalent of
    `check_indicators_na`, so this `_meta` is the ONLY signal the health check has — it has
    to tell the truth. `_meta` also carries `incomplete` (the ticker names) and `tried`
    (the full per-series cascade trail).
  UI thresholds (hedgelab convention): percentile/rank ≤10 green (cheap), ≥70 orange,
  ≥90 red (panic); negative VRP orange (realized above implied = stress).
- **Four Horsemen — Recession Watch** (`FourHorsemen.js`, full-width card after the
  Economic Indicators grid, added 2026-07-23) — the classic "Four Horsemen of the
  Apocalypse" chart as ONE overlay (owner explicitly wanted the overlay, not small
  multiples): **Initial Jobless Claims** (ICSA, weekly, red), **Unemployment Rate**
  (UNRATE, monthly, green), **10Y−2Y spread** (reuses `fred.yieldCurve` — NOT duplicated
  in the payload, blue), and **US Bankruptcies** (quarterly, non-FRED — see below, light
  gray), with NBER recession shading (`fred.recessions`) behind all four. The units are
  incomparable, so each series is min-max normalized into its own (slightly overlapping)
  vertical band — **log-scaled for the three positive series** (linear flattens 40 years
  under the 2020 claims spike), linear for the spread (crosses zero; dashed
  inversion line at 0). Custom SVG in the component (not MiniChart): shared timeframe
  tabs (ALL=1979→ / 20Y / 10Y / 5Y / 1Y), series thinned to ≤1500 points, inline boxed
  labels pinned to each line, and hand-annotation-style direction notes at each line's
  right end (`trendOf`: sign of a least-squares trendline over the last 12 months of the
  RAW history, computed once in the parent and passed into the overlay — NEVER from the
  thinned/zoomed chart data, which once made the verdict flip between zoom tabs and
  devices; "flat" = fitted yearly change < 2% of the mean (count series) or < 0.08 pts
  (rate series); labels state the window ("· 1y"); flat claims reads "watch this line"). Stat chips above the chart double as
  the legend and carry the numbers/YoY/status; warn badges: claims YoY > +10%,
  Sahm ≥ 0.5, spread < 0, bankruptcies YoY > +10%; header badge counts "N of 4 riding".
  All data rides on the `/api/fred` payload's `horsemen` block; no new endpoint. Because
  the card needs full histories, the ICSA and UNRATE `FRED_REQUESTS` limits are `100000`
  — which is WHY `unrate12moLow` (Sahm rule) must stay `unrate.slice(0, 12)`: a min over
  the full history would break it. (MiniChart kept its `weekly` cadence /
  `defaultTimeframe` / `fmt` props from the first iteration — unused by this card now
  but tested and harmless.)
  - **Mobile (below 640px, via `matchMedia`; SSR/jsdom default to the wide variant):**
    the overlay switches to `OVERLAY_DIMS.compact` — a narrower/TALLER SVG canvas
    (720×800 vs 1200×430) with proportionally larger type, thicker strokes, `shortLabel`
    series names, `LABEL_AT_COMPACT` staggering (pulls labels left, away from the
    right-edge direction notes), fewer year ticks, and ≤700 points. Without this the
    wide viewBox scales down to an unreadable ~135px-tall sliver on phones. The stat
    chips are an EXPLICIT `minmax(0, 1fr)` grid (2×2 phone / 4-across desktop) — do not
    "simplify" it back to `repeat(auto-fit, minmax(150px, 1fr))`: auto-fit's intrinsic
    sizing let long chip content widen the whole card past the viewport.
  - **Visual QA recipe (works for any dashboard component):** bundle the component
    standalone with esbuild — `NODE_PATH=dashboard/node_modules npx esbuild harness.jsx
    --bundle --jsx=automatic --loader:.js=jsx --define:process.env.NODE_ENV='"production"'`
    (`--jsx=automatic` is required; Next injects the JSX runtime, so component files
    never import React) — into an HTML shell that inlines `app/globals.css`, feed it a
    downloaded production `/api/fred` JSON as the prop, and screenshot with headless
    Chrome. Iterating label/annotation placement this way beats deploying to look.
    GOTCHA: desktop headless Chrome clamps `--window-size` width to a ~500px minimum —
    a 390-wide screenshot silently CROPS a 500px viewport and looks exactly like an
    overflow bug; simulate phones by constraining a wrapper div to 390px instead.
  - **Bankruptcies source** (`lib/bankruptcies.js` + `lib/data/bankruptciesBaked.json`):
    the AOUSC publishes Table F-2 (business + nonbusiness filings, 12-month period ending
    each quarter) as a small XLSX at a predictable URL
    (`uscourts.gov/sites/default/files/document/bf_f2_<MMDD>.<YYYY>.xlsx`). Layers:
    **live** — try recent quarter-ends newest-first, falling back to scraping the
    quarter's F-2 landing page for a renamed link, time-boxed by `deadlineMs` (15 s) so a
    uscourts outage can't stall the route → **baked** — the full 2001→present quarterly
    history JSON committed in-repo (regenerate with
    `uv run scripts/build_bankruptcies_history.py`) → serve()'s /tmp last-known-good.
    The XLSX parse is a dependency-free mini ZIP reader (Node zlib) and is
    **column-anchored** (finds the Business/Nonbusiness header columns; cells can be
    formulas and zero cells are absent from the XML, so positional rules are unreliable)
    with a hard sanity check (business + nonbusiness ≈ total) so a reshuffled table can
    never ship a wrong number. The card shows BUSINESS filings (the classic recession
    line); freshness deadline 150d (quarterly print + ~4-6 wk publish lag), graceful
    staleness like S&P EPS (old real number in orange beats N/A). `spEps`-style guard in
    GET(); `_meta.messages` gets a `Bankruptcies: uscourts|baked|unavailable` line.
    Verified live 2026-07-23: resolver lands on `2026-03-31` (591,850 total /
    25,960 business — matches the AOUSC news release, YoY +11.4%).
    **Auto-rebake (added 2026-07-25):** the baked file was previously regenerated ONLY by
    hand, making it the one tier that could rot silently — a broken uscourts would fall
    back to a bake that stopped growing, and the card would show a confident number for the
    ~150 days it takes the freshness deadline to notice. `.github/workflows/rebake-bankruptcies.yml`
    now runs monthly (8th, 11:00 UTC — after the ~6wk publish lag) and opens a PR when the
    bake grows. `quarters()` derives its end from `latest_quarter_end()` (was a HARDCODED
    `(2026,3,31)`, which would have quietly stopped extending), and `main()` **refuses to
    write a bake smaller than the existing one** and exits non-zero — a degraded uscourts
    must never gut years of history that no live tier can rebuild. **SILENCE IS THE HEALTHY
    STATE**: it Telegrams only when a PR is opened or the run fails, so no news = no new
    quarter. Test-fired 2026-07-25 (`workflow_dispatch`): walked all 99 quarters, newest
    `2026-03-31 total=591,850 business=25,960` (matches live), printed `no change (99
    quarters)`, opened no PR, sent no Telegram — i.e. the silent-when-healthy path works.
    **The notify could not fire in the failure it exists to catch (fixed 2026-08-06).** The
    PR work and the Telegram were two steps: the first did `URL=$(gh pr create … | tail -1)`
    under `set -uo pipefail` + `bash -e`, and the second was `if: env.MSG != ''` with no
    `if: always()`. A failing `gh pr create` (e.g. the "Allow GitHub Actions to create and
    approve pull requests" repo setting being OFF, which killed self-improve on 2026-08-05 —
    see §7) propagated its exit code out of the ASSIGNMENT and killed the step **before**
    the `echo "MSG=…"`, so the notify saw an empty `MSG` and was skipped. Net: branch
    pushed, no PR, **no Telegram** — and since silence is this workflow's healthy state,
    the failure read as success. A failing `git push` had the same shape. Both are now ONE
    step with `if: always()`, using self-improve's pattern (`if PR_OUT=$(gh pr create …)`
    / `if ! PUSH_OUT=$(git push …)`, a `::warning::` with the real error, and a
    `/compare/<branch>?expand=1` fallback link), sending only when `MSG` is non-empty so
    "no new quarter" stays silent. It also tests `steps.rebake.outcome != "success"` rather
    than `= "failure"`, so a **skipped** rebuild is not read as healthy. Shape-tested in
    `tests/test_workflows.py` (both this and `self-improve.yml`).
  - **INDEPENDENT SOURCES for the three FRED-fed lines** (`lib/horsemen.js`, added
    2026-07-25, PRs #35+#36). Before this, claims/unemployment/spread had exactly ONE live
    provider (the FRED API) behind ONE api key — while copper/gold got 4-5 providers per leg
    and the vol table 3-4 per cell. The most decision-relevant card was the least redundant
    one. Each line now cascades, and the tier-2s are deliberately the **ORIGIN publishers**
    (FRED merely republishes them), so a TOTAL FRED outage is survivable, not just a bad key.

    | line | primary | tier 2 | tier 3 | status |
    |---|---|---|---|---|
    | spread | FRED `T10Y2Y` | **US Treasury** yield-curve CSV (keyless) | `fredgraph` | ✅ prod-verified |
    | unemployment | FRED `UNRATE` | **BLS v2** `LNS14000000` (keyless) | `fredgraph` | ✅ prod-verified |
    | claims | FRED `ICSA` | — none exists — | `fredgraph` | ⚠️ single publisher |

    Both tier-2s reproduce FRED EXACTLY (verified against live data 2026-07-25):
    Treasury 4.69 − 4.33 = **0.36** = `T10Y2Y`; BLS June 2026 = **4.2%** = `UNRATE`.
    Treasury serves ONE calendar year per request, so we fetch the current + prior year.
    - **⚠️ `fredgraph.csv` (tier 3 everywhere) IS A PHANTOM — never count it as redundancy.**
      The intent was that it shares FRED's servers but not its api key, so it would survive a
      revoked key. **It works in local `next dev` (3107 pts) and FAILS on Vercel**
      (`fredcsv:err`, prod-verified via `?_fail=fred`); plain `curl` also fails from a
      residential connection (`HTTP/2 INTERNAL_ERROR`, then hangs). `fred.stlouisfed.org`'s
      WEB paths (`/graph/fredgraph.csv`, `/data/<ID>.txt`) gate on something browser-like;
      only the keyed `api.stlouisfed.org` is reliable from a server. Kept as a harmless
      best-effort last attempt (one failed request in an already-degraded path). **The local
      dev success is a trap — it is why this shipped in #35 looking like real redundancy.**
    - **⚠️ CLAIMS HAS ONLY ONE LIVE PUBLISHER — a known, ACCEPTED limit, not a gap to fix.**
      Nobody else publishes seasonally-adjusted weekly claims in a serverless-friendly form.
      DOL's ETA `oui.doleta.gov/unemploy/csv/ar539.csv` is the ORIGIN but is a 13MB
      **state-major** file (it honors `Range` → 206, but the newest national week is scattered
      across 53 jurisdiction blocks, so a range fetch cannot isolate it) and it is **NSA**.
      NSA is NOT a drop-in for SA: it swings ±30% seasonally, so charting it on the same line
      would MANUFACTURE FALSE RECESSION SIGNALS. Stale-but-correct SA beats
      fresh-but-incomparable NSA. DBnomics is out too — its FRED mirror is **404 dead**
      (re-verified 2026-07-25), no DOL provider. Claims is protected by the PERSISTENCE
      layers instead (/tmp last-known-good + the twice-daily `dashboard_lkg` snapshot with 5y
      of history), served frozen and honestly flagged stale.
    - **Costs nothing on the happy path** — `needsRepair()` only fires a cascade when the
      primary series is empty, `unavailable`, or already past its staleness deadline. That
      last trigger is what an ORIGIN source uniquely fixes: when FRED quietly stops updating
      one series the /tmp last-known-good is equally stale, but BLS/Treasury still publish.
      Prod happy path confirms zero cascade messages in `_meta`.
    - **Never regresses** — `isUpgrade()` adopts a fallback only when genuinely NEWER than the
      primary, so a lagging BLS print can't overwrite a fresh FRED one.
    - **Degraded history is BY DESIGN**: Treasury ~2y, BLS ~10y. Stat chips, the "N of 4
      riding" badge and the 12-month trend notes all still work; only the ALL/20Y tabs draw a
      shorter line. A short real chart beats a blank card.
    - **GOTCHA (BLS):** the keyless tier caps a request at 10 years and, when you ask for more,
      **silently returns the OLDEST 10 years** rather than erroring — a naive
      `startyear=1948` request yields 1948-1957 and the series looks dead. Always anchor the
      span to the CURRENT year. A free `BLS_API_KEY` raises the quota 25→500/day, NOT the span.
    - **GOTCHA (BLS):** period `M13` is the ANNUAL AVERAGE, not a 13th month — dropping it is
      mandatory or it lands as an extra point and corrupts the trend fit.
    - **GOTCHA (Treasury):** read the `2 Yr`/`10 Yr` columns BY HEADER NAME. Treasury has
      inserted tenors before (`1.5 Month` is recent), and a positional read that grabbed
      `20 Yr` instead of `2 Yr` would print a **FALSE INVERSION** (−0.49 vs +0.36) — the most
      consequential way this parser could be wrong. Regression-tested.
    - **Total-outage merge:** with 0/17 series loaded, returning the live payload would blank
      every other card, but returning the cache would throw away live recession data. So the
      route overlays the live lines onto the best cached base (/tmp last-good, else the Sheet
      tier) via `mergeHorsemenOverBase` and serves the union — marked `stale` + `hasErrors`,
      and **never stored back** (`serve()`'s `shouldStore` option, separate from `isGood`:
      storing a mostly-cached payload would refresh its `savedAt` and let stale data outlive
      the 7-day window forever). A missing `FRED_API_KEY` therefore no longer throws out of
      `produce()` — that IS the outage these fallbacks exist for.
    - **GOTCHA (#36, both found ONLY by testing on prod):** (a) this merge path reads /tmp
      **DIRECTLY** via `loadLastGood()`, bypassing `serve()`'s guarded reader — so
      `?_fail=lastgood` was silently ignored and the Sheet tier was unreachable alongside live
      horsemen. **Any direct cache read must re-check the fault set itself.** (b) The merged
      payload inherited the base snapshot's `_meta.source` ("St. Louis Fed") and messages
      ("Loaded 17/17 series") VERBATIM while carrying `loadedCount: 0` — self-contradictory,
      and it read as fresh live FRED data. **A merge-over-cache path must relabel BOTH:**
      `source` → `"<baseLabel incl. LKG timestamp> + live Horsemen (<lines>)"`, and inherited
      messages get a `cached: ` prefix. The premise of this card is that stale data must never
      look fresh; this was a place where it did.
    - **PROD VERIFICATION (2026-07-25, authoritative — local dev is NOT).** Happy path:
      `Loaded 17/17`, `hasErrors:false`, NO cascade messages, all four lines fresh with full
      history. Degraded paths, both rendering **4/4 lines** (pre-#35 the card vanished
      entirely in both):

      | injected fault | result |
      |---|---|
      | `?_fail=fred` | spread `treasury:ok(390)`, unemployment `bls:ok(113)`, claims `fredcsv:err` |
      | `?_fail=fred,lastgood` | live BLS+Treasury merged over the Sheet base; `source` = `Google Sheet (last-known-good …) + live Horsemen (unemployment, spread)` |
      | `?_fail=fred,lastgood,hm_treasury,hm_bls,hm_fredcsv` | Sheet tier serves all four (claims 260 pts, unemployment 118, spread 251, bankruptcies 99) |

      **Re-verify with `?_fail=` after ANY change here, on PROD not local** — three separate
      defects in this feature (the phantom tier, the ignored `lastgood` fault, the
      cached-as-live labelling) passed a green local test suite and were only exposed by
      hitting the deployed dashboard.
  - **Sheet last-resort tier now carries the card — LIVE and populated** (`lib/sheetLkg.js`
    + the scraper's `build_horsemen_pairs`, 2026-07-25). `reconstructFred` previously emitted
    no `horsemen` key and no history at all, so the deepest fallback restored every card
    EXCEPT this one: the component's `hasAnySeries` check failed and it rendered
    "N/A — Unavailable" — the single most decision-relevant card was the FIRST thing to
    vanish in a deep outage. The `dashboard_lkg` tab now holds 78 keys including all four
    lines (claims 260 pts / unemployment 118 / spread 251 / bankruptcies 99), confirmed by a
    manual `workflow_dispatch` of the scraper right after the companion PR merged — **a
    schema change here is not live until the scraper next runs** (cron `0 14,2 * * *`; dispatch
    it manually rather than waiting). The
    helper tab now also carries, per line, a thinned **packed history** in one cell
    (`YYYY-MM-DD:value|…`, `parsePackedHistory`) — claims 5y, unemployment 10y, spread 5y,
    bankruptcies 30y, largest ~4.7KB against Sheets' 50k cell limit. Cross-repo contract
    keys: `horsemen.<claims|unemployment|bankruptcies>.{value,asOf,history}`,
    `horsemen.bankruptcies.{total,changePct,status}`, `yieldCurve.history`. Backward
    compatible both ways (a tab written by the old scraper still parses; a line present as
    history-only still yields a current value). Writer lives in **financial-dashboard-history**
    (`build_horsemen_pairs`/`pack_history`) — don't rename keys in one repo without the other.
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
  the fresh level and datahub the chart; `spEps.tried` shows the trail). The bankruptcies
  cascade adds **`bk_uscourts`** (kill the live uscourts tier → baked serves, possibly
  stale-marked) and **`bk_baked`** (kill the baked history; both together → the
  unavailable/N-A path); `horsemen.bankruptcies.tried` shows the trail. The Four Horsemen
  fallback cascades add **`hm_treasury` / `hm_bls` / `hm_fredcsv`** (each disables that
  provider for every line that uses it). These only bite when the FRED primary has already
  failed — always pair them with `fred`. Each line's `source` + `tried` fields show which
  provider answered. The canonical ladder to re-verify after any change (run it on PROD):
  `?_fail=fred` → independent providers; `?_fail=fred,lastgood` → live horsemen merged over
  the Sheet base; `?_fail=fred,lastgood,hm_treasury,hm_bls,hm_fredcsv` → the Sheet tier
  alone. Note `lastgood` must be honoured by BOTH `serve()` and any code that calls
  `loadLastGood()` directly (see the #36 gotcha in §3). `/api/sheets` adds **`vix_fred`**
  (kill the VIX pill's FRED-computed fear/greed tag → falls back to the sheet's C2 value;
  see "VIX pill fear/greed tag" above).

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
- `dashboard/lib/{store,sources,fetcher,faults,freshness,finance,copperGold,spEps,bankruptcies,constants}.js` —
  the never-throw store + last-known-good, direct data sources, fetch/proxy helpers, fault
  injection, FRED freshness, math (RSI/dailyChange/copperGoldRatio), copper/gold cascade,
  S&P 500 EPS parsers + cascade, the AOUSC bankruptcies resolver (mini ZIP/XLSX reader +
  live→baked cascade; baked history in `dashboard/lib/data/bankruptciesBaked.json`),
  and shared constants (FRED IDs, freshness deadlines, URLs).
- `dashboard/app/page.js` — dashboard page + the `.system-status-bar` footer.
- `dashboard/components/*.js` — UI (MarketModal, PolymarketTable, SpyChart, Gauge,
  EconomicIndicatorGrid, BullChecklist, ExtraMarketsGrid, MarketPulse, MiniChart,
  FourHorsemen, CustomIndicatorBar, Skeleton, ErrorBoundary, MarketModal.example).
- `dashboard/{jest.config.js,jest.setup.js,next.config.js,package.json}` — build/test config.

**Self-healing / ops (scripts + workflows)**
- `scripts/health_check.py` — the daily self-check prober (see §7); pure check functions +
  HTTP probes + `--notify`/`--summary`.
- `scripts/collect_health_context.py` — distills recent health reports into the weekly
  agent's digest (`health-digest.md`).
- `scripts/build_bankruptcies_history.py` — regenerates the Four Horsemen card's baked
  bankruptcies history from uscourts.gov F-2 tables (run with `uv run`; rarely needed —
  the dashboard's live tier keeps the newest quarter fresh on its own).
- `tests/` — pytest: `test_polymarket_fetcher`, `test_run_report`, `test_utils`,
  `test_health_check`, `test_collect_health_context`, `test_workflows` (shape tests for the
  two "silence is healthy" workflows — needs `PyYAML`, in `requirements-dev.txt`).
  Dashboard Jest tests live under `dashboard/**/__tests__/`.
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

### ⚠️ The governing principle for every check in this file
**A green check must prove the PRIMARY path ran. If a FALLBACK can satisfy the check, the
check is a false negative** — and this system is built almost entirely out of fallbacks, so
that is the default failure mode, not an edge case. Every silent outage this repo has had
was of exactly this shape: something degraded, a backup absorbed it, and the monitor kept
reporting green. Known instances, all fixed 2026-08-06:
- `check_report_delivered` returned `ok` when only the **GHA backstop** delivered — hid a
  dead EventBridge→Lambda path for **two months** (§2 gotcha #0).
- The **P/E tile** silently served Shiller CAPE when the TTM scrape broke → `check_pe_source`.
- **`/api/vol`** accepted any non-empty series, so a frozen feed served months-old vol as
  current, and `hasErrors` only tripped if EVERY cell was null → staleness gate + per-ticker
  `hasErrors`.
- The **four Lambda-primary routes** fall back to direct sources with `hasErrors:false`, so
  a dead Lambda was invisible → `lambda_primary_path`.
- **`yieldCurve`/`profitMargin`/`spEps`** sat outside the N/A sweep, so a frozen `T10Y2Y`
  could never warn → they are in `fred_metrics_for_na_check` now.
- **`rebake-bankruptcies.yml`** could not Telegram in the failure it exists to catch, while
  "silence is the healthy state" made that read as success.

When you add or change a check, ask: *what would this report if the primary died and the
backup covered?* If the answer is "ok", it is not a check. The mirror-image rule still
applies though — **unknowable ≠ known-bad** (an unreadable CloudWatch or an unparseable
payload must never manufacture an alarm), and normal reporting lag must never alarm.

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
- **`indicators_na`** — inspects `/api/fred`'s `indicators` + `checklist` + `horsemen` +
  the top-level `yieldCurve`/`profitMargin`/`spEps` cards
  (via `fred_metrics_for_na_check`); an unexpected `null`/`unavailable`/overdue metric
  is `warn`; a metric on the `KNOWN_DISCONTINUED` allowlist (currently `lei`) is expected and
  never alarmed.
- **`lambda_primary_path`** — **is the Lambda still serving the dashboard's HTTP data
  path, or is everything quietly on fallbacks?** `/api/spy`, `/api/market-extra`,
  `/api/spy-daily-move` and `/api/polymarket` are Lambda-primary; when the Lambda does not
  answer they fall back to direct public sources and return `hasErrors:false`
  **on purpose** (a full direct build IS healthy data — hard-coding `hasErrors:true` once
  made SPY perpetually red), recording the Lambda failure only in a `_meta.messages` string
  nothing reads. The prober only ever touches Vercel. So a lost `apigateway` invoke grant —
  the EXACT failure that hid on the EventBridge grant for two months (§2 gotcha #0) — would
  flip all four routes to fallbacks with every check still green. The check reads **which
  source won**: every dashboard fallback builder labels itself `… (fallback)`
  (`Polygon + Finnhub (fallback)`, `Direct sources (fallback)`, `Finnhub (fallback)`,
  `Polymarket Gamma API (fallback)`) and no Lambda label ever contains that marker
  (`Polygon + Finnhub Spot`, `yfinance/Polygon/Finnhub/FRED/ER-API`, `Google Sheets`), so
  one marker separates the paths for all four routes without a per-route allowlist that
  would rot the next time a waterfall tier is renamed. Unreadable payloads are **skipped,
  not counted against the Lambda** (`endpoint_<name>` already owns an outage; unknowable
  must never become known-bad). Verify it end-to-end with `?_fail=lambda`, which forces
  every one of the four onto its fallback.
- **`report_delivered_today`** — from delivery evidence gathered by the workflow (CloudWatch
  `REPORT_DELIVERED`/`REPORT_FAILED` markers in a **rolling 24 h** window + whether
  `daily_report.yml` succeeded in the same rolling 24 h). Lambda marker present → `ok`;
  `REPORT_FAILED`, or no marker with no GHA run → `critical` + `remediation:
  auto:redispatch_daily_report`; CloudWatch unreadable with no corroboration → `warn`
  ("could not confirm"), never a false critical.
  **A GHA success alone is NOT `ok`.** There are two senders — the Lambda (primary,
  EventBridge ~08:15 UTC) and the runner (backstop, ~09:45 UTC). CloudWatch readable + no
  Lambda marker + GHA success → **`warn`: "delivered by BACKSTOP only"**, because the report
  arrived but the primary is dead. Treating that as `ok` is what hid the 2026-06-01 →
  2026-08-06 EventBridge outage for two months. CloudWatch *unreadable* still returns `ok`
  (unknowable ≠ known-bad, so a missing IAM grant can't false-alarm).
  **Both windows are rolling 24 h on purpose — never a UTC calendar day.** A calendar-day
  query run just past 00:00 UTC finds zero runs "today", escalates to `critical`, and
  auto-redispatches a **duplicate report** (happened 2026-08-07T00:02Z; the same bug class
  was fixed once before in PR #5 and reintroduced by the GHA cross-check).
- **`known_issue_config_urls`**, **`secret_leak`** (gitleaks), **`ci_health`** (latest run
  per **active** workflow) — see the table below.

### When the owner pastes you a health alert, act on it like this
Each finding has an `id`. Map id → meaning → fix:

| Finding `id` | What it means | How to act |
|---|---|---|
| `report_delivered_today` | Couldn't confirm today's Telegram report went out | `cloudwatch_readable:false` → the `logs:FilterLogEvents` IAM grant is missing (owner action), **not a real outage**. A `REPORT_FAILED` marker → a real send failure: read the Lambda's CloudWatch logs; the daily run may already have re-dispatched `daily_report.yml`. |
| `report_delivered_today` **severity `warn`, title "delivered by BACKSTOP only"** | The report *did* reach the owner, but the **runner backstop sent it and the Lambda did not** — the primary path is down and used to be invisible | Check `AWS/Events` **`FailedInvocations`** for rule `daily-financial-report-trigger` (a steady 1/day = the Lambda is rejecting EventBridge), then the Lambda's **resource policy** (needs an `events.amazonaws.com` invoke grant, Sid `AllowEventBridgeDailyReport`) and its **env vars**. See gotcha #0 in §2 — recreating the function wipes both. |
| `endpoint_<name>` | `/api/<name>` returned non-200, invalid JSON, a bare `NaN`, or `_meta.hasErrors` | Hit the live URL, read `_meta.messages`; fix per §3 (never-throw, sanitize NaN). Slow first loads + transient degradation are already retried, so a flagged endpoint is genuinely failing. |
| `indicators_na` | A dashboard indicator is N/A **unexpectedly** | `detail` names the metric; repair/extend its fallback. Known-discontinued metrics are allowlisted (`KNOWN_DISCONTINUED`) and never alarmed. |
| `lambda_primary_path` | The dashboard is serving correct numbers **from its fallbacks** — the API Gateway → Lambda hop is failing | `evidence.fallback_routes` names which routes fell back. Check the Lambda's **resource policy** for an `apigateway.amazonaws.com` invoke grant (§2 "Wiring a new API Gateway"), that Vercel's `LAMBDA_URL` still points at the gateway base (NOT the dead Function URL), and the function's recent CloudWatch errors. Nothing looks broken on the page — that is exactly why this check exists. |
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
  `self-improve/run-<run_id>`, commits, pushes, opens a PR via `gh pr create`,
  and posts a Telegram summary with the PR link. It **never merges**; the owner reviews + merges.
- **GOTCHA — do NOT put `[skip ci]` in that commit** (it was there until 2026-08-05). `ci.yml`
  fires only on `pull_request` → main and `deploy-lambda.yml` only on push → main, so on a
  feature-branch push the marker suppressed **nothing** — but on the PR it suppressed
  `backend-tests` + `dashboard-tests`, which are **required checks** on `main`. The agent's own
  PRs were unmergeable by design (first seen on #38: zero check runs, blocked). It would also
  skip the Lambda deploy if it rode into a merge commit.
- The agent's hard rules (`.github/self-improve-prompt.md`): be conservative (a clean no-op is
  success), one small single-concern change, cross-check this AGENTS.md first, and **never
  touch** secrets/keys, `aws/template.yaml`, live AWS config, `.env`, or `.github/workflows/`.
- **GOTCHA — `gh pr create` needs a REPO SETTING, not just `pull-requests: write`.** Settings →
  Actions → General → **"Allow GitHub Actions to create and approve pull requests"** must be ON
  (API: `PUT /repos/{owner}/{repo}/actions/permissions/workflow`,
  `can_approve_pull_request_reviews: true`). It was OFF, and on **2026-08-05** (run
  `31019492456`) the agent made its first real change in 8 weekly runs, pushed the branch fine,
  and then `gh pr create` was refused → the job failed. **Every prior green run was a no-op, so
  this whole code path had never once executed** — a workflow that has "passed" 8 times can
  still have a completely untested branch. If you ever see the branch pushed but no PR, check
  this setting first.
- **GOTCHA — never swallow that step's output.** The old line was
  `URL=$(gh pr create … 2>&1 | tail -1)`. With `set -o pipefail` under GitHub's default
  `bash -e`, gh's non-zero status propagated out of the ASSIGNMENT and killed the step, while
  `2>&1` had already redirected the real error message into `$URL` where nobody saw it. Net
  effect: no PR, no error text, and — because the step died before the notify — **no Telegram
  either**; the only signal was GitHub's own "run failed" email. The PR-create call is now
  wrapped in `if PR_OUT=$(…); then … else …`, which prints the failure as a `::warning::` and
  still Telegrams, falling back to a `/compare/<branch>?expand=1` link. **The branch is already
  pushed by that point, so the work is never lost — the notify matters more than the exit
  code.**
- **To fix anything here:** branch → fix + test → PR (CI gates it) → owner merges. Never push
  to `main`. Roll back via a PR's **Revert** button or a `known-good-*` git tag.
