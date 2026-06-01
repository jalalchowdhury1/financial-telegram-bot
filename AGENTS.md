# AGENTS.md — Financial Telegram Bot & Dashboard

> **This is the single source of truth for anyone (human or AI) touching this repo.**
> Read it fully before changing code, deploying, or "fixing" anything. It replaces the
> old scattered guides (`docs/AI_MAINTAINER_GUIDE.md`, the `docs/superpowers/specs/*`
> design docs, and the dashboard test/verification reports) — those were point-in-time
> and have been deleted. If something here is wrong, fix *this* file.

This project is the owner's **most important** tool. Production stability matters more
than speed. When in doubt, prefer a graceful fallback over a hard change, and verify a
deploy actually worked before claiming success.

---

## 1. What this is

One AWS **Lambda** (`financial-telegram-report`, `us-east-1`, python3.11, 512 MB, 300 s
timeout, account `463256610967`) that does **two** unrelated jobs from the same
`lambda_handler.py`:

1. **Daily Telegram report** — an EventBridge schedule invokes the Lambda; it assembles a
   market brief (Google-Sheet indicators + SPY snapshot via `bot/fetchers.py`) and sends
   it to Telegram (`bot/utils.py:send_to_telegram`). Entry: `handle_eventbridge`.
   *(Note: there's also an independent report path — the `daily_report.yml` GitHub Action
   runs `python -m bot.main report` on the runner via `run_report()`, not the Lambda.)*
2. **Dashboard data API** — serves read-only market-data endpoints (`/api/spy`,
   `/api/spy-daily-move`, `/api/market-extra`, `/api/polymarket`) consumed by the
   Next.js dashboard. Entry: `handle_http_api`.

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
  sources, so the dashboard keeps working even if the Lambda is down.

---

## 2. Deploying the Lambda (read before any backend change)

**You normally don't deploy by hand — just push to `main`.** GitHub Actions
(`.github/workflows/deploy-lambda.yml`) deploys automatically when a commit changes
`lambda_handler.py`, `bot/**`, `aws/requirements-lambda.txt`, or the deploy workflow
itself. You can also run it manually: Actions ▸ **Deploy to AWS Lambda** ▸ Run workflow.

The deploy: builds the zip → uploads to S3 (`financial-telegram-bot-deployments`, the
package is >50 MB so direct upload won't work) → `update-function-code` (serialized via a
`concurrency` group + retried on `ResourceConflictException`) → waits → **smoke-tests the
live API Gateway** (`GET /api/spy` must return 200, valid JSON, no bare `NaN`). It runs as
the scoped IAM user **`github-deploy-bot`** (not root).

### ⚠️ Config drift — the deploy only updates *code*
The deploy runs `update-function-code` ONLY. **`aws/template.yaml` (the SAM template) is
never applied.** Lambda env vars, memory/timeout, the EventBridge schedule, and the
Function URL are all hand-managed in the AWS console and can drift from the template.
**Editing `template.yaml` does NOT change the live function** — change those in the
console (or extend the workflow) and update the template to match.

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
  and cold start depend on it.
- If you ever build locally, clean up `package/` and `*.zip` so they don't pollute git.

### Three Lambda gotchas that have caused real outages
1. **yfinance must cache to `/tmp`.** Lambda's filesystem is read-only except `/tmp`.
   Without a writable TZ/cookie cache, every Yahoo call is a fresh cookieless scrape and
   Yahoo **429-rate-limits / IP-bans** the function. `bot/fetchers.py` already does
   `os.makedirs('/tmp/yfinance')` + `yf.set_tz_cache_location('/tmp/yfinance')` before any
   yfinance call — keep it that way.
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

---

## 3. The dashboard (`/dashboard`)

Deploys to Vercel automatically on push to `main`. Tests: from `dashboard/`,
`npm test` (Jest) and `npm run build` must pass before merge; `npm run test:watch` for
watch mode; `--testPathPattern <Name>` filters.

### Conventions (enforce on every change)
- **Never hardcode secrets** — the repo is public. Keys live in **Vercel env vars**
  (`LAMBDA_URL`, `POLYGON_KEY`, `FINNHUB_KEY`, `FRED_API_KEY`, optional `RAPIDAPI_KEY`),
  Production + Preview. (A RapidAPI key was once committed here; don't repeat it.)
- **Never-throw routes.** Every `/api/*` GET wraps its *entire* body in
  `lib/store.js: serve(key, produce, opts)`, which cascades **live → fallbacks →
  durable last-known-good (in `/tmp`) → safe default** and always returns **HTTP 200 with
  valid JSON**. Define any `liveKey`/`faults` vars *before* `serve()` so nothing can throw
  outside the try/catch (that would 500 with an empty body and break the guarantee).
- **Extract** new features into standalone components in `dashboard/components/`; add
  `'use client'` to any component with client-side state/interactivity.
- **External links** must use `target="_blank" rel="noopener noreferrer"`.
- Probability-bar colors (MarketModal `getOddsColor`): `<0.2` red, `0.2–0.4` orange,
  `0.4–0.6` yellow, `0.6–0.8` green, `≥0.8` bright green. The field is **`bet.odds`** (a
  decimal 0–1, rendered as a %) — not `bet.probability`. The modal currently links to the
  bare `https://polymarket.com` homepage; per-market deep links were deliberately avoided
  as unreliable (the API doesn't surface a usable slug).

### `market-extra` + the status footer (where a "red" bug lived)
- `market-extra` fills only the metrics the Lambda returned `null`, **never overwrites a
  good Lambda value**, and **must reconcile `_meta` after backfilling** (recompute
  `hasErrors` + the "unavailable: N" message from what's *still* missing). If you skip
  this, the page-bottom status footer shows red even when every metric was filled.
- The footer (`dashboard/app/page.js`, `.system-status-bar`) colors each source from its
  route's `_meta`: red if `hasErrors`/`unavailable`, yellow for degraded sources
  (Stooq/FRED proxies), green otherwise. **Green must mean the data is actually healthy** —
  don't hardcode a status; derive it.

### FRED route specifics (`/api/fred`) — subtle, don't regress
- The route uses `export const fetchCache = 'default-cache'` and stays dynamic by reading
  the request, so Next's Data Cache via per-fetch `revalidate` works (don't switch it to
  `force-dynamic` — that would force `no-store` and disable the caching below). FRED
  upstream + the P/E scrapes are cached **30 min** (`REVALIDATE_SECONDS = 1800`).
- FRED's free API ~120 req/min + shared-IP throttle → random `429`s. Fetch in small
  batches (`BATCH_SIZE = 4`, `STAGGER_MS ≈ 150`); retry `429` with backoff
  `[400, 900, 1800] ms`; **do not cache a final failure**. Mask any `api_key=…` in
  `_meta.messages` (`maskKey`).
- FRED dates observations at the *start* of the period and publishes weeks late, so fresh
  series legitimately look old. Per-metric freshness deadlines (`FRED_FRESHNESS`, days):
  daily=7, weekly=14, monthly=80, JOLTS/quarterly larger. Too-old/NaN → value forced to
  `null` so the UI shows N/A; `stale:true` (had data, too old) ≠ `unavailable:true`.
- **Copper/Gold ratio** (the `indicators.copperGold` tile) replaced the old `LEI`/`USSLIND`
  series, which FRED **discontinued/froze in 2020** (it could only ever show N/A). Unlike a
  single FRED series, copper+gold are each fetched from **multiple price sources** (Yahoo
  primary → Stooq fallback per leg; see `fetchCopperGold` in the FRED route + the pure
  `copperGoldRatio` in `lib/finance.js`), so the tile is robust and a genuine N/A there is a
  real signal the daily health-check will flag.
- The `?_fail=` fault-injection harness (`lib/faults.js`) is intentionally **kept in
  production** — it only degrades the caller's own response and never writes caches.

---

## 4. Known issues / open items (as of 2026-05-31)

- **`bot/config.py` is missing `URLS` keys** `SPY_DAILY_MOVE`, `SPY_INDICATORS`,
  `STOOQ_SPY` → those Lambda data tiers `KeyError` and are dead (e.g. `/api/spy-daily-move`
  from the Lambda always returns null; the dashboard masks it with a Finnhub fallback).
  Fixing needs the owner's Google-Sheet IDs/gids; `STOOQ_SPY` = `https://stooq.com/q/d/l/?s=spy.us&i=d`.
- **Daily report can fail silently:** `bot/main.py`'s `__main__` doesn't `sys.exit(1)` on
  failure, so a broken run (or a Markdown-400 from Telegram) goes green and the retry
  harness in `daily_report.yml` never fires. Make it exit non-zero on failure.
- **Telegram delivery** sends the whole report as one `parse_mode='Markdown'` message over
  un-escaped sheet content — an unbalanced `_ * [ ]` ` causes a 400 and drops the report.
  Retry as plain text on 400; chunk >4096 chars.
- **Secrets:** a RapidAPI key was committed and removed from code — it must still be
  rotated (it's in git history). AWS CI moved off root keys to the `github-deploy-bot` IAM
  user; the old root access keys should be deactivated/deleted.

## 5. Never-break checklist (before you commit)
1. Did you sanitize pandas/math floats (no bare `NaN`) before any API response?
2. Do yfinance calls still cache to `/tmp`?
3. Are Lambda deps built for `manylinux2014_x86_64` (i.e., changes go through CI, not a
   local zip)?
4. Does SPY still override stale Polygon spot with live Finnhub?
5. Is every new/changed `/api/*` route still never-throw (whole body inside `serve()`),
   returning 200 + valid JSON, with no hardcoded secrets?
6. Did you keep `bot/` lite and put new UI in `dashboard/components/`?
7. For backend changes: did the **Deploy to AWS Lambda** run go green (incl. the API
   Gateway smoke test)? For dashboard changes: does `npm test` + `npm run build` pass?

## 6. Map / quick reference
- `lambda_handler.py` — Lambda entry; `_clean_nans`, `_ok`, `handle_http_api`,
  `handle_eventbridge`.
- `bot/fetchers.py` — the data waterfall (yfinance→Polygon→Finnhub→Stooq→FRED→Sheets).
- `bot/config.py` — `URLS` (Google-Sheet/data source URLs) + FRED series IDs.
- `bot/utils.py` — env loading + `send_to_telegram`.
- `bot/main.py` — local/runner entry (`python -m bot.main report`).
- `dashboard/app/api/*/route.js` — per-metric routes (Lambda-primary + fallbacks).
- `dashboard/lib/{store,sources,fetcher,faults,freshness,finance,constants}.js` — the
  never-throw store, direct sources, retry, fault injection, FRED freshness, math.
- `dashboard/app/page.js` — dashboard page + the `.system-status-bar` footer.
- `aws/template.yaml` — SAM template (reference only; **not** applied by CI — see §2).
- `.github/workflows/` — `deploy-lambda.yml` (auto-deploys the Lambda) and
  `daily_report.yml` (runner-based daily report with a self-retry harness).
