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
  series, which FRED **discontinued/froze in 2020**. It is a leading growth/rates gauge
  (~1.4 = copper $/lb ÷ gold $/oz ×1000). The tile shows the **level + its ~1-month and
  ~3-month change** (the trend is what matters, not the level). Because this route runs on
  **Vercel datacenter IPs where Yahoo + Stooq are blocked/JS-walled** (the old 2-source
  version went permanently N/A), each leg now cascades through several independent,
  datacenter-reachable sources, all normalized to the same unit so the ratio stays
  consistent regardless of which answered (see `fetchCopperGold` + `copperSources`/
  `goldSources` in the FRED route, the pure cascade in `lib/copperGold.js`, and the
  parsers `cnbcQuotes`/`cnbcHistory`/`goldApiSpot` in `lib/sources.js`):
  - **Copper $/lb**: CNBC `@HG.1` (keyless, daily history) → FRED `PCOPPUSDM` (key; ÷2204.6226
    from $/tonne; monthly) → gold-api.com `HG` (keyless spot) → Yahoo `HG=F` (self-heal).
  - **Gold $/oz**: CNBC `@GC.1` (keyless, daily history) → Polygon `C:XAUUSD` (key) → FRED
    `GOLDPMGBD228NLBM` (key; daily) → gold-api.com `XAU` (keyless spot) → Yahoo `GC=F`.
  CNBC + FRED give history → the 1mo/3mo delta; spot-only sources still give the level. A
  genuine N/A here (all sources down) is a real signal the daily health-check flags.
- The `?_fail=` fault-injection harness (`lib/faults.js`) is intentionally **kept in
  production** — it only degrades the caller's own response and never writes caches.
  **Fault names:** `fred` (forces an invalid FRED key → all FRED *series* fail, exercising
  last-known-good; note this does **not** disable the copper/gold FRED leg — that's
  `cg_fred`), `lastgood` (also skip reading last-known-good, to reach the safe default), and
  the per-source copper/gold gates **`cg_cnbc` / `cg_fred` / `cg_polygon` / `cg_goldapi` /
  `cg_yahoo`** (each disables that source in **both** legs). E.g. `?_fail=cg_cnbc` forces both
  legs past CNBC; `?_fail=cg_cnbc,cg_fred,cg_polygon` forces them down to the keyless gold-api
  spot tier. The served object's `tried` field shows the per-leg cascade trail (e.g.
  `cnbc:off → polygon:off → fred:ok`).

---

## 4. Known issues / open items (updated 2026-06-01)

**✅ Resolved 2026-06-01** (kept here so the history is legible):
- `bot/config.py` `URLS` now has `SPY_DAILY_MOVE`, `SPY_INDICATORS`, `STOOQ_SPY` (mirrored
  from `dashboard/lib/constants.js`) — the Lambda SPY fallback tiers no longer `KeyError`.
  *Note:* Stooq's `q/d/l` download endpoint now gates behind an apikey, so `STOOQ_SPY` is a
  graceful **dead fallback** (the yfinance/Polygon/Finnhub tiers cover SPY).
- Silent daily-report failure fixed: `bot/main.py` `run_report()` returns `False` on empty
  content / failed send and `__main__` `sys.exit(1)` → green == sent, retry harness fires.
- Telegram delivery hardened: `send_to_telegram` retries as plain text on a Markdown 400 and
  chunks >4096 chars.
- The discontinued LEI tile was replaced by the multi-sourced **Copper/Gold ratio** (§3).

**Still open (needs the owner):**
- **Secrets:** the leaked RapidAPI key (in git history) must still be **rotated**; the old
  root AWS access keys should be deactivated/deleted (CI uses the scoped `github-deploy-bot`).
- **Delivery monitoring:** grant `github-deploy-bot` `logs:FilterLogEvents` on
  `/aws/lambda/financial-telegram-report` so the health-check can confirm delivery from
  CloudWatch. Until then `report_delivered_today` safely degrades to a `warn` ("could not
  confirm") — never a false `critical`.

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
- `scripts/health_check.py` — the daily self-check prober (see §7).
- `scripts/collect_health_context.py` — distills recent health reports into the weekly
  agent's digest.
- `aws/template.yaml` — SAM template (reference only; **not** applied by CI — see §2).
- `.github/workflows/` — `deploy-lambda.yml` (auto-deploys the Lambda), `daily_report.yml`
  (runner-based daily report + self-retry harness), `ci.yml` (pytest + jest + build on every
  PR — the merge gate), `health-check.yml` (daily self-check + Telegram alert), and
  `self-improve.yml` (weekly Wed agent that opens PRs — see §7).

---

## 7. The self-healing health-check — and how to read its alerts

The repo monitors itself. **Daily** (`health-check.yml`, 14:00 UTC) a pure-code prober
(`scripts/health_check.py`) checks the LIVE system and, on any warn/critical, sends the owner
a Telegram alert **written to be pasted straight into a Claude session**. **Weekly**
(`self-improve.yml`, Wed 13:00 UTC) a headless Claude agent reads the recent reports + this
file and opens PRs the owner approves (it never self-merges; branch protection enforces it).

### When the owner pastes you a health alert, act on it like this
Each finding has an `id`. Map id → meaning → fix:

| Finding `id` | What it means | How to act |
|---|---|---|
| `report_delivered_today` | Couldn't confirm today's Telegram report went out | `cloudwatch_readable:false` → the `logs:FilterLogEvents` IAM grant is missing (owner action), **not a real outage**. A `REPORT_FAILED` marker → a real send failure: read the Lambda's CloudWatch logs; the daily run may already have re-dispatched `daily_report.yml`. |
| `endpoint_<name>` | `/api/<name>` returned non-200, invalid JSON, a bare `NaN`, or `_meta.hasErrors` | Hit the live URL, read `_meta.messages`; fix per §3 (never-throw, sanitize NaN). Slow first loads are already retried, so a flagged endpoint is genuinely failing. |
| `indicators_na` | A dashboard indicator is N/A **unexpectedly** | `detail` names the metric; repair/extend its fallback. Known-discontinued metrics are allowlisted (`KNOWN_DISCONTINUED` in the prober) and never alarmed. |
| `known_issue_config_urls` | `bot/config.py` `URLS` missing a required key | Add the key — the URLs live in `dashboard/lib/constants.js`. |
| `secret_leak` | gitleaks found a credential in the repo | **Rotate it immediately**, then remove the literal (env vars only). |
| `ci_health` | An **active** workflow's **latest** run failed | Open that run, read the failure, fix + PR. Historical/fixed failures and deleted workflows are already excluded. |

### Conventions
- Severity `ok` < `warn` < `critical`; the alert lists **only non-ok** findings. A
  `remediation: auto:redispatch_daily_report` tag means the daily workflow already retried.
- **History = artifacts, not commits.** Each daily run uploads `health-report.json` (90-day
  retention); the weekly agent downloads the recent ones. Nothing is pushed to `main`
  (branch protection blocks bot pushes).
- Delivery uses a **24h rolling CloudWatch window** for the Lambda's
  `REPORT_DELIVERED`/`REPORT_FAILED` markers (`bot/utils.report_marker`), cross-checked with
  the `daily_report.yml` run status.
- **To fix anything here:** branch → fix + test → PR (CI gates it) → owner merges. Never push
  to `main`. Roll back via a PR's **Revert** button or the `known-good-*` git tag.
