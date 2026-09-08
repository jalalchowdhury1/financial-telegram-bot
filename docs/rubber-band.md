# Rubber Band Radar — research record (spec v1.0, 2026-09-02)

**Question it answers, daily:** is the machine's core assumption — QQQ is a rubber band, oversold
dips snap back and overbought rips fade — still true *right now*? Every threshold below was
measured, not chosen. Where the record is bad it says so.

- Engine: `scripts/rubber_band.py` (pure maths, `tests/test_rubber_band.py`); runs nightly on the
  Mac mini (launchd `com.jalal.rubber-band`, weekdays 18:30 ET, `docs/com.jalal.rubber-band.plist.example`).
- Publishes one JSON snapshot to a secret gist → `/api/rubber-band` (never-throw) →
  `<RubberBandRadar/>` on the dashboard and one 🪢 line in the Telegram brief.
- Alerts the 📡 thread **only when a dial changes colour** (or a run fails). Never twice.

## Data (verified 2026-09-02)
- **Prices:** yfinance QQQ, `auto_adjust=True`. A throwaway Composer symphony
  (`RSI(QQQ,10) < 32 → TQQQ else BIL`) was backtested and its allocation days compared with the
  engine's event days: **adjusted closes match Composer 194/194 days since 2010; raw closes
  produce 4 extra dips** (2011-06-20, 2014-03-27, 2015-06-29, 2016-06-17). So the dials see the
  same data the live machine trades on.
- RSI: full-history Wilder RSI-10, seeded like `comp_eval.Store.rsi_arr` (golden fixture
  `tests/fixtures/rubber_band_rsi_golden.json`, agreement < 1e-5).
- The 1971→1999 history behind the decade tables is the ARBOR `deep_series.json` QQQ proxy
  (a scaled NDX/Nasdaq series). It and yfinance share 358 of the 368/390 dips since 1999 —
  same regime story, slightly different single days. Since 1999 the record below was re-run
  on the production (yfinance) data and holds: 0 STOPs, 2 long fast-line alarms, 0 rip alarms.
- Machine curves: Composer backtests of Main / C3 / m1 / C8-T (same payload as the ARBOR
  harness). **Current drawdown is stable run-to-run; the historical "worst ever" wobbles
  ±1–2 pts** between Composer runs as their data revises. Lines have 10-pt amber margins.

## The five dials
"excess" = mean next-day return after the last N events − the market's own mean daily return
over the same span. Only events whose payoff day is ≤ today count (no look-ahead).

| # | Dial | Events | N | Colour rule | Record |
|---|------|--------|---|-------------|--------|
| 1 | **slow** (the one that can say STOP) | RSI-10 < 32 (≈ machine's `RSI(TQQQ,10) < 31`) | 30 | red = excess < 0; **STOP after 60 straight red days** | Red share of days: 70s 92%, 80s 90%, 90s 21%, 00s 1.6%, 10s 0.1%, 20s 0%. 15 STOPs, all ≤ 1991, **zero since 1993**. |
| 2 | **fast** (LOOK only) | same | 20 | same colours; never acts | Since 1993: 12 red spells (2 long: 2001-10→2002-04, 2006/07), **all false alarms**. In the 1970s it led the slow line by 200+ days in 3 of 5 flips. N=10 was rejected (25/25 false). |
| 3 | **age** | span of the 30 slow events | — | green ≤ 3.3y, amber ≤ 4.0y, red > 4.0y (p75/p90 of the span since 1993) | Trust gauge, not a forecast. Historical STOPs fired with spans < 2y (evidence gets dense in a real flip). |
| 4 | **rip** | RSI-10 > 79 (the machine's sell trigger) | 30 | hot = excess > 0; **red after 60 straight hot days** | Hot share: 70s 99%, 80s 80%, 90s 55%, **0% since 2000**. Next-day excess after a rip by decade: +19/+16/+2/−46/−11/−32 bp. RSI>70 variants were too noisy. |
| 5 | **machines** | Composer backtest curves | — | through a written line = red; within 10 pts = amber; ≥ 9 months underwater = red; m1 lagging C3 two complete months = red (the written exit rule) | Lines from the money-radar plan §3.2: Main −40 (watch), C3 −54, C8-T −31, m1 none. |

Cross-check printed on dial 1: 10-day drawdown > 6% events (a real machine trigger), N=30.
Red share 96/84/12/13/0/19% by decade; it catches 2008-01 (true, −10.7% fwd) but was false in
2001-10 and 2020-03. Shown, never acted on.

**Verdict** = worst of slow/rip/machines; a red fast line alone only lifts green to amber;
too little data = amber, never green.

## Sensitivity / red-team (all on the 1971→ store)
- STOP sustain 40–120 days → identical STOP set. hold=1 is cleanest (hold 5 adds 2 false STOPs).
- Threshold 30 vs 32: same STOP set; 32 chosen to match the machine.
- SPY instead of QQQ: noisier (a false 2001 STOP).
- Alternative dip definitions (single-day −2%, 10-day drawdown): the 1970s/80s flip appears in
  every one — it is not an RSI artefact. Single-day −2% dips show recent weakness (36–50% red
  in the 2010s/20s) — noted, not used.
- Noise: the 30-event mean has se ≈ 0.5%; today's +0.63% is ~1.2 se above zero, i.e. green but
  not far from the line. The 60-day sustain is what makes STOP robust, not one day's reading.

## Blind spots (say them out loud)
- **Grinding bears stay green.** 2001 and 2008 never tripped the slow dial — dips still paid
  next day while the market bled for months. The machine-health lines are the backstop there.
- The rip dial has not been red since 2000; its alarm has never fired on data the live
  machine actually traded. Its evidence is 1970–90s.
- Every threshold is from one market (QQQ/NDX). The 1972–91 flips are 15 STOPs from one
  regime change, not 15 independent trials.
- Composer backtest curves are proxies for the legs, not account values.

## Today (as of 2026-09-01, production data)
slow +0.63% (hit 63%, 30 dips over 2.8y, se 0.54%) · fast +0.64% · age 2.8y · rip −0.07% ·
machines: Main −8.8% / C3 −8.2% / m1 −8.1% / C8-T −4.4% vs lines, 1 month underwater, lag 0.
All five green.

## Operations
- Run by hand: `.venv/bin/python scripts/rubber_band.py run [--no-publish] [--no-alert] [--out FILE]`;
  inspect: `scripts/rubber_band.py show FILE`. State + gist id: `~/.config/rubber-band/state.json`.
- Guards: refuses to publish < 5,000 bars; drops a still-forming bar before 16:05 ET; a
  failed run posts `🪢 rubber-band run FAILED` to the alert thread and exits 1.
- Dashboard route flags `_meta.stale` after 4 days (a missed run); health check
  `check_rubber_band` warns with the age and points at the Mac mini log.
- The only decision the radar does NOT make: what action a sustained red triggers. That is
  the owner's call, written in the money-radar plan.

## v1.1 — 2026-09-08: rules re-tested on the 1971→ store, decision layer added

**Slow / fast STOP = below zero on 45 of the last 60 days** (was: 60 straight days). A single good
day no longer resets the clock. Same test rig reproduced the v1.0 result exactly (15 STOPs, all
1972-91, 0 since 1993) before anything was changed.

| rule | first fire | % of 1972-91 in STOP | episodes | false since 1993 |
|---|---|---|---|---|
| 60 straight (v1.0) | 1972-01-27 | 71% | 15 | 0 |
| 40 straight | 1971-12-30 | 77% | 16 | 0 |
| **45 of last 60 (v1.1)** | **1972-01-06** | **88%** | **8** | **0** |
| 25 straight | 1971-12-08 | 81% | 18 | 1 (2002-04-01) |

Rejected, with the reason: a depth floor (−0.15%: coverage halves to 44%; −0.50%: first fire Apr
1980, eight years late — the bad era was *mildly* negative for years, worst reading −0.96%);
CUSUM (never releases — one episode 1971→1995); "fast lowers the bar to 15 days" (false STOP
2002-03-15). Near-miss check: max negative-days-in-60 since 2000 = 33 (May 2002) vs 45.

**Rip = above zero on 45 of the last 60 days** (was 60 straight): first fire 1972-03-17 vs
1972-07-18, 7 episodes vs 9, 0 since 1993.

**Machines — judged against each leg's OWN completed history** (`leg_health` now returns
`worst_dd_prior_pct`, `longest_underwater_prior_months`, `days_before_peak`, `ret_window_pct`):
- red: drawdown deeper than any that ever completed before the current peak; amber at 85% of it.
- red: underwater longer than the longest completed stretch; amber at 75% (min 3 months).
- records only judge once a leg has 250 days before its peak.
- written lines (Main −40, C3 −54, C8-T −31) kept as red / amber within 10 pts. Only Main's line
  sits inside its own record (−51.8%); C3 (−43.8%) and C8-T (−20.3%) hit their record first.
- **retired:** m1-vs-C3 lag ≥ 2 months (a 2-month run happens ~25% of the time by chance) — still
  computed and shown, never in the verdict. Flat "9 months underwater" replaced by the record rule.
- **new (off until a hedges symphony id is pasted into SPEC):** hedges are not hedging = book
  (68/20/12) down more than 10% over 20 days while the hedge sleeve did not rise. The −10% is the
  one untested number here — backtest on the deep curves before trusting it.
- Baselines on 2026-09-08: Main record −51.8% / 7 mo · C3 −43.8% / 5 mo · m1 −44.8% / 5 mo ·
  C8-T −20.3% / 5 mo (Composer history since 2013-10 / 2016). The deep 1971→ proxies show −99%
  for every leg — the machines would not have survived the 1970s; that is what the slow dial is for.

**Blind spot unchanged:** no slow/rip variant catches 2001 or 2008 (dips kept paying next day
while the market bled). The machines dial is the grinding-bear defence — hence the redesign.

**Decision layer: `scripts/defensive_trigger.py`** (runs after the nightly, plus hourly `nag`
via com.jalal.defensive-nag 08:20–22:20; state `~/.config/rubber-band/defensive.json`).
INVESTED → PENDING_DEFENSIVE (slow or rip red on 1 close, or machines red on 5 consecutive
closes) → DONE → DEFENSIVE (every algo on, 50% of the book parked as cash in Composer) →
10 consecutive green closes with slow excess > +0.2% → PENDING_REENTRY → DONE → INVESTED.
Closes counted by `asOf` (holiday re-runs never count twice). Alarm clears before action →
stand down. Re-entry hysteresis test (fire = 45/60): exit at +0.0% = 5 round trips 1972-92,
+0.2% = 2 round trips, back in 1992-07-13, 0 days defensive after 1996; +0.5% re-enters 1997.
DONE arrives by reply in the 📡 thread (getUpdates, hourly) or `defensive_trigger.py ack` from
the concierge. Stale radar (> 4 business days without a close) → one warning a day.

Follow-ups: dashboard labels (`red_days`/`hot_days` now mean "of the last 60", `stop_after`/
`red_after` = 45; `lag_months` is info-only); paste the hedges id; commit this change.

### Hedges leg armed — 2026-09-08 evening
- Standalone symphony `TOnRRwxkXtdL0S3yXpNm` = "Hedge sleeve — GLD/BTAL 50/50 (radar leg) | NOT FUNDED": GLD 50 / BTAL 50, daily rebalance, the C6/C8 sleeve nodes copied verbatim (weights 15→50). Tree kept at `docs/hedge_sleeve_TOnRRwxkXtdL0S3yXpNm.json`. Curve 2011-09-13 → today (3,768 days). Favoriting has no API endpoint (`/watchlist`, `/favorites` → 404) — one tap in the app.
- Hedge-failure rule backtest (Composer curves, common window 2016-02-25 → 2026-09-08, 2,629 days): `book20 ≤ −10% AND hedge20 ≤ 0` true on 16 days in 8 runs; runs ≥ 5 closes = **1** (2017-06-27 → 07-07, 8 closes, book −14.5%, hedge −4.4%) → the trigger would have fired once in 10.5 years, a false alarm costing ~2 weeks half in cash. Worst book20 ever = −20.6% (2025-04-08): hedges were UP, rule correctly silent. Of 68 days with book20 ≤ −10, hedges were up on 52. Thresholds −8/−12/−15 → 30/7/1 red days. Kept −10.

### Dashboard v1.1 — 2026-09-08 late
- `RubberBandRadar.js` rewritten for the v1.1 rules: tap / double-click / Enter on any dial opens its ELI5 panel (what it measures, today's numbers, red when, amber when, the record) — one panel at a time, drawn under the dial row so phones stay readable. Legs table now shows the hedges leg, "under water now / record" and the record drawdown (the self-referential baselines), colours drawdowns against both the written line and the record, and prints the hedge check + the lag as information-only.
- New "What happens on red" strip = the decision layer: live mode badge (INVESTED / GO DEFENSIVE — waiting / DEFENSIVE — half in cash / RE-ENTER — waiting), red-close and green-run counters, and the rule in one breath on tap.
- Plumbing: the snapshot now carries `defensive` (mode, since, streaks, pending, rules). `rubber_band.run` stamps the last-known trigger state; `defensive_trigger.py` re-stamps and re-publishes the gist (`gh gist edit`) after every evaluate/ack/set or any mode change from the hourly nag, so the dashboard is never more than one action behind. `rubber-band.json` in the state dir is the file that gets stamped.
- Tests: `dashboard/__tests__/RubberBandRadar.test.js` (tap/double-click/one-at-a-time/hedges/trigger + a v1.0-snapshot fallback) alongside the older `components/__tests__` suite; Python 41 tests.

### Main retired from the machines dial — 2026-09-08 (Jalal: "Main won't be there anymore")
- The radar now watches only what the book will be after switch day (23 Sep 2026, Tranche Map S2.1 "Main + Shartino → cash"): **C3 (line −54), m1, hedges** (the Roth 68/20/12) and **C8-T (line −31)** (taxable). Main's written line (−40) stays in the money-radar plan for the record; C3 is its slimmed cousin and carries the "is the machine doing something new?" job. Until switch day the trigger's steps name C3/m1/hedges — if it fires before 23 Sep, apply them to whatever is actually held.
