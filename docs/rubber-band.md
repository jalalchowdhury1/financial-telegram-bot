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
