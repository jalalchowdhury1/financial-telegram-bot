"""Rubber Band Radar engine — pure-maths tests (no network).

The engine lives in scripts/rubber_band.py. These tests pin the maths that the dashboard
dials, the Telegram line and the nightly alerts all depend on. Golden RSI values come
from comp_eval.rsi_arr (composer-auto-research/slimmer) so the radar and the machine's
own evaluator agree bit-for-bit on what "oversold" means.
"""
import importlib.util
import json
import pytest
import os

_spec = importlib.util.spec_from_file_location(
    "rubber_band",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "rubber_band.py"),
)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "rubber_band_rsi_golden.json")


# --- RSI -------------------------------------------------------------------
def test_rsi_wilder_matches_comp_eval_golden():
    g = json.load(open(FIX))
    out = rb.rsi_wilder(g["closes"], 10)
    assert len(out) == len(g["closes"])
    for mine, ref in zip(out, g["rsi10"]):
        if ref is None:
            assert mine is None
        else:
            assert abs(mine - ref) < 1e-5


def test_rsi_wilder_warmup_is_none_and_all_up_is_100():
    px = [float(i) for i in range(1, 30)]          # monotonically rising
    out = rb.rsi_wilder(px, 10)
    assert out[:10] == [None] * 10
    assert all(v == 100.0 for v in out[10:])


def test_rsi_wilder_too_short_series_is_all_none():
    assert rb.rsi_wilder([1.0, 2.0, 3.0], 10) == [None, None, None]


# --- returns & events -------------------------------------------------------
def test_daily_returns():
    r = rb.daily_returns([100.0, 110.0, 99.0])
    assert r[0] is None and r[1:] == pytest.approx([0.10, -0.10])


def test_dip_events_excludes_events_whose_payoff_is_not_yet_known():
    rsi = [None, 10.0, 50.0, 20.0, 60.0, 25.0]      # dips at 1, 3, 5 — but 5 is the last bar
    assert rb.events_below(rsi, 32, n=len(rsi), hold=1) == [1, 3]


def test_rip_events_uses_strict_greater_than():
    rsi = [None, 79.0, 79.5, 90.0, 10.0]
    assert rb.events_above(rsi, 79, n=len(rsi), hold=1) == [2, 3]


def test_drawdown_events_flag_ten_day_falls_deeper_than_six_percent():
    px = [100.0] * 15 + [93.0, 95.0, 100.0]        # 15: -7% from the 10-day high; 16: -5%
    assert rb.events_drawdown(px, days=10, below=-0.06, n=len(px), hold=1) == [15]


# --- the rolling window ------------------------------------------------------
def _toy():
    # 9 closes; events at 2, 4, 6. Returns: r[i] = px[i]/px[i-1]-1
    px = [100, 100, 90, 99, 90, 108, 90, 99, 100]
    ret = rb.daily_returns([float(p) for p in px])
    return px, ret


def test_window_stats_uses_last_n_paid_events_and_span_from_oldest():
    px, ret = _toy()
    st = rb.window_stats([2, 4, 6], ret, n_events=2, t=7, hold=1)
    # window = events 4 and 6 (payoffs at 5 and 7); span from 4 to 7 = 3 days
    m_ev = (ret[5] + ret[7]) / 2
    m_all = (ret[5] + ret[6] + ret[7]) / 3
    assert st["span_days"] == 3
    assert abs(st["mean_event"] - m_ev) < 1e-12
    assert abs(st["mean_all"] - m_all) < 1e-12
    assert abs(st["excess"] - (m_ev - m_all)) < 1e-12
    assert st["hit"] == 1.0
    assert st["n"] == 2
    assert st["se"] > 0


def test_window_stats_is_none_until_n_events_have_paid_off():
    px, ret = _toy()
    assert rb.window_stats([2, 4, 6], ret, n_events=3, t=6, hold=1) is None   # event 6 pays at 7
    assert rb.window_stats([2, 4, 6], ret, n_events=3, t=7, hold=1) is not None


def test_series_excess_gives_one_value_per_day_from_first_full_window():
    px, ret = _toy()
    s = rb.series_excess([2, 4, 6], ret, n_events=2, n=len(px), hold=1)
    assert len(s) == len(px)
    assert s[:5] == [None] * 5                       # 2nd payoff lands on day 5
    assert all(v is not None for v in s[5:])


def test_run_length_at_end_counts_consecutive_days_on_the_wrong_side():
    assert rb.run_length_at_end([0.1, -0.1, -0.2, -0.3], negative=True) == 3
    assert rb.run_length_at_end([-0.1, 0.1, 0.2], negative=True) == 0
    assert rb.run_length_at_end([-0.1, 0.1, 0.2], negative=False) == 2
    assert rb.run_length_at_end([None, None], negative=True) == 0


# --- colours (the tested thresholds; see docs/rubber-band.md) ---------------
def test_dip_colour_green_amber_red():
    assert rb.colour_dip(0.005, red_days=0, stop_after=60) == "green"
    assert rb.colour_dip(0.0, red_days=0, stop_after=60) == "green"
    assert rb.colour_dip(-0.001, red_days=59, stop_after=60) == "amber"
    assert rb.colour_dip(-0.001, red_days=60, stop_after=60) == "red"


def test_age_colour_thresholds_in_years():
    assert rb.colour_age(2.8) == "green"
    assert rb.colour_age(3.3) == "green"
    assert rb.colour_age(3.31) == "amber"
    assert rb.colour_age(4.01) == "red"


def test_rip_colour_hot_only_when_rips_keep_going():
    assert rb.colour_rip(-0.001, hot_days=0, red_after=60) == "green"
    assert rb.colour_rip(0.0, hot_days=0, red_after=60) == "green"
    assert rb.colour_rip(0.001, hot_days=10, red_after=60) == "amber"
    assert rb.colour_rip(0.001, hot_days=60, red_after=60) == "red"


# --- machines ------------------------------------------------------------------
def test_drawdown_and_months_underwater():
    dates = ["2026-01-05", "2026-02-02", "2026-03-02", "2026-04-01", "2026-05-01"]
    vals = [100.0, 120.0, 90.0, 100.0, 108.0]
    m = rb.leg_health(dates, vals)
    assert m["dd_pct"] == -10.0            # 108 vs peak 120
    assert m["peak_date"] == "2026-02-02"
    assert m["months_underwater"] == 3     # Feb 2 -> May 1
    assert m["worst_dd_pct"] == -25.0


def test_leg_health_at_a_new_high_is_zero():
    m = rb.leg_health(["2026-01-05", "2026-01-06"], [100.0, 101.0])
    assert m["dd_pct"] == 0.0 and m["months_underwater"] == 0


def test_monthly_lag_counts_full_months_where_a_trails_b():
    dates = ["2026-05-29", "2026-06-15", "2026-06-30", "2026-07-15", "2026-07-31", "2026-08-14", "2026-08-31", "2026-09-01"]
    a = [100, 101, 102, 103, 103, 104, 103, 103.5]       # Jun +2%, Jul +0.98%, Aug 0%
    b = [100, 100, 101, 104, 105, 105, 106, 106.2]       # Jun +1%, Jul +3.96%, Aug +0.95%
    assert rb.lag_months(dates, a, dates, b) == 2         # Jul and Aug (the last two full months)
    assert rb.lag_months(dates, b, dates, a) == 0


def test_machine_colour_breach_is_red_near_is_amber():
    legs = [{"name": "C3", "dd_pct": -55.0, "line_pct": -54, "months_underwater": 1},
            {"name": "C8-T", "dd_pct": -5.0, "line_pct": -31, "months_underwater": 1}]
    assert rb.colour_machines(legs, lag_months=0)["colour"] == "red"
    legs[0]["dd_pct"] = -45.0                            # within 10 pts of the line
    assert rb.colour_machines(legs, lag_months=0)["colour"] == "amber"
    legs[0]["dd_pct"] = -20.0
    assert rb.colour_machines(legs, lag_months=0)["colour"] == "green"
    assert rb.colour_machines(legs, lag_months=2)["colour"] == "red"       # exit-m1 rule
    legs[1]["months_underwater"] = 9
    assert rb.colour_machines(legs, lag_months=0)["colour"] == "red"


# --- snapshot & change detection --------------------------------------------------
def _synthetic_closes(n=2600, seed=7):
    import random
    rnd = random.Random(seed)
    px = [100.0]
    for _ in range(n - 1):
        px.append(px[-1] * (1 + rnd.gauss(0.0004, 0.014)))
    return px


def test_build_snapshot_has_five_dials_verdict_and_history():
    px = _synthetic_closes()
    dates = [f"D{i:05d}" for i in range(len(px))]
    curves = {"C3": (dates[-300:], px[-300:]), "m1": (dates[-300:], px[-300:]), "C8-T": (dates[-300:], px[-300:]), "Main": (dates[-300:], px[-300:])}
    snap = rb.build_snapshot(dates, px, curves, spec=rb.SPEC, generated_at="2026-09-02T22:30:00Z")
    assert snap["asOf"] == dates[-1]
    for k in ("slow", "fast", "age", "rip", "machines"):
        assert snap["dials"][k]["colour"] in ("green", "amber", "red", "grey"), k
    assert snap["verdict"]["colour"] in ("green", "amber", "red")
    assert isinstance(snap["verdict"]["text"], str) and snap["verdict"]["text"]
    assert snap["spec"]["version"] == rb.SPEC["version"]
    assert len(snap["history"]) > 100 and set(snap["history"][-1]) >= {"d", "slow", "fast", "rip"}
    assert snap["dials"]["slow"]["n"] == rb.SPEC["dip"]["slow_n"]


def test_build_snapshot_with_too_little_history_goes_grey_not_crash():
    px = _synthetic_closes(n=40)
    dates = [f"D{i:05d}" for i in range(len(px))]
    snap = rb.build_snapshot(dates, px, {}, spec=rb.SPEC, generated_at="x")
    assert snap["dials"]["slow"]["colour"] == "grey"
    assert snap["dials"]["machines"]["colour"] == "grey"
    assert snap["verdict"]["colour"] in ("green", "amber", "red")


def test_colour_changes_reports_only_dials_that_changed():
    old = {"dials": {"slow": {"colour": "green"}, "fast": {"colour": "green"}, "rip": {"colour": "green"}, "age": {"colour": "green"}, "machines": {"colour": "green"}}, "verdict": {"colour": "green"}}
    new = json.loads(json.dumps(old))
    new["dials"]["fast"]["colour"] = "amber"
    new["verdict"]["colour"] = "amber"
    ch = rb.colour_changes(old, new)
    assert [c["dial"] for c in ch] == ["fast", "verdict"]
    assert ch[0] == {"dial": "fast", "from": "green", "to": "amber"}
    assert rb.colour_changes(None, new) == []       # first run: nothing to compare, no alert storm


# --- live-path guards -------------------------------------------------------------
def test_trim_intraday_drops_todays_bar_before_the_close_only():
    from datetime import datetime
    dates = ["2026-09-01", "2026-09-02"]; px = [1.0, 2.0]
    # 15:59 ET on the 2nd: today's bar is still forming -> drop it
    d, p = rb.trim_intraday(dates, px, now_et=datetime(2026, 9, 2, 15, 59))
    assert d == ["2026-09-01"] and p == [1.0]
    # 18:30 ET on the 2nd: the close is final -> keep it
    d, p = rb.trim_intraday(dates, px, now_et=datetime(2026, 9, 2, 18, 30))
    assert d == dates and p == px
    # next morning: yesterday's bar is obviously final
    d, p = rb.trim_intraday(dates, px, now_et=datetime(2026, 9, 3, 9, 0))
    assert d == dates


def test_run_refuses_a_truncated_price_history(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "STATE_DIR", str(tmp_path))
    short = ([f"D{i:05d}" for i in range(400)], [100.0 + (i % 7) for i in range(400)])
    with pytest.raises(RuntimeError, match="too short"):
        rb.run(publish=False, alert=False, fetch_closes=lambda: short, fetch_curves=lambda: {}, log=lambda *_: None)


def test_run_end_to_end_with_injected_sources_alerts_only_on_change(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "STATE_DIR", str(tmp_path))
    px = _synthetic_closes(n=6000)
    dates = [f"2000-01-{i:05d}" for i in range(len(px))]       # not real dates; only ordering matters
    sent = []
    args = dict(publish=False, fetch_closes=lambda: (dates, px), fetch_curves=lambda: {},
                notifier=lambda text: sent.append(text) or True, log=lambda *_: None)
    snap1 = rb.run(**args)                       # first run: no previous state -> no alert
    assert sent == [] and snap1["asOf"] == dates[-1]
    snap2 = rb.run(**args)                       # same data -> same colours -> no alert
    assert sent == []
    # now force a colour change in the stored state and rerun -> exactly one alert
    st = rb.load_state(); st["last_snapshot"]["dials"]["slow"]["colour"] = "red"; rb.save_state(st)
    rb.run(**args)
    assert len(sent) == 1 and "slow" in sent[0]


def test_main_reports_a_failed_run_to_the_alert_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(rb, "STATE_DIR", str(tmp_path))
    sent = []
    monkeypatch.setattr(rb, "send_alert", lambda text: sent.append(text) or True)
    def boom(): raise RuntimeError("yfinance returned no rows")
    monkeypatch.setattr(rb, "fetch_qqq_closes", boom)
    rc = rb.main(["rubber_band.py", "run", "--no-publish"])
    assert rc != 0 and len(sent) == 1 and "FAILED" in sent[0] and "yfinance" in sent[0]
