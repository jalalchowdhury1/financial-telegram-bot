#!/usr/bin/env python3
"""
Rubber Band Radar — "is the dip-buying regime still alive?" — five dials, nightly.

Runs on the Mac mini after the close, writes one JSON snapshot that the dashboard
(/api/rubber-band -> <RubberBandRadar/>) and the Telegram brief both read, and
alerts the 📡 thread only when a dial changes colour. Pure maths lives in the
top half (no I/O, unit-tested in tests/test_rubber_band.py); network and shell
live in the bottom half behind injectable functions.

The five dials (every threshold below was tested on QQQ 1971→, see docs/rubber-band.md):
  1 slow   — last 30 oversold dips (Wilder RSI-10 < 32 ≈ the machine's TQQQ<31 trigger):
             next-day return minus the market's own drift over the same span. Below zero for
             60 straight days = STOP (fired 15× 1972-91, never since 1993).
  2 fast   — same maths, last 20 dips. LOOK only: leads the slow line by months in a real
             flip, but every alarm since 1993 was false.
  3 age    — how many years the 30 dips span (fresh evidence < 3.3y, stale > 4y).
  4 rip    — last 30 overbought days (RSI-10 > 79 = the machine's sell trigger): if the
             market keeps rising after a rip for 60 straight days, the 1970s are back.
  5 machines — each leg's backtest drawdown vs the written tripwire lines, months underwater,
             m1-vs-C3 lag.

CLI:
    python scripts/rubber_band.py run [--out FILE] [--no-publish] [--no-alert]
    python scripts/rubber_band.py show FILE
"""
import bisect
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- The spec: every number here is a tested, written-down decision -----------------
SPEC = {
    "version": "1.0",
    "ticker": "QQQ",
    "rsi_window": 10,
    "dip": {"rsi_below": 32, "slow_n": 30, "fast_n": 20, "hold_days": 1, "stop_after_red_days": 60},
    "crosscheck": {"dd_days": 10, "dd_below": -0.06, "n": 30},
    "age": {"amber_years": 3.3, "red_years": 4.0},          # p75 / p90 of the span since 1993
    "rip": {"rsi_above": 79, "n": 30, "red_after_hot_days": 60},
    "machines": {
        # written lines from the money-radar plan §3.2; None = no line written for that leg
        "legs": [
            {"name": "Main", "id": "xL9KQGN5FWIPcA8WAop9", "line_pct": -40},
            {"name": "C3",   "id": "oJF4TTzhjbS8YrfOEqvK", "line_pct": -54},
            {"name": "m1",   "id": "oYAwQVVRyUFHln4sWbFV", "line_pct": None},
            {"name": "C8-T", "id": "LyRMxIoIqQ4X1ywtf1az", "line_pct": -31},
        ],
        "near_line_pts": 10,
        "underwater_months_line": 9,
        "lag_pair": ["m1", "C3"],
        "lag_months_line": 2,
    },
    "history_days": 750,
}

TRADING_DAYS_PER_YEAR = 252


# =====================================================================================
# Pure maths
# =====================================================================================
def rsi_wilder(px, w):
    """Full-history Wilder RSI: seed = SMA of the first w changes, then
    avg = (avg*(w-1) + change)/w. Bit-for-bit the same as comp_eval.Store.rsi_arr."""
    n = len(px)
    out = [None] * n
    if n <= w:
        return out
    g = l = 0.0
    for i in range(1, w + 1):
        ch = px[i] - px[i - 1]
        if ch > 0:
            g += ch
        else:
            l -= ch
    ag, al = g / w, l / w
    out[w] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(w + 1, n):
        ch = px[i] - px[i - 1]
        ag = (ag * (w - 1) + max(ch, 0.0)) / w
        al = (al * (w - 1) + max(-ch, 0.0)) / w
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def daily_returns(px):
    return [None] + [px[i] / px[i - 1] - 1 for i in range(1, len(px))]


def events_below(rsi, thr, n, hold):
    """Days whose RSI is below thr AND whose payoff (close hold days later) is already known.
    The last bar can be an event but never counts until its payoff prints — no look-ahead."""
    return [i for i, v in enumerate(rsi) if v is not None and v < thr and i + hold < n]


def events_above(rsi, thr, n, hold):
    return [i for i, v in enumerate(rsi) if v is not None and v > thr and i + hold < n]


def events_drawdown(px, days, below, n, hold):
    out = []
    for i in range(days, n):
        if px[i] / max(px[i - days:i + 1]) - 1 < below and i + hold < n:
            out.append(i)
    return out


def _hold_return(ret, i, hold):
    r = 1.0
    for k in range(i + 1, i + hold + 1):
        r *= 1 + ret[k]
    return r - 1


def window_stats(events, ret, n_events, t, hold, _prefix=None):
    """Stats of the last n_events events whose payoff landed on or before day t.
    excess = mean event payoff − mean market return over the same span (per `hold` days)."""
    paydays = [i + hold for i in events]
    j = bisect.bisect_right(paydays, t)
    if j < n_events:
        return None
    win = events[j - n_events:j]
    a = win[0]
    pays = [_hold_return(ret, i, hold) for i in win]
    m_ev = sum(pays) / n_events
    if _prefix is not None:
        m_all = (_prefix[t] - _prefix[a]) / (t - a) * hold
    else:
        m_all = sum(ret[k] for k in range(a + 1, t + 1)) / (t - a) * hold
    se = statistics.stdev(pays) / math.sqrt(n_events) if n_events > 1 else 0.0
    return {
        "n": n_events,
        "mean_event": m_ev,
        "mean_all": m_all,
        "excess": m_ev - m_all,
        "hit": sum(1 for p in pays if p > 0) / n_events,
        "span_days": t - a,
        "se": se,
        "first_event_index": a,
        "last_event_index": win[-1],
    }


def _prefix_sums(ret):
    p = [0.0] * len(ret)
    for i in range(1, len(ret)):
        p[i] = p[i - 1] + ret[i]
    return p


def series_excess(events, ret, n_events, n, hold):
    """One excess value per day (None until n_events payoffs exist)."""
    pre = _prefix_sums(ret)
    out = [None] * n
    for t in range(n):
        st = window_stats(events, ret, n_events, t, hold, _prefix=pre)
        if st is not None:
            out[t] = st["excess"]
    return out


def run_length_at_end(vals, negative=True):
    k = 0
    for v in reversed(vals):
        if v is None:
            break
        if (v < 0) if negative else (v > 0):
            k += 1
        else:
            break
    return k


def colour_dip(excess, red_days, stop_after):
    if excess >= 0:
        return "green"
    return "red" if red_days >= stop_after else "amber"


def colour_age(years, spec=SPEC):
    if years <= spec["age"]["amber_years"]:
        return "green"
    if years <= spec["age"]["red_years"]:
        return "amber"
    return "red"


def colour_rip(excess, hot_days, red_after):
    if excess <= 0:
        return "green"
    return "red" if hot_days >= red_after else "amber"


# --- machines ---------------------------------------------------------------------------
def _months_between(d1, d2, fallback_days=None):
    """Calendar months between two ISO dates; non-ISO labels fall back to trading days / 21."""
    try:
        y1, m1 = int(d1[:4]), int(d1[5:7])
        y2, m2 = int(d2[:4]), int(d2[5:7])
        return (y2 - y1) * 12 + (m2 - m1)
    except ValueError:
        return (fallback_days or 0) // 21


def leg_health(dates, vals):
    peak, peak_date, peak_i, worst = -1.0, None, 0, 0.0
    for i, (d, v) in enumerate(zip(dates, vals)):
        if v > peak:
            peak, peak_date, peak_i = v, d, i
        dd = v / peak - 1
        worst = min(worst, dd)
    last = vals[-1]
    dd_pct = round((last / peak - 1) * 100, 2)
    return {
        "dd_pct": dd_pct,
        "peak_date": peak_date,
        "months_underwater": 0 if dd_pct >= 0 else _months_between(peak_date, dates[-1], len(vals) - 1 - peak_i),
        "worst_dd_pct": round(worst * 100, 2),
        "asOf": dates[-1],
    }


def _month_end_values(dates, vals):
    """{YYYY-MM: last value in that month}, in date order."""
    out = {}
    for d, v in zip(dates, vals):
        out[d[:7]] = v
    return out


def lag_months(dates_a, a, dates_b, b):
    """Consecutive most-recent COMPLETE months in which leg a's monthly return trailed leg b's.
    The month containing the last data point is treated as incomplete and skipped."""
    ma, mb = _month_end_values(dates_a, a), _month_end_values(dates_b, b)
    months = sorted(set(ma) & set(mb))
    if not months:
        return 0
    months = months[:-1]                       # drop the (incomplete) current month
    k = 0
    for i in range(len(months) - 1, 0, -1):
        m, prev = months[i], months[i - 1]
        ra = ma[m] / ma[prev] - 1
        rb_ = mb[m] / mb[prev] - 1
        if ra < rb_:
            k += 1
        else:
            break
    return k


def colour_machines(legs, lag_months, spec=SPEC):
    m = spec["machines"]
    reasons, colour = [], "green"
    for leg in legs:
        line = leg.get("line_pct")
        dd = leg.get("dd_pct")
        if dd is None:
            continue
        if line is not None and dd <= line:
            colour = "red"
            reasons.append(f"{leg['name']} is through its line ({dd:+.0f}% vs {line:+.0f}%)")
        elif line is not None and dd <= line + m["near_line_pts"] and colour != "red":
            colour = "amber"
            reasons.append(f"{leg['name']} is within {m['near_line_pts']} points of its line ({dd:+.0f}% vs {line:+.0f}%)")
        if (leg.get("months_underwater") or 0) >= m["underwater_months_line"]:
            colour = "red"
            reasons.append(f"{leg['name']} has been underwater {leg['months_underwater']} months (line: {m['underwater_months_line']})")
    if lag_months >= m["lag_months_line"]:
        colour = "red"
        reasons.append(f"{m['lag_pair'][0]} has lagged {m['lag_pair'][1]} {lag_months} months running — the written rule says exit {m['lag_pair'][0]}")
    return {"colour": colour, "reasons": reasons}


# --- verdict ------------------------------------------------------------------------------
_RANK = {"green": 0, "grey": 0, "amber": 1, "red": 2}


def verdict(dials):
    slow, fast, age, rip, mach = (dials[k] for k in ("slow", "fast", "age", "rip", "machines"))
    worst = "green"
    for c in (slow["colour"], rip["colour"], mach["colour"]):
        if _RANK[c] > _RANK[worst]:
            worst = c
    if worst == "green" and fast["colour"] == "red":
        worst = "amber"
    if slow["colour"] == "grey" and worst == "green":
        worst = "amber"                                    # can't vouch for the band = caution, never green
    bits = []
    if slow["colour"] == "grey":
        bits.append("Not enough history yet to judge the rubber band.")
    elif slow["colour"] == "green":
        bits.append(f"The rubber band is working: the last {slow['n']} dips paid {slow['excess_pct']:+.2f}% more than an ordinary day.")
    elif slow["colour"] == "amber":
        bits.append(f"Dip-buying has lost money for {slow['red_days']} days running — watch, it becomes STOP at {slow['stop_after']}.")
    else:
        bits.append(f"STOP: dip-buying has lost money for {slow['red_days']} straight days — this is what 1973–1990 looked like.")
    if fast["colour"] == "red":
        bits.append("The fast line is red (a LOOK, not an order — every fast alarm since 1993 was false).")
    if age["colour"] != "green" and age["colour"] != "grey":
        bits.append(f"The evidence is {age['years']}y old — quiet market, few dips; a real flip would refresh it within months.")
    if rip["colour"] == "red":
        bits.append("Rips keep running instead of fading — the 1970s pattern; the sell-the-rip half is broken.")
    elif rip["colour"] == "amber":
        bits.append(f"Rips have kept running for {rip['hot_days']} days — watch.")
    if mach["colour"] == "red":
        bits.append("A written tripwire is hit: " + "; ".join(mach["reasons"]) + ".")
    elif mach["colour"] == "amber":
        bits.append("A leg is near its line: " + "; ".join(mach["reasons"]) + ".")
    elif mach["colour"] == "green":
        bits.append("All legs inside their lines.")
    return {"colour": worst, "text": " ".join(bits)}


# --- snapshot ---------------------------------------------------------------------------
def _pct(x, nd=3):
    return None if x is None else round(100 * x, nd)


def build_snapshot(dates, px, curves, spec=SPEC, generated_at=None):
    """dates/px: full QQQ close history (oldest first). curves: {leg name: (dates, values)}."""
    n = len(px)
    w, hold = spec["rsi_window"], spec["dip"]["hold_days"]
    rsi = rsi_wilder(px, w)
    ret = daily_returns(px)
    t = n - 1
    grey = {"colour": "grey"}

    dips = events_below(rsi, spec["dip"]["rsi_below"], n, hold)
    rips = events_above(rsi, spec["rip"]["rsi_above"], n, hold)
    dds = events_drawdown(px, spec["crosscheck"]["dd_days"], spec["crosscheck"]["dd_below"], n, hold)

    slow_s = series_excess(dips, ret, spec["dip"]["slow_n"], n, hold)
    fast_s = series_excess(dips, ret, spec["dip"]["fast_n"], n, hold)
    rip_s = series_excess(rips, ret, spec["rip"]["n"], n, hold)

    def dip_dial(n_events, series, stop_after):
        st = window_stats(dips, ret, n_events, t, hold)
        if st is None:
            return dict(grey, n=n_events, reason="fewer than %d paid-off dips in the data" % n_events)
        red_days = run_length_at_end(series, negative=True)
        return {
            "colour": colour_dip(st["excess"], red_days, stop_after),
            "n": n_events,
            "excess_pct": _pct(st["excess"]),
            "mean_event_pct": _pct(st["mean_event"]),
            "mean_all_pct": _pct(st["mean_all"]),
            "se_pct": _pct(st["se"]),
            "hit": round(st["hit"], 3),
            "span_days": st["span_days"],
            "span_years": round(st["span_days"] / TRADING_DAYS_PER_YEAR, 1),
            "first_event": dates[st["first_event_index"]],
            "last_event": dates[st["last_event_index"]],
            "red_days": red_days,
            "stop_after": stop_after,
        }

    slow = dip_dial(spec["dip"]["slow_n"], slow_s, spec["dip"]["stop_after_red_days"])
    fast = dip_dial(spec["dip"]["fast_n"], fast_s, spec["dip"]["stop_after_red_days"])
    fast["look_only"] = True

    cc = window_stats(dds, ret, spec["crosscheck"]["n"], t, hold)
    slow["crosscheck"] = (
        {"kind": "10-day drawdown > 6%", "excess_pct": _pct(cc["excess"]), "hit": round(cc["hit"], 3),
         "n": cc["n"], "span_years": round(cc["span_days"] / TRADING_DAYS_PER_YEAR, 1),
         "agrees": (cc["excess"] >= 0) == (slow.get("excess_pct", 0) >= 0)}
        if cc is not None and slow["colour"] != "grey" else None
    )

    if slow["colour"] == "grey":
        age = dict(grey)
    else:
        age = {"colour": colour_age(slow["span_years"], spec), "years": slow["span_years"],
               "amber_years": spec["age"]["amber_years"], "red_years": spec["age"]["red_years"],
               "events_last_12m": sum(1 for i in dips if i > t - TRADING_DAYS_PER_YEAR)}

    rs = window_stats(rips, ret, spec["rip"]["n"], t, hold)
    if rs is None:
        rip = dict(grey, n=spec["rip"]["n"])
    else:
        hot_days = run_length_at_end(rip_s, negative=False)
        rip = {
            "colour": colour_rip(rs["excess"], hot_days, spec["rip"]["red_after_hot_days"]),
            "n": rs["n"], "excess_pct": _pct(rs["excess"]), "mean_event_pct": _pct(rs["mean_event"]),
            "se_pct": _pct(rs["se"]), "hit": round(rs["hit"], 3),
            "span_years": round(rs["span_days"] / TRADING_DAYS_PER_YEAR, 1),
            "first_event": dates[rs["first_event_index"]], "last_event": dates[rs["last_event_index"]],
            "hot_days": hot_days, "red_after": spec["rip"]["red_after_hot_days"],
        }

    legs = []
    for leg in spec["machines"]["legs"]:
        c = curves.get(leg["name"])
        row = {"name": leg["name"], "line_pct": leg["line_pct"]}
        if c and len(c[1]) >= 2:
            row.update(leg_health(c[0], c[1]))
        else:
            row.update({"dd_pct": None, "months_underwater": None, "asOf": None, "missing": True})
        legs.append(row)
    a_name, b_name = spec["machines"]["lag_pair"]
    lag = 0
    if curves.get(a_name) and curves.get(b_name):
        lag = lag_months(curves[a_name][0], curves[a_name][1], curves[b_name][0], curves[b_name][1])
    if all(l.get("missing") for l in legs):
        machines = dict(grey, legs=legs, lag_months=None, reasons=["no machine curves this run"])
    else:
        cm = colour_machines(legs, lag, spec)
        machines = {"colour": cm["colour"], "reasons": cm["reasons"], "legs": legs, "lag_months": lag,
                    "lag_pair": spec["machines"]["lag_pair"],
                    "underwater_months_line": spec["machines"]["underwater_months_line"]}

    dials = {"slow": slow, "fast": fast, "age": age, "rip": rip, "machines": machines}
    hist = []
    for i in range(max(0, n - spec["history_days"]), n):
        hist.append({"d": dates[i], "px": round(px[i], 2), "rsi": None if rsi[i] is None else round(rsi[i], 1),
                     "slow": _pct(slow_s[i]), "fast": _pct(fast_s[i]), "rip": _pct(rip_s[i])})
    return {
        "asOf": dates[-1],
        "generatedAt": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": spec,
        "verdict": verdict(dials),
        "dials": dials,
        "counts": {"dips_total": len(dips), "rips_total": len(rips), "bars": n, "first_bar": dates[0]},
        "history": hist,
    }


def colour_changes(old, new):
    if not old:
        return []
    out = []
    for k in ("slow", "fast", "age", "rip", "machines"):
        a = ((old.get("dials") or {}).get(k) or {}).get("colour")
        b = ((new.get("dials") or {}).get(k) or {}).get("colour")
        if a != b:
            out.append({"dial": k, "from": a, "to": b})
    a, b = (old.get("verdict") or {}).get("colour"), (new.get("verdict") or {}).get("colour")
    if a != b:
        out.append({"dial": "verdict", "from": a, "to": b})
    return out


# =====================================================================================
# I/O — sources, publishing, alerts (everything injectable for tests)
# =====================================================================================
COMPOSER_BASE = "https://api.composer.trade/api/v0.1"
COMPOSER_ENV_FILE = os.environ.get("COMPOSER_ENV_FILE",
                                   os.path.expanduser("~/PycharmProjects/composer-auto-research/.env"))
STATE_DIR = os.path.expanduser(os.environ.get("RUBBER_BAND_STATE_DIR", "~/.config/rubber-band"))


MIN_BARS = 5000          # QQQ has ~6,900 bars since 1999; far fewer = a truncated feed, refuse to publish
CLOSE_FINAL_ET = (16, 5)  # a bar dated today is only trusted after 16:05 ET


def trim_intraday(dates, px, now_et=None):
    """Drop a bar dated today if the session hasn't closed yet (a forming intraday bar
    would otherwise be published as a close)."""
    if now_et is None:
        try:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
        except Exception:
            now_et = datetime.now()
    if dates and dates[-1] == now_et.strftime("%Y-%m-%d") and (now_et.hour, now_et.minute) < CLOSE_FINAL_ET:
        return dates[:-1], px[:-1]
    return dates, px


def fetch_qqq_closes(ticker="QQQ", adjusted=True):
    """Full-history daily closes from yfinance (the Mac mini can reach it; Vercel cannot).
    ADJUSTED closes: verified bit-for-bit against Composer's own RSI(QQQ,10)<32 trigger
    days (194/194 since 2010); raw closes disagree on 4 days. See docs/rubber-band.md."""
    import yfinance as yf
    df = yf.Ticker(ticker).history(period="max", auto_adjust=adjusted, actions=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no rows for %s" % ticker)
    df = df[df["Close"].notna()]
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    return trim_intraday(dates, [float(v) for v in df["Close"].tolist()])


def _composer_headers():
    from dotenv import load_dotenv
    if os.path.exists(COMPOSER_ENV_FILE):
        load_dotenv(COMPOSER_ENV_FILE)
    kid, sec = os.environ.get("COMPOSER_KEY_ID"), os.environ.get("COMPOSER_SECRET")
    if not kid or not sec:
        raise RuntimeError("COMPOSER_KEY_ID / COMPOSER_SECRET not set (see COMPOSER_ENV_FILE)")
    return {"Content-Type": "application/json", "Accept": "application/json",
            "x-api-key-id": kid, "authorization": f"Bearer {sec}"}


def _strip(node, root=False):
    if not root:
        for k in ("name", "description", "collapsed?"):
            node.pop(k, None)
    for c in node.get("children") or []:
        _strip(c)


def fetch_curve(symphony_id):
    """(dates, values) of a symphony's Composer backtest through its last available day."""
    import requests
    H = _composer_headers()
    r = requests.get(f"{COMPOSER_BASE}/symphonies/{symphony_id}/score", headers=H, timeout=60)
    r.raise_for_status()
    tree = r.json()
    _strip(tree, root=True)
    raw = tree if "description" in tree else {**tree, "description": tree.get("name") or "x"}
    body = {"capital": 10000, "slippage_percent": 0.0001, "apply_reg_fee": True, "apply_taf_fee": True,
            "benchmark_tickers": ["SPY"],
            "symphony": {"raw_value": raw, "encoding_type": "edn", "encoded_value": ""}}
    for attempt in range(6):
        r = requests.post(f"{COMPOSER_BASE}/backtest", json=body, headers=H, timeout=300)
        if r.status_code == 200:
            dv = r.json().get("dvm_capital") or {}
            curve = next((v for k, v in dv.items() if k != "SPY"), None)
            if not curve:
                raise RuntimeError("backtest returned no curve for %s" % symphony_id)
            days = sorted(curve, key=int)
            epoch = date(1970, 1, 1)
            return ([(epoch + timedelta(days=int(k))).isoformat() for k in days],
                    [float(curve[k]) for k in days])
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(20, 2 * (2 ** attempt)))
            continue
        raise RuntimeError("Composer backtest HTTP %s: %s" % (r.status_code, r.text[:200]))
    raise RuntimeError("Composer backtest kept failing for %s" % symphony_id)


def fetch_all_curves(spec=SPEC, fetch=fetch_curve, log=print):
    """Never-throw: a leg that fails is simply absent (the dial shows it as missing)."""
    out = {}
    for leg in spec["machines"]["legs"]:
        try:
            out[leg["name"]] = fetch(leg["id"])
            log(f"  curve {leg['name']}: {len(out[leg['name']][0])} days to {out[leg['name']][0][-1]}")
        except Exception as e:                       # noqa: BLE001 — surfaced in the snapshot, not fatal
            log(f"  curve {leg['name']}: FAILED ({e})")
    return out


def _state_path(name):
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, name)


def load_state():
    p = _state_path("state.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(state):
    json.dump(state, open(_state_path("state.json"), "w"), indent=1)


def publish_gist(snapshot_path, state):
    """Secret gist = the snapshot's home. Created once, edited every night. Returns raw URL."""
    gid = state.get("gist_id")
    if not gid:
        out = subprocess.run(["gh", "gist", "create", "--filename", "rubber-band.json", snapshot_path],
                             capture_output=True, text=True, check=True).stdout.strip()
        gid = out.rstrip("/").split("/")[-1]
        state["gist_id"] = gid
        save_state(state)
    else:
        subprocess.run(["gh", "gist", "edit", gid, "-f", "rubber-band.json", snapshot_path],
                       capture_output=True, text=True, check=True)
    user = subprocess.run(["gh", "api", "user", "--jq", ".login"], capture_output=True, text=True, check=True).stdout.strip()
    return f"https://gist.githubusercontent.com/{user}/{gid}/raw/rubber-band.json"


def send_alert(text):
    """Telegram to the alerts thread (token from the repo's .env, chat from the environment)."""
    import requests
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("RUBBER_BAND_ALERT_CHAT")
    if not token or not chat:
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=20)
    return r.status_code == 200


EMOJI = {"green": "🟢", "amber": "🟡", "red": "🔴", "grey": "⚪"}


def brief_line(snap):
    d = snap["dials"]
    lights = "".join(EMOJI[d[k]["colour"]] for k in ("slow", "fast", "age", "rip", "machines"))
    return f"🪢 Rubber band {lights} — {snap['verdict']['text']} (as of {snap['asOf']})"


def alert_text(snap, changes):
    lines = [f"🪢 <b>Rubber Band Radar changed</b> (as of {snap['asOf']})"]
    for c in changes:
        lines.append(f"• {c['dial']}: {EMOJI.get(c['from'], '?')} → {EMOJI.get(c['to'], '?')}")
    lines.append(snap["verdict"]["text"])
    return "\n".join(lines)


def run(out_path=None, publish=True, alert=True, fetch_closes=None,
        fetch_curves=None, publisher=None, notifier=None, log=print):
    # defaults resolved at call time so tests (and monkeypatching) can swap the I/O
    fetch_closes = fetch_closes or fetch_qqq_closes
    fetch_curves = fetch_curves or fetch_all_curves
    publisher = publisher or publish_gist
    notifier = notifier or send_alert
    log(f"rubber-band run {datetime.now().isoformat(timespec='seconds')}")
    dates, px = fetch_closes()
    if len(px) < MIN_BARS:
        raise RuntimeError(f"price history too short ({len(px)} bars < {MIN_BARS}) — refusing to publish")
    log(f"  closes: {len(px)} bars {dates[0]} → {dates[-1]}")
    curves = fetch_curves()
    snap = build_snapshot(dates, px, curves)
    state = load_state()
    prev = state.get("last_snapshot")
    changes = colour_changes(prev, snap)
    snap["changesSinceLastRun"] = changes
    out_path = out_path or _state_path("rubber-band.json")
    with open(out_path, "w") as f:
        json.dump(snap, f, indent=1)
    log("  " + brief_line(snap))
    if publish:
        url = publisher(out_path, state)
        log(f"  published → {url}")
        state["published_url"] = url
    if alert and changes:
        ok = notifier(alert_text(snap, changes))
        log(f"  alert ({len(changes)} changes): {'sent' if ok else 'NOT sent'}")
    state["last_snapshot"] = {"asOf": snap["asOf"], "dials": {k: {"colour": v["colour"]} for k, v in snap["dials"].items()},
                              "verdict": {"colour": snap["verdict"]["colour"]}}
    state["last_run"] = snap["generatedAt"]
    save_state(state)
    return snap


def main(argv):
    if len(argv) >= 2 and argv[1] == "run":
        try:
            run(out_path=(argv[argv.index("--out") + 1] if "--out" in argv else None),
                publish="--no-publish" not in argv, alert="--no-alert" not in argv)
            return 0
        except Exception as e:                   # noqa: BLE001 — a silent nightly failure is the worst outcome
            msg = f"🪢 rubber-band run FAILED: {type(e).__name__}: {str(e)[:300]}"
            print(msg)
            if "--no-alert" not in argv:
                send_alert(msg)
            return 1
    if len(argv) >= 3 and argv[1] == "show":
        snap = json.load(open(argv[2]))
        print(brief_line(snap))
        for k, v in snap["dials"].items():
            print(f"  {k:9s} {EMOJI[v['colour']]} {json.dumps({kk: vv for kk, vv in v.items() if kk not in ('legs', 'crosscheck')})}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
