#!/usr/bin/env python3
"""
Defensive trigger — turns the Rubber Band Radar's colours into the ONE decision Jalal never has to make.

The radar (rubber_band.py, nightly 18:30) measures. This script decides and nags. It never touches
Composer: Jalal keeps the hands ("nothing moves without Jalal", "NEVER touch the live symphony").
Alerts carry ✅ Done / ⏰ 2 h buttons; taps land in health-hub KV (api/defensive.js webhook) and are read
back here once an hour. This script no longer calls getUpdates, so the webhook can stay set on the bot.

State machine (~/.config/rubber-band/defensive.json):

  INVESTED --fire--> PENDING_DEFENSIVE --DONE--> DEFENSIVE --10 green closes--> PENDING_REENTRY --DONE--> INVESTED
      ^                   | alarm clears                                            | streak breaks
      +----stand down-----+                                    DEFENSIVE <----hold--+

Fire  = slow red on 1 close, or rip red on 1 close (both already carry the 45-of-60-day test inside the
        radar), or machines red on 5 consecutive closes (the only dial that can flip overnight).
Re-entry = verdict green on 10 consecutive closes with the dip edge above +0.2% on each one
        (1971-> test: 2 round trips in 1972-92, back in by Jul 1992, never in cash since).
Closes are counted by the snapshot's asOf, so a holiday re-run never counts twice.
DEFENSIVE = every algo stays on, 50% of the book parked as cash in Composer (Jalal, 2026-09-08).

CLI:
  defensive_trigger.py evaluate [--snapshot F] [--dry]   after the nightly radar run
  defensive_trigger.py nag [--dry]                       hourly: pick DONE up from the 📡 thread, remind
  defensive_trigger.py ack                               Jalal said DONE through the concierge
  defensive_trigger.py set INVESTED|DEFENSIVE            manual correction (no message)
  defensive_trigger.py steps                             preview today's GO DEFENSIVE / RE-ENTER steps (no send)
  defensive_trigger.py status
"""
import html
import json
import os
import subprocess
import shutil
import re
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tranche_state as tranche                      # noqa: E402 — the Tranche Map's tick boxes + INSTRUMENTS.json

STATE_DIR = os.path.expanduser(os.environ.get("RUBBER_BAND_STATE_DIR", "~/.config/rubber-band"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FALLBACK_CHAT = "7956935476"                         # 📡 alerts thread, never the family group
FIRE_AFTER = {"slow": 1, "rip": 1, "machines": 5}    # consecutive red CLOSES per dial
REENTRY_CLOSES = 10
REENTRY_EDGE_PCT = 0.2                               # slow dial's excess must be above this on every green close
STALE_BUSINESS_DAYS = 4                              # no new close for this long = the trigger is blind, say so
STEPS_EVERY_N_REMINDERS = 6                          # repeat the full steps every ~6 hours of nagging
ACK_RE = re.compile(r"^\s*(done|✅|defensive done|re-?entered|back in|cash is in|cash is back)\b", re.I)
MODES = ("INVESTED", "PENDING_DEFENSIVE", "DEFENSIVE", "PENDING_REENTRY")
def BUTTONS(kind="PENDING_DEFENSIVE"):
    """Inline buttons whose labels say the OUTCOME (Jalal, 11 Sep 2026); taps land in health-hub KV."""
    done = "✅ Done — cash parked" if kind == "PENDING_DEFENSIVE" else "✅ Done — cash back in"
    return {"inline_keyboard": [[{"text": done, "callback_data": "dt:done"},
                                 {"text": "⏰ Quiet 2 h", "callback_data": "dt:snooze"}]]}
_RANK = {"green": 0, "grey": 0, "amber": 1, "red": 2}


# --- state --------------------------------------------------------------------------------
def _path(name):
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, name)


def fresh_state():
    return {"mode": "INVESTED", "last_asof": None, "streak": {"slow": 0, "rip": 0, "machines": 0},
            "green_streak": 0, "pending": None, "history": []}


def load_state(path=None):
    p = path or _path("defensive.json")
    st = fresh_state()
    if os.path.exists(p):
        st.update(json.load(open(p)))
    return st


def save_state(st):
    json.dump(st, open(_path("defensive.json"), "w"), indent=1)


def _record(st, now, asof, to, note):
    st["history"] = (st.get("history") or [])[-199:] + [{"t": now.isoformat(), "asof": asof, "to": to, "note": note}]


# --- telegram (stdlib only; token pulled by NAME from the repo .env, never sourced wholesale) ---
def _token():
    try:
        for line in open(os.path.join(REPO, ".env")):
            if line.startswith("TELEGRAM_TOKEN="):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return None


def _chat():
    return str(os.environ.get("RUBBER_BAND_ALERT_CHAT") or FALLBACK_CHAT)


def send_telegram(text, buttons=None):
    tok = _token()
    if not tok:
        print("  no TELEGRAM_TOKEN — not sent")
        return False
    payload = {"chat_id": _chat(), "text": text, "parse_mode": "HTML"}
    if buttons:
        payload["reply_markup"] = json.dumps(buttons)
    body = urllib.parse.urlencode(payload).encode()
    try:
        req = urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=body)
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception as e:                       # noqa: BLE001 — a failed send is logged, never fatal
        print(f"  telegram send failed: {e}")
        return False


def _send_with_buttons(send, text, kind="PENDING_DEFENSIVE"):
    """Try sending with buttons; fall back to plain text if the callable only accepts one arg."""
    try:
        return send(text, BUTTONS(kind))
    except TypeError:
        return send(text)


def _env_value(name):
    """One value by NAME from the repo .env (launchd never sources it); a real env var wins."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        for line in open(os.path.join(REPO, ".env")):
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return None


def get_taps():
    """The latest button tap, read ONCE from the health-hub tap endpoint (api/defensive.js → KV).
    {} when nothing was tapped, None when the endpoint is unconfigured or unreachable."""
    url, key = _env_value("DEFENSIVE_TAP_URL"), _env_value("DEFENSIVE_TAP_KEY")
    if not url or not key:
        print("  no DEFENSIVE_TAP_URL/KEY in .env — taps unavailable")
        return None
    try:
        with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode({'k': key})}", timeout=20) as r:
            return json.load(r).get("tap") or {}
    except Exception as e:                       # noqa: BLE001 — the reminder still goes out without taps
        print(f"  tap endpoint failed: {e}")
        return None


def _book(snap):
    return ((snap.get("spec") or {}).get("machines") or {}).get("book") or {"C3": 0.68, "m1": 0.20, "hedges": 0.12}


def default_ctx(mode):
    """What the Tranche Map says today: funded symphonies + collision notes for `mode`. Empty when the files are absent."""
    return tranche.tranche_context(mode)


_LOGIN = {"jalal": "Your Composer login", "nabila": "Nabila's Composer login"}


def _by_login(funded):
    for login in ("jalal", "nabila"):
        mine = [f for f in funded if f["login"] == login]
        if mine:
            yield _LOGIN.get(login, f"{login}'s Composer login"), mine


def steps_defensive(snap, ctx=None):
    """Names the symphonies that hold money TODAY (Tranche Map ticks), grouped by whose Composer login."""
    funded = (ctx or {}).get("funded") or []
    if funded:
        lines, n = [], 0
        for heading, mine in _by_login(funded):
            lines.append(f"<i>{heading}</i>")
            for f in mine:
                n += 1
                lines.append(f"{n}. Composer → <b>{html.escape(f['name'])}</b> ({html.escape(f['account'])}) → Withdraw → <b>50%</b> of its value → confirm")
        return "\n".join(lines) + "\nCash stays parked in Composer. Fills at the next close (~3:50pm ET)."
    lines = [f"{i}. Composer → <b>{name}</b> → Withdraw → <b>50%</b> of its value → confirm"
             for i, name in enumerate(_book(snap), 1)]
    return ("\n".join(lines) + "\nCash stays parked in Composer. Fills at the next close (~3:50pm ET)."
            "\n⚠️ Instrument list unavailable — these are the radar's look-through names. Use the symphonies that "
            "actually hold money (TRANCHE-EXECUTION.md / INSTRUMENTS.json).")


def steps_reentry(snap, ctx=None):
    funded = (ctx or {}).get("funded") or []
    if funded:
        parts = [f"{heading}: " + ", ".join(f"<b>{html.escape(f['name'])}</b>" for f in mine) for heading, mine in _by_login(funded)]
        return ("Composer → each of these → Invest the parked half back (same login you parked it from):\n"
                + "\n".join(f"• {p}" for p in parts))
    book = _book(snap)
    w = " / ".join(f"{int(round(v * 100))}" for v in book.values())
    names = " / ".join(book)
    return f"Composer → each algo → Invest the parked cash back so the book is {w} ({names}) again."


def _notes(ctx):
    notes = (ctx or {}).get("notes") or []
    return ("\n\n" + "\n".join(f"⚠️ {html.escape(n)}" for n in notes)) if notes else ""


def fire_reasons(st, snap):
    d, out = snap["dials"], []
    if st["streak"]["slow"] >= FIRE_AFTER["slow"]:
        out.append(f"STOP on the dip edge: {d['slow'].get('red_days')} of the last {d['slow'].get('window')} days "
                   f"lost money. In 55 years of QQQ data this fired only in 1972–91, never since 1993.")
    if st["streak"]["rip"] >= FIRE_AFTER["rip"]:
        out.append(f"Rips stopped fading: {d['rip'].get('hot_days')} of the last {d['rip'].get('window')} days "
                   f"kept running — the 1970s pattern.")
    if st["streak"]["machines"] >= FIRE_AFTER["machines"]:
        out.append(f"Machines red {st['streak']['machines']} closes running: "
                   + "; ".join(d["machines"].get("reasons") or ["(no detail)"]) + ".")
    return out


def msg_fire(snap, reasons, ctx=None):
    why = "\n".join(f"• {r}" for r in reasons)
    return (f"🛑 <b>GO DEFENSIVE — move 50% of the book to cash</b>\n"
            f"Radar as of {snap['asOf']}:\n{why}\n\n"
            f"Do this now — 2 minutes, every algo stays on:\n{steps_defensive(snap, ctx)}{_notes(ctx)}")


def msg_reenter(snap, edge, ctx=None):
    return (f"🟢 <b>RE-ENTER — put the parked cash back</b>\n"
            f"Radar green {REENTRY_CLOSES} closes running; dip edge {edge:+.2f}% (as of {snap['asOf']}).\n"
            f"{steps_reentry(snap, ctx)}{_notes(ctx)}")


def msg_stand_down(snap):
    return (f"✅ <b>Stand down</b> — the alarm cleared before you acted. No action, the book stays as it is.\n"
            f"Radar as of {snap['asOf']}: {snap['verdict']['text']}")


def msg_hold(snap):
    return (f"⛔ <b>Hold — stay defensive.</b> The green streak broke before you re-entered.\n"
            f"Radar as of {snap['asOf']}: {snap['verdict']['text']}\n"
            f"Re-entry needs {REENTRY_CLOSES} green closes again.")


def msg_ack(mode, today):
    if mode == "DEFENSIVE":
        return (f"📌 Logged: the book is <b>DEFENSIVE</b> (50% cash) from {today}. "
                f"You'll get RE-ENTER after {REENTRY_CLOSES} green closes with the dip edge above +{REENTRY_EDGE_PCT}%.")
    return f"📌 Logged: the book is <b>INVESTED</b> again from {today}. The trigger is re-armed."


def msg_reminder(st, now_local):
    p = st["pending"]
    what = "GO DEFENSIVE — 50% to cash" if p["kind"] == "PENDING_DEFENSIVE" else "RE-ENTER — cash back in"
    sent = datetime.fromisoformat(p["sent_at"]).astimezone(now_local.tzinfo).strftime("%a %H:%M")
    text = (f"⏰ <b>{what}</b> still open · sent {sent} · #{p['reminders']}")
    if p["reminders"] % STEPS_EVERY_N_REMINDERS == 0 and p.get("steps"):
        text += "\n\n" + p["steps"]
    return text


def msg_stale(last_asof, days):
    return (f"⚠️ <b>Defensive trigger is blind</b>: no new radar close since {last_asof} ({days} business days). "
            f"Check com.jalal.rubber-band / ~/Library/Logs/rubber-band.log.")


# --- the decision ---------------------------------------------------------------------------
def dial_colours(snap):
    return {k: snap["dials"][k]["colour"] for k in ("slow", "fast", "age", "rip", "machines")}


def evaluate(snap, st, send=send_telegram, now=None, log=print, ctx=None):
    """One new close in, at most one transition + one message out.
    `ctx`: callable(mode) → Tranche Map context (tests inject it); the default reads the files on this machine."""
    now = now or datetime.now(timezone.utc)
    ctx_for = ctx if callable(ctx) else ((lambda mode: ctx) if ctx is not None else default_ctx)
    asof = snap["asOf"]
    if asof == st.get("last_asof"):
        log(f"  no new close (asOf {asof}) — nothing counted")
        return st, []
    col = dial_colours(snap)
    for k in ("slow", "rip", "machines"):
        st["streak"][k] = st["streak"][k] + 1 if col[k] == "red" else 0
    edge = snap["dials"]["slow"].get("excess_pct")
    green_ok = snap["verdict"]["colour"] == "green" and edge is not None and edge > REENTRY_EDGE_PCT
    st["green_streak"] = st["green_streak"] + 1 if green_ok else 0
    st["last_asof"] = asof
    reasons = fire_reasons(st, snap)
    fire = bool(reasons)
    sent = []

    def go(mode, text, note, steps=None):
        st["mode"] = mode
        st["pending"] = ({"kind": mode, "sent_at": now.isoformat(), "asof": asof, "reminders": 0, "steps": steps}
                         if mode.startswith("PENDING") else None)
        _record(st, now, asof, mode, note)
        if (_send_with_buttons(send, text, mode) if mode.startswith("PENDING") else send(text)):
            sent.append(text)
        log(f"  → {mode}: {note}")

    m = st["mode"]
    if m == "INVESTED" and fire:
        c = ctx_for("PENDING_DEFENSIVE")
        go("PENDING_DEFENSIVE", msg_fire(snap, reasons, c), reasons[0][:90], steps_defensive(snap, c))
    elif m == "PENDING_DEFENSIVE" and not fire:
        go("INVESTED", msg_stand_down(snap), "alarm cleared before action")
    elif m == "DEFENSIVE" and st["green_streak"] >= REENTRY_CLOSES:
        c = ctx_for("PENDING_REENTRY")
        go("PENDING_REENTRY", msg_reenter(snap, edge, c), f"{REENTRY_CLOSES} green closes, edge {edge:+.2f}%", steps_reentry(snap, c))
    elif m == "PENDING_REENTRY" and not green_ok:
        go("DEFENSIVE", msg_hold(snap), "green streak broke before re-entry")
    else:
        log(f"  {m} · red streaks {st['streak']} · green streak {st['green_streak']} · {asof} — no change")
    return st, sent


def ack(st, send=send_telegram, now=None, log=print):
    now = now or datetime.now(timezone.utc)
    today = now.date().isoformat()
    m = st["mode"]
    if m == "PENDING_DEFENSIVE":
        st.update(mode="DEFENSIVE", pending=None, green_streak=0, defensive_since=today)
    elif m == "PENDING_REENTRY":
        st.update(mode="INVESTED", pending=None, streak={"slow": 0, "rip": 0, "machines": 0}, invested_since=today)
    else:
        log(f"  nothing pending (mode {m})")
        return st, False
    _record(st, now, st.get("last_asof"), st["mode"], "Jalal said DONE")
    send(msg_ack(st["mode"], today))
    log(f"  → {st['mode']} (acknowledged)")
    return st, True


def business_days_between(a, b):
    """Weekdays strictly after a, up to and including b."""
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def nag(st, send=send_telegram, taps=get_taps, now=None, log=print):
    """Hourly: read taps from KV, remind while something is pending, shout if the radar is stale."""
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        now_local = now.astimezone(ZoneInfo("America/New_York"))
    except Exception:                            # noqa: BLE001
        now_local = now
    sent = []

    tap = taps()
    if tap is None:
        log("  tap endpoint unavailable — DONE through the concierge still works")
    else:
        p = st.get("pending")
        if p and tap:
            tap_at = datetime.fromisoformat(tap["at"].replace("Z", "+00:00"))
            sent_at = datetime.fromisoformat(p["sent_at"])
            if tap_at >= sent_at:
                if tap["kind"] == "done":
                    st, ok = ack(st, send, now, log)
                    if ok:
                        sent.append("ack")
                elif tap["kind"] == "snooze":
                    p["snooze_until"] = (tap_at + timedelta(hours=2)).isoformat()
                    log(f"  snoozed until {p['snooze_until']}")

    p = st.get("pending")
    if p:
        snooze_until = p.get("snooze_until")
        snoozed = bool(snooze_until) and now < datetime.fromisoformat(snooze_until.replace("Z", "+00:00"))
        if not snoozed:                          # a ⏰ 2 h tap skips the reminder, never the stale check below
            p["reminders"] = p.get("reminders", 0) + 1
            text = msg_reminder(st, now_local)
            if _send_with_buttons(send, text, p["kind"]):
                sent.append(text)

    if st.get("last_asof"):
        days = business_days_between(date.fromisoformat(st["last_asof"]), now_local.date())
        today = now_local.date().isoformat()
        if days > STALE_BUSINESS_DAYS and st.get("stale_warned") != today:
            st["stale_warned"] = today
            if send(msg_stale(st["last_asof"], days)):
                sent.append("stale")
    return st, sent


# --- CLI -------------------------------------------------------------------------------------
# --- dashboard hand-off ---------------------------------------------------------------------
def summary(st):
    """The trigger's state the way the dashboard shows it (rides inside the radar snapshot as `defensive`)."""
    hist = st.get("history") or []
    pend = st.get("pending")
    return {"mode": st.get("mode", "INVESTED"), "since": hist[-1]["t"] if hist else None,
            "last_asof": st.get("last_asof"), "streak": st.get("streak"), "green_streak": st.get("green_streak"),
            "pending": ({"kind": pend.get("kind"), "sent_at": pend.get("sent_at"), "reminders": pend.get("reminders", 0)} if pend else None),
            "defensive_since": st.get("defensive_since"), "invested_since": st.get("invested_since"),
            "rules": {"fire_after": FIRE_AFTER, "reentry_closes": REENTRY_CLOSES, "reentry_edge_pct": REENTRY_EDGE_PCT}}


def republish(st, log=print, gh_path=None):
    """Stamp summary(st) into the last published radar snapshot and push it to the gist. Never throws."""
    try:
        snap_path = _path("rubber-band.json")
        if not os.path.exists(snap_path):
            log("  republish skipped (no snapshot yet)")
            return False
        snap = json.load(open(snap_path))
        snap["defensive"] = summary(st)
        with open(snap_path, "w") as f:
            json.dump(snap, f, indent=1)
        rb_state_path = _path("state.json")
        gid = json.load(open(rb_state_path)).get("gist_id") if os.path.exists(rb_state_path) else None
        gh = gh_path or shutil.which("gh") or next((g for g in ("/usr/local/bin/gh", "/opt/homebrew/bin/gh") if os.path.exists(g)), None)
        if not gid or not gh:
            log(f"  republish skipped (gist {gid!r}, gh {gh!r})")
            return False
        r = subprocess.run([gh, "gist", "edit", gid, "-f", "rubber-band.json", snap_path], capture_output=True, text=True, timeout=90)
        log("  republish → gist " + ("ok" if r.returncode == 0 else "FAILED " + (r.stderr or r.stdout).strip()[:120]))
        return r.returncode == 0
    except Exception as e:                        # noqa: BLE001 — the dashboard stamp must never break the trigger
        log(f"  republish failed: {type(e).__name__}: {str(e)[:120]}")
        return False


def main(argv):
    args = argv[1:]
    dry = "--dry" in args
    cmd = args[0] if args and not args[0].startswith("--") else "status"
    st = load_state()
    mode_before = st["mode"]
    send = (lambda t: print("--- would send ---\n" + t + "\n---") or True) if dry else send_telegram
    now = datetime.now(timezone.utc)
    print(f"defensive-trigger {cmd} {now.isoformat(timespec='seconds')} mode={st['mode']}")
    if cmd == "evaluate":
        path = args[args.index("--snapshot") + 1] if "--snapshot" in args else _path("rubber-band.json")
        st, _ = evaluate(json.load(open(path)), st, send=send, now=now)
    elif cmd == "nag":
        st, _ = nag(st, send=send, taps=(lambda: {}) if dry else get_taps, now=now)
    elif cmd == "ack":
        st, _ = ack(st, send=send, now=now)
    elif cmd == "set":
        mode = args[1].upper()
        if mode not in MODES:
            print(f"mode must be one of {MODES}")
            return 2
        st.update(mode=mode, pending=None)
        _record(st, now, st.get("last_asof"), mode, "set by hand")
        print(f"  mode set to {mode}")
    elif cmd == "steps":
        path = args[args.index("--snapshot") + 1] if "--snapshot" in args else _path("rubber-band.json")
        snap = json.load(open(path)) if os.path.exists(path) else {"spec": {}}
        for mode, fn in (("PENDING_DEFENSIVE", steps_defensive), ("PENDING_REENTRY", steps_reentry)):
            c = default_ctx(mode)
            ticked = ", ".join(k for k, v in c["ticks"].items() if v) or "none"
            print(f"--- {mode} (instrument list {'loaded' if c['loaded'] else 'MISSING'}; ticked: {ticked})\n{fn(snap, c)}{_notes(c)}\n")
        return 0
    elif cmd == "status":
        print(json.dumps({k: st.get(k) for k in ("mode", "last_asof", "streak", "green_streak", "pending",
                                                  "defensive_since", "invested_since")}, indent=1))
        return 0
    else:
        print(__doc__)
        return 2
    if not dry:
        save_state(st)
        with open(_path("defensive.last_ok"), "w") as f:
            f.write(now.isoformat())
        if cmd in ("evaluate", "ack", "set") or st["mode"] != mode_before:
            republish(st)                          # the dashboard shows mode + streaks from the gist
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
