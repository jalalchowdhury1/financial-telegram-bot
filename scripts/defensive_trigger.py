#!/usr/bin/env python3
"""
Defensive trigger — turns the Rubber Band Radar's colours into the ONE decision Jalal never has to make.

The radar (rubber_band.py, nightly 18:30) measures. This script decides and nags. It never touches
Composer: Jalal keeps the hands ("nothing moves without Jalal", "NEVER touch the live symphony").

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
  defensive_trigger.py status
"""
import json
import os
import subprocess
import shutil
import re
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

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
_RANK = {"green": 0, "grey": 0, "amber": 1, "red": 2}


# --- state --------------------------------------------------------------------------------
def _path(name):
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, name)


def fresh_state():
    return {"mode": "INVESTED", "last_asof": None, "streak": {"slow": 0, "rip": 0, "machines": 0},
            "green_streak": 0, "pending": None, "history": [], "tg_offset": 0}


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


def send_telegram(text):
    tok = _token()
    if not tok:
        print("  no TELEGRAM_TOKEN — not sent")
        return False
    body = urllib.parse.urlencode({"chat_id": _chat(), "text": text, "parse_mode": "HTML"}).encode()
    try:
        req = urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=body)
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception as e:                       # noqa: BLE001 — a failed send is logged, never fatal
        print(f"  telegram send failed: {e}")
        return False


def get_updates(offset):
    """Messages since `offset`; None when Telegram is unreachable or another poller owns the bot (409)."""
    tok = _token()
    if not tok:
        return None
    q = urllib.parse.urlencode({"offset": offset, "timeout": 0, "allowed_updates": json.dumps(["message"])})
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/getUpdates?{q}", timeout=20) as r:
            return json.load(r).get("result") or []
    except Exception as e:                       # noqa: BLE001
        print(f"  telegram getUpdates failed: {e}")
        return None


# --- messages -------------------------------------------------------------------------------
def _book(snap):
    return ((snap.get("spec") or {}).get("machines") or {}).get("book") or {"C3": 0.68, "m1": 0.20, "hedges": 0.12}


def steps_defensive(snap):
    lines = [f"{i}. Composer → <b>{name}</b> → Withdraw → <b>50%</b> of its value → confirm"
             for i, name in enumerate(_book(snap), 1)]
    return "\n".join(lines) + "\nCash stays parked in Composer. Fills at the next close (~3:50pm ET)."


def steps_reentry(snap):
    book = _book(snap)
    w = " / ".join(f"{int(round(v * 100))}" for v in book.values())
    names = " / ".join(book)
    return f"Composer → each algo → Invest the parked cash back so the book is {w} ({names}) again."


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


def msg_fire(snap, reasons):
    why = "\n".join(f"• {r}" for r in reasons)
    return (f"🛑 <b>GO DEFENSIVE — move 50% of the book to cash</b>\n"
            f"Radar as of {snap['asOf']}:\n{why}\n\n"
            f"Do this now — 2 minutes, every algo stays on:\n{steps_defensive(snap)}\n\n"
            f"Reply <b>DONE</b> here when it's in. I'll remind you every hour until then.")


def msg_reenter(snap, edge):
    return (f"🟢 <b>RE-ENTER — put the parked cash back</b>\n"
            f"Radar green {REENTRY_CLOSES} closes running; dip edge {edge:+.2f}% (as of {snap['asOf']}).\n"
            f"{steps_reentry(snap)}\n\nReply <b>DONE</b> here when it's in.")


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
    text = (f"⏰ Still waiting on <b>{what}</b> (sent {sent}). Reply <b>DONE</b> here when it's in. "
            f"Reminder {p['reminders']}.")
    if p["reminders"] % STEPS_EVERY_N_REMINDERS == 0 and p.get("steps"):
        text += "\n\n" + p["steps"]
    return text


def msg_stale(last_asof, days):
    return (f"⚠️ <b>Defensive trigger is blind</b>: no new radar close since {last_asof} ({days} business days). "
            f"Check com.jalal.rubber-band / ~/Library/Logs/rubber-band.log.")


# --- the decision ---------------------------------------------------------------------------
def dial_colours(snap):
    return {k: snap["dials"][k]["colour"] for k in ("slow", "fast", "age", "rip", "machines")}


def evaluate(snap, st, send=send_telegram, now=None, log=print):
    """One new close in, at most one transition + one message out."""
    now = now or datetime.now(timezone.utc)
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
        if send(text):
            sent.append(text)
        log(f"  → {mode}: {note}")

    m = st["mode"]
    if m == "INVESTED" and fire:
        go("PENDING_DEFENSIVE", msg_fire(snap, reasons), reasons[0][:90], steps_defensive(snap))
    elif m == "PENDING_DEFENSIVE" and not fire:
        go("INVESTED", msg_stand_down(snap), "alarm cleared before action")
    elif m == "DEFENSIVE" and st["green_streak"] >= REENTRY_CLOSES:
        go("PENDING_REENTRY", msg_reenter(snap, edge), f"{REENTRY_CLOSES} green closes, edge {edge:+.2f}%", steps_reentry(snap))
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


def nag(st, send=send_telegram, updates=get_updates, now=None, log=print):
    """Hourly: pick DONE up from the thread, remind while something is pending, shout if the radar is stale."""
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        now_local = now.astimezone(ZoneInfo("America/New_York"))
    except Exception:                            # noqa: BLE001
        now_local = now
    sent = []
    res = updates(st.get("tg_offset", 0))
    if res is None:
        log("  telegram updates unavailable (another poller?) — DONE through the concierge still works")
    else:
        for u in res:
            st["tg_offset"] = max(st.get("tg_offset", 0), u["update_id"] + 1)
            msg = u.get("message") or {}
            if str((msg.get("chat") or {}).get("id")) != _chat():
                continue
            p = st.get("pending")
            if p and ACK_RE.match(msg.get("text") or "") and \
                    msg.get("date", 0) >= int(datetime.fromisoformat(p["sent_at"]).timestamp()):
                st, ok = ack(st, send, now, log)
                if ok:
                    sent.append("ack")
    p = st.get("pending")
    if p:
        p["reminders"] = p.get("reminders", 0) + 1
        text = msg_reminder(st, now_local)
        if send(text):
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
        st, _ = nag(st, send=send, updates=(lambda o: []) if dry else get_updates, now=now)
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
