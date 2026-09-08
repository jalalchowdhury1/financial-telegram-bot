"""
Tranche Map wiring — the ONE reader of the plan's tick boxes and instrument list.

Files (both on the Mac mini, outside this repo; override with env for tests):
  TRANCHE_MD           ~/concierge/triggers/TRANCHE-EXECUTION.md   the dated checklist, `[ ]` → `[x]` when done
  TRANCHE_INSTRUMENTS  ~/concierge/triggers/INSTRUMENTS.json       every account, instrument, flow, collision rule

Answers two questions for defensive_trigger.py without ever touching a broker or an account:
  * which Composer symphonies hold money TODAY — decided by the plan's ticks, never by balances, and
  * which collision rule applies right now (switch day / the drain) while the Tranche Map executes.
Jalal's rule (8 Sep 2026): stay as far away from the money as possible — no API reads, no auto-ticks.
Missing or unreadable files degrade to an empty context; callers fall back to the radar's look-through book.
"""
import json
import os
import re

TRANCHE_MD = os.path.expanduser(os.environ.get("TRANCHE_MD", "~/concierge/triggers/TRANCHE-EXECUTION.md"))
INSTRUMENTS_PATH = os.path.expanduser(os.environ.get("TRANCHE_INSTRUMENTS", "~/concierge/triggers/INSTRUMENTS.json"))

# | done | date Www | step | owner | what | duration / note |
_ROW = re.compile(r"^\|\s*\[( |x|X)\]\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\w{3}\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|")


def parse_steps(md_text):
    """Every dated table row → {id, date, owner, what, note, done}, in file order."""
    out = []
    for line in md_text.splitlines():
        m = _ROW.match(line)
        if m:
            out.append({"id": m.group(3), "date": m.group(2), "owner": m.group(4), "what": m.group(5),
                        "note": m.group(6), "done": m.group(1) != " "})
    return out


def ticks(steps):
    return {s["id"]: s["done"] for s in steps}


def load_steps(path=None):
    try:
        with open(path or TRANCHE_MD, encoding="utf-8") as f:
            return parse_steps(f.read())
    except OSError:
        return []


def load_instruments(path=None):
    try:
        with open(path or INSTRUMENTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_funded(instrument, tk):
    """funded: {"from": step} and/or {"until": step}, {"standing": true}, or null (never holds money)."""
    f = instrument.get("funded")
    if not f:
        return False
    if f.get("standing"):
        return True
    if f.get("from") and not tk.get(f["from"], False):
        return False
    if f.get("until") and tk.get(f["until"], False):
        return False
    return True


def funded_symphonies(instruments, tk):
    """Composer symphonies that hold money today and take part in a defensive move, in file order.
    Each: {name, login, account} — `name` is what Jalal clicks in Composer, `login` whose Composer login."""
    if not instruments:
        return []
    accounts = {a["id"]: a for a in instruments.get("accounts", [])}
    out = []
    for i in instruments.get("instruments", []):
        if i.get("kind") != "symphony" or i.get("defensive") != "withdraw-half":
            continue
        if not is_funded(i, tk):
            continue
        a = accounts.get(i.get("account") or "", {})
        out.append({"name": i.get("name") or i["id"], "login": a.get("login") or "jalal",
                    "account": a.get("label") or i.get("account") or "Composer"})
    return out


def collision_notes(instruments, tk, mode):
    """Plain-language rules that apply RIGHT NOW: every `when` condition (ticked / unticked / modes) must hold."""
    if not instruments:
        return []
    out = []
    for c in (instruments.get("defensive") or {}).get("collisions", []):
        w = c.get("when") or {}
        if not all(tk.get(s, False) for s in w.get("ticked", [])):
            continue
        if any(tk.get(s, False) for s in w.get("unticked", [])):
            continue
        if w.get("modes") and mode not in w["modes"]:
            continue
        out.append(c["text"])
    return out


def tranche_context(mode, md_path=None, instruments_path=None):
    """Everything the trigger needs for one message: funded symphonies, notes for `mode`, the ticks."""
    steps = load_steps(md_path)
    tk = ticks(steps)
    ins = load_instruments(instruments_path)
    return {"funded": funded_symphonies(ins, tk), "notes": collision_notes(ins, tk, mode),
            "ticks": tk, "loaded": ins is not None}
