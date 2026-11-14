#!/usr/bin/env python3
"""Main / Shartino split for the dashboard's Indicators card (14 Nov 2026).

No broker or Composer call. It starts from the two totals Jalal last read off Composer
(the anchor, kept only on this Mac in ~/.config/split-anchor.json) and rolls each one
forward with the night shift's own replay curve of that algo. Only percentages leave the
Mac: {asOf, main, target, band, anchored, history of main %} goes to a secret gist that /api/split relays.

  split.py run                       publish today's split
  split.py set MAIN_USD SHARTINO_USD [YYYY-MM-DD]   new anchor (after he moves money). Read the
                                     totals while the market is closed. No date = the last close
                                     (today from 16:00 New York time, else yesterday). Re-anchor
                                     after any edit of a live tree too: the curve replays the NEW
                                     tree over the days the money ran the old one.
"""
import datetime, json, os, subprocess, sys, tempfile, zoneinfo

CFG = os.path.expanduser("~/.config/split-anchor.json")
HIST = os.path.expanduser("~/.config/split-history.json")
CURVES = os.path.expanduser("~/PycharmProjects/composer-auto-research/nightshift/state/curve-{}.json")
GH = "/opt/homebrew/bin/gh"
DAY0 = datetime.date(1970, 1, 1)


def rolled(usd, sid, anchor):
    """usd on the anchor day -> (usd at the curve's last close, that close's date)."""
    curve = {int(k): v for k, v in json.load(open(CURVES.format(sid))).items()}
    a = max(d for d in curve if d <= (datetime.date.fromisoformat(anchor) - DAY0).days)
    last = max(curve)
    return usd * curve[last] / curve[a], str(DAY0 + datetime.timedelta(last))


def snapshot(cfg):
    main, d1 = rolled(cfg["main"]["usd"], cfg["main"]["sid"], cfg["date"])
    shar, d2 = rolled(cfg["shartino"]["usd"], cfg["shartino"]["sid"], cfg["date"])
    if d1 != d2: raise SystemExit(f"curves end on different days: {d1} / {d2}")
    return {"asOf": d1, "main": round(100 * main / (main + shar), 2), "target": cfg["target"],
            "band": cfg["band"], "anchored": cfg["date"],
            "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def with_history(snap, cfg):
    """snap + history [[day, main %], ...] for the tile's chart. The log (Mac only) gains one day per
    close and survives a new anchor, so a money move shows as a step. Born with the 3 months before
    the anchor, rolled BACKWARDS on the curves: true only if no money moved in those months, so
    `logFrom` tells the chart where the real log starts."""
    try: log = json.load(open(HIST))
    except FileNotFoundError:
        c = {k: {int(d): v for d, v in json.load(open(CURVES.format(cfg[k]["sid"]))).items()} for k in ("main", "shartino")}
        a = max(d for d in c["main"] if d <= (datetime.date.fromisoformat(cfg["date"]) - DAY0).days)
        usd = lambda k, d: cfg[k]["usd"] * c[k][d] / c[k][a]
        log = {"from": str(DAY0 + datetime.timedelta(a)),
               "days": {str(DAY0 + datetime.timedelta(d)): round(100 * usd("main", d) / (usd("main", d) + usd("shartino", d)), 2)
                        for d in sorted(c["main"]) if a - 92 <= d < a and d in c["shartino"]}}
    log["days"][snap["asOf"]] = snap["main"]
    json.dump(log, open(HIST + ".tmp", "w")); os.replace(HIST + ".tmp", HIST)
    return {**snap, "logFrom": log["from"], "history": sorted(log["days"].items())[-400:]}


def write_cfg(cfg):
    json.dump(cfg, open(CFG + ".tmp", "w"), indent=1)
    os.replace(CFG + ".tmp", CFG)


def publish(snap, cfg):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "split.json")
        json.dump(snap, open(path, "w"))
        if cfg.get("gist"):
            subprocess.run([GH, "gist", "edit", cfg["gist"], "-f", "split.json", path], check=True, timeout=60)
        else:
            out = subprocess.run([GH, "gist", "create", "--filename", "split.json", path], check=True, timeout=60,
                                 capture_output=True, text=True).stdout
            cfg["gist"] = out.strip().rsplit("/", 1)[-1]
            write_cfg(cfg)


def last_close_day(now=None):
    """The day whose close the totals on his screen are: today from 16:00 New York time, else yesterday
    (a weekend or holiday maps back to the last session inside rolled())."""
    now = now or datetime.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    return str(now.date() if now.hour >= 16 else now.date() - datetime.timedelta(1))


def new_anchor(cfg, main, shartino, date=None):
    """cfg with a new anchor; raises before anything is saved when a value is off."""
    date = str(datetime.date.fromisoformat(date)) if date else last_close_day()
    if min(main, shartino) <= 0 or date > str(datetime.date.today()):
        raise SystemExit("set: both totals must be above 0 and the date not in the future")
    cfg = {**cfg, "date": date, "main": {**cfg["main"], "usd": main}, "shartino": {**cfg["shartino"], "usd": shartino}}
    snapshot(cfg)   # a bad anchor fails here, before it is saved
    return cfg


if __name__ == "__main__":
    cfg = json.load(open(CFG))
    if sys.argv[1] == "set":
        cfg = new_anchor(cfg, float(sys.argv[2]), float(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
        write_cfg(cfg)
    snap = with_history(snapshot(cfg), cfg)
    publish(snap, cfg)
    print(f"=== split {json.dumps({**snap, 'history': len(snap['history'])})} ===")
