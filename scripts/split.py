#!/usr/bin/env python3
"""Main / Shartino split for the dashboard's Indicators card (14 Nov 2026).

No broker or Composer call. It starts from the two totals Jalal last read off Composer
(the anchor, kept only on this Mac in ~/.config/split-anchor.json) and rolls each one
forward with the night shift's own replay curve of that algo. Only percentages leave the
Mac: {asOf, main, target, band, anchored} goes to a secret gist that /api/split relays.

  split.py run                       publish today's split
  split.py set MAIN_USD SHARTINO_USD [YYYY-MM-DD]   new anchor (after he moves money)
"""
import datetime, json, os, subprocess, sys, tempfile

CFG = os.path.expanduser("~/.config/split-anchor.json")
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
    return {"asOf": d1, "main": round(100 * main / (main + shar), 1), "target": cfg["target"],
            "band": cfg["band"], "anchored": cfg["date"]}


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
            json.dump(cfg, open(CFG, "w"), indent=1)


if __name__ == "__main__":
    cfg = json.load(open(CFG))
    if sys.argv[1] == "set":
        cfg["main"]["usd"], cfg["shartino"]["usd"] = float(sys.argv[2]), float(sys.argv[3])
        cfg["date"] = sys.argv[4] if len(sys.argv) > 4 else str(datetime.date.today())
        json.dump(cfg, open(CFG, "w"), indent=1)
    snap = snapshot(cfg)
    publish(snap, cfg)
    print(f"=== split {json.dumps(snap)} ===")
