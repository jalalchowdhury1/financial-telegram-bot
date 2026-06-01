#!/usr/bin/env python3
"""Distill recent health-history into one bounded digest for the weekly self-improve agent.

Reads health/history/*.json (written by the Phase 1 daily workflow), groups non-ok
findings by id, flags recurring ones, and emits a compact Markdown digest to stdout
and to health-digest.md. Keeping the agent's input small + focused controls token cost
and keeps it on-target.
"""
import glob
import json
import os
import sys


def load_recent_history(history_dir, limit=7):
    paths = sorted(glob.glob(os.path.join(history_dir, "*.json")))
    reports = []
    for p in paths[-limit:]:
        try:
            with open(p) as fh:
                reports.append(json.load(fh))
        except Exception:
            continue
    return reports


def build_digest(reports):
    by_id = {}
    for rep in reports:
        for f in rep.get("findings", []):
            if f.get("severity") == "ok":
                continue
            entry = by_id.setdefault(f["id"], {"finding": f, "days": 0})
            entry["days"] += 1
            entry["finding"] = f  # keep latest
    if not by_id:
        return "All health-history findings are OK over the window. No action needed."
    lines = ["# Health digest (unresolved findings)\n"]
    for fid, entry in sorted(by_id.items(), key=lambda kv: -kv[1]["days"]):
        f = entry["finding"]
        recurring = " (recurring)" if entry["days"] > 1 else ""
        lines.append(f"## {fid} — {f['severity']}{recurring}, seen {entry['days']} day(s)")
        lines.append(f"- {f.get('title', '')}")
        if f.get("detail"):
            lines.append(f"- detail: {f['detail']}")
        if f.get("remediation") and f["remediation"] != "none":
            lines.append(f"- suggested: {f['remediation']}")
        lines.append("")
    return "\n".join(lines)


def main():
    history_dir = os.environ.get("HEALTH_HISTORY_DIR", "health/history")
    digest = build_digest(load_recent_history(history_dir))
    with open("health-digest.md", "w") as fh:
        fh.write(digest)
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
