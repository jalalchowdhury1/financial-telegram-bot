"""Defensive trigger — state machine tests (no network, no clock)."""
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

_spec = importlib.util.spec_from_file_location(
    "defensive_trigger", os.path.join(os.path.dirname(__file__), "..", "scripts", "defensive_trigger.py"))
dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dt)

T0 = datetime(2026, 9, 15, 22, 40, tzinfo=timezone.utc)


def snap(asof, slow="green", rip="green", machines="green", edge=0.6, reasons=None, red_days=3, hot_days=2):
    worst = max((slow, rip, machines), key=lambda c: dt._RANK[c])
    return {"asOf": asof, "verdict": {"colour": worst, "text": f"verdict {worst}"},
            "spec": {"machines": {"book": {"C3": 0.68, "m1": 0.20, "hedges": 0.12}}},
            "dials": {"slow": {"colour": slow, "excess_pct": edge, "red_days": red_days, "window": 60},
                      "fast": {"colour": "green"}, "age": {"colour": "green"},
                      "rip": {"colour": rip, "hot_days": hot_days, "window": 60},
                      "machines": {"colour": machines, "reasons": reasons or []}}}


def closes(start="2026-09-15", n=1):
    d = datetime.fromisoformat(start)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.date().isoformat())
        d += timedelta(days=1)
    return out


def run(seq, st=None, sent=None):
    st = st or dt.fresh_state()
    sent = sent if sent is not None else []
    for i, s in enumerate(seq):
        st, _ = dt.evaluate(s, st, send=lambda t: sent.append(t) or True, now=T0 + timedelta(days=i), log=lambda *_: None)
    return st, sent


def test_all_green_stays_invested_and_silent():
    st, sent = run([snap(d) for d in closes(n=12)])
    assert st["mode"] == "INVESTED" and sent == [] and st["green_streak"] == 12


def test_slow_red_fires_on_one_close_and_a_repeat_snapshot_does_not_count_twice():
    d = closes(n=2)
    st, sent = run([snap(d[0], slow="red", red_days=47)])
    assert st["mode"] == "PENDING_DEFENSIVE" and len(sent) == 1
    assert "GO DEFENSIVE" in sent[0] and "47 of the last 60" in sent[0] and "C3" in sent[0] and "50%" in sent[0]
    st, sent = run([snap(d[0], slow="red")], st, sent)              # holiday re-run: same asOf
    assert st["streak"]["slow"] == 1 and len(sent) == 1
    st, sent = run([snap(d[1])], st, sent)                          # alarm clears before he acted
    assert st["mode"] == "INVESTED" and len(sent) == 2 and "Stand down" in sent[1]


def test_machines_needs_five_consecutive_red_closes():
    d = closes(n=6)
    st, sent = run([snap(x, machines="red", reasons=["C3 is in its deepest drawdown ever"]) for x in d[:4]])
    assert st["mode"] == "INVESTED" and sent == [] and st["streak"]["machines"] == 4
    st, sent = run([snap(d[4])], st, sent)                          # one green close resets the count
    assert st["streak"]["machines"] == 0 and sent == []
    st, sent = run([snap(x, machines="red", reasons=["C3 is in its deepest drawdown ever"]) for x in closes("2026-09-22", 5)])
    assert st["mode"] == "PENDING_DEFENSIVE" and len(sent) == 1 and "deepest drawdown ever" in sent[0]


def test_rip_red_fires_on_one_close():
    st, sent = run([snap(closes()[0], rip="red", hot_days=46)])
    assert st["mode"] == "PENDING_DEFENSIVE" and "Rips stopped fading" in sent[0]


def test_ack_then_ten_green_closes_with_edge_then_ack_again():
    sent = []
    st, _ = run([snap(closes()[0], slow="red")], sent=sent)
    st, ok = dt.ack(st, send=lambda t: sent.append(t) or True, now=T0, log=lambda *_: None)
    assert ok and st["mode"] == "DEFENSIVE" and "DEFENSIVE" in sent[-1] and st["green_streak"] == 0
    greens = closes("2026-09-16", 10)
    st, _ = run([snap(x, edge=0.5) for x in greens[:9]], st, sent)
    assert st["mode"] == "DEFENSIVE" and st["green_streak"] == 9
    st, _ = run([snap(greens[9], edge=0.5)], st, sent)
    assert st["mode"] == "PENDING_REENTRY" and "RE-ENTER" in sent[-1] and "68 / 20 / 12" in sent[-1]
    st, ok = dt.ack(st, send=lambda t: sent.append(t) or True, now=T0, log=lambda *_: None)
    assert ok and st["mode"] == "INVESTED" and st["streak"] == {"slow": 0, "rip": 0, "machines": 0}


def test_green_streak_needs_the_edge_above_threshold_and_breaks_on_amber():
    sent = []
    st, _ = run([snap(closes()[0], slow="red")], sent=sent)
    st, _ = dt.ack(st, send=lambda *_: True, now=T0, log=lambda *_: None)
    days = closes("2026-09-16", 12)
    st, _ = run([snap(x, edge=0.5) for x in days[:5]] + [snap(days[5], edge=0.1)], st, sent)
    assert st["green_streak"] == 0                                     # green but the edge is too thin
    st, _ = run([snap(x, edge=0.5) for x in days[6:12]], st, sent)
    assert st["green_streak"] == 6 and st["mode"] == "DEFENSIVE"
    st, _ = run([snap(x, edge=0.5) for x in closes("2026-10-02", 4)], st, sent)
    assert st["mode"] == "PENDING_REENTRY"
    st, _ = run([snap(closes("2026-10-08")[0], machines="amber")], st, sent)
    assert st["mode"] == "DEFENSIVE" and "stay defensive" in sent[-1]


def test_ack_with_nothing_pending_is_a_noop():
    st, ok = dt.ack(dt.fresh_state(), send=lambda *_: True, now=T0, log=lambda *_: None)
    assert not ok and st["mode"] == "INVESTED"


def test_nag_reminds_only_while_pending_and_repeats_steps_every_sixth_time():
    sent = []
    st, _ = run([snap(closes()[0], slow="red")], sent=sent)
    for i in range(6):
        st, out = dt.nag(st, send=lambda t: sent.append(t) or True, updates=lambda o: [], now=T0 + timedelta(hours=i + 1), log=lambda *_: None)
    assert st["pending"]["reminders"] == 6 and "Reminder 6" in sent[-1] and "Withdraw" in sent[-1]
    assert "Withdraw" not in sent[-2]                                  # reminder 5 was the short form
    st, out = dt.nag(dt.fresh_state(), send=lambda t: sent.append(t) or True, updates=lambda o: [], now=T0, log=lambda *_: None)
    assert out == []


def test_nag_picks_up_done_from_the_right_chat_after_the_alert_only(monkeypatch):
    monkeypatch.setenv("RUBBER_BAND_ALERT_CHAT", "7956935476")
    sent = []
    st, _ = run([snap(closes()[0], slow="red")], sent=sent)
    t_sent = int(datetime.fromisoformat(st["pending"]["sent_at"]).timestamp())
    upd = [{"update_id": 10, "message": {"chat": {"id": 7956935476}, "date": t_sent - 100, "text": "done"}},   # before the alert
           {"update_id": 11, "message": {"chat": {"id": 1}, "date": t_sent + 100, "text": "done"}},            # wrong chat
           {"update_id": 12, "message": {"chat": {"id": 7956935476}, "date": t_sent + 100, "text": "Done ✅"}}]
    st, out = dt.nag(st, send=lambda t: sent.append(t) or True, updates=lambda o: upd, now=T0 + timedelta(hours=1), log=lambda *_: None)
    assert st["mode"] == "DEFENSIVE" and st["tg_offset"] == 13 and "ack" in out and st["pending"] is None
    assert not any("Still waiting" in s for s in sent)


def test_nag_survives_a_poller_conflict_and_warns_once_a_day_when_the_radar_is_stale():
    sent = []
    st, _ = run([snap("2026-09-15")], sent=sent)
    later = datetime(2026, 9, 23, 14, 0, tzinfo=timezone.utc)        # 6 business days, no new close
    st, out = dt.nag(st, send=lambda t: sent.append(t) or True, updates=lambda o: None, now=later, log=lambda *_: None)
    assert out == ["stale"] and "blind" in sent[-1]
    st, out = dt.nag(st, send=lambda t: sent.append(t) or True, updates=lambda o: None, now=later + timedelta(hours=1), log=lambda *_: None)
    assert out == []


def test_cli_dry_run_never_writes_state(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "STATE_DIR", str(tmp_path))
    p = tmp_path / "s.json"
    p.write_text(json.dumps(snap("2026-09-15", slow="red")))
    assert dt.main(["x", "evaluate", "--snapshot", str(p), "--dry"]) == 0
    assert not (tmp_path / "defensive.json").exists()
    monkeypatch.setattr(dt, "send_telegram", lambda t: True)
    assert dt.main(["x", "evaluate", "--snapshot", str(p)]) == 0
    st = json.load(open(tmp_path / "defensive.json"))
    assert st["mode"] == "PENDING_DEFENSIVE" and (tmp_path / "defensive.last_ok").exists()


# --- dashboard hand-off (v1.1: the trigger's state rides inside the published snapshot) ---
def test_summary_shape_and_republish(tmp_path, monkeypatch):
    st = dt.fresh_state()
    st["history"] = [{"t": "2026-09-08T22:31:00+00:00", "asof": "2026-09-08", "to": "INVESTED", "note": "x"}]
    s = dt.summary(st)
    assert s["mode"] == "INVESTED" and s["since"] == "2026-09-08T22:31:00+00:00" and s["pending"] is None
    assert s["rules"]["fire_after"]["machines"] == 5 and s["rules"]["reentry_closes"] == 10
    monkeypatch.setattr(dt, "STATE_DIR", str(tmp_path))
    logs = []
    assert dt.republish(st, log=logs.append) is False and "no snapshot" in logs[-1]        # nothing published yet → skip
    json.dump({"asOf": "2026-09-08", "dials": {}}, open(tmp_path / "rubber-band.json", "w"))
    assert dt.republish(st, log=logs.append, gh_path="/nonexistent/gh") is False           # no gist id → skip, but the file is stamped
    assert json.load(open(tmp_path / "rubber-band.json"))["defensive"]["mode"] == "INVESTED"
    json.dump({"gist_id": "abc123"}, open(tmp_path / "state.json", "w"))
    fake_gh = tmp_path / "gh"
    fake_gh.write_text("#!/bin/sh\necho \"$@\" > \"$(dirname \"$0\")/gh.args\"\n")
    fake_gh.chmod(0o755)
    assert dt.republish(st, log=logs.append, gh_path=str(fake_gh)) is True
    assert "gist edit abc123 -f rubber-band.json" in (tmp_path / "gh.args").read_text()
