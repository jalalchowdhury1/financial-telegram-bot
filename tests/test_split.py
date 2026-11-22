"""scripts/split.py: the split maths on FAKE curves (no gist, no real anchor)."""
import json, sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import split

DAY = 20770  # 2026-11-13, a Friday


@pytest.fixture
def curves(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "CURVES", str(tmp_path / "curve-{}.json"))
    monkeypatch.setattr(split, "HIST", str(tmp_path / "history.json"))
    def write(sid, curve): json.dump({str(k): v for k, v in curve.items()}, open(tmp_path / f"curve-{sid}.json", "w"))
    return write


def cfg(date="2026-11-13"):
    return {"date": date, "main": {"sid": "M", "usd": 700.0}, "shartino": {"sid": "S", "usd": 300.0}, "target": 70, "band": 10}


def test_anchor_day_is_the_anchor_split(curves):
    curves("M", {DAY - 1: 5.0, DAY: 9.0}); curves("S", {DAY - 1: 2.0, DAY: 2.0})
    s = split.snapshot(cfg())
    assert (s["asOf"], s["main"], s["anchored"]) == ("2026-11-13", 70.0, "2026-11-13")


def test_rolls_each_total_with_its_own_curve(curves):
    curves("M", {DAY: 10.0, DAY + 3: 12.0}); curves("S", {DAY: 4.0, DAY + 3: 4.2})   # Main +20%, Shartino +5%
    s = split.snapshot(cfg())
    assert s["asOf"] == "2026-11-16" and s["main"] == round(100 * 840 / (840 + 315), 2) == 72.73


def test_weekend_anchor_uses_the_last_close_before_it(curves):
    curves("M", {DAY: 10.0, DAY + 3: 5.0}); curves("S", {DAY: 4.0, DAY + 3: 4.0})    # Main halves on Monday
    assert split.snapshot(cfg("2026-11-15"))["main"] == round(100 * 350 / 650, 2)


def test_history_rewritten_by_a_new_tree_only_needs_the_ratio(curves):
    curves("M", {DAY: 1e9, DAY + 3: 1.1e9}); curves("S", {DAY: 3.0, DAY + 3: 3.0})
    assert split.snapshot(cfg())["main"] == round(100 * 770 / 1070, 2)


def test_curves_ending_on_different_days_publish_nothing(curves):
    curves("M", {DAY: 10.0, DAY + 3: 12.0}); curves("S", {DAY: 4.0})
    with pytest.raises(SystemExit): split.snapshot(cfg())


def test_anchor_before_the_curve_starts_fails_loudly(curves):
    curves("M", {DAY: 10.0}); curves("S", {DAY: 4.0})
    with pytest.raises(ValueError): split.snapshot(cfg("2020-01-01"))


def test_no_date_means_the_last_close_not_today(curves):
    import datetime, zoneinfo
    ny = zoneinfo.ZoneInfo("America/New_York")
    assert split.last_close_day(datetime.datetime(2026, 11, 17, 8, 0, tzinfo=ny)) == "2026-11-16"    # Tuesday before the open
    assert split.last_close_day(datetime.datetime(2026, 11, 18, 0, 30, tzinfo=ny)) == "2026-11-17"   # after midnight
    assert split.last_close_day(datetime.datetime(2026, 11, 17, 16, 5, tzinfo=ny)) == "2026-11-17"   # after the close


def test_a_bad_set_is_refused_before_it_is_saved(curves):
    curves("M", {DAY: 10.0}); curves("S", {DAY: 4.0})
    for bad in ((700.0, 300.0, "11/13/2026"), (700.0, 300.0, "2026-11-5"), (0.0, 300.0, "2026-11-13"), (700.0, 300.0, "2999-01-01")):
        with pytest.raises((SystemExit, ValueError)): split.new_anchor(cfg(), *bad)
    ok = split.new_anchor(cfg(), 600.0, 400.0, "2026-11-13")
    assert (ok["main"], ok["date"]) == ({"sid": "M", "usd": 600.0}, "2026-11-13") and split.snapshot(ok)["main"] == 60.0


def test_history_is_born_with_the_months_before_the_anchor_then_gains_a_day_per_close(curves):
    m = {DAY - 200: 1.0, DAY - 2: 8.0, DAY - 1: 10.0, DAY: 10.0}; s = {DAY - 200: 4.0, DAY - 2: 4.0, DAY - 1: 4.0, DAY: 4.0}
    curves("M", m); curves("S", s)
    one = split.with_history(split.snapshot(cfg()), cfg())
    assert one["logFrom"] == "2026-11-13"
    assert one["history"] == [("2026-11-11", round(100 * 560 / 860, 2)), ("2026-11-12", 70.0), ("2026-11-13", 70.0)]   # 200 days back is left out
    curves("M", {**m, DAY + 3: 12.0}); curves("S", {**s, DAY + 3: 4.0})
    two = split.with_history(split.snapshot(cfg()), cfg())
    assert [d for d, _ in two["history"]] == ["2026-11-11", "2026-11-12", "2026-11-13", "2026-11-16"] and two["history"][-1][1] == two["main"]
    moved = {**cfg("2026-11-16"), "main": {"sid": "M", "usd": 600.0}, "shartino": {"sid": "S", "usd": 400.0}}   # he moved money: the old days stay, the new day steps
    three = split.with_history(split.snapshot(moved), moved)
    assert [list(x) for x in three["history"][-2:]] == [["2026-11-13", 70.0], ["2026-11-16", 60.0]]


def test_ballast_swap_and_topup(curves, tmp_path):
    """22 Nov: the ballast seat can be another algo with its own curve file and name; the top-up card is due on its day."""
    curves("M", {DAY: 10.0, DAY + 3: 12.0}); curves("S", {DAY: 4.0, DAY + 3: 4.2}); own = tmp_path / "own.json"; own.write_text(json.dumps({str(DAY): 4.0, str(DAY + 3): 4.2}))
    c = {**cfg(), "shartino": {"sid": "X", "usd": 300.0, "name": "Opposite 3", "curve": str(own), "start": "2019-11-26"}, "topup": {"next": "2027-02-22", "every": 91}}
    s = split.snapshot(c)
    assert s["other"] == "Opposite 3" and s["main"] == round(100 * 840 / 1155, 2) and split.snapshot(cfg())["other"] == "Shartino"
    assert split.topup_due(s, c, "2027-02-21") is None and split.topup_due(s, cfg(), "2027-02-22") is None
    assert split.topup_due(s, c, "2027-02-22") == "<b>⚖️ Top-up day · Main 73 / Opposite 3 27</b>\nMove 4% of Main → Opposite 3 to be back at 70 / 30."
    assert split.topup_due({**s, "main": 70.4}, c, "2027-02-22") == ""
