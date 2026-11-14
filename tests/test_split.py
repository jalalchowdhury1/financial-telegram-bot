"""scripts/split.py: the split maths on FAKE curves (no gist, no real anchor)."""
import json, sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import split

DAY = 20770  # 2026-11-13, a Friday


@pytest.fixture
def curves(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "CURVES", str(tmp_path / "curve-{}.json"))
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
