"""tranche_state — reads the Tranche Map's tick boxes + INSTRUMENTS.json. No network, no clock, no broker."""
import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location(
    "tranche_state", os.path.join(os.path.dirname(__file__), "..", "scripts", "tranche_state.py"))
ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts)

MD = """# Tranche Map — EXECUTION SCHEDULE
| done | date | step | owner | what | duration / note |
|---|---|---|---|---|---|
| [x] | 2026-09-15 Tue | S1.1 | Jalal | Message Composer support, 3 questions | reply 1-3 biz days |
| [ ] | 2026-09-23 Wed | S2.1 | Jalal | SWITCH DAY. One sale: Main + Shartino → cash | 1 afternoon · radar rule |
| [ ] | 2026-10-06 Tue | S2.4 | Jalal | Sweep each landing from checking → JPM SD | |
| [ ] | 2026-10-24 Sat | S2.6 | Nabila | Day 31: fund book B | |
not a row | [ ] | 2026-01-01 Thu | S9.9 | x | y | z |
"""

INS = {
    "accounts": [
        {"id": "composer-taxable", "label": "Composer · taxable · Jalal", "owner": "Jalal", "login": "jalal"},
        {"id": "composer-roth-nabila", "label": "Composer Roth IRA · Nabila", "owner": "Nabila", "login": "nabila"},
        {"id": "jpm-sd-jalal", "label": "JPM Self-Directed", "owner": "Jalal"},
    ],
    "instruments": [
        {"id": "Main", "name": "Main", "kind": "symphony", "account": "composer-taxable",
         "funded": {"until": "S2.1"}, "defensive": "withdraw-half"},
        {"id": "C8-T", "name": "C8-T", "kind": "symphony", "account": "composer-taxable",
         "funded": {"from": "S2.1"}, "defensive": "withdraw-half"},
        {"id": "C3-nabila", "name": "C3", "kind": "symphony", "account": "composer-roth-nabila",
         "funded": {"from": "S2.6"}, "defensive": "withdraw-half"},
        {"id": "C9-nabila", "name": "C9", "kind": "symphony", "account": "composer-roth-nabila",
         "funded": {"from": "S2.6"}, "defensive": "withdraw-half"},
        {"id": "hedges-mirror", "name": "Hedge sleeve mirror", "kind": "symphony", "account": None,
         "funded": None, "defensive": "never"},
        {"id": "SGOV", "name": "SGOV", "kind": "etf", "account": "jpm-sd-jalal",
         "funded": {"from": "S2.4"}, "defensive": "never"},
        {"id": "SPY-2323", "name": "SPY", "kind": "etf", "account": "jpm-2323-nabila",
         "funded": {"standing": True}, "defensive": "never"},
    ],
    "defensive": {"collisions": [
        {"id": "switch-day", "when": {"unticked": ["S2.1"], "modes": ["PENDING_DEFENSIVE", "DEFENSIVE"]},
         "text": "HALF into C8-T"},
        {"id": "drain", "when": {"ticked": ["S2.1"], "unticked": ["S2.4"], "modes": ["DEFENSIVE", "PENDING_REENTRY"]},
         "text": "drain is not parked cash"},
    ]},
}


def test_parse_steps_reads_tick_date_id_owner_what_and_note_and_skips_non_rows():
    steps = ts.parse_steps(MD)
    assert [s["id"] for s in steps] == ["S1.1", "S2.1", "S2.4", "S2.6"]
    assert steps[0]["done"] is True and steps[1]["done"] is False
    assert steps[0]["date"] == "2026-09-15" and steps[0]["owner"] == "Jalal"
    assert steps[0]["what"].startswith("Message Composer support")
    assert steps[0]["note"] == "reply 1-3 biz days" and steps[2]["note"] == ""
    assert ts.ticks(steps) == {"S1.1": True, "S2.1": False, "S2.4": False, "S2.6": False}


def test_before_switch_day_only_the_old_taxable_symphonies_hold_money():
    tk = ts.ticks(ts.parse_steps(MD))
    got = ts.funded_symphonies(INS, tk)
    assert [(f["name"], f["login"]) for f in got] == [("Main", "jalal")]


def test_after_switch_day_c8t_replaces_main_and_the_mirror_and_etfs_never_appear():
    got = ts.funded_symphonies(INS, {"S2.1": True})
    assert [(f["name"], f["login"], f["account"]) for f in got] == [("C8-T", "jalal", "Composer · taxable · Jalal")]


def test_book_b_appears_under_nabilas_login_once_s26_is_ticked():
    got = ts.funded_symphonies(INS, {"S2.1": True, "S2.6": True})
    assert [(f["name"], f["login"]) for f in got] == [("C8-T", "jalal"), ("C3", "nabila"), ("C9", "nabila")]


ALL_UNTICKED = {"S1.1": False, "S2.1": False, "S2.4": False, "S2.6": False}


def test_collision_notes_are_gated_by_ticks_and_by_mode():
    assert ts.collision_notes(INS, ALL_UNTICKED, "PENDING_DEFENSIVE") == ["HALF into C8-T"]
    assert ts.collision_notes(INS, ALL_UNTICKED, "INVESTED") == []
    assert ts.collision_notes(INS, {**ALL_UNTICKED, "S2.1": True}, "DEFENSIVE") == ["drain is not parked cash"]
    assert ts.collision_notes(INS, {**ALL_UNTICKED, "S2.1": True}, "PENDING_DEFENSIVE") == []
    assert ts.collision_notes(INS, {**ALL_UNTICKED, "S2.1": True, "S2.4": True}, "DEFENSIVE") == []


def test_a_rule_naming_a_step_the_calendar_lacks_is_skipped_not_fired_forever():
    assert ts.collision_notes(INS, {"S1.1": True}, "PENDING_DEFENSIVE") == []   # S2.1 unknown → the switch-day rule stays quiet
    assert ts.collision_notes(INS, {}, "PENDING_DEFENSIVE") == []


def test_a_present_but_empty_step_id_never_means_funded_forever():
    assert ts.is_funded({"funded": {"until": None}}, {"S2.1": True}) is False
    assert ts.is_funded({"funded": {"from": ""}}, {"S2.1": True}) is False
    assert ts.is_funded({"funded": {"from": "S2.1", "until": "S9.9"}}, {"S2.1": True, "S9.9": False}) is True
    assert ts.is_funded({"funded": {"from": "S2.1", "until": "S9.9"}}, {"S2.1": True, "S9.9": True}) is False


def test_context_reads_both_files_and_degrades_to_empty_when_they_are_missing(tmp_path):
    md = tmp_path / "t.md"
    md.write_text(MD)
    ins = tmp_path / "i.json"
    ins.write_text(json.dumps(INS))
    ctx = ts.tranche_context("PENDING_DEFENSIVE", md_path=str(md), instruments_path=str(ins))
    assert ctx["loaded"] is True and [f["name"] for f in ctx["funded"]] == ["Main"]
    assert ctx["notes"] == ["HALF into C8-T"] and ctx["ticks"]["S1.1"] is True
    empty = ts.tranche_context("INVESTED", md_path=str(tmp_path / "none.md"), instruments_path=str(tmp_path / "none.json"))
    assert empty == {"funded": [], "notes": [], "ticks": {}, "loaded": False, "md_loaded": False}


def test_an_unreadable_calendar_names_nothing_even_when_the_instrument_list_is_present(tmp_path):
    """Red team 8 Sep: md missing → ticks {} used to read as 'nothing ticked yet' → Main + Shartino named after they were sold."""
    ins = tmp_path / "i.json"
    ins.write_text(json.dumps(INS))
    ctx = ts.tranche_context("PENDING_DEFENSIVE", md_path=str(tmp_path / "renamed.md"), instruments_path=str(ins))
    assert ctx["loaded"] is False and ctx["md_loaded"] is False
    assert ctx["funded"] == [] and ctx["notes"] == [] and ctx["ticks"] == {}
    empty_md = tmp_path / "empty.md"
    empty_md.write_text("# no table yet\n")
    ctx2 = ts.tranche_context("PENDING_DEFENSIVE", md_path=str(empty_md), instruments_path=str(ins))
    assert ctx2["loaded"] is False and ctx2["funded"] == []
