"""The radar's machine legs and the Tranche Map's INSTRUMENTS.json must agree — one list, not two.

Runs only where the instrument file exists (the Mac mini); CI skips it. If this fails, the radar is watching a
leg the plan no longer funds (or with a different written line) — fix INSTRUMENTS.json or rubber_band.SPEC, never
the test. Lesson from 8 Sep 2026: Main sat in the radar two weeks after the plan had it sold.
"""
import importlib.util
import json
import os

import pytest

# The real file, on purpose: test_defensive_trigger.py points TRANCHE_INSTRUMENTS at /nonexistent to stay hermetic,
# and this test exists precisely to read what the Mac mini actually has.
PATH = os.path.expanduser(os.environ.get("INSTRUMENTS_SYNC_PATH", "~/concierge/triggers/INSTRUMENTS.json"))

pytestmark = pytest.mark.skipif(not os.path.exists(PATH), reason="INSTRUMENTS.json lives on the Mac mini only")


def _spec():
    s = importlib.util.spec_from_file_location(
        "rubber_band", os.path.join(os.path.dirname(__file__), "..", "scripts", "rubber_band.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.SPEC


def test_every_radar_leg_is_an_instrument_with_the_same_written_line():
    legs = _spec()["machines"]["legs"]
    ins = json.load(open(PATH))["instruments"]
    for leg in legs:
        matches = [i for i in ins if (i.get("radar") or {}).get("leg") == leg["name"]]
        assert matches, f"radar leg {leg['name']} is not in INSTRUMENTS.json"
        for i in matches:
            assert i["radar"].get("line_pct") == leg["line_pct"], f"{leg['name']}: line {i['radar'].get('line_pct')} vs SPEC {leg['line_pct']}"
            if i.get("symphony_id") and i.get("funded") is not None:
                # a funded instrument the radar watches must be the same symphony (mirrors are watched via `via_mirror`)
                assert i["symphony_id"] == leg["id"] or i["radar"].get("via_mirror") == leg["id"], \
                    f"{leg['name']}: symphony {i['symphony_id']} vs SPEC {leg['id']}"


def test_no_retired_symphony_is_still_a_radar_leg():
    legs = {leg["id"] for leg in _spec()["machines"]["legs"]}
    ins = json.load(open(PATH))["instruments"]
    retired = [i for i in ins if (i.get("funded") or {}).get("until") and i.get("symphony_id") in legs]
    assert not retired, f"still on the radar although the plan sells them: {[i['id'] for i in retired]}"
