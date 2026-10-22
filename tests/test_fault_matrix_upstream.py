"""fault_matrix: a FRED outage stays a FAIL but is named as one (FAKE data; 21 Oct 2026 57/60)."""
import io
import os
import re
import sys
import urllib.error
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import fault_matrix  # noqa: E402

MAINT = b"<!doctype html><title>Site Maintenance | St. Louis Fed</title>We'll be back soon!"
COPY = {"_meta": {"source": "Google Sheet (last-known-good 2026-10-22T01:30:35Z)", "stale": True,
                  "lastGoodAt": "2026-10-22T01:30:35Z"}}


def _body(b):
    r = mock.Mock()
    r.read.return_value = b
    return r


class FredUpstream(unittest.TestCase):
    def test_maintenance_page_is_down(self):
        with mock.patch("urllib.request.urlopen", return_value=_body(MAINT)):
            self.assertEqual(fault_matrix.fred_upstream_down(), "FRED serving its maintenance page")

    def test_keyless_json_400_is_up(self):
        err = urllib.error.HTTPError("u", 400, "x", {}, io.BytesIO(b'{"error_code":400}'))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            self.assertIsNone(fault_matrix.fred_upstream_down())

    def test_fred_copy_still_fails_but_says_upstream(self):
        with mock.patch.object(fault_matrix, "fetch", return_value=(200, COPY, 1.0, None)), \
                mock.patch.object(fault_matrix, "fred_upstream_down", return_value="FRED serving its maintenance page"):
            ok, msg, _ = fault_matrix.check("fred", "", "live")
        self.assertFalse(ok)  # never loosened: a copy where live is expected is a FAIL
        self.assertIn("upstream down", msg)

    def test_fred_copy_with_fred_up_has_no_excuse(self):
        with mock.patch.object(fault_matrix, "fetch", return_value=(200, COPY, 1.0, None)), \
                mock.patch.object(fault_matrix, "fred_upstream_down", return_value=None):
            ok, msg, _ = fault_matrix.check("fred", "", "live")
        self.assertFalse(ok)
        self.assertNotIn("upstream", msg)


class RouteKeepsFailReason(unittest.TestCase):
    def test_merged_outage_payload_keeps_series_failed(self):
        src = open(os.path.join(os.path.dirname(__file__), "..", "dashboard", "app", "api", "fred", "route.js")).read()
        m = re.search(r"messages: messages\.filter\(\(m\) => (.+?)\),\n", src)
        self.assertIsNotNone(m, "merged-outage messages filter moved; keep the FRED fail reason in it")
        self.assertIn("Series failed", m.group(1))


if __name__ == "__main__":
    unittest.main()


class FredExtras(unittest.TestCase):
    """FAKE payloads: the four non-FRED numbers must stay live on top of a FRED-down copy."""
    LIVE = {"_meta": {"stale": True}, "spEps": {"current": 298.0, "source": "multpl"},
            "peRatio": 26.5, "peSource": "multpl",
            "indicators": {"copperGold": {"value": 1.58, "source": "copper:cnbc"}},
            "horsemen": {"bankruptcies": {"current": 27054, "source": "uscourts"}}}
    DROPPED = {"_meta": {"stale": True}, "peRatio": 26.3, "peSource": None,
               "indicators": {"copperGold": {"value": 1.57}}, "horsemen": {"bankruptcies": {"current": 26941}}}

    def test_live_overlay_passes(self):
        want = dict.fromkeys(fault_matrix.FRED_EXTRAS, "live")
        self.assertTrue(all(ok for ok, _ in fault_matrix.judge_extras(self.LIVE, want)))

    def test_cached_copy_hiding_live_values_fails(self):
        want = dict.fromkeys(fault_matrix.FRED_EXTRAS, "live")
        self.assertFalse(any(ok for ok, _ in fault_matrix.judge_extras(self.DROPPED, want)))

    def test_all_sources_off_is_an_unlabelled_stale_copy(self):
        want = dict.fromkeys(fault_matrix.FRED_EXTRAS, "copy")
        self.assertTrue(all(ok for ok, _ in fault_matrix.judge_extras(self.DROPPED, want)))
        self.assertFalse(all(ok for ok, _ in fault_matrix.judge_extras(self.LIVE, want)))


class CatchUp(unittest.TestCase):
    """--catchup (04:30): re-run only when the 21:40 drill did not end N/N (FAKE logs)."""
    def _needed(self, text):
        p = os.path.join(os.path.dirname(__file__), "_catchup_fake.log")
        with open(p, "w") as f:
            f.write(text)
        try:
            return fault_matrix.catchup_needed(p)
        finally:
            os.remove(p)

    def test_green_last_run_skips(self):
        self.assertFalse(self._needed("=== 57/60 passed ===\nx\n=== 72/72 passed ===\n"))

    def test_red_last_run_reruns(self):
        self.assertTrue(self._needed("=== 72/72 passed ===\nFAIL fred\n=== 57/60 passed ===\n"))

    def test_crashed_or_missing_run_reruns(self):
        self.assertTrue(self._needed("=== 72/72 passed ===\nTraceback (most recent call last):\n"))
        self.assertTrue(fault_matrix.catchup_needed("/nonexistent/ftb.log"))
