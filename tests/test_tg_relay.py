"""tg_relay.open_tg: relay first, same request straight to Telegram when the relay is down (FAKE data)."""
import io
import os
import sys
import urllib.error
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import tg_relay  # noqa: E402


def _err(code):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b"{}"))


class OpenTg(unittest.TestCase):
    def setUp(self):
        os.environ.pop("TG_API_BASE", None)

    def run_with(self, *answers):
        calls = []

        def fake(req, timeout):
            calls.append((req.full_url, req.data))
            a = answers[len(calls) - 1]
            if isinstance(a, Exception):
                raise a
            return a
        with mock.patch("urllib.request.urlopen", side_effect=fake):
            out = tg_relay.open_tg("T", "sendMessage", b"FAKE", "defensive")
        return out, calls

    def test_relay_ok(self):
        out, calls = self.run_with("ok")
        self.assertEqual(out, "ok")
        self.assertEqual(calls, [("https://jalal-health.vercel.app/api/tg/defensive/botT/sendMessage", b"FAKE")])

    def test_fallback_on_5xx_401_404_and_network(self):
        for first in (_err(502), _err(401), _err(404), OSError("down")):
            out, calls = self.run_with(first, "direct")
            self.assertEqual(out, "direct")
            self.assertEqual(calls[1], ("https://api.telegram.org/botT/sendMessage", b"FAKE"))

    def test_telegram_400_is_raised_not_resent(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.run_with(_err(400), "direct")

    def test_rollback_env_goes_direct(self):
        os.environ["TG_API_BASE"] = "https://api.telegram.org"
        try:
            _, calls = self.run_with("ok")
        finally:
            os.environ.pop("TG_API_BASE")
        self.assertEqual(calls[0][0], "https://api.telegram.org/botT/sendMessage")


if __name__ == "__main__":
    unittest.main()
