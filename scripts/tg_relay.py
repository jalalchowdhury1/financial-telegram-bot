"""Telegram Inbox relay (health-hub): the same Bot API request, turned into a doorbell + card for Jalal.

open_tg(token, method, data, src) tries https://jalal-health.vercel.app/api/tg/<src> first (RELAY_TIMEOUT s). On a network
error, timeout, 5xx, 401 or 404 the SAME request goes straight to api.telegram.org, so a relay outage never
loses a message. Any other 4xx (e.g. 400 bad HTML) is Telegram's real answer and is raised as before.
Rollback: TG_API_BASE=https://api.telegram.org.
The relay's own worst case is ~4 s flags + ~4 s work, then it forwards the message itself (plus a cold start), so
the client waits longer than that: at 8 s it gave up while the relay still sent, and the direct send made a twin.
"""
import os
import urllib.error
import urllib.request

DIRECT = "https://api.telegram.org"
RELAY = "https://jalal-health.vercel.app/api/tg"
RELAY_TIMEOUT = 15


def open_tg(token, method, data, src, headers=None, timeout=20):
    """urllib.request.urlopen for a Bot API call, relay first. Returns the response (caller closes it)."""
    base = os.environ.get("TG_API_BASE") or f"{RELAY}/{src}"
    if not base.startswith(DIRECT):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(f"{base}/bot{token}/{method}", data=data, headers=headers or {}), timeout=RELAY_TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code not in (401, 404):
                raise
        except Exception:                      # noqa: BLE001 — relay unreachable → direct
            pass
    return urllib.request.urlopen(
        urllib.request.Request(f"{DIRECT}/bot{token}/{method}", data=data, headers=headers or {}), timeout=timeout)
