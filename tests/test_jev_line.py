"""Jev regime line for the daily brief (2026-09-19).

Contract: with JEV_PILLS_URL unset nothing is fetched and the brief is unchanged;
with it set, every failure path yields None and the brief still goes out.
"""
import os
from unittest.mock import MagicMock, patch

import requests

from bot.jev_line import fetch_jev_line, format_jev_line

SAMPLE = {
    'enabled': True, 'mode': 'on', 'asOf': '2026-09-19T08:00:00Z',
    'pills': {
        'regime': {'verdict': 'risk-on', 'p': 0.82, 'by': 'jev', 'reason': 'x'},
        'recession': {'verdict': 'low', 'p': 1, 'by': 'rule', 'reason': 'y'},
        'breadth': {'verdict': 'narrow', 'p': 0.71, 'by': 'jev', 'reason': 'z'},
        'hedging': {'verdict': 'cheap', 'p': 0.9, 'by': 'jev', 'reason': 'w'},
        'conflict': {'verdict': 'mild-divergence', 'p': 0.66, 'by': 'jev', 'reason': 'v'},
    },
    'conflictPairs': [{'pair': 'sentiment vs price', 'detail': 'fg 29 vs +6.3% over MA200'}],
    'since': {'date': '2026-09-18', 'direction': 'softening',
              'changed': [{'pill': 'breadth', 'from': 'rolling-over', 'to': 'narrow'}], 'noBaseline': False},
}


def test_format_full_payload():
    line = format_jev_line(SAMPLE)
    assert line.startswith('🧭 *Jev regime*\n')
    assert 'Regime risk-on · Recession low (rule) · Breadth narrow · Hedges cheap · Conflict mild-divergence: sentiment vs price' in line
    assert line.endswith('_since yesterday: softening (breadth rolling-over→narrow)_')


def test_format_no_baseline_and_no_change():
    p = dict(SAMPLE, since={'noBaseline': True, 'direction': 'none', 'changed': []})
    assert format_jev_line(p).endswith('_since yesterday: n/a_')
    p = dict(SAMPLE, since={'direction': 'none', 'changed': []})
    assert format_jev_line(p).endswith('_since yesterday: no change_')
    p = dict(SAMPLE, since=None)
    assert format_jev_line(p).endswith('_since yesterday: n/a_')


def test_format_unusable_payloads_give_none():
    assert format_jev_line(None) is None
    assert format_jev_line({'enabled': False}) is None
    assert format_jev_line({'enabled': True, 'pills': None}) is None
    assert format_jev_line({'enabled': True, 'pills': {'regime': {'verdict': None}}}) is None
    assert format_jev_line('garbage') is None


def test_format_strips_markdown_control_chars():
    p = dict(SAMPLE, pills={'regime': {'verdict': 'risk_on*', 'by': 'jev'}}, conflictPairs=[], since=None)
    line = format_jev_line(p)
    assert 'risk-on' in line and 'risk_on' not in line and 'on*' not in line


def test_fetch_empty_url_never_calls_network():
    with patch('bot.jev_line.requests.get') as get:
        assert fetch_jev_line('') is None
        get.assert_not_called()


def test_fetch_swallows_every_failure():
    with patch('bot.jev_line.requests.get', side_effect=requests.Timeout('slow')):
        assert fetch_jev_line('https://x.test/api/jev-pills') is None
    bad = MagicMock(); bad.raise_for_status.side_effect = requests.HTTPError('500')
    with patch('bot.jev_line.requests.get', return_value=bad):
        assert fetch_jev_line('https://x.test/api/jev-pills') is None
    nojson = MagicMock(); nojson.json.side_effect = ValueError('not json')
    with patch('bot.jev_line.requests.get', return_value=nojson):
        assert fetch_jev_line('https://x.test/api/jev-pills') is None


def test_fetch_formats_good_reply():
    ok = MagicMock(); ok.json.return_value = SAMPLE
    with patch('bot.jev_line.requests.get', return_value=ok) as get:
        line = fetch_jev_line('https://x.test/api/jev-pills')
    assert 'Regime risk-on' in line
    assert get.call_args.kwargs['timeout'] == 8.0


def _run_handler(monkeypatch):
    import lambda_handler
    monkeypatch.setattr(lambda_handler, 'fetch_google_sheet_indicators', lambda: 'sheet text')
    monkeypatch.setattr(lambda_handler, 'fetch_spy_with_fallback', lambda **k: {'error': 'skip'})
    sent = {}
    monkeypatch.setattr(lambda_handler, 'digest_post', lambda *a, **k: sent.update(text=a[1]) or True)
    env = {'TELEGRAM_TOKEN': 't', 'TELEGRAM_CHAT_ID': 'c', 'FRED_API_KEY': 'f'}
    lambda_handler.handle_eventbridge(env, '2026-09-19')
    return sent['text']


def test_handler_without_env_var_never_fetches(monkeypatch):
    monkeypatch.delenv('JEV_PILLS_URL', raising=False)
    with patch('lambda_handler.fetch_jev_line') as f:
        text = _run_handler(monkeypatch)
    f.assert_not_called()
    assert 'Jev' not in text


def test_handler_with_env_var_appends_line(monkeypatch):
    monkeypatch.setenv('JEV_PILLS_URL', 'https://x.test/api/jev-pills')
    with patch('lambda_handler.fetch_jev_line', return_value='🧭 *Jev regime*\nRegime risk-on') as f:
        text = _run_handler(monkeypatch)
    f.assert_called_once_with('https://x.test/api/jev-pills')
    assert text.endswith('Regime risk-on')


def test_handler_with_env_var_but_no_line_is_unchanged(monkeypatch):
    monkeypatch.setenv('JEV_PILLS_URL', 'https://x.test/api/jev-pills')
    with patch('lambda_handler.fetch_jev_line', return_value=None):
        text = _run_handler(monkeypatch)
    assert text == 'sheet text'
