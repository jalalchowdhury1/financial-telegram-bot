import lambda_handler


def _brief(monkeypatch, spy):
    monkeypatch.delenv('JEV_PILLS_URL', raising=False)
    monkeypatch.setattr(lambda_handler, 'fetch_google_sheet_indicators', lambda: 'NotSoBoring : ON\nFrontRunner : BIL')
    monkeypatch.setattr(lambda_handler, 'fetch_spy_with_fallback', lambda **k: spy)
    sent = {}
    monkeypatch.setattr(lambda_handler, 'digest_post', lambda *a, **k: sent.update(text=a[1], sig=k['sig']) or True)
    lambda_handler.handle_eventbridge({'TELEGRAM_TOKEN': 't', 'TELEGRAM_CHAT_ID': 'c'}, 'now')
    return sent


def test_lost_section_is_named_and_never_a_quiet_heartbeat(monkeypatch):
    sent = _brief(monkeypatch, {'error': 'all tiers failed'})
    assert sent['text'].endswith('⚠️ Missing today: SPY')
    assert sent['sig'] == ''


def test_full_brief_keeps_its_signal_sig(monkeypatch):
    sent = _brief(monkeypatch, {'current': 500.0})
    assert 'Missing' not in sent['text']
    assert sent['sig'] == 'NotSoBoring=ON|FrontRunner=BIL'


def test_non_public_path_rejects_an_empty_secret_header():
    event = {'rawPath': '/api/send', 'requestContext': {'http': {'method': 'GET'}}, 'headers': {'x-bot-secret': ''}}
    assert lambda_handler.handle_http_api(event, {'FRED_API_KEY': 'f'})['statusCode'] == 401
