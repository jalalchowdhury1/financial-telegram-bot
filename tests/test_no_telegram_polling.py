"""Nothing in this repo may poll Telegram or delete a webhook.

TELEGRAM_TOKEN here is @TweetSyn_bot, the alerts bot. Its updates are owned by
health-hub's webhook (/api/defensive: Silent-digest card buttons, defensive-trigger
taps). `python -m bot.main` (Render's startCommand, re-run on every push to main) used
to call run_polling(), which deletes the webhook on start: 26 wipes 26 Sep-10 Oct 2026,
each one a dead-button window until health-hub's 5-min tick re-registered it."""
import pathlib
import re
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r'run_polling\s*\(|start_polling\s*\(|infinity_polling\s*\(|deleteWebhook|delete_webhook\s*\('
    r'|remove_webhook\s*\(|/getUpdates|get_updates\s*\(|Updater\s*\(|Application\.builder\s*\(')


def _python_sources():
    files = [ROOT / 'lambda_handler.py']
    for d in ('bot', 'scripts'):
        files += sorted((ROOT / d).rglob('*.py'))
    return [p for p in files if p.is_file()]


def test_no_source_polls_telegram_or_deletes_a_webhook():
    hits = [f'{p.relative_to(ROOT)}:{i}: {line.strip()}'
            for p in _python_sources()
            for i, line in enumerate(p.read_text(errors='ignore').splitlines(), 1)
            if FORBIDDEN.search(line) and not line.lstrip().startswith('#')]
    assert not hits, 'Telegram polling/deleteWebhook on the alerts-bot token:\n' + '\n'.join(hits)


def test_serve_mode_runs_flask_in_the_foreground_and_never_touches_telegram():
    import bot.main as m
    with patch.object(m, 'load_environment_variables', return_value={}), \
         patch.object(m, 'BackgroundScheduler') as sched, \
         patch.object(m, 'run_flask') as flask, \
         patch('requests.post') as post, patch('requests.get') as get:
        m.main()
    sched.return_value.start.assert_called_once()
    flask.assert_called_once_with()
    post.assert_not_called()
    get.assert_not_called()
