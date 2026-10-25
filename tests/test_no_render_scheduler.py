"""Bare `python -m bot.main` (Render's startCommand) is a health server only.

Its 04:15 APScheduler job sent a second daily report next to the Lambda's, from a
service nothing uses any more, so it was removed (25 Oct 2026). The Flask server stays
so Render still sees a live service. `python -m bot.main report` (daily_report.yml's
backstop) is a one-shot and is not affected."""
import pathlib
import threading
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_serve_mode_starts_no_scheduler_and_schedules_no_job():
    import bot.main as m
    before = set(threading.enumerate())
    with patch.object(m, 'load_environment_variables', return_value={}), \
         patch.object(m, 'run_flask') as flask, \
         patch.object(m, 'run_report') as report:
        m.main()
    flask.assert_called_once_with()
    report.assert_not_called()
    assert set(threading.enumerate()) <= before, 'main() started a background thread'
    assert not hasattr(m, 'global_scheduler')


def test_bot_main_source_has_no_scheduler():
    src = (ROOT / 'bot' / 'main.py').read_text()
    for word in ('BackgroundScheduler', 'add_job', 'CronTrigger', 'apscheduler'):
        assert word not in src, f'bot/main.py mentions {word}'
