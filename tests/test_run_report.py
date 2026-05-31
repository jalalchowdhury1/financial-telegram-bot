from unittest.mock import patch
import bot.main as m

ENV = {'TELEGRAM_TOKEN': 't', 'TELEGRAM_CHAT_ID': 'c', 'FRED_API_KEY': 'f'}


@patch('bot.main.load_environment_variables', return_value=ENV)
@patch('bot.main.fetch_google_sheet_indicators', return_value="")
@patch('bot.main.send_to_telegram')
def test_run_report_false_and_no_send_on_empty_content(mock_send, _fetch, _env):
    assert m.run_report() is False
    mock_send.assert_not_called()


@patch('bot.main.load_environment_variables', return_value=ENV)
@patch('bot.main.fetch_google_sheet_indicators', return_value="some report")
@patch('bot.main.send_to_telegram', return_value=False)
def test_run_report_false_on_send_failure(_send, _fetch, _env):
    assert m.run_report() is False


@patch('bot.main.load_environment_variables', return_value=ENV)
@patch('bot.main.fetch_google_sheet_indicators', return_value="some report")
@patch('bot.main.send_to_telegram', return_value=True)
def test_run_report_true_on_success(mock_send, _fetch, _env):
    assert m.run_report() is True
    mock_send.assert_called_once()
