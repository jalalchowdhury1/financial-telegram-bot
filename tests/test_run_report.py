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


def test_signal_sig_only_the_two_signals_and_empty_when_na():
    import lambda_handler
    rep = "🛡️ NotSoBoring : ON\n\n🔑 FrontRunner : BIL (T-Bill ETF)\n\n🎢 VIX: 17 | 18 | GREED13\nPrice : $580.10"
    assert lambda_handler.signal_sig(rep) == "NotSoBoring=ON|FrontRunner=BIL (T-Bill ETF)"
    assert lambda_handler.signal_sig(rep.replace("$580.10", "$571.00")) == lambda_handler.signal_sig(rep)
    assert lambda_handler.signal_sig(rep.replace("NotSoBoring : ON", "NotSoBoring : N/A")) == ""
    assert lambda_handler.signal_sig("Price : $580") == ""
