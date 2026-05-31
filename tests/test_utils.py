from unittest.mock import patch, MagicMock
from bot.utils import report_marker, send_to_telegram, _split_message


def test_report_marker_success_is_greppable():
    assert report_marker(True, sections=2, errors=1) == "REPORT_DELIVERED ok=true sections=2 errors=1"


def test_report_marker_failure_includes_reason():
    marker = report_marker(False, reason="empty_content")
    assert marker.startswith("REPORT_FAILED")
    assert "ok=false" in marker
    assert "reason=empty_content" in marker


@patch('bot.utils.requests.post')
def test_send_falls_back_to_plain_text_on_400(mock_post):
    mock_post.side_effect = [MagicMock(status_code=400), MagicMock(status_code=200)]
    ok = send_to_telegram("tok", "chat", caption="*unbalanced_markdown")
    assert ok is True
    assert mock_post.call_count == 2
    # The retry must NOT carry parse_mode (that's what triggered the 400).
    _, retry_kwargs = mock_post.call_args_list[1]
    assert 'parse_mode' not in retry_kwargs['data']


@patch('bot.utils.requests.post')
def test_send_returns_false_when_both_attempts_fail(mock_post):
    mock_post.return_value = MagicMock(status_code=400)
    assert send_to_telegram("tok", "chat", caption="x") is False


@patch('bot.utils.requests.post')
def test_send_chunks_messages_over_the_limit(mock_post):
    mock_post.return_value = MagicMock(status_code=200)
    long_text = "\n".join("line %d" % i for i in range(2000))  # well over 4096 chars
    assert send_to_telegram("tok", "chat", caption=long_text) is True
    assert mock_post.call_count >= 2


def test_split_message_respects_limit_and_splits_long_lines():
    text = "\n".join("x" * 100 for _ in range(100))   # ~10,099 chars
    chunks = _split_message(text, limit=4096)
    assert len(chunks) > 1
    assert all(len(c) <= 4096 for c in chunks)
    # one giant line longer than the limit must be hard-split, not dropped
    chunks2 = _split_message("y" * 9000, limit=4096)
    assert all(len(c) <= 4096 for c in chunks2)
    assert sum(c.count("y") for c in chunks2) == 9000
