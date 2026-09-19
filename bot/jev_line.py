"""Optional one-line Jev regime chip for the daily Telegram brief (2026-09-19).

Gated by the ``JEV_PILLS_URL`` env var (hand-managed in the Lambda console — see the
config-drift note in AGENTS.md §2). Unset ⇒ this module is never called and the brief
is byte-for-byte what it was. Set ⇒ one GET to the dashboard's ``/api/jev-pills`` with
a short timeout; ANY failure (timeout, HTTP error, bad JSON, pills disabled, no pills)
⇒ ``None`` and the brief goes out without the line. Never raises.

Output is legacy Telegram Markdown (the brief is sent with parse_mode "Markdown"), so
no bare ``*`` or ``_`` may appear inside the text — verdicts use hyphens only.
"""
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

TIMEOUT_S = 8.0
ORDER: List[str] = ['regime', 'recession', 'breadth', 'hedging', 'conflict']
LABEL: Dict[str, str] = {
    'regime': 'Regime', 'recession': 'Recession', 'breadth': 'Breadth',
    'hedging': 'Hedges', 'conflict': 'Conflict',
}


def _clean(s: Any) -> str:
    """Strip the two legacy-Markdown control characters so a verdict can't break the brief."""
    return str(s).replace('*', '').replace('_', '-')


def format_jev_line(payload: Any) -> Optional[str]:
    """Render the /api/jev-pills payload as one Telegram section, or None if unusable."""
    if not isinstance(payload, dict) or payload.get('enabled') is False:
        return None
    pills = payload.get('pills') or {}
    parts: List[str] = []
    for key in ORDER:
        pill = pills.get(key) if isinstance(pills, dict) else None
        verdict = (pill or {}).get('verdict') if isinstance(pill, dict) else None
        if not verdict:
            continue
        by = ' (rule)' if pill.get('by') != 'jev' else ''
        parts.append(f'{LABEL[key]} {_clean(verdict)}{by}')
    if not parts:
        return None

    pairs = payload.get('conflictPairs') or []
    names = [_clean(p.get('pair')) for p in pairs if isinstance(p, dict) and p.get('pair')]
    if names and parts[-1].startswith('Conflict'):
        parts[-1] += ': ' + '; '.join(names[:2])

    since = payload.get('since') or {}
    direction = since.get('direction') if isinstance(since, dict) else None
    if not isinstance(since, dict) or since.get('noBaseline') or not direction:
        since_text = 'since yesterday: n/a'
    elif direction == 'none':
        since_text = 'since yesterday: no change'
    else:
        moves = [
            f"{_clean(c.get('pill'))} {_clean(c.get('from'))}→{_clean(c.get('to'))}"
            for c in (since.get('changed') or [])[:3] if isinstance(c, dict)
        ]
        since_text = f'since yesterday: {_clean(direction)}' + (f" ({', '.join(moves)})" if moves else '')

    return '🧭 *Jev regime*\n' + ' · '.join(parts) + f'\n_{since_text}_'


def fetch_jev_line(url: str, timeout: float = TIMEOUT_S) -> Optional[str]:
    """GET the pills and format them. Empty url or any failure ⇒ None (brief unchanged)."""
    if not url:
        return None
    try:
        r = requests.get(url, timeout=timeout, headers={'User-Agent': 'financial-telegram-bot/lambda'})
        r.raise_for_status()
        return format_jev_line(r.json())
    except Exception as e:  # noqa: BLE001 — any failure = no line, never a failed brief
        logger.warning(f'Jev line skipped: {type(e).__name__}: {str(e)[:120]}')
        return None
