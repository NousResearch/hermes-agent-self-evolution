#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ENV_PATH = Path('/Users/kirniy/.hermes/.env')
STATE_PATH = Path('/Users/kirniy/.hermes/google_key_pool_state.json')
KEY_PATTERNS = [
    re.compile(r'^(GOOGLE_API_KEY)(?:_(\d+))?$'),
    re.compile(r'^(GEMINI_API_KEY)(?:_(\d+))?$'),
]
COOLDOWN_SECONDS = 6 * 60 * 60


def parse_env(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    seen = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not value:
            continue
        for pattern in KEY_PATTERNS:
            m = pattern.match(key)
            if m:
                order = int(m.group(2) or '1')
                if value in seen:
                    break
                seen.add(value)
                entries.append({'env_key': key, 'value': value, 'order': order})
                break
    entries.sort(key=lambda item: (item['order'], item['env_key']))
    return entries


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {'bad': {}, 'cursor': 0}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {'bad': {}, 'cursor': 0}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2) + '\n')


def select_key() -> int:
    entries = parse_env(ENV_PATH)
    if not entries:
        print('No GOOGLE_API_KEY / GEMINI_API_KEY pool entries found in ~/.hermes/.env', file=sys.stderr)
        return 1

    state = load_state()
    now = time.time()
    bad = {k: v for k, v in (state.get('bad') or {}).items() if float(v.get('until', 0)) > now}
    state['bad'] = bad
    cursor = int(state.get('cursor', 0) or 0)

    ordered = entries[cursor:] + entries[:cursor]
    chosen = None
    for idx, entry in enumerate(ordered):
        if entry['env_key'] in bad:
            continue
        chosen = entry
        state['cursor'] = (cursor + idx + 1) % len(entries)
        break

    if chosen is None:
        # All keys cooling down. Reset cooldowns and try first key again.
        state['bad'] = {}
        chosen = ordered[0]
        state['cursor'] = (cursor + 1) % len(entries)

    save_state(state)
    print(chosen['env_key'])
    print(chosen['value'])
    return 0


def mark_bad(env_key: str, reason: str) -> int:
    entries = parse_env(ENV_PATH)
    if not any(entry['env_key'] == env_key for entry in entries):
        print(f'Unknown env key: {env_key}', file=sys.stderr)
        return 1
    state = load_state()
    bad = state.setdefault('bad', {})
    bad[env_key] = {
        'reason': reason,
        'at': time.time(),
        'until': time.time() + COOLDOWN_SECONDS,
    }
    save_state(state)
    print(f'marked bad: {env_key} ({reason})')
    return 0


def reset_state() -> int:
    save_state({'bad': {}, 'cursor': 0})
    print('reset')
    return 0


def count_keys() -> int:
    print(len(parse_env(ENV_PATH)))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print('usage: google_key_pool.py select|mark-bad <ENV_KEY> <reason>|reset|count', file=sys.stderr)
        return 1
    cmd = argv[1]
    if cmd == 'select':
        return select_key()
    if cmd == 'mark-bad' and len(argv) >= 4:
        return mark_bad(argv[2], ' '.join(argv[3:]))
    if cmd == 'reset':
        return reset_state()
    if cmd == 'count':
        return count_keys()
    print('usage: google_key_pool.py select|mark-bad <ENV_KEY> <reason>|reset|count', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
