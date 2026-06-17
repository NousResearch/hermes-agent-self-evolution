"""Multi-group session miner with group filtering.

PR #4: Mines session traces from Hermes DB with group-level filtering.
This enables evolution to learn from specific groups while respecting
privacy boundaries (e.g., excluding business-critical groups).

Usage:
    from evolution.core.group_session_miner import GroupSessionMiner
    
    miner = GroupSessionMiner(
        target_groups=["-1003855677539", "-1003773897557"],
        excluded_groups=["-1003806480938"],  # Baked in BLR — never mine
        min_messages=5,
        days_lookback=7,
    )
    traces = miner.mine_for_skill("github-code-review")
"""

import json
import sqlite3
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

HERMES_HOME = Path.home() / ".hermes"
SESSION_DB = HERMES_HOME / "sessions.db"


class GroupSessionMiner:
    """Mine session traces with group-level filtering."""

    def __init__(
        self,
        target_groups: Optional[List[str]] = None,
        excluded_groups: Optional[List[str]] = None,
        min_messages: int = 5,
        days_lookback: int = 7,
        sanitize: bool = True,
    ):
        self.target_groups = set(target_groups) if target_groups else set()
        self.excluded_groups = set(excluded_groups) if excluded_groups else set()
        self.min_messages = min_messages
        self.days_lookback = days_lookback
        self.sanitize = sanitize

    def _get_db_connection(self) -> Optional[sqlite3.Connection]:
        """Connect to Hermes session database."""
        if not SESSION_DB.exists():
            return None
        try:
            return sqlite3.connect(str(SESSION_DB))
        except sqlite3.Error:
            return None

    def _build_query(self) -> tuple:
        """Build SQL query with group filters."""
        query = """
            SELECT id, chat_id, role, content, timestamp, tool_calls, tool_outputs
            FROM messages
            WHERE timestamp > datetime('now', '-{} days')
            AND role IN ('user', 'assistant', 'tool')
        """.format(self.days_lookback)
        
        params = []
        
        # Target groups filter
        if self.target_groups:
            placeholders = ','.join('?' for _ in self.target_groups)
            query += f" AND chat_id IN ({placeholders})"
            params.extend(self.target_groups)
        
        # Excluded groups filter (CRITICAL for privacy)
        if self.excluded_groups:
            placeholders = ','.join('?' for _ in self.excluded_groups)
            query += f" AND chat_id NOT IN ({placeholders})"
            params.extend(self.excluded_groups)
        
        query += " ORDER BY chat_id, timestamp"
        
        return query, params

    def _sanitize_content(self, content: str) -> str:
        """Redact sensitive patterns from content."""
        patterns = [
            r'\b(?:\d{4}[- ]?){3}\d{4}\b',  # Credit cards
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'(?:api[_-]?key|apikey|token|secret)\s*[:=]\s*["\']?[A-Za-z0-9_\-/+=]{20,}',
            r'\b\d{9,10}:[A-Za-z0-9_-]{35}\b',  # Telegram tokens
            r'telegram:\s*-\d{10,}',  # Group IDs in content
        ]
        
        for pattern in patterns:
            content = re.sub(pattern, '[REDACTED]', content, flags=re.IGNORECASE)
        
        return content

    def mine_for_skill(self, skill_name: str) -> List[Dict]:
        """Mine session traces relevant to a specific skill.
        
        Returns list of dicts: {chat_id, role, content, timestamp, skill_relevance}
        """
        conn = self._get_db_connection()
        if not conn:
            return []
        
        query, params = self._build_query()
        cursor = conn.execute(query, params)
        
        traces = []
        skill_keywords = skill_name.lower().replace('-', ' ').replace('_', ' ').split()
        
        for row in cursor.fetchall():
            msg_id, chat_id, role, content, timestamp, tool_calls, tool_outputs = row
            
            if not content:
                continue
            
            # Sanitize if enabled
            if self.sanitize:
                content = self._sanitize_content(content)
            
            # Calculate skill relevance (simple keyword matching)
            content_lower = content.lower()
            relevance = 0
            for keyword in skill_keywords:
                if len(keyword) > 3 and keyword in content_lower:
                    relevance += 1
            
            # Include if any keyword matches or it's a tool call
            if relevance > 0 or tool_calls or role == 'tool':
                traces.append({
                    'id': msg_id,
                    'chat_id': str(chat_id),
                    'role': role,
                    'content': content,
                    'timestamp': timestamp,
                    'tool_calls': json.loads(tool_calls) if tool_calls else None,
                    'tool_outputs': json.loads(tool_outputs) if tool_outputs else None,
                    'skill_relevance': relevance,
                })
        
        conn.close()
        
        # Filter by minimum messages per group
        group_counts = {}
        for t in traces:
            group_counts[t['chat_id']] = group_counts.get(t['chat_id'], 0) + 1
        
        valid_groups = {g for g, c in group_counts.items() if c >= self.min_messages}
        traces = [t for t in traces if t['chat_id'] in valid_groups]
        
        return traces

    def get_group_stats(self) -> Dict[str, int]:
        """Return message counts per group in lookback period."""
        conn = self._get_db_connection()
        if not conn:
            return {}
        
        query, params = self._build_query()
        # Replace SELECT with count
        count_query = query.replace(
            "SELECT id, chat_id, role, content, timestamp, tool_calls, tool_outputs",
            "SELECT chat_id, COUNT(*)"
        ).replace("ORDER BY chat_id, timestamp", "GROUP BY chat_id")
        
        cursor = conn.execute(count_query, params)
        stats = {str(row[0]): row[1] for row in cursor.fetchall()}
        conn.close()
        
        return stats


def main():
    """CLI for testing group session mining."""
    import click
    
    @click.command()
    @click.option('--skill', required=True)
    @click.option('--target-groups', multiple=True)
    @click.option('--exclude-groups', multiple=True)
    @click.option('--min-messages', default=5)
    @click.option('--days', default=7)
    def cli(skill, target_groups, exclude_groups, min_messages, days):
        miner = GroupSessionMiner(
            target_groups=list(target_groups) if target_groups else None,
            excluded_groups=list(exclude_groups) if exclude_groups else None,
            min_messages=min_messages,
            days_lookback=days,
        )
        
        print(f"Mining sessions for skill: {skill}")
        print(f"Target groups: {target_groups or 'ALL'}")
        print(f"Excluded: {exclude_groups or 'NONE'}")
        
        stats = miner.get_group_stats()
        print(f"\nGroup stats (last {days} days):")
        for group, count in sorted(stats.items(), key=lambda x: -x[1]):
            marker = "✓" if (not target_groups or group in target_groups) else "○"
            print(f"  {marker} {group}: {count} messages")
        
        traces = miner.mine_for_skill(skill)
        print(f"\nMined {len(traces)} traces for '{skill}'")
    
    cli()


if __name__ == "__main__":
    main()
