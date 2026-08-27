"""Run-outcome reporting that cannot fail silently.

The weekly harness ended with a WhatsApp summary that crashed on send
(``_standalone_send() takes 2 positional arguments but 3 were given``). Paired
with an evaluation source that produced nothing, the steady state became: fail
every skill, tell no one. A self-improvement system that cannot report its own
failures is worse than one that does not run.

Three rules here:

1. **Delivery is layered.** A signed webhook first, then whatever channels are
   configured, and always a status file on local disk. The file channel has no
   network dependency, so there is always a record.
2. **Delivery failure is itself reported.** A channel that throws is caught,
   recorded, and surfaced in the result — never swallowed.
3. **Delivery never decides the exit code.** Whether the *run* succeeded and
   whether the *notification* was delivered are separate facts, and the caller
   gets both.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

DEFAULT_TIMEOUT_S = 10


@dataclass
class DeliveryResult:
    """Whether one channel accepted the message."""

    channel: str
    delivered: bool
    detail: str = ""

    def render(self) -> str:
        mark = "ok" if self.delivered else "FAILED"
        return f"{self.channel}: {mark}{' — ' + self.detail if self.detail else ''}"


@dataclass
class NotificationOutcome:
    """The result of trying every configured channel."""

    results: list[DeliveryResult] = field(default_factory=list)

    @property
    def delivered(self) -> bool:
        return any(r.delivered for r in self.results)

    @property
    def failures(self) -> list[DeliveryResult]:
        return [r for r in self.results if not r.delivered]

    def render(self) -> str:
        return "; ".join(r.render() for r in self.results) or "no channels configured"


class Channel(Protocol):
    name: str

    def send(self, subject: str, body: str) -> DeliveryResult: ...


class FileChannel:
    """Append to a local status log and keep a latest-run snapshot.

    Always available, no network. This is the channel that guarantees a broken
    run leaves evidence somewhere even when every remote channel is down.
    """

    name = "file"

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def send(self, subject: str, body: str) -> DeliveryResult:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            entry = f"\n{'=' * 72}\n{stamp}  {subject}\n{'=' * 72}\n{body}\n"
            with open(self.directory / "runs.log", "a", encoding="utf-8") as fh:
                fh.write(entry)
            (self.directory / "latest.txt").write_text(
                f"{stamp}  {subject}\n\n{body}\n", encoding="utf-8"
            )
            return DeliveryResult(self.name, True, str(self.directory / "latest.txt"))
        except OSError as exc:
            return DeliveryResult(self.name, False, str(exc))


class WebhookChannel:
    """POST to a Hermes webhook route, HMAC-signed with the V2 scheme.

    The signature is ``HMAC-SHA256(secret, "<ts>.<body>")`` — the same scheme
    Hermes' own inbound routes verify.
    """

    name = "webhook"

    def __init__(
        self,
        url: str,
        secret: str,
        field_name: str = "report",
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ):
        self.url = url
        self.secret = secret
        self.field_name = field_name
        self.timeout_s = timeout_s

    def send(self, subject: str, body: str) -> DeliveryResult:
        if not self.url or not self.secret:
            return DeliveryResult(self.name, False, "url or secret not configured")

        payload = json.dumps({self.field_name: f"{subject}\n\n{body}"}).encode()
        timestamp = str(int(time.time()))
        signature = hmac.new(
            self.secret.encode(),
            timestamp.encode() + b"." + payload,
            hashlib.sha256,
        ).hexdigest()

        request = urllib.request.Request(
            self.url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Hermes-Timestamp": timestamp,
                "X-Hermes-Signature": signature,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                status = getattr(response, "status", 200)
                if 200 <= status < 300:
                    return DeliveryResult(self.name, True, f"HTTP {status}")
                return DeliveryResult(self.name, False, f"HTTP {status}")
        except urllib.error.HTTPError as exc:
            return DeliveryResult(self.name, False, f"HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001 — any transport failure is a failure
            return DeliveryResult(self.name, False, f"{type(exc).__name__}: {exc}")


class Notifier:
    """Sends a run summary through every configured channel."""

    def __init__(self, channels: Optional[list[Channel]] = None):
        self.channels: list[Channel] = channels or []

    @classmethod
    def from_env(cls, status_dir: Optional[Path] = None) -> "Notifier":
        """Build from environment configuration.

        ``EVOLUTION_WEBHOOK_URL`` plus one of ``EVOLUTION_WEBHOOK_SECRET`` or
        ``EVOLUTION_WEBHOOK_SECRET_FILE`` enables the webhook. The file channel
        is always added last so there is a local record regardless.
        """
        channels: list[Channel] = []

        url = os.getenv("EVOLUTION_WEBHOOK_URL", "").strip()
        secret = os.getenv("EVOLUTION_WEBHOOK_SECRET", "").strip()
        secret_file = os.getenv("EVOLUTION_WEBHOOK_SECRET_FILE", "").strip()
        if not secret and secret_file:
            try:
                secret = Path(secret_file).read_text().strip()
            except OSError:
                secret = ""
        if url and secret:
            channels.append(WebhookChannel(url=url, secret=secret))

        directory = status_dir or Path(
            os.getenv("EVOLUTION_STATUS_DIR", "") or Path.cwd() / "output" / "_status"
        )
        channels.append(FileChannel(directory))

        return cls(channels)

    def send(self, subject: str, body: str) -> NotificationOutcome:
        outcome = NotificationOutcome()
        for channel in self.channels:
            try:
                outcome.results.append(channel.send(subject, body))
            except Exception as exc:  # noqa: BLE001 — a channel bug is not fatal
                outcome.results.append(
                    DeliveryResult(
                        getattr(channel, "name", "unknown"),
                        False,
                        f"channel raised {type(exc).__name__}: {exc}",
                    )
                )
        return outcome


@dataclass
class RunSummary:
    """One evolution run's outcome, in a shape suitable for a message."""

    subject: str
    succeeded: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def exit_code(self) -> int:
        """Non-zero when anything failed.

        A run that could not evaluate a single skill must not exit 0. That is
        what let the broken weekly job look healthy to every process watching
        it.
        """
        return 0 if self.ok else 1

    def render(self) -> str:
        lines = [
            f"{len(self.succeeded)} succeeded, {len(self.failed)} failed, "
            f"{len(self.skipped)} skipped",
        ]
        for name in self.succeeded:
            lines.append(f"  ok    {name}")
        for name, reason in self.failed:
            lines.append(f"  FAIL  {name} — {reason}")
        for name, reason in self.skipped:
            lines.append(f"  skip  {name} — {reason}")
        if self.notes:
            lines.append("")
            lines.extend(self.notes)
        return "\n".join(lines)


def report_run(
    summary: RunSummary,
    notifier: Optional[Notifier] = None,
    status_dir: Optional[Path] = None,
) -> tuple[int, NotificationOutcome]:
    """Deliver a run summary and return (exit_code, delivery_outcome).

    The exit code reflects the *run*, not the notification — a delivered
    message about a failed run still exits non-zero, and an undelivered
    message about a successful run still exits zero. The caller is expected to
    print the delivery outcome either way.
    """
    notifier = notifier or Notifier.from_env(status_dir)
    outcome = notifier.send(summary.subject, summary.render())
    return summary.exit_code, outcome
