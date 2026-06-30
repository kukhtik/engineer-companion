"""Local usage tracker for Ollama Cloud.

Note on server-side quota
-------------------------
Ollama Cloud does NOT expose server-side remaining quota via any API endpoint or
response header.  The upstream feature request (issue #15663) was closed as a
duplicate with no resolution.  Ollama's documented limits are:

  • 5-hour session window  (GPU-time based, exact threshold undisclosed)
  • 7-day rolling week window

This module tracks usage LOCALLY by persisting a timestamped event log to disk
and computing rolling sums over those two windows.  It is a *local approximation*
only — it counts your recorded calls, not GPU-seconds consumed on the server.

Usage::

    from core.cloud_usage import CloudUsageTracker
    tracker = CloudUsageTracker()
    tracker.record_request(prompt_tokens=120, completion_tokens=340)
    print(tracker.snapshot())
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_5H_SECS = 5 * 3600
_7D_SECS = 7 * 24 * 3600
_DEFAULT_RATELIMIT_SECONDS = _5H_SECS  # conservative: assume full 5-h window blackout


class CloudUsageTracker:
    """Persist and query local Ollama Cloud usage statistics.

    All data lives in a single JSON file::

        data_root() / "cloud_usage.json"

    The file format is::

        {
            "events": [
                {"ts": <epoch float>, "prompt_tokens": int, "completion_tokens": int},
                ...
            ],
            "rate_limited_until": <epoch float> | null
        }

    Thread-safety: writes are atomic (write-to-temp-then-rename on POSIX;
    best-effort on Windows where rename over an existing file is allowed).
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            from windows.app_paths import data_root
            data_dir = data_root()
        self._path = Path(data_dir) / "cloud_usage.json"
        self._state: dict[str, Any] = {"events": [], "rate_limited_until": None}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load state from disk (idempotent; creates empty state if file missing)."""
        try:
            raw = self._path.read_text(encoding="utf-8")
            loaded = json.loads(raw)
            if isinstance(loaded.get("events"), list):
                self._state["events"] = loaded["events"]
            self._state["rate_limited_until"] = loaded.get("rate_limited_until")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass  # fresh state

    def _save(self) -> None:
        """Atomically write state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False), encoding="utf-8")
        try:
            # On Windows, os.replace works atomically enough for our purposes.
            os.replace(str(tmp), str(self._path))
        except OSError:
            # Last-resort: direct write
            self._path.write_text(json.dumps(self._state, ensure_ascii=False), encoding="utf-8")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _prune(self, now: float) -> None:
        """Drop events older than 7 days (the longest tracking window)."""
        cutoff = now - _7D_SECS
        self._state["events"] = [e for e in self._state["events"] if e["ts"] >= cutoff]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_request(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Append a usage event and persist.

        Args:
            prompt_tokens:      Number of prompt tokens consumed.
            completion_tokens:  Number of completion tokens generated.
        """
        now = time.time()
        self._state["events"].append({
            "ts": now,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
        })
        self._prune(now)
        self._save()

    def mark_rate_limited(self, retry_after_seconds: float | None = None) -> None:
        """Record that we hit a 429 and set a local cooldown.

        Args:
            retry_after_seconds: If the server returned Retry-After, pass it
                                 here.  Defaults to 5 hours (full session window).
        """
        wait = retry_after_seconds if retry_after_seconds is not None else _DEFAULT_RATELIMIT_SECONDS
        self._state["rate_limited_until"] = time.time() + wait
        self._save()

    def is_rate_limited(self) -> bool:
        """Return True if we are currently within a locally-recorded cooldown."""
        until = self._state.get("rate_limited_until")
        if until is None:
            return False
        return time.time() < until

    def snapshot(self) -> dict[str, Any]:
        """Return a summary of local usage statistics.

        Returns::

            {
                "session": {
                    "requests": int,
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "window_hours": 5,
                },
                "week": {
                    "requests": int,
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "window_days": 7,
                },
                "rate_limited_until": float | null,
            }

        Note: these are LOCAL approximations.  Ollama does not expose server-side
        remaining quota (see module docstring).
        """
        now = time.time()
        session_cutoff = now - _5H_SECS
        week_cutoff = now - _7D_SECS

        session_events = [e for e in self._state["events"] if e["ts"] >= session_cutoff]
        week_events = [e for e in self._state["events"] if e["ts"] >= week_cutoff]

        def _sum(events: list[dict]) -> dict:
            return {
                "requests": len(events),
                "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in events),
                "completion_tokens": sum(e.get("completion_tokens", 0) for e in events),
            }

        return {
            "session": {**_sum(session_events), "window_hours": 5},
            "week": {**_sum(week_events), "window_days": 7},
            "rate_limited_until": self._state.get("rate_limited_until"),
        }
