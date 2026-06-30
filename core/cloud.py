"""Ollama Cloud generation backend.

API spec (verified):
  Chat:   POST https://ollama.com/api/chat
          body  {"model": "...", "messages": [...], "stream": true}
          auth  Authorization: Bearer <OLLAMA_API_KEY>
          NDJSON stream: each line {"message":{"role":"assistant","content":"..."}, "done":false}
                         final line {"done":true, "prompt_eval_count":N, "eval_count":M}
  Models: GET  https://ollama.com/api/tags  (same auth)
          resp {"models":[{"name":"..."},...]}

Context limit: cloud models cap at 16,384 tokens — callers must stay within that.
Rate-limiting: 429 HTTP status.  No quota-remaining headers are exposed (upstream
issue #15663 closed as won't-fix); use CloudUsageTracker for local approximation.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CloudError(Exception):
    """Generic Ollama Cloud error (network, HTTP non-429, JSON parse)."""


class CloudRateLimitError(CloudError):
    """HTTP 429 — rate limit hit.  retry_after is seconds to wait if known."""

    def __init__(self, message: str = "Ollama Cloud rate limit exceeded", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Curated fallback model list (used when /api/tags fails)
# ---------------------------------------------------------------------------

FALLBACK_MODELS: list[str] = [
    "gpt-oss:120b-cloud",
    "gpt-oss:20b-cloud",
    "qwen3.5:cloud",
    "qwen3-coder:480b-cloud",
    "deepseek-v3.1:671b-cloud",
    "glm-5.1",
    "kimi-k2.6",
    "minimax-m3",
]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaCloudClient:
    """Thin client for the Ollama Cloud API.

    Args:
        api_key:  Bearer token for Authorization header.
        base_url: API root (default ``https://ollama.com``).
        timeout:  Seconds to wait for a response (default 120).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ollama.com",
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = None  # lazily created requests.Session

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_session(self):
        """Return a cached requests.Session with auth header pre-set."""
        if self._session is not None:
            return self._session
        try:
            import requests as _requests
            sess = _requests.Session()
            sess.headers.update({"Authorization": f"Bearer {self.api_key}"})
            self._session = sess
            return self._session
        except ImportError:
            return None  # will fall back to urllib

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _post_stream_urllib(self, url: str, payload: dict) -> Iterator[bytes]:
        """urllib fallback for streaming POST (when requests is unavailable)."""
        import urllib.request
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise CloudRateLimitError() from exc
            raise CloudError(f"HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise CloudError(f"Network error: {exc.reason}") from exc
        for line in resp:
            yield line

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        on_done: Callable[[dict], None] | None = None,
    ) -> Iterator[str]:
        """Stream chat completions from Ollama Cloud.

        Yields each text piece from ``message.content`` as it arrives.
        When the final ``done:true`` line is received, calls ``on_done`` with
        ``{"prompt_tokens": N, "completion_tokens": M}`` before returning.

        Raises:
            CloudRateLimitError: on HTTP 429.
            CloudError: on any other HTTP or network error.
        """
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options

        sess = self._get_session()

        if sess is not None:
            # --- requests path ---
            try:
                resp = sess.post(url, json=payload, stream=True, timeout=self.timeout)
            except Exception as exc:
                raise CloudError(f"Network error: {exc}") from exc

            if resp.status_code == 429:
                retry_after: float | None = None
                try:
                    retry_after = float(resp.headers.get("Retry-After", ""))
                except (TypeError, ValueError):
                    pass
                raise CloudRateLimitError(retry_after=retry_after)
            if not resp.ok:
                raise CloudError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line_text = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
                yield from self._process_line(line_text, on_done)
        else:
            # --- urllib fallback ---
            try:
                for raw_line in self._post_stream_urllib(url, payload):
                    line_text = raw_line.decode("utf-8", errors="replace").strip()
                    if not line_text:
                        continue
                    yield from self._process_line(line_text, on_done)
            except (CloudRateLimitError, CloudError):
                raise
            except Exception as exc:
                raise CloudError(f"Stream error: {exc}") from exc

    def _process_line(
        self, line: str, on_done: Callable[[dict], None] | None
    ) -> Iterator[str]:
        """Parse one NDJSON line; yield content piece or fire on_done."""
        line = line.strip()
        if not line:
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return  # skip malformed lines silently

        # Non-final line: yield the content piece
        if not obj.get("done", False):
            content = obj.get("message", {}).get("content", "")
            if content:
                yield content
            return

        # Final line (done=true): fire on_done with token counts
        if on_done is not None:
            prompt_tokens = obj.get("prompt_eval_count", 0) or 0
            completion_tokens = obj.get("eval_count", 0) or 0
            on_done({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})

    def list_models(self) -> list[str]:
        """Return available cloud model names.

        Tries GET /api/tags; falls back to FALLBACK_MODELS on any error.
        """
        url = f"{self.base_url}/api/tags"
        sess = self._get_session()

        try:
            if sess is not None:
                resp = sess.get(url, timeout=self.timeout)
                if not resp.ok:
                    return list(FALLBACK_MODELS)
                data = resp.json()
            else:
                import urllib.request
                req = urllib.request.Request(url, headers=self._headers())
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())

            models = data.get("models", [])
            return [m["name"] for m in models if "name" in m]
        except Exception:
            return list(FALLBACK_MODELS)
