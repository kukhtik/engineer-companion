"""Tests for Ollama Cloud backend and usage tracker.

All HTTP calls are mocked — no network access.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from core.cloud import (
    FALLBACK_MODELS,
    CloudError,
    CloudRateLimitError,
    OllamaCloudClient,
)
from core.cloud_usage import CloudUsageTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ndjson_lines(pieces: list[str], prompt_eval_count: int = 10, eval_count: int = 20) -> list[bytes]:
    """Build a list of NDJSON bytes as the streaming API would return them."""
    lines: list[bytes] = []
    for p in pieces:
        obj = {"message": {"role": "assistant", "content": p}, "done": False}
        lines.append(json.dumps(obj).encode())
    # Final done=true line
    done_obj = {
        "done": True,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
    lines.append(json.dumps(done_obj).encode())
    return lines


class _FakeResponse:
    """Minimal fake requests.Response for streaming."""

    def __init__(self, status_code: int = 200, lines: list[bytes] | None = None, headers: dict | None = None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._lines = lines or []
        self.headers = headers or {}
        self.text = ""

    def iter_lines(self):
        return iter(self._lines)

    def json(self):
        # For list_models test
        return json.loads(self._lines[0]) if self._lines else {}

    def raise_for_status(self):
        if not self.ok:
            raise Exception(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------
# OllamaCloudClient — chat_stream
# ---------------------------------------------------------------------------

class TestChatStream:
    def _client(self) -> OllamaCloudClient:
        return OllamaCloudClient(api_key="test-key")

    def test_yields_pieces_and_calls_on_done(self):
        """chat_stream must yield each content piece and fire on_done with token counts."""
        lines = _make_ndjson_lines(["Hello", " world", "!"], prompt_eval_count=5, eval_count=3)
        fake_resp = _FakeResponse(200, lines)

        done_payloads = []

        with patch("requests.Session.post", return_value=fake_resp):
            client = self._client()
            client._get_session()  # ensure session is created
            pieces = list(
                client.chat_stream("gpt-oss:120b-cloud", [{"role": "user", "content": "hi"}],
                                   on_done=lambda d: done_payloads.append(d))
            )

        assert pieces == ["Hello", " world", "!"]
        assert len(done_payloads) == 1
        assert done_payloads[0]["prompt_tokens"] == 5
        assert done_payloads[0]["completion_tokens"] == 3

    def test_429_raises_rate_limit_error(self):
        """HTTP 429 must raise CloudRateLimitError."""
        fake_resp = _FakeResponse(429)
        fake_resp.ok = False

        with patch("requests.Session.post", return_value=fake_resp):
            client = self._client()
            client._get_session()
            with pytest.raises(CloudRateLimitError):
                list(client.chat_stream("gpt-oss:120b-cloud", [{"role": "user", "content": "hi"}]))

    def test_5xx_raises_cloud_error(self):
        """Non-429 HTTP errors must raise CloudError (not CloudRateLimitError)."""
        fake_resp = _FakeResponse(503)
        fake_resp.ok = False
        fake_resp.text = "Service Unavailable"

        with patch("requests.Session.post", return_value=fake_resp):
            client = self._client()
            client._get_session()
            with pytest.raises(CloudError) as exc_info:
                list(client.chat_stream("gpt-oss:120b-cloud", [{"role": "user", "content": "hi"}]))
            assert not isinstance(exc_info.value, CloudRateLimitError)

    def test_network_error_raises_cloud_error(self):
        """Network-level exceptions must be wrapped as CloudError."""
        import requests

        with patch("requests.Session.post", side_effect=requests.ConnectionError("no network")):
            client = self._client()
            client._get_session()
            with pytest.raises(CloudError):
                list(client.chat_stream("gpt-oss:120b-cloud", [{"role": "user", "content": "hi"}]))

    def test_empty_stream_no_error(self):
        """An empty stream (only done line) must not raise and yield nothing."""
        lines = _make_ndjson_lines([], prompt_eval_count=2, eval_count=0)
        fake_resp = _FakeResponse(200, lines)

        with patch("requests.Session.post", return_value=fake_resp):
            client = self._client()
            client._get_session()
            pieces = list(client.chat_stream("gpt-oss:120b-cloud", []))
        assert pieces == []

    def test_malformed_lines_skipped(self):
        """Non-JSON lines in the stream must be silently skipped."""
        lines = [b"not-json", b"", json.dumps({"message": {"content": "ok"}, "done": False}).encode(),
                 json.dumps({"done": True, "eval_count": 1, "prompt_eval_count": 1}).encode()]
        fake_resp = _FakeResponse(200, lines)

        with patch("requests.Session.post", return_value=fake_resp):
            client = self._client()
            client._get_session()
            pieces = list(client.chat_stream("gpt-oss:120b-cloud", []))
        assert pieces == ["ok"]

    def test_on_done_not_required(self):
        """chat_stream must work without on_done argument."""
        lines = _make_ndjson_lines(["A"])
        fake_resp = _FakeResponse(200, lines)

        with patch("requests.Session.post", return_value=fake_resp):
            client = self._client()
            client._get_session()
            pieces = list(client.chat_stream("gpt-oss:120b-cloud", [], on_done=None))
        assert pieces == ["A"]


# ---------------------------------------------------------------------------
# OllamaCloudClient — list_models
# ---------------------------------------------------------------------------

class TestListModels:
    def _client(self) -> OllamaCloudClient:
        return OllamaCloudClient(api_key="test-key")

    def test_parses_api_tags_response(self):
        """list_models must return names from /api/tags."""
        payload = {"models": [{"name": "gpt-oss:120b-cloud"}, {"name": "qwen3.5:cloud"}]}
        fake_resp = _FakeResponse(200, [json.dumps(payload).encode()])
        fake_resp.ok = True

        with patch("requests.Session.get", return_value=fake_resp):
            client = self._client()
            client._get_session()
            models = client.list_models()

        assert "gpt-oss:120b-cloud" in models
        assert "qwen3.5:cloud" in models

    def test_returns_fallback_on_http_error(self):
        """list_models must return FALLBACK_MODELS on HTTP error."""
        fake_resp = _FakeResponse(500)
        fake_resp.ok = False

        with patch("requests.Session.get", return_value=fake_resp):
            client = self._client()
            client._get_session()
            models = client.list_models()

        assert models == FALLBACK_MODELS

    def test_returns_fallback_on_network_error(self):
        """list_models must return FALLBACK_MODELS on network exception."""
        import requests

        with patch("requests.Session.get", side_effect=requests.ConnectionError()):
            client = self._client()
            client._get_session()
            models = client.list_models()

        assert models == FALLBACK_MODELS

    def test_returns_fallback_on_empty_models(self):
        """list_models can return empty list if API returns empty models array."""
        payload = {"models": []}
        fake_resp = _FakeResponse(200, [json.dumps(payload).encode()])
        fake_resp.ok = True

        with patch("requests.Session.get", return_value=fake_resp):
            client = self._client()
            client._get_session()
            models = client.list_models()

        assert models == []  # empty from API is valid


# ---------------------------------------------------------------------------
# CloudUsageTracker
# ---------------------------------------------------------------------------

class TestCloudUsageTracker:
    def _tracker(self, tmp_path) -> CloudUsageTracker:
        return CloudUsageTracker(data_dir=tmp_path)

    def test_empty_snapshot(self, tmp_path):
        t = self._tracker(tmp_path)
        snap = t.snapshot()
        assert snap["session"]["requests"] == 0
        assert snap["week"]["requests"] == 0
        assert snap["rate_limited_until"] is None

    def test_record_request_increments_session_and_week(self, tmp_path):
        t = self._tracker(tmp_path)
        t.record_request(prompt_tokens=100, completion_tokens=200)
        snap = t.snapshot()
        assert snap["session"]["requests"] == 1
        assert snap["session"]["prompt_tokens"] == 100
        assert snap["session"]["completion_tokens"] == 200
        assert snap["week"]["requests"] == 1

    def test_multiple_requests_accumulate(self, tmp_path):
        t = self._tracker(tmp_path)
        t.record_request(10, 20)
        t.record_request(30, 40)
        snap = t.snapshot()
        assert snap["session"]["requests"] == 2
        assert snap["session"]["prompt_tokens"] == 40
        assert snap["session"]["completion_tokens"] == 60

    def test_session_window_5h(self, tmp_path):
        """Events older than 5h should not appear in session count."""
        t = self._tracker(tmp_path)
        now = time.time()
        # Inject an old event (6 hours ago) directly
        t._state["events"].append({
            "ts": now - 6 * 3600,
            "prompt_tokens": 999,
            "completion_tokens": 999,
        })
        # Fresh event
        t.record_request(10, 20)
        snap = t.snapshot()
        assert snap["session"]["requests"] == 1  # only the fresh one
        assert snap["week"]["requests"] == 2     # both within 7 days

    def test_week_window_7d(self, tmp_path):
        """Events older than 7 days should not appear in week count."""
        t = self._tracker(tmp_path)
        now = time.time()
        t._state["events"].append({
            "ts": now - 8 * 24 * 3600,
            "prompt_tokens": 500,
            "completion_tokens": 500,
        })
        t.record_request(10, 20)
        snap = t.snapshot()
        assert snap["week"]["requests"] == 1  # old event pruned

    def test_prune_older_than_7d(self, tmp_path):
        """record_request should prune events older than 7 days."""
        t = self._tracker(tmp_path)
        now = time.time()
        t._state["events"] = [
            {"ts": now - 8 * 24 * 3600, "prompt_tokens": 1, "completion_tokens": 1},
            {"ts": now - 6 * 24 * 3600, "prompt_tokens": 2, "completion_tokens": 2},
        ]
        t.record_request(3, 3)
        # After pruning, only the 6d-old and fresh event remain
        assert len(t._state["events"]) == 2

    def test_persist_and_reload(self, tmp_path):
        """State must survive a tracker reload from disk."""
        t1 = self._tracker(tmp_path)
        t1.record_request(77, 88)
        # Create a second tracker from the same dir
        t2 = CloudUsageTracker(data_dir=tmp_path)
        snap = t2.snapshot()
        assert snap["session"]["requests"] == 1
        assert snap["session"]["prompt_tokens"] == 77

    def test_mark_rate_limited(self, tmp_path):
        t = self._tracker(tmp_path)
        t.mark_rate_limited(retry_after_seconds=300)
        snap = t.snapshot()
        assert snap["rate_limited_until"] is not None
        assert snap["rate_limited_until"] > time.time()
        assert t.is_rate_limited() is True

    def test_mark_rate_limited_default(self, tmp_path):
        """Default cooldown is 5 hours."""
        t = self._tracker(tmp_path)
        t.mark_rate_limited()
        snap = t.snapshot()
        expected_min = time.time() + 5 * 3600 - 5
        assert snap["rate_limited_until"] >= expected_min

    def test_rate_limited_until_persists(self, tmp_path):
        t1 = self._tracker(tmp_path)
        t1.mark_rate_limited(retry_after_seconds=600)
        t2 = CloudUsageTracker(data_dir=tmp_path)
        assert t2.is_rate_limited() is True

    def test_snapshot_window_labels(self, tmp_path):
        t = self._tracker(tmp_path)
        snap = t.snapshot()
        assert snap["session"]["window_hours"] == 5
        assert snap["week"]["window_days"] == 7


# ---------------------------------------------------------------------------
# RAGQueryPipeline.ask_streaming — cloud path
# ---------------------------------------------------------------------------

def _make_fake_retriever(hits=None):
    """Return a mock Retriever whose search_streaming fires embed/search and returns hits."""
    from core.query import SearchResult
    if hits is None:
        hits = [
            SearchResult(text="Relevant context text.", source="doc.pdf", page=1, section="Sec", score=0.9),
        ]

    class FakeRetriever:
        def search_streaming(self, query, on_event):
            on_event({"stage": "embed"})
            on_event({"stage": "search", "found": len(hits)})
            return hits

    return FakeRetriever()


def _make_cloud_client(pieces: list[str], prompt_eval: int = 5, eval_count: int = 10) -> OllamaCloudClient:
    """Return an OllamaCloudClient whose chat_stream yields pieces without hitting network."""
    client = MagicMock(spec=OllamaCloudClient)

    def _fake_chat_stream(model, messages, options=None, on_done=None):
        yield from pieces
        if on_done:
            on_done({"prompt_tokens": prompt_eval, "completion_tokens": eval_count})

    client.chat_stream.side_effect = _fake_chat_stream
    return client


class TestAskStreamingCloudPath:
    def test_event_sequence_embed_search_prompt_generatestart_tokens_done(self, tmp_path):
        """Cloud path must emit: embed→search→prompt→generate_start→token*→done."""
        from core.query import RAGQueryPipeline

        pipeline = RAGQueryPipeline(db_path=Path("/fake"), llm_model_path=None)
        pipeline.retriever = _make_fake_retriever()

        cloud_client = _make_cloud_client(["Ответ ", "на вопрос."])
        tracker = CloudUsageTracker(data_dir=tmp_path)

        events = []
        result = pipeline.ask_streaming(
            query="что такое MLC?",
            on_event=events.append,
            backend="cloud",
            cloud_model="gpt-oss:120b-cloud",
            cloud_client=cloud_client,
            usage_tracker=tracker,
        )

        stages = [e["stage"] for e in events]
        assert "embed" in stages
        assert "search" in stages
        assert "prompt" in stages
        assert "generate_start" in stages
        token_events = [e for e in events if e["stage"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["text"] == "Ответ "
        assert token_events[1]["text"] == "на вопрос."
        assert stages[-1] == "done"

        assert result["answer"] != ""
        assert result["sources"][0]["source"] == "doc.pdf"

    def test_cloud_records_usage(self, tmp_path):
        """After a successful cloud call, usage_tracker must have 1 recorded request."""
        from core.query import RAGQueryPipeline

        pipeline = RAGQueryPipeline(db_path=Path("/fake"), llm_model_path=None)
        pipeline.retriever = _make_fake_retriever()
        cloud_client = _make_cloud_client(["OK"], prompt_eval=7, eval_count=15)
        tracker = CloudUsageTracker(data_dir=tmp_path)

        pipeline.ask_streaming(
            "test", on_event=lambda e: None, backend="cloud",
            cloud_model="qwen3.5:cloud", cloud_client=cloud_client,
            usage_tracker=tracker,
        )

        snap = tracker.snapshot()
        assert snap["session"]["requests"] == 1
        assert snap["session"]["prompt_tokens"] == 7
        assert snap["session"]["completion_tokens"] == 15

    def test_429_emits_error_event_and_marks_rate_limited(self, tmp_path):
        """CloudRateLimitError must emit error event and mark tracker rate-limited."""
        from core.query import RAGQueryPipeline

        pipeline = RAGQueryPipeline(db_path=Path("/fake"), llm_model_path=None)
        pipeline.retriever = _make_fake_retriever()

        client_429 = MagicMock(spec=OllamaCloudClient)
        client_429.chat_stream.side_effect = CloudRateLimitError(retry_after=300)

        tracker = CloudUsageTracker(data_dir=tmp_path)
        events = []

        result = pipeline.ask_streaming(
            "test", on_event=events.append, backend="cloud",
            cloud_model="gpt-oss:120b-cloud", cloud_client=client_429,
            usage_tracker=tracker,
        )

        error_events = [e for e in events if e["stage"] == "error"]
        assert len(error_events) == 1
        assert "Лимит" in error_events[0]["message"] or "лимит" in error_events[0]["message"].lower()

        assert tracker.is_rate_limited() is True
        assert result["answer"] == ""

    def test_cloud_error_emits_error_event(self, tmp_path):
        """Generic CloudError must emit an error event and return empty answer."""
        from core.query import RAGQueryPipeline

        pipeline = RAGQueryPipeline(db_path=Path("/fake"), llm_model_path=None)
        pipeline.retriever = _make_fake_retriever()

        client_err = MagicMock(spec=OllamaCloudClient)
        client_err.chat_stream.side_effect = CloudError("timeout")

        events = []
        result = pipeline.ask_streaming(
            "test", on_event=events.append, backend="cloud",
            cloud_model="gpt-oss:120b-cloud", cloud_client=client_err,
            usage_tracker=None,
        )

        error_events = [e for e in events if e["stage"] == "error"]
        assert len(error_events) == 1
        assert result["answer"] == ""

    def test_local_backend_unchanged(self, tmp_path):
        """backend='local' with no LLM must still emit embed/search/prompt/done events."""
        from core.query import RAGQueryPipeline

        pipeline = RAGQueryPipeline(db_path=Path("/fake"), llm_model_path=None)
        pipeline.retriever = _make_fake_retriever()

        events = []
        result = pipeline.ask_streaming(
            "test query", on_event=events.append,
            backend="local",  # default path
        )

        stages = [e["stage"] for e in events]
        assert "embed" in stages
        assert "search" in stages
        assert "prompt" in stages
        assert "done" in stages
        # No LLM → answer is the "not loaded" message
        assert "LLM" in result["answer"] or result["answer"] == ""

    def test_cloud_no_client_emits_error(self, tmp_path):
        """backend='cloud' with no client must emit error and return."""
        from core.query import RAGQueryPipeline

        pipeline = RAGQueryPipeline(db_path=Path("/fake"), llm_model_path=None)
        pipeline.retriever = _make_fake_retriever()

        events = []
        result = pipeline.ask_streaming(
            "test", on_event=events.append, backend="cloud",
            cloud_model=None, cloud_client=None,
        )

        assert any(e["stage"] == "error" for e in events)
        assert result["answer"] != "" or result["prompt_tokens"] == 0

    def test_source_tags_stripped_from_cloud_answer(self, tmp_path):
        """[ИСТОЧНИК N] tags in the cloud answer must be stripped from result.answer."""
        from core.query import RAGQueryPipeline

        pipeline = RAGQueryPipeline(db_path=Path("/fake"), llm_model_path=None)
        pipeline.retriever = _make_fake_retriever()

        cloud_client = _make_cloud_client(["[ИСТОЧНИК 1] Ответ здесь."])
        result = pipeline.ask_streaming(
            "test", on_event=lambda e: None, backend="cloud",
            cloud_model="gpt-oss:120b-cloud", cloud_client=cloud_client,
        )

        assert "[ИСТОЧНИК" not in result["answer"]
        assert "Ответ здесь." in result["answer"]
