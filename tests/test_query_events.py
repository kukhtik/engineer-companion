"""Tests for RAGQueryPipeline.ask_streaming event contract.

All heavy models are mocked — no real embedder, reranker, or LLM loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import numpy as np
import pytest
from core.query import RAGQueryPipeline, SearchResult, Retriever


# ---------------------------------------------------------------------------
# Shared fixtures / fakes
# ---------------------------------------------------------------------------

class FakeEmbedder:
    def encode(self, query, normalize_embeddings=True, device="cpu"):
        return np.array([0.1] * 64)


class FakeTable:
    def __init__(self, records):
        self._records = records

    def search(self, emb):
        return self

    def metric(self, m):
        return self

    def limit(self, n):
        return self

    def to_list(self):
        return self._records


def _make_records(n: int) -> list[dict]:
    return [
        {"text": f"chunk {i}", "source": f"doc{i}.pdf", "page": i + 1,
         "section": f"Sec{i}", "_distance": 0.1 * i}
        for i in range(n)
    ]


def _make_pipeline_no_rerank(records=None, llm=None) -> RAGQueryPipeline:
    """Build a pipeline with mocked retriever internals, reranking disabled."""
    p = RAGQueryPipeline(
        db_path=Path("/tmp/fake.db"),
        rerank_model="",  # disable reranking
        llm_model_path=None,
    )
    p.retriever._model = FakeEmbedder()
    p.retriever._table = FakeTable(records or _make_records(3))
    if llm is not None:
        p._llm = llm
    return p


def _make_pipeline_with_rerank(records=None, top_k=8, rerank_top_k=2, llm=None) -> RAGQueryPipeline:
    """Build a pipeline with a fake reranker that simply reverses order."""
    p = RAGQueryPipeline(
        db_path=Path("/tmp/fake.db"),
        rerank_model="fake-reranker",
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        llm_model_path=None,
    )
    p.retriever._model = FakeEmbedder()
    p.retriever._table = FakeTable(_make_records(top_k) if records is None else records)

    class FakeReranker:
        def predict(self, pairs):
            # Return descending scores: [N-1, N-2, ..., 0]
            return list(range(len(pairs) - 1, -1, -1))

    p.retriever._reranker = FakeReranker()
    if llm is not None:
        p._llm = llm
    return p


class FakeLLMNoStream:
    """LLM that does NOT support streaming (returns whole response at once)."""
    def create_completion(self, prompt, max_tokens, temperature, stop, stream=False):
        if stream:
            raise TypeError("streaming not supported")
        return {
            "choices": [{"text": " The answer is 42."}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class FakeLLMStream:
    """LLM that supports streaming."""
    TOKENS = [" The", " answer", " is", " 42."]

    def create_completion(self, prompt, max_tokens, temperature, stop, stream=False):
        if stream:
            for tok in self.TOKENS:
                yield {"choices": [{"text": tok}]}
        else:
            return {
                "choices": [{"text": "".join(self.TOKENS)}],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(self.TOKENS)},
            }


# ---------------------------------------------------------------------------
# Tests: event order and content
# ---------------------------------------------------------------------------

class TestAskStreamingNoRerank:
    def test_event_stages_in_order(self):
        p = _make_pipeline_no_rerank(records=_make_records(4), llm=FakeLLMStream())
        events = []
        result = p.ask_streaming("test query", events.append)
        stages = [e["stage"] for e in events]
        # Must appear in this order; "rerank" must NOT appear
        assert "embed" in stages
        assert "search" in stages
        assert "rerank" not in stages
        assert "prompt" in stages
        assert "generate_start" in stages
        assert "token" in stages
        assert stages[-1] == "done"
        # Strict ordering of structural stages
        embed_i = stages.index("embed")
        search_i = stages.index("search")
        prompt_i = stages.index("prompt")
        gen_i = stages.index("generate_start")
        done_i = stages.index("done")
        assert embed_i < search_i < prompt_i < gen_i < done_i

    def test_search_found_count(self):
        records = _make_records(5)
        p = _make_pipeline_no_rerank(records=records, llm=FakeLLMStream())
        events = []
        p.ask_streaming("q", events.append)
        search_evt = next(e for e in events if e["stage"] == "search")
        assert search_evt["found"] == 5

    def test_prompt_sources_count(self):
        records = _make_records(3)
        p = _make_pipeline_no_rerank(records=records, llm=FakeLLMStream())
        events = []
        p.ask_streaming("q", events.append)
        prompt_evt = next(e for e in events if e["stage"] == "prompt")
        assert prompt_evt["sources"] == 3

    def test_tokens_concatenate_to_final_answer(self):
        p = _make_pipeline_no_rerank(llm=FakeLLMStream())
        events = []
        result = p.ask_streaming("q", events.append)
        token_events = [e for e in events if e["stage"] == "token"]
        concatenated = "".join(e["text"] for e in token_events).strip()
        assert concatenated == result["answer"]

    def test_done_event_has_all_fields(self):
        p = _make_pipeline_no_rerank(llm=FakeLLMStream())
        events = []
        result = p.ask_streaming("q", events.append)
        done_evt = next(e for e in events if e["stage"] == "done")
        assert "answer" in done_evt
        assert "sources" in done_evt
        assert "prompt_tokens" in done_evt
        assert "completion_tokens" in done_evt
        # done event answer matches return value
        assert done_evt["answer"] == result["answer"]

    def test_no_llm_emits_done_immediately(self):
        p = _make_pipeline_no_rerank(llm=None)
        # No LLM path set
        events = []
        result = p.ask_streaming("q", events.append)
        stages = [e["stage"] for e in events]
        # Should not have generate_start or token
        assert "generate_start" not in stages
        assert "token" not in stages
        # Should have done
        assert "done" in stages
        assert "[LLM" in result["answer"]


class TestAskStreamingWithRerank:
    def test_rerank_event_fires(self):
        records = _make_records(8)
        p = _make_pipeline_with_rerank(records=records, top_k=8, rerank_top_k=2, llm=FakeLLMStream())
        events = []
        p.ask_streaming("q", events.append)
        rerank_evt = next((e for e in events if e["stage"] == "rerank"), None)
        assert rerank_evt is not None
        assert rerank_evt["from"] == 8
        assert rerank_evt["to"] == 2

    def test_rerank_event_between_search_and_prompt(self):
        records = _make_records(4)
        p = _make_pipeline_with_rerank(records=records, top_k=4, rerank_top_k=2, llm=FakeLLMStream())
        events = []
        p.ask_streaming("q", events.append)
        stages = [e["stage"] for e in events]
        search_i = stages.index("search")
        rerank_i = stages.index("rerank")
        prompt_i = stages.index("prompt")
        assert search_i < rerank_i < prompt_i

    def test_prompt_sources_equals_rerank_top_k(self):
        records = _make_records(6)
        p = _make_pipeline_with_rerank(records=records, top_k=6, rerank_top_k=3, llm=FakeLLMStream())
        events = []
        p.ask_streaming("q", events.append)
        prompt_evt = next(e for e in events if e["stage"] == "prompt")
        assert prompt_evt["sources"] == 3

    def test_no_rerank_event_when_rerank_disabled(self):
        records = _make_records(3)
        p = _make_pipeline_no_rerank(records=records, llm=FakeLLMStream())
        events = []
        p.ask_streaming("q", events.append)
        assert not any(e["stage"] == "rerank" for e in events)

    def test_no_rerank_event_when_empty_results(self):
        """Rerank must NOT fire when search returns 0 hits."""
        p = _make_pipeline_with_rerank(records=[], top_k=8, rerank_top_k=5, llm=FakeLLMStream())
        events = []
        p.ask_streaming("q", events.append)
        assert not any(e["stage"] == "rerank" for e in events)


class TestAskOriginalUnchanged:
    """The original ask() must still work correctly."""

    def test_ask_returns_answer_and_sources(self):
        records = _make_records(3)
        p = _make_pipeline_no_rerank(records=records, llm=FakeLLMNoStream())
        result = p.ask("test query")
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) == 3

    def test_ask_no_llm_returns_error_message(self):
        p = _make_pipeline_no_rerank(records=_make_records(2))
        result = p.ask("test")
        assert "[LLM" in result["answer"]
