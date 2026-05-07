"""Unit tests for core indexer + query pipeline (no GUI, no LLM)."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest
from core.indexer import Chunker, Chunk, PdfTextExtractor, _estimate_tokens
from core.query import PromptBuilder, SearchResult, Retriever


class TestEstimateTokens:
    def test_typical_technical_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        assert _estimate_tokens(text) == 6  # 9 words * 0.75

    def test_empty_string(self):
        assert _estimate_tokens("") == 0


class TestChunker:
    def test_single_short_paragraph(self):
        c = Chunker(chunk_size=512, overlap=64)
        para = " ".join(["word"] * 100)
        chunks = list(c.chunk_paragraph(para, "doc.pdf", 1, "Section A"))
        assert len(chunks) == 1
        assert chunks[0].source == "doc.pdf"
        assert chunks[0].page == 1
        assert chunks[0].section == "Section A"
        assert chunks[0].chunk_index == 0

    def test_long_paragraph_splits(self):
        c = Chunker(chunk_size=10, overlap=2)
        para = " ".join([f"w{i}" for i in range(30)])
        chunks = list(c.chunk_paragraph(para, "doc.pdf", 2, ""))
        assert len(chunks) == 4  # stride 8
        assert chunks[0].text.startswith("w0")
        assert chunks[1].text.startswith("w8")

    def test_chunk_index_increments(self):
        c = Chunker(chunk_size=5, overlap=1)
        para = " ".join([f"x{i}" for i in range(12)])
        chunks = list(c.chunk_paragraph(para, "doc.pdf", 1, ""))
        indices = [ch.chunk_index for ch in chunks]
        assert indices == list(range(len(chunks)))


class TestPromptBuilder:
    def test_build_includes_sources_and_query(self):
        pb = PromptBuilder()
        hits = [
            SearchResult(text="foo bar", source="a.pdf", page=1, section="S1", score=0.1),
            SearchResult(text="baz qux", source="b.pdf", page=5, section="S2", score=0.2),
        ]
        prompt = pb.build("what is it?", hits)
        assert "a.pdf" in prompt
        assert "стр.1" in prompt
        assert "what is it?" in prompt
        assert PromptBuilder.SYSTEM_PERSONA in prompt


class TestRetrieverSearchMock:
    """Mock LanceDB table to test Retriever.search logic without real DB."""

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

    def test_search_returns_results(self):
        r = Retriever(db_path=Path("/tmp/fake.db"))
        fake_records = [
            {"text": "result one", "source": "r1.pdf", "page": 10, "section": "Sec", "_distance": 0.123},
        ]
        r._table = self.FakeTable(fake_records)

        class FakeModel:
            def encode(self, query, normalize_embeddings=True, device="cpu"):
                import numpy as np
                return np.array([0.1] * 384)

        r._model = FakeModel()

        results = r.search("test query")
        assert len(results) == 1
        assert results[0].text == "result one"
        assert results[0].source == "r1.pdf"
        assert results[0].page == 10
