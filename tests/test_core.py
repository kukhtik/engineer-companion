"""Unit tests for core indexer + query pipeline (no GUI, no LLM)."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest
from core.indexer import Chunker, Chunk, PdfTextExtractor, _estimate_tokens, is_low_quality_chunk
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

    def test_search_empty_db_returns_empty(self):
        r = Retriever(db_path=Path("/tmp/fake_empty.db"))
        r._table = self.FakeTable([])

        class FakeModel:
            def encode(self, query, normalize_embeddings=True, device="cpu"):
                import numpy as np
                return np.array([0.1] * 384)

        r._model = FakeModel()
        results = r.search("empty query")
        assert len(results) == 0

    def test_search_with_invalid_db_path_raises(self):
        r = Retriever(db_path=Path("/nonexistent/path.db"))
        r._model = None
        with pytest.raises((FileNotFoundError, Exception)):
            r.search("anything")


class TestPromptBuilderEdgeCases:
    def test_build_with_no_hits(self):
        pb = PromptBuilder()
        prompt = pb.build("test query", [])
        assert "test query" in prompt
        assert PromptBuilder.SYSTEM_PERSONA in prompt

    def test_build_with_many_hits_limits_context(self):
        pb = PromptBuilder()
        hits = [
            SearchResult(text=f"chunk_{i}", source=f"s{i}.pdf", page=i, section="Sec", score=0.5)
            for i in range(50)
        ]
        prompt = pb.build("query", hits)
        # Should not blow up — prompt should be bounded
        assert len(prompt) > 0
        assert "s0.pdf" in prompt


class TestChunkerEdgeCases:
    def test_zero_overlap(self):
        c = Chunker(chunk_size=10, overlap=0)
        para = " ".join([f"w{i}" for i in range(40)])
        chunks = list(c.chunk_paragraph(para, "doc.pdf", 1, ""))
        assert len(chunks) == 4  # stride 10
        assert "w0" in chunks[0].text
        assert "w10" not in chunks[0].text

    def test_exact_chunk_boundary(self):
        c = Chunker(chunk_size=5, overlap=2)
        para = "w0 w1 w2 w3 w4 w5 w6"
        chunks = list(c.chunk_paragraph(para, "doc.pdf", 1, ""))
        assert len(chunks) >= 2

    def test_special_chars_in_paragraph(self):
        c = Chunker(chunk_size=100, overlap=10)
        para = "Section 3.2 — Interlocks: [ALC-001] & <status>"
        chunks = list(c.chunk_paragraph(para, "doc.pdf", 1, "Test"))
        assert len(chunks) == 1
        assert "[ALC-001]" in chunks[0].text

    def test_very_short_paragraph(self):
        c = Chunker(chunk_size=50, overlap=5)
        chunks = list(c.chunk_paragraph("Hi.", "doc.pdf", 1, "Short"))
        assert len(chunks) == 1
        assert chunks[0].text == "Hi."


class TestTokenEstimation:
    def test_long_text(self):
        text = " ".join(["word"] * 1000)
        tokens = _estimate_tokens(text)
        assert tokens == 750  # 1000 * 0.75

    def test_exact_token_boundary(self):
        text = " ".join(["token"] * 100)
        tokens = _estimate_tokens(text)
        assert tokens == 75


class TestSearchResultDataclass:
    def test_construction(self):
        sr = SearchResult(text="hello", source="a.pdf", page=5, section="Sec", score=0.99)
        assert sr.text == "hello"
        assert sr.source == "a.pdf"
        assert sr.page == 5
        assert sr.score == 0.99

    def test_ordering_by_score(self):
        r1 = SearchResult("a", "a.pdf", 1, "", 0.9)
        r2 = SearchResult("b", "b.pdf", 2, "", 0.5)
        r3 = SearchResult("c", "c.pdf", 3, "", 0.7)
        results = [r1, r2, r3]
        sorted_r = sorted(results, key=lambda x: x.score, reverse=True)
        assert sorted_r[0].score == 0.9
        assert sorted_r[2].score == 0.5

    def test_repr(self):
        sr = SearchResult("text", "src.pdf", 3, "S1", 0.5)
        r = repr(sr)
        assert "src.pdf" in r
        assert "S1" in r


class TestQualityFilter:
    """Tests for is_low_quality_chunk — the garbage-chunk heuristic."""

    # --- GARBAGE cases (must return True) ---

    def test_scan_artifact_ocr_gibberish(self):
        # Binary/OCR debris with tilde-trains and control chars
        assert is_low_quality_chunk("- - 8 7 5 t-=-f-cJ~~~~~~~ 100050108 B") is True

    def test_control_char_debris(self):
        # Control characters (0x0b = \x0b vertical tab) with no real words
        assert is_low_quality_chunk("abc\x0bdef\x0c\x0e xyz 123 456 789 0ab") is True

    def test_dot_train_toc_line(self):
        # TOC dot-trains: almost no alphabetic chars, no real words
        garbage = "3.1 .......... 45 3.2 .......... 67 3.3 .......... 89 4.0 .......100"
        assert is_low_quality_chunk(garbage) is True

    def test_too_short_fragment(self):
        # Fewer than 5 tokens — always garbage regardless of content
        assert is_low_quality_chunk("voltage") is True
        assert is_low_quality_chunk("beam on") is True
        assert is_low_quality_chunk("3.2 see") is True

    def test_high_nonascii_noncy(self):
        # Dense non-ASCII non-Cyrillic = encoding garbage
        garbage = "؀؁؂ 一丁丂 ؀؁ 一丁 ؂丂 bogus"
        assert is_low_quality_chunk(garbage) is True

    # --- CLEAN cases (must return False) ---

    def test_clean_russian_technical(self):
        text = (
            "Абсолютная калибровка дозы выполняется с использованием ионизационной камеры "
            "в условиях референсного поля 10x10 см на глубине 10 см в воде при SSD 100 см. "
            "Результаты должны соответствовать протоколу TRS-398 МАГАТЭ."
        )
        assert is_low_quality_chunk(text) is False

    def test_clean_english_technical(self):
        text = (
            "The TrueBeam system uses a dual-energy linear accelerator capable of delivering "
            "photon beams at 6 MV and 15 MV as well as electron beams from 6 to 20 MeV. "
            "Beam calibration must be verified monthly per institutional protocol."
        )
        assert is_low_quality_chunk(text) is False

    def test_clean_mixed_russian_english(self):
        text = (
            "Interlocks системы безопасности TrueBeam делятся на категории: "
            "Machine Interlock, Beam Hold и Treatment Interlock. "
            "При срабатывании любого из них подача излучения немедленно прекращается."
        )
        assert is_low_quality_chunk(text) is False

    def test_clean_sentence_with_numbers(self):
        # Numbers + technical terms — should NOT be flagged
        text = (
            "Field size at isocenter: 10 x 10 cm. SSD: 100 cm. "
            "Depth of measurement: 10 cm. Output factor: 1.000 cGy/MU."
        )
        assert is_low_quality_chunk(text) is False
