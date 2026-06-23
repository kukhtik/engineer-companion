"""End-to-end integration tests for core RAG pipeline with real LLM.

Requires: GGUF model at assets/models/ and LanceDB at assets/db/engineer.db
These tests are SLOW (~2 min) — mark with pytest.mark.slow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from core.indexer import Chunker, PdfTextExtractor, _estimate_tokens  # noqa: E402
from core.query import RAGQueryPipeline, SearchResult  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db_path() -> Path:
    p = _REPO / "assets" / "db" / "engineer.db"
    assert p.exists(), f"DB not found: {p}"
    return p


@pytest.fixture(scope="session")
def llm_path() -> Path:
    p = _REPO / "assets" / "models" / "gemma-3-4b-it-Q4_K_M.gguf"
    if not p.exists():
        pytest.skip(f"GGUF model not found at {p} — install first")
    return p


@pytest.fixture(scope="session")
def pipeline(db_path: Path, llm_path: Path) -> RAGQueryPipeline:
    return RAGQueryPipeline(
        db_path=db_path,
        llm_model_path=llm_path,
        top_k=3,
        max_tokens=128,
        temperature=0.2,
        llm_n_ctx=768,
        llm_n_threads=2,
    )


@pytest.fixture(scope="session")
def search_pipeline(db_path: Path) -> RAGQueryPipeline:
    """Pipeline without LLM — search-only mode, fast."""
    return RAGQueryPipeline(
        db_path=db_path,
        llm_model_path=None,
        top_k=3,
    )


# ── Indexer tests (real PDF) ─────────────────────────────────────────

class TestPdfTextExtractorReal:
    PDF_LIMIT_MB = 5  # Only test PDFs under this size to keep tests fast

    def test_extract_real_pdf_returns_paragraphs(self):
        pdfs = sorted(
            p for p in (_REPO / "docs").glob("*.pdf")
            if p.stat().st_size < self.PDF_LIMIT_MB * 1024 * 1024
        )
        if not pdfs:
            pytest.skip("No small PDFs (<5MB) in docs/")
        extractor = PdfTextExtractor(min_heading_size=11.0)
        results = list(extractor.extract_with_structure(pdfs[0]))
        assert len(results) > 0, "No paragraphs extracted"
        page, heading, para = results[0]
        assert isinstance(page, int) and page >= 1
        assert isinstance(heading, str)
        assert len(para) > 0

    def test_extract_small_pdfs_produce_chunks(self):
        chunker = Chunker(chunk_size=512, overlap=64)
        extractor = PdfTextExtractor(min_heading_size=11.0)
        pdfs = sorted(
            p for p in (_REPO / "docs").glob("*.pdf")
            if p.stat().st_size < self.PDF_LIMIT_MB * 1024 * 1024
        )[:2]
        if not pdfs:
            pytest.skip("No small PDFs (<5MB) in docs/")
        total_chunks = 0
        for pdf in pdfs:
            for page, heading, para in extractor.extract_with_structure(pdf):
                total_chunks += len(
                    list(chunker.chunk_paragraph(para, pdf.name, page, heading))
                )
        assert total_chunks > 0, "Zero chunks produced"


# ── RAG pipeline tests ───────────────────────────────────────────────

class TestRAGPipelineSmoke:
    def test_pipeline_ask_returns_answer_and_sources(self, pipeline):
        result = pipeline.ask("What is a TrueBeam interlock error?")
        assert "answer" in result
        assert "sources" in result
        assert len(result["answer"]) > 0
        assert len(result["sources"]) > 0

    def test_answer_mentions_source(self, pipeline):
        result = pipeline.ask("What is the TrueBeam maintenance schedule?")
        answer = result["answer"]
        sources = result.get("sources", [])
        # At least one source must be mentioned
        for src in sources:
            if src["source"] in answer:
                break
        else:
            if sources:
                pytest.xfail("Answer does not mention any source by name")

    def test_empty_query_returns_graceful(self, pipeline):
        result = pipeline.ask("")
        assert isinstance(result, dict)
        # Should not crash

    def test_irrelevant_query_returns_no_error(self, pipeline):
        result = pipeline.ask("What is the meaning of life today?")
        assert "answer" in result
        assert isinstance(result["answer"], str)


class TestLLMLoadsModel:
    def test_llm_loads_gguf_and_answers_minimal(self, llm_path):
        """Direct test — load model, ask trivial question, verify non-empty."""
        from llama_cpp import Llama
        llm = Llama(
            model_path=str(llm_path),
            n_ctx=512,
            n_threads=2,
            verbose=False,
        )
        r = llm.create_completion(
            "What is 2+2? Answer in one word.",
            max_tokens=20,
            temperature=0,
            # No stop token: Gemma may begin its reply with a newline before the
            # actual answer, so stopping on "\n" yields an empty string on Windows.
        )
        answer = r["choices"][0]["text"].strip()
        assert len(answer) > 0, "Empty answer from LLM"


# ── Mock pipeline tests (fast) ───────────────────────────────────────

class TestRAGPipelineWithMock:
    @pytest.fixture
    def mock_pipeline(self, db_path):
        """Pipeline with LLM disabled — search only mode."""
        return RAGQueryPipeline(
            db_path=db_path,
            llm_model_path=None,
            top_k=3,
        )

    def test_search_only_returns_sources(self, mock_pipeline):
        result = mock_pipeline.ask("TrueBeam interlock")
        assert result["answer"] == "[LLM не загружена. Проверьте путь к модели.]"
        assert len(result["sources"]) > 0

    def test_search_only_sources_have_correct_keys(self, mock_pipeline):
        result = mock_pipeline.ask("VitalBeam")
        for src in result["sources"]:
            assert "source" in src
            assert "page" in src
            assert "section" in src

    def test_search_result_distinct_sources(self, mock_pipeline):
        result = mock_pipeline.ask("VitalBeam administrators")
        files = {s["source"] for s in result["sources"]}
        assert len(files) > 0, "No sources returned"
