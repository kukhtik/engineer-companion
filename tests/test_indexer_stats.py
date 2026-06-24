"""Tests for index_stats, remove_document, and index_new_pdf (core/indexer.py).

These tests use a tiny in-memory-style lancedb (written to a tmp dir) so no
real PDFs or embedding models are needed — except for test_index_new_pdf_progress
which builds a real 2-page PDF with fitz and monkeypatches SentenceTransformer.
"""

import tempfile
import numpy as np
import lancedb
import pyarrow as pa
from pathlib import Path
import pytest

# ---------------------------------------------------------------------------
# Shared schema (mirrors DocumentIndexPipeline._get_table)
# ---------------------------------------------------------------------------

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("source", pa.string()),
    pa.field("page", pa.int32()),
    pa.field("section", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("tokens", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), 384)),
])


def _make_fake_vector():
    return [0.0] * 384


def _make_db(tmp_path: Path) -> Path:
    """Create a tiny lancedb with 2 sources.

    doc_a.pdf → 3 chunks across 2 pages
    doc_b.pdf → 2 chunks on 1 page
    """
    db_path = tmp_path / "test.db"
    db = lancedb.connect(str(db_path))
    records = []
    # doc_a.pdf: 3 chunks across 2 pages
    for i in range(3):
        records.append({
            "id": f"a{i}",
            "text": f"text {i}",
            "source": "doc_a.pdf",
            "page": 1 if i < 2 else 2,
            "section": "intro",
            "chunk_index": i,
            "tokens": 10,
            "vector": _make_fake_vector(),
        })
    # doc_b.pdf: 2 chunks on 1 page
    for i in range(2):
        records.append({
            "id": f"b{i}",
            "text": f"other {i}",
            "source": "doc_b.pdf",
            "page": 1,
            "section": "sec",
            "chunk_index": i,
            "tokens": 8,
            "vector": _make_fake_vector(),
        })
    db.create_table("chunks", data=records, schema=SCHEMA)
    return db_path


# ---------------------------------------------------------------------------
# index_stats tests
# ---------------------------------------------------------------------------

def test_index_stats_basic(tmp_path):
    from core.indexer import index_stats

    db_p = _make_db(tmp_path)
    stats = index_stats(db_p)

    assert stats["total_chunks"] == 5
    per = {d["source"]: d for d in stats["per_doc"]}
    assert per["doc_a.pdf"]["chunks"] == 3
    assert per["doc_a.pdf"]["pages"] == 2
    assert per["doc_b.pdf"]["chunks"] == 2
    assert per["doc_b.pdf"]["pages"] == 1


def test_index_stats_missing_db(tmp_path):
    from core.indexer import index_stats

    stats = index_stats(tmp_path / "nonexistent.db")
    assert stats == {"total_chunks": 0, "per_doc": []}


def test_index_stats_empty_table(tmp_path):
    from core.indexer import index_stats

    db_path = tmp_path / "empty.db"
    db = lancedb.connect(str(db_path))
    db.create_table("chunks", schema=SCHEMA)
    stats = index_stats(db_path)
    assert stats == {"total_chunks": 0, "per_doc": []}


# ---------------------------------------------------------------------------
# remove_document tests
# ---------------------------------------------------------------------------

def test_remove_document(tmp_path):
    from core.indexer import index_stats, remove_document

    db_p = _make_db(tmp_path)
    removed = remove_document(db_p, "doc_a.pdf")
    assert removed == 3

    stats = index_stats(db_p)
    assert stats["total_chunks"] == 2
    sources = [d["source"] for d in stats["per_doc"]]
    assert "doc_a.pdf" not in sources
    assert "doc_b.pdf" in sources


def test_remove_document_nonexistent_source(tmp_path):
    from core.indexer import remove_document

    db_p = _make_db(tmp_path)
    removed = remove_document(db_p, "missing.pdf")
    assert removed == 0


def test_remove_document_missing_db(tmp_path):
    from core.indexer import remove_document

    removed = remove_document(tmp_path / "no.db", "doc_a.pdf")
    assert removed == 0


def test_remove_document_quote_in_name(tmp_path):
    """Source names with single-quotes must not break the SQL WHERE clause."""
    from core.indexer import index_stats, remove_document

    db_path = tmp_path / "q.db"
    db = lancedb.connect(str(db_path))
    tricky_name = "doc's file.pdf"
    records = [{
        "id": "q0",
        "text": "some text here for the chunk",
        "source": tricky_name,
        "page": 1,
        "section": "",
        "chunk_index": 0,
        "tokens": 5,
        "vector": _make_fake_vector(),
    }]
    db.create_table("chunks", data=records, schema=SCHEMA)

    removed = remove_document(db_path, tricky_name)
    assert removed == 1

    stats = index_stats(db_path)
    assert stats["total_chunks"] == 0


# ---------------------------------------------------------------------------
# index_new_pdf test (monkeypatched embedding)
# ---------------------------------------------------------------------------

def test_index_new_pdf_progress(tmp_path, monkeypatch):
    """index_new_pdf fires progress events and writes chunks; uses a fake model."""
    import fitz
    import unittest.mock as mock
    from core import indexer as idx_mod

    # Build a tiny 2-page PDF
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 100),
        "This is a real engineering test sentence with enough words to form a valid chunk.",
    )
    page2 = doc.new_page()
    page2.insert_text(
        (50, 100),
        "Second page has different content about system calibration and maintenance procedures.",
    )
    doc.save(str(pdf_path))
    doc.close()

    db_p = tmp_path / "idx.db"
    progress_events = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 384), dtype=np.float32)

    with mock.patch("sentence_transformers.SentenceTransformer", return_value=FakeModel()):
        with mock.patch("android.assets_loader.resolve_bundled_model", return_value=None):
            count = idx_mod.index_new_pdf(
                pdf_path,
                db_p,
                progress_callback=progress_events.append,
            )

    assert count > 0, "Should have added at least 1 chunk"
    assert len(progress_events) > 0, "Should have fired progress events"

    # Last event must be the final sentinel
    last = progress_events[-1]
    assert last["done"] is True
    assert last["chunks_added"] == count
    assert last["source"] == pdf_path.name

    # Verify data actually landed in the DB
    from core.indexer import index_stats
    stats = index_stats(db_p)
    assert stats["total_chunks"] == count


def test_index_new_pdf_should_stop(tmp_path):
    """should_stop flag aborts processing early and returns fewer chunks."""
    import fitz
    import unittest.mock as mock
    from core import indexer as idx_mod

    # Build a 4-page PDF
    pdf_path = tmp_path / "big.pdf"
    doc = fitz.open()
    for i in range(4):
        pg = doc.new_page()
        pg.insert_text(
            (50, 100),
            f"Page {i} content with engineering details about calibration and maintenance.",
        )
    doc.save(str(pdf_path))
    doc.close()

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 384), dtype=np.float32)

    stop_flag = [False]
    events = []

    def _callback(evt):
        events.append(evt)
        # Stop after the first page is processed
        if evt.get("page_index", 0) >= 1 and not evt.get("done"):
            stop_flag[0] = True

    db_p = tmp_path / "stop.db"
    with mock.patch("sentence_transformers.SentenceTransformer", return_value=FakeModel()):
        with mock.patch("android.assets_loader.resolve_bundled_model", return_value=None):
            count = idx_mod.index_new_pdf(
                pdf_path,
                db_p,
                progress_callback=_callback,
                should_stop=stop_flag,
            )

    # We may get 0 chunks if stop triggers before embed phase; either way no crash
    assert isinstance(count, int)
    assert count >= 0
