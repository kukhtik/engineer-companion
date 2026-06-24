"""Tests for LibraryDialog — headless instantiation, table population, worker mock."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import lancedb
import numpy as np
import pyarrow as pa
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

# Schema shared with indexer
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


@pytest.fixture(scope="session")
def app():
    a = QApplication.instance() or QApplication([])
    yield a
    a.processEvents()


def _seed_db(tmp_path: Path) -> Path:
    """Create a tiny 2-doc lancedb for test use."""
    db_p = tmp_path / "lib.db"
    db = lancedb.connect(str(db_p))
    records = []
    for i in range(3):
        records.append({
            "id": f"a{i}", "text": f"text {i}", "source": "alpha.pdf",
            "page": 1, "section": "intro", "chunk_index": i,
            "tokens": 10, "vector": [0.0] * 384,
        })
    records.append({
        "id": "b0", "text": "beta text", "source": "beta.pdf",
        "page": 1, "section": "sec", "chunk_index": 0,
        "tokens": 8, "vector": [0.0] * 384,
    })
    db.create_table("chunks", data=records, schema=SCHEMA)
    return db_p


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestLibraryDialogInstantiation:
    def test_dialog_opens_headless(self, app, tmp_path):
        from windows.library_dialog import LibraryDialog
        db_p = _seed_db(tmp_path)
        dlg = LibraryDialog(None, db_path=db_p)
        dlg.show()
        app.processEvents()
        assert isinstance(dlg, QDialog)
        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_dialog_has_required_buttons(self, app, tmp_path):
        from windows.library_dialog import LibraryDialog
        db_p = _seed_db(tmp_path)
        dlg = LibraryDialog(None, db_path=db_p)
        app.processEvents()
        assert hasattr(dlg, "add_btn")
        assert hasattr(dlg, "remove_btn")
        assert hasattr(dlg, "reindex_btn")
        assert hasattr(dlg, "stop_btn")
        assert hasattr(dlg, "refresh_btn")
        assert dlg.add_btn.isEnabled()
        assert not dlg.stop_btn.isEnabled()
        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_table_has_4_columns(self, app, tmp_path):
        from windows.library_dialog import LibraryDialog
        db_p = _seed_db(tmp_path)
        dlg = LibraryDialog(None, db_path=db_p)
        app.processEvents()
        assert dlg.docs_table.columnCount() == 4
        dlg.close()
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# Stats display
# ---------------------------------------------------------------------------

class TestLibraryDialogStats:
    def test_header_shows_doc_and_chunk_counts(self, app, tmp_path):
        from windows.library_dialog import LibraryDialog
        db_p = _seed_db(tmp_path)
        dlg = LibraryDialog(None, db_path=db_p)
        app.processEvents()
        text = dlg.header_label.text()
        assert "2" in text   # 2 documents
        assert "4" in text   # 4 total chunks
        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_table_lists_indexed_docs(self, app, tmp_path):
        from windows.library_dialog import LibraryDialog
        db_p = _seed_db(tmp_path)
        dlg = LibraryDialog(None, db_path=db_p)
        app.processEvents()
        row_count = dlg.docs_table.rowCount()
        assert row_count == 2
        sources = {
            dlg.docs_table.item(r, 0).text()
            for r in range(row_count)
        }
        assert "alpha.pdf" in sources
        assert "beta.pdf" in sources
        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_empty_db_shows_zeros(self, app, tmp_path):
        from windows.library_dialog import LibraryDialog
        dlg = LibraryDialog(None, db_path=tmp_path / "no.db")
        app.processEvents()
        assert dlg.docs_table.rowCount() == 0
        assert "0" in dlg.header_label.text()
        dlg.close()
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# IndexWorker mock test
# ---------------------------------------------------------------------------

class TestIndexWorkerMocked:
    def test_worker_emits_progress_and_finished(self, app, tmp_path):
        """IndexWorker calls index_new_pdf and forwards signals — monkeypatched."""
        from windows.library_dialog import IndexWorker

        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")  # minimal placeholder

        progress_events = []
        finished_events = []

        stop_flag = [False]

        # Mock index_new_pdf to fire a couple of progress events then return 42
        def _fake_index_new_pdf(pdf_path, db_path, progress_callback=None, should_stop=None):
            if progress_callback:
                progress_callback({"source": "fake.pdf", "page_index": 1,
                                   "total_pages": 2, "chunks_added": 10, "done": False})
                progress_callback({"source": "fake.pdf", "page_index": 2,
                                   "total_pages": 2, "chunks_added": 42, "done": True})
            return 42

        worker = IndexWorker(fake_pdf, tmp_path / "test.db", stop_flag)
        worker.progress.connect(lambda e: progress_events.append(e))
        worker.finished.connect(lambda r: finished_events.append(r))

        with patch("core.indexer.index_new_pdf", _fake_index_new_pdf):
            worker.start()
            deadline = time.time() + 5.0
            while not finished_events and time.time() < deadline:
                app.processEvents()
                time.sleep(0.01)
            worker.wait(5000)

        assert len(progress_events) == 2
        assert len(finished_events) == 1
        assert finished_events[0]["chunks_added"] == 42
        assert finished_events[0]["source"] == "fake.pdf"


# ---------------------------------------------------------------------------
# Main window integration
# ---------------------------------------------------------------------------

class TestMainWindowLibraryIntegration:
    def test_library_menu_action_exists(self, app):
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        app.processEvents()

        menubar = w.menuBar()
        lib_action = None
        for top_action in menubar.actions():
            menu = top_action.menu()
            if menu is None:
                continue
            for action in menu.actions():
                if "Библиотека" in action.text():
                    lib_action = action
                    break
            if lib_action is not None:
                break

        assert lib_action is not None, '"Библиотека" action not found in menus'
        assert lib_action.isEnabled()

        w.close()
        w.deleteLater()
        app.processEvents()

    def test_index_status_label_in_status_bar(self, app):
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        app.processEvents()
        assert hasattr(w, "index_status_label")
        text = w.index_status_label.text()
        # Should either show real counts or "—"
        assert "Документов:" in text
        assert "Чанков:" in text
        w.close()
        w.deleteLater()
        app.processEvents()

    def test_refresh_index_status_does_not_crash(self, app):
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        app.processEvents()
        # Should not raise even if DB is missing
        w._refresh_index_status()
        app.processEvents()
        w.close()
        w.deleteLater()
        app.processEvents()
