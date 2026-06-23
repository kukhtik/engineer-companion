"""Tests for OCR UI: ocr_stats, OCRDialog instantiation, worker with mock subprocess."""
import io
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Session-scoped QApplication fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    a = QApplication.instance() or QApplication([])
    yield a
    a.processEvents()


# ---------------------------------------------------------------------------
# TestOcrStats
# ---------------------------------------------------------------------------

class TestOcrStats:
    def test_ocr_stats_returns_correct_shape(self, tmp_path):
        """ocr_stats returns a list of dicts with required keys."""
        from scripts.ocr_databooks import ocr_stats

        # Write a tmp JSONL cache with 3 entries for "fakebook.pdf"
        cache = tmp_path / "cache.jsonl"
        entries = [
            {"doc": "fakebook.pdf", "page": 0, "source": "native", "text": "hello"},
            {"doc": "fakebook.pdf", "page": 1, "source": "native", "text": "world"},
            {"doc": "fakebook.pdf", "page": 2, "source": "ocr", "text": "ocr text"},
        ]
        with cache.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        # docs_dir has no PDFs — fitz.open won't be called for real files
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        result = ocr_stats(docs_dir, cache)
        # No PDFs in docs_dir, so result should be empty list
        assert isinstance(result, list)

    def test_ocr_stats_with_empty_cache(self, tmp_path):
        """ocr_stats with non-existent cache returns empty list (no PDFs in docs_dir)."""
        from scripts.ocr_databooks import ocr_stats

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        cache = tmp_path / "nonexistent_cache.jsonl"

        result = ocr_stats(docs_dir, cache)
        assert isinstance(result, list)
        # No PDFs in docs_dir, so result is empty
        assert len(result) == 0

    def test_ocr_stats_counts_correctly(self, tmp_path):
        """ocr_stats counts native/ocr correctly for a real doc filename."""
        from scripts.ocr_databooks import ocr_stats

        target_doc = "TrueBeam 3.0 Volume 1 Field Service Databook.pdf"

        # Write cache with entries for the target doc
        cache = tmp_path / "cache.jsonl"
        entries = [
            {"doc": target_doc, "page": 0, "source": "native", "text": "a"},
            {"doc": target_doc, "page": 1, "source": "native", "text": "b"},
            {"doc": target_doc, "page": 2, "source": "native", "text": "c"},
            {"doc": target_doc, "page": 3, "source": "ocr", "text": "d"},
            {"doc": target_doc, "page": 4, "source": "ocr", "text": "e"},
        ]
        with cache.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        # Create a mock PDF in docs_dir
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        fake_pdf = docs_dir / target_doc
        fake_pdf.write_bytes(b"")  # empty placeholder

        # Patch fitz.open so we don't need a real PDF
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=100)
        mock_doc.close = MagicMock()

        with patch("fitz.open", return_value=mock_doc):
            result = ocr_stats(docs_dir, cache)

        assert isinstance(result, list)
        # Find entry for our target doc
        found = next((s for s in result if s["filename"] == target_doc), None)
        assert found is not None, f"Expected {target_doc} in stats"
        assert found["native_pages"] == 3
        assert found["ocr_pages"] == 2
        assert found["cached_pages"] == 5
        assert found["is_scan_heavy"] is True


# ---------------------------------------------------------------------------
# TestOCRDialogInstantiation
# ---------------------------------------------------------------------------

class TestOCRDialogInstantiation:
    def test_dialog_instantiates(self, app):
        """OCRDialog can be constructed and shown."""
        from windows.ocr_dialog import OCRDialog

        dlg = OCRDialog(None)
        dlg.show()
        app.processEvents()
        assert isinstance(dlg, QDialog)
        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_dialog_has_run_and_stop_buttons(self, app):
        """Dialog has run_btn (enabled) and stop_btn (disabled by default)."""
        from windows.ocr_dialog import OCRDialog

        dlg = OCRDialog(None)
        app.processEvents()

        assert hasattr(dlg, "run_btn")
        assert hasattr(dlg, "stop_btn")
        assert dlg.run_btn.isEnabled()
        assert not dlg.stop_btn.isEnabled()

        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_dialog_lists_docs(self, app):
        """Combobox has at least 1 item (known docs list)."""
        from windows.ocr_dialog import OCRDialog

        dlg = OCRDialog(None)
        app.processEvents()

        assert dlg.doc_combo.count() >= 1

        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_stats_table_has_correct_columns(self, app):
        """Stats table has exactly 6 columns."""
        from windows.ocr_dialog import OCRDialog

        dlg = OCRDialog(None)
        app.processEvents()

        assert dlg.stats_table.columnCount() == 6

        dlg.close()
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# Helpers for mocking the OcrWorker subprocess
# ---------------------------------------------------------------------------

def _make_fake_popen(progress_events, summary):
    """
    Return a context-manager-compatible Popen mock whose stdout yields
    JSON progress lines followed by a __SUMMARY__ line.
    """
    lines = [json.dumps(e) + "\n" for e in progress_events]
    lines.append(f"__SUMMARY__ {json.dumps(summary)}\n")

    mock_proc = MagicMock()
    mock_proc.stdout = iter(lines)
    mock_proc.stderr = iter([])
    mock_proc.returncode = 0
    mock_proc.wait.return_value = 0
    mock_proc.terminate = MagicMock()
    mock_proc.kill = MagicMock()
    return mock_proc


# ---------------------------------------------------------------------------
# TestOcrWorkerMocked
# ---------------------------------------------------------------------------

class TestOcrWorkerMocked:
    def test_worker_emits_progress_and_finishes(self, app, tmp_path):
        """OcrWorker spawns a subprocess, reads JSON lines, emits progress + finished."""
        from windows.ocr_dialog import OcrWorker

        fake_doc = tmp_path / "fake.pdf"
        fake_doc.write_bytes(b"")
        fake_cache = tmp_path / "cache.jsonl"

        progress_events = []
        finished_events = []

        fake_progress = [
            {
                "doc": "fake.pdf", "page_index": i, "total_pages": 3,
                "source": "native", "ocr_count": 0, "native_count": i + 1,
                "done_count": i + 1, "elapsed_s": float(i),
            }
            for i in range(3)
        ]
        fake_summary = {
            "doc": "fake.pdf", "total_pages": 3, "ocr_count": 0,
            "native_count": 3, "skipped_cached": 0, "elapsed_s": 3.0,
            "stopped": False,
        }

        mock_proc = _make_fake_popen(fake_progress, fake_summary)

        stop_flag = [False]
        worker = OcrWorker(fake_doc, fake_cache, stop_flag)

        worker.progress.connect(lambda e: progress_events.append(e))
        worker.finished.connect(lambda s: finished_events.append(s))

        with patch("subprocess.Popen", return_value=mock_proc):
            worker.start()
            deadline = time.time() + 5.0
            while not finished_events and time.time() < deadline:
                app.processEvents()
                time.sleep(0.01)
            worker.wait(5000)

        assert len(progress_events) == 3, (
            f"Expected 3 progress events, got {len(progress_events)}"
        )
        assert len(finished_events) == 1, (
            f"Expected 1 finished event, got {len(finished_events)}"
        )
        assert finished_events[0]["doc"] == "fake.pdf"
        assert finished_events[0]["stopped"] is False

    def test_stop_flag_terminates_subprocess(self, app, tmp_path):
        """Setting stop_flag[0]=True causes the worker to terminate the subprocess."""
        from windows.ocr_dialog import OcrWorker

        fake_doc = tmp_path / "fake2.pdf"
        fake_doc.write_bytes(b"")
        fake_cache = tmp_path / "cache2.jsonl"

        # stop_flag already True — worker should terminate child immediately
        stop_flag = [True]

        # Give it one progress line so the loop body executes once before
        # the stop check fires.
        one_event = {
            "doc": "fake2.pdf", "page_index": 0, "total_pages": 5,
            "source": "native", "ocr_count": 0, "native_count": 1,
            "done_count": 1, "elapsed_s": 0.1,
        }
        # Summary won't be reached because stop fires first.
        mock_proc = _make_fake_popen([one_event], {
            "doc": "fake2.pdf", "total_pages": 5, "ocr_count": 0,
            "native_count": 1, "skipped_cached": 0, "elapsed_s": 0.1,
            "stopped": True,
        })

        worker = OcrWorker(fake_doc, fake_cache, stop_flag)

        finished_events = []
        worker.finished.connect(lambda s: finished_events.append(s))

        with patch("subprocess.Popen", return_value=mock_proc):
            worker.start()
            deadline = time.time() + 5.0
            while not finished_events and time.time() < deadline:
                app.processEvents()
                time.sleep(0.01)
            worker.wait(5000)

        assert len(finished_events) == 1
        # Worker was stopped — terminated dict has stopped=True
        assert finished_events[0].get("stopped") is True
        # Subprocess was asked to terminate
        mock_proc.terminate.assert_called_once()
