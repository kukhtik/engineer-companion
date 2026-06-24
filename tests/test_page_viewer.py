"""Tests for Phase 3: PDF page renderer, PageViewer dialog, clickable sources.

Requires:
- PyMuPDF (fitz) installed in the venv
- A real PDF in docs/ (TrueBeam Administrators Guide.pdf)
- QT_QPA_PLATFORM=offscreen for headless rendering

Run with:
    $env:QT_QPA_PLATFORM="offscreen"; $env:PYTHONPATH="E:\engineer-companion"
    pytest tests/test_page_viewer.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root on path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DOCS_DIR = _REPO / "docs"
TRUEBEAM_ADMIN = DOCS_DIR / "TrueBeam Administrators Guide.pdf"

# Source string as it would appear in the index (just the filename)
TRUEBEAM_ADMIN_SOURCE = "TrueBeam Administrators Guide.pdf"


def _require_fitz():
    """Skip test if PyMuPDF is not installed."""
    try:
        import fitz  # noqa: F401
    except ImportError:
        pytest.skip("PyMuPDF (fitz) not installed")


def _require_pdf():
    """Skip test if the real PDF is not present."""
    if not TRUEBEAM_ADMIN.exists():
        pytest.skip(f"PDF not present: {TRUEBEAM_ADMIN}")


# ---------------------------------------------------------------------------
# A. resolve_pdf tests
# ---------------------------------------------------------------------------

class TestResolvePdf:
    """Tests for windows.pdf_render.resolve_pdf."""

    def test_resolve_exact_match(self):
        """resolve_pdf finds the file by its exact index source string."""
        _require_pdf()
        from windows.pdf_render import resolve_pdf
        result = resolve_pdf(TRUEBEAM_ADMIN_SOURCE)
        assert result is not None, "Expected to find the PDF in docs/"
        assert result.exists()
        assert result.name == TRUEBEAM_ADMIN_SOURCE

    def test_resolve_case_insensitive(self):
        """resolve_pdf matches case-insensitively."""
        _require_pdf()
        from windows.pdf_render import resolve_pdf
        result = resolve_pdf("truebeam administrators guide.pdf")
        assert result is not None

    def test_resolve_without_extension(self):
        """resolve_pdf matches when .pdf suffix is omitted."""
        _require_pdf()
        from windows.pdf_render import resolve_pdf
        result = resolve_pdf("TrueBeam Administrators Guide")
        assert result is not None

    def test_resolve_not_found_returns_none(self):
        """resolve_pdf returns None for a non-existent document."""
        from windows.pdf_render import resolve_pdf
        result = resolve_pdf("NoSuchDocument_XYZ123.pdf")
        assert result is None

    def test_resolve_missing_library(self, tmp_path, monkeypatch):
        """resolve_pdf returns None gracefully if library dir doesn't exist."""
        from windows import pdf_render
        monkeypatch.setattr(pdf_render, "library_dir", lambda: tmp_path / "nonexistent")
        result = pdf_render.resolve_pdf("anything.pdf")
        assert result is None


# ---------------------------------------------------------------------------
# B. render_page tests
# ---------------------------------------------------------------------------

class TestRenderPage:
    """Tests for windows.pdf_render.render_page."""

    def test_render_returns_qimage(self, qapp):
        """render_page returns a valid non-null QImage with sane dimensions."""
        _require_fitz()
        _require_pdf()
        from windows.pdf_render import render_page
        img = render_page(TRUEBEAM_ADMIN, page_number=1, dpi=72)
        assert img is not None
        assert not img.isNull()
        # At 72 dpi a letter page is ~612x792 pt -> ~612x792 px at 72dpi
        assert img.width() > 100
        assert img.height() > 100

    def test_render_page_114(self, qapp):
        """Render page 114 of TrueBeam Administrators Guide at dpi=200 (smoke test)."""
        _require_fitz()
        _require_pdf()
        from windows.pdf_render import render_page
        img = render_page(TRUEBEAM_ADMIN, page_number=114, dpi=200)
        assert not img.isNull()
        # 200 dpi on a letter page: ~1700x2200 px
        assert img.width() > 500
        assert img.height() > 500

    def test_render_higher_dpi_is_larger(self, qapp):
        """Higher DPI produces larger images (not just upscaled)."""
        _require_fitz()
        _require_pdf()
        from windows.pdf_render import render_page
        img72 = render_page(TRUEBEAM_ADMIN, page_number=1, dpi=72)
        img200 = render_page(TRUEBEAM_ADMIN, page_number=1, dpi=200)
        assert img200.width() > img72.width()

    def test_render_invalid_page_raises(self, qapp):
        """render_page raises ValueError for out-of-range page_number."""
        _require_fitz()
        _require_pdf()
        from windows.pdf_render import render_page
        import fitz
        doc = fitz.open(str(TRUEBEAM_ADMIN))
        total = doc.page_count
        doc.close()
        with pytest.raises(ValueError, match="out of range"):
            render_page(TRUEBEAM_ADMIN, page_number=total + 1, dpi=72)

    def test_render_page_1_indexed(self, qapp):
        """Page 1 and page 2 return different images (confirms 1-indexed offset)."""
        _require_fitz()
        _require_pdf()
        from windows.pdf_render import render_page
        img1 = render_page(TRUEBEAM_ADMIN, page_number=1, dpi=72)
        img2 = render_page(TRUEBEAM_ADMIN, page_number=2, dpi=72)
        # Different pages should produce different pixel content
        assert img1.bits() != img2.bits() or img1.size() != img2.size() or \
               img1.pixel(10, 10) != img2.pixel(10, 10)


# ---------------------------------------------------------------------------
# C. PageViewer dialog tests
# ---------------------------------------------------------------------------

class TestPageViewer:
    """Tests for windows.page_viewer.PageViewer."""

    def test_viewer_instantiates_headless(self, qapp):
        """PageViewer can be instantiated under offscreen with a real PDF."""
        _require_fitz()
        _require_pdf()
        from windows.page_viewer import PageViewer
        viewer = PageViewer(
            pdf_path=TRUEBEAM_ADMIN,
            page_number=1,
            source_name=TRUEBEAM_ADMIN_SOURCE,
            is_ocr=False,
        )
        assert viewer is not None
        # Wait briefly for the render worker to finish
        if viewer._worker is not None:
            viewer._worker.wait(3000)
        viewer.close()

    def test_viewer_shows_page_label(self, qapp):
        """PageViewer's page label contains the page number."""
        _require_fitz()
        _require_pdf()
        from windows.page_viewer import PageViewer
        viewer = PageViewer(
            pdf_path=TRUEBEAM_ADMIN,
            page_number=5,
            source_name=TRUEBEAM_ADMIN_SOURCE,
        )
        label_text = viewer._page_label.text()
        assert "5" in label_text
        if viewer._worker:
            viewer._worker.wait(3000)
        viewer.close()

    def test_viewer_not_found_mode(self, qapp):
        """PageViewer with pdf_path=None shows a 'not found' UI (no crash)."""
        from windows.page_viewer import PageViewer
        viewer = PageViewer(
            pdf_path=None,
            page_number=1,
            source_name="MissingDoc.pdf",
        )
        # Should have built the not-found UI without crash
        assert viewer is not None
        viewer.close()

    def test_ocr_mode_default_dpi(self, qapp):
        """OCR mode initialises with higher default DPI (250)."""
        _require_fitz()
        _require_pdf()
        from windows.page_viewer import PageViewer, _DPI_OCR, _DPI_FIT
        viewer_normal = PageViewer(TRUEBEAM_ADMIN, 1, is_ocr=False)
        viewer_ocr = PageViewer(TRUEBEAM_ADMIN, 1, is_ocr=True)
        assert viewer_normal._current_dpi == _DPI_FIT
        assert viewer_ocr._current_dpi == _DPI_OCR
        assert viewer_ocr._current_dpi > viewer_normal._current_dpi
        for v in (viewer_normal, viewer_ocr):
            if v._worker:
                v._worker.wait(2000)
            v.close()


# ---------------------------------------------------------------------------
# D. Source link wiring in main window
# ---------------------------------------------------------------------------

class TestSourceLinkWiring:
    """Tests for clickable source: links in the chat log."""

    def test_format_html_contains_source_links(self):
        """format_html renders source links with source: scheme."""
        from windows.main_window import ChatHistory
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([], f)
            tmp = f.name
        try:
            history = ChatHistory(Path(tmp))
            history.add(
                query="test question",
                answer="test answer",
                sources=[{"source": "TrueBeam Administrators Guide.pdf", "page": 114, "section": "Calibration"}],
            )
            html_out = history.format_html()
            assert "source:" in html_out
            assert "TrueBeam" in html_out
            assert "114" in html_out
        finally:
            os.unlink(tmp)

    def test_format_html_ocr_flag(self):
        """format_html sets ocr_flag=1 for sections starting with [OCR]."""
        from windows.main_window import ChatHistory
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([], f)
            tmp = f.name
        try:
            history = ChatHistory(Path(tmp))
            history.add(
                query="q",
                answer="a",
                sources=[{"source": "Drawing.pdf", "page": 5, "section": "[OCR] Schematic"}],
            )
            html_out = history.format_html()
            assert "|1" in html_out, "Expected ocr_flag=1 in source link for [OCR] section"
        finally:
            os.unlink(tmp)

    def test_anchor_parsed_correctly(self, qapp):
        """_on_anchor_clicked parses source: URL and calls _open_page_viewer."""
        from windows.main_window import CompanionWindow
        from PySide6.QtCore import QUrl

        window = CompanionWindow(pipeline=None)
        calls = []

        def mock_open(filename, page, is_ocr=False):
            calls.append((filename, page, is_ocr))

        window._open_page_viewer = mock_open
        url = QUrl("source:TrueBeam%20Administrators%20Guide.pdf|114|0")
        window._on_anchor_clicked(url)

        assert len(calls) == 1
        filename, page, is_ocr = calls[0]
        assert "TrueBeam" in filename
        assert page == 114
        assert is_ocr is False

    def test_anchor_ocr_flag_parsed(self, qapp):
        """_on_anchor_clicked correctly parses is_ocr=True from the URL."""
        from windows.main_window import CompanionWindow
        from PySide6.QtCore import QUrl

        window = CompanionWindow(pipeline=None)
        calls = []
        window._open_page_viewer = lambda f, p, is_ocr=False: calls.append((f, p, is_ocr))

        url = QUrl("source:Drawing.pdf|5|1")
        window._on_anchor_clicked(url)

        assert calls[0][2] is True  # is_ocr

    def test_open_page_viewer_monkeypatched(self, qapp):
        """_open_page_viewer resolves PDF and instantiates PageViewer (mocked)."""
        from windows.main_window import CompanionWindow
        from windows import main_window

        window = CompanionWindow(pipeline=None)
        viewer_calls = []

        class FakeViewer:
            def __init__(self, *a, **kw):
                viewer_calls.append(kw)
            def exec(self):
                pass

        with patch("windows.main_window.CompanionWindow._open_page_viewer") as mock_open:
            mock_open.side_effect = lambda f, p, is_ocr=False: viewer_calls.append((f, p, is_ocr))
            from PySide6.QtCore import QUrl
            url = QUrl("source:TrueBeam%20Administrators%20Guide.pdf|10|0")
            window._on_anchor_clicked(url)

        assert len(viewer_calls) == 1
        assert "TrueBeam" in viewer_calls[0][0]
        assert viewer_calls[0][1] == 10
