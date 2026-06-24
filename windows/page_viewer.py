"""PDF Page Viewer dialog.

Shows a single rendered PDF page with zoom controls and prev/next navigation.
Rendering is done off-thread (QThread worker) so the UI doesn't freeze.
Fade-in animation via windows/anim.py.

Usage:
    viewer = PageViewer(pdf_path, page_number, parent=parent, is_ocr=False)
    viewer.exec()

If pdf_path is None (PDF not found in library), shows a friendly message
with a button to open the Library dialog to import it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl
from PySide6.QtGui import QImage, QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QSizePolicy,
)

from windows.anim import fade_in


# DPI constants
_DPI_FIT = 150        # base DPI for fit-to-width
_DPI_100 = 150        # "100%" (screen resolution approx)
_DPI_OCR = 250        # higher default for OCR/scanned drawing pages
_DPI_MAX = 400        # maximum allowed DPI
_DPI_STEP = 50        # DPI increment per zoom in/out step


class _RenderWorker(QThread):
    """Off-thread page renderer."""
    done = Signal(QImage)
    error = Signal(str)

    def __init__(self, pdf_path: Path, page_number: int, dpi: int) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.page_number = page_number
        self.dpi = dpi

    def run(self) -> None:
        try:
            from windows.pdf_render import render_page
            img = render_page(self.pdf_path, self.page_number, dpi=self.dpi)
            self.done.emit(img)
        except Exception as exc:
            self.error.emit(str(exc))


class PageViewer(QDialog):
    """Dialog that shows a single PDF page with zoom / prev / next controls.

    Args:
        pdf_path: Path to the PDF file, or None if the file is not in the library.
        page_number: 1-indexed page number to show initially.
        source_name: Original source filename (for display and library prompt).
        parent: Parent widget.
        is_ocr: If True (section started with "[OCR] "), default to higher DPI.
    """

    def __init__(
        self,
        pdf_path: Optional[Path],
        page_number: int,
        source_name: str = "",
        parent: Optional[QWidget] = None,
        is_ocr: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Просмотр страницы")
        self.setMinimumSize(700, 600)
        self.resize(900, 700)

        self._pdf_path = pdf_path
        self._current_page = page_number
        self._total_pages = 0
        self._source_name = source_name
        self._is_ocr = is_ocr
        self._current_dpi = _DPI_OCR if is_ocr else _DPI_FIT
        self._worker: Optional[_RenderWorker] = None

        self._build_ui(pdf_path is None)

        if pdf_path is not None:
            # Count pages
            try:
                import fitz
                doc = fitz.open(str(pdf_path))
                self._total_pages = doc.page_count
                doc.close()
            except Exception:
                self._total_pages = max(page_number, 1)
            self._update_page_label()
            self._start_render()

        # Fade in the dialog
        fade_in(self, duration=180)

    def _build_ui(self, not_found: bool) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        if not_found:
            self._build_not_found_ui(layout)
            return

        # --- Controls row ---
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self._prev_btn = QPushButton("◀ Пред.")
        self._prev_btn.clicked.connect(self._on_prev)
        ctrl.addWidget(self._prev_btn)

        self._next_btn = QPushButton("След. ▶")
        self._next_btn.clicked.connect(self._on_next)
        ctrl.addWidget(self._next_btn)

        ctrl.addStretch(1)

        self._page_label = QLabel("стр. — / —")
        self._page_label.setObjectName("muted")
        ctrl.addWidget(self._page_label)

        ctrl.addStretch(1)

        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedWidth(32)
        zoom_out_btn.clicked.connect(self._on_zoom_out)
        ctrl.addWidget(zoom_out_btn)

        self._zoom_label = QLabel("150 dpi")
        self._zoom_label.setObjectName("muted")
        self._zoom_label.setMinimumWidth(60)
        self._zoom_label.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self._zoom_label)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(32)
        zoom_in_btn.clicked.connect(self._on_zoom_in)
        ctrl.addWidget(zoom_in_btn)

        fit_btn = QPushButton("По ширине")
        fit_btn.clicked.connect(self._on_fit_width)
        ctrl.addWidget(fit_btn)

        layout.addLayout(ctrl)

        # --- Render status label ---
        self._render_status = QLabel("")
        self._render_status.setObjectName("muted")
        self._render_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._render_status)

        # --- Scroll area with page image ---
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignCenter)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll.setWidget(self._img_label)
        layout.addWidget(self._scroll, stretch=1)

        # --- Bottom row: open original ---
        bottom = QHBoxLayout()
        open_pdf_btn = QPushButton("Открыть оригинал PDF")
        open_pdf_btn.clicked.connect(self._on_open_original)
        bottom.addStretch(1)
        bottom.addWidget(open_pdf_btn)
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    def _build_not_found_ui(self, layout: QVBoxLayout) -> None:
        layout.addStretch(1)
        msg = QLabel(
            f"PDF «{self._source_name}» не найден в библиотеке.\n"
            "Импортируйте документ через «Файл → Библиотека и индекс…»."
        )
        msg.setAlignment(Qt.AlignCenter)
        msg.setWordWrap(True)
        layout.addWidget(msg)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        lib_btn = QPushButton("Открыть библиотеку…")
        lib_btn.clicked.connect(self._on_open_library)
        btn_row.addWidget(lib_btn)
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        layout.addStretch(1)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _start_render(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(500)

        self._render_status.setText("рендер…")
        self._zoom_label.setText(f"{self._current_dpi} dpi")

        worker = _RenderWorker(self._pdf_path, self._current_page, self._current_dpi)
        worker.done.connect(self._on_render_done)
        worker.error.connect(self._on_render_error)
        worker.finished.connect(self._cleanup_worker)
        worker.start()
        self._worker = worker

    def _on_render_done(self, img: QImage) -> None:
        self._render_status.setText("")
        pixmap = QPixmap.fromImage(img)
        self._img_label.setPixmap(pixmap)
        self._img_label.setMinimumSize(QSize(pixmap.width(), pixmap.height()))

    def _on_render_error(self, msg: str) -> None:
        self._render_status.setText(f"Ошибка рендеринга: {msg}")

    def _cleanup_worker(self) -> None:
        self._worker = None

    # ------------------------------------------------------------------
    # Navigation & zoom
    # ------------------------------------------------------------------

    def _update_page_label(self) -> None:
        self._page_label.setText(f"стр. {self._current_page} / {self._total_pages}")
        if hasattr(self, "_prev_btn"):
            self._prev_btn.setEnabled(self._current_page > 1)
            self._next_btn.setEnabled(self._current_page < self._total_pages)

    def _on_prev(self) -> None:
        if self._current_page > 1:
            self._current_page -= 1
            self._update_page_label()
            self._start_render()

    def _on_next(self) -> None:
        if self._current_page < self._total_pages:
            self._current_page += 1
            self._update_page_label()
            self._start_render()

    def _on_zoom_in(self) -> None:
        self._current_dpi = min(self._current_dpi + _DPI_STEP, _DPI_MAX)
        self._zoom_label.setText(f"{self._current_dpi} dpi")
        self._start_render()

    def _on_zoom_out(self) -> None:
        self._current_dpi = max(self._current_dpi - _DPI_STEP, 72)
        self._zoom_label.setText(f"{self._current_dpi} dpi")
        self._start_render()

    def _on_fit_width(self) -> None:
        self._current_dpi = _DPI_FIT
        self._zoom_label.setText(f"{self._current_dpi} dpi")
        self._start_render()

    # ------------------------------------------------------------------
    # External actions
    # ------------------------------------------------------------------

    def _on_open_original(self) -> None:
        if self._pdf_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._pdf_path)))

    def _on_open_library(self) -> None:
        # Try to open library dialog from parent or standalone
        parent = self.parent()
        if parent is not None and hasattr(parent, "_on_open_library"):
            self.accept()
            parent._on_open_library()
        else:
            from windows.app_paths import db_path as _default_db_path
            from windows.library_dialog import LibraryDialog
            dlg = LibraryDialog(self, db_path=_default_db_path())
            dlg.exec()
