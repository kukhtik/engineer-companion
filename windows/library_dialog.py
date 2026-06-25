"""Library & Index management dialog.

Shows indexed documents with chunk/page counts, allows adding new PDFs
(copy → index with live progress) and removing documents from the index.

Threading model:
- IndexWorker(QThread) calls core.indexer.index_new_pdf IN-PROCESS
  (sentence-transformers / torch are fine in a QThread; only paddle
   must stay in a subprocess — see ocr_dialog.py).
- Stop via a mutable list flag passed to index_new_pdf's should_stop param.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, QEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QApplication,
)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from windows.app_paths import db_path as _default_db_path, library_dir  # noqa: E402


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class IndexWorker(QThread):
    """In-process indexing worker for a single PDF.

    Calls core.indexer.index_new_pdf directly (torch/sentence-transformers
    are safe in a QThread).  Progress dicts are forwarded to the UI thread
    via the ``progress`` signal.

    Stop by setting ``stop_flag[0] = True`` from the main thread.
    """

    progress = Signal(dict)    # {"source", "page_index", "total_pages", "chunks_added", "done"}
    finished = Signal(dict)    # {"source", "chunks_added"}
    error = Signal(str)

    def __init__(self, pdf_path: Path, db_path: Path, stop_flag: list) -> None:
        super().__init__()
        self._pdf_path = pdf_path
        self._db_path = db_path
        self._stop_flag = stop_flag  # mutable [False]; set [0]=True to abort

    def run(self) -> None:
        try:
            from core.indexer import index_new_pdf

            def _cb(evt: dict) -> None:
                self.progress.emit(evt)

            count = index_new_pdf(
                self._pdf_path,
                self._db_path,
                progress_callback=_cb,
                should_stop=self._stop_flag,
            )
            self.finished.emit({"source": self._pdf_path.name, "chunks_added": count})
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class LibraryDialog(QDialog):
    """Библиотека и индекс — управление документами в векторном хранилище."""

    def __init__(self, parent=None, db_path: Optional[Path] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Библиотека и индекс")
        self.setMinimumSize(860, 600)

        self._db_path: Path = db_path or _default_db_path()
        self._worker: Optional[IndexWorker] = None
        self._stop_flag: list = [False]
        self._shown_once = False

        self._build_ui()
        self._apply_style()
        self._refresh_stats()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ---- Header stats ----
        self.header_label = QLabel("Документов: — · Чанков: —")
        self.header_label.setObjectName("muted")
        root.addWidget(self.header_label)

        # ---- Documents table ----
        table_group = QGroupBox("Проиндексированные документы")
        table_v = QVBoxLayout(table_group)
        table_v.setSpacing(6)

        self.docs_table = QTableWidget(0, 4)
        self.docs_table.setHorizontalHeaderLabels([
            "Документ", "Страниц", "Чанков", "Статус"
        ])
        self.docs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 4):
            self.docs_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self.docs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.docs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.docs_table.setSelectionMode(QTableWidget.SingleSelection)
        self.docs_table.verticalHeader().setVisible(False)
        self.docs_table.itemSelectionChanged.connect(self._on_selection_changed)
        table_v.addWidget(self.docs_table)

        # Table action buttons row
        tbl_btn_row = QHBoxLayout()
        tbl_btn_row.setSpacing(8)

        self.refresh_btn = QPushButton("Обновить")
        self.refresh_btn.clicked.connect(self._refresh_stats)
        tbl_btn_row.addWidget(self.refresh_btn)

        self.remove_btn = QPushButton("Удалить из индекса")
        self.remove_btn.setProperty("destructive", "true")
        self.remove_btn.setEnabled(False)
        self.remove_btn.clicked.connect(self._on_remove)
        tbl_btn_row.addWidget(self.remove_btn)

        self.reindex_btn = QPushButton("Переиндексировать")
        self.reindex_btn.setEnabled(False)
        self.reindex_btn.clicked.connect(self._on_reindex)
        tbl_btn_row.addWidget(self.reindex_btn)

        tbl_btn_row.addStretch()
        table_v.addLayout(tbl_btn_row)
        root.addWidget(table_group, stretch=1)

        # ---- Add document section ----
        add_group = QGroupBox("Добавить документы")
        add_v = QVBoxLayout(add_group)
        add_v.setSpacing(8)

        add_btn_row = QHBoxLayout()
        add_btn_row.setSpacing(8)

        self.add_btn = QPushButton("Добавить документ…")
        self.add_btn.clicked.connect(self._on_add)
        add_btn_row.addWidget(self.add_btn)

        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        add_btn_row.addWidget(self.stop_btn)

        add_btn_row.addStretch()
        add_v.addLayout(add_btn_row)

        # Progress area (hidden until indexing starts)
        self._progress_widget = QWidget()
        prog_v = QVBoxLayout(self._progress_widget)
        prog_v.setContentsMargins(0, 0, 0, 0)
        prog_v.setSpacing(4)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        prog_v.addWidget(self.progress_bar)

        self.progress_label = QLabel("Индексация: —")
        self.progress_label.setWordWrap(True)
        prog_v.addWidget(self.progress_label)

        self._progress_widget.setVisible(False)
        add_v.addWidget(self._progress_widget)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("muted")
        add_v.addWidget(self.status_label)

        root.addWidget(add_group)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def showEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._shown_once:
            self._shown_once = True
            from windows.anim import fade_in
            # Fade in the whole dialog content (self) on first show
            fade_in(self, duration=200)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        """Apply the active application theme (matches the rest of the UI)."""
        app = QApplication.instance()
        if app is not None:
            # Inherit the app-level stylesheet — no separate setStyleSheet needed.
            # But fall back to DribbbleDarkQt for standalone testing.
            if not app.styleSheet():
                try:
                    from design.tokens import DribbbleDarkQt
                    self.setStyleSheet(DribbbleDarkQt().as_stylesheet())
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Stats / table population
    # ------------------------------------------------------------------

    def _refresh_stats(self) -> None:
        """Re-query the DB and repopulate the table."""
        from core.indexer import index_stats
        stats = index_stats(self._db_path)

        total_docs = len(stats["per_doc"])
        total_chunks = stats["total_chunks"]
        self.header_label.setText(
            f"Документов: {total_docs} · Чанков: {total_chunks}"
        )

        self.docs_table.setRowCount(len(stats["per_doc"]))
        for row, doc in enumerate(stats["per_doc"]):
            self.docs_table.setItem(row, 0, QTableWidgetItem(doc["source"]))
            self.docs_table.setItem(row, 1, QTableWidgetItem(str(doc["pages"])))
            self.docs_table.setItem(row, 2, QTableWidgetItem(str(doc["chunks"])))
            self.docs_table.setItem(row, 3, QTableWidgetItem("✓ В индексе"))

        self._on_selection_changed()

    def _selected_source(self) -> Optional[str]:
        """Return the source filename of the selected row, or None."""
        selected = self.docs_table.selectedItems()
        if not selected:
            return None
        row = self.docs_table.currentRow()
        item = self.docs_table.item(row, 0)
        return item.text() if item else None

    # ------------------------------------------------------------------
    # Button state management
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        has_sel = self._selected_source() is not None
        busy = self._worker is not None
        self.remove_btn.setEnabled(has_sel and not busy)
        self.reindex_btn.setEnabled(has_sel and not busy)

    def _set_busy(self, busy: bool) -> None:
        self.add_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.refresh_btn.setEnabled(not busy)
        self._on_selection_changed()
        if busy:
            self._progress_widget.setVisible(True)
            from windows.anim import fade_in
            fade_in(self._progress_widget, duration=180)
        else:
            self._progress_widget.setVisible(False)

    # ------------------------------------------------------------------
    # Add document
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите PDF для добавления",
            str(library_dir()),
            "PDF-файлы (*.pdf);;Все файлы (*)",
        )
        if not paths:
            return

        # Process one file at a time; queue them sequentially
        self._pending_paths = [Path(p) for p in paths]
        self._index_next()

    def _index_next(self) -> None:
        if not self._pending_paths:
            self._refresh_stats()
            self.status_label.setText("Готово: все файлы добавлены в индекс.")
            return

        pdf_path = self._pending_paths.pop(0)

        # Copy into library_dir if not already there
        lib = library_dir()
        lib.mkdir(parents=True, exist_ok=True)
        dest = lib / pdf_path.name

        if dest.exists() and dest != pdf_path:
            answer = QMessageBox.question(
                self,
                "Файл уже существует",
                f"Файл «{pdf_path.name}» уже есть в библиотеке.\nПерезаписать?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                # Skip this file, move on
                self._index_next()
                return
            shutil.copy2(str(pdf_path), str(dest))
        elif dest != pdf_path:
            shutil.copy2(str(pdf_path), str(dest))

        # Start indexing worker
        self._stop_flag = [False]
        self._worker = IndexWorker(dest, self._db_path, self._stop_flag)
        self._worker.progress.connect(self._on_index_progress)
        self._worker.finished.connect(self._on_index_finished)
        self._worker.error.connect(self._on_index_error)

        self._set_busy(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Индексация: {dest.name} — стр. 0/?, чанков: 0")
        self.status_label.setText(f"Запускаем индексацию: {dest.name}…")
        self._worker.start()

    def _on_index_progress(self, evt: dict) -> None:
        page_idx = evt.get("page_index", 0)
        total_pages = evt.get("total_pages", 1)
        chunks_added = evt.get("chunks_added", 0)
        source = evt.get("source", "")

        self.progress_bar.setMaximum(max(total_pages, 1))
        self.progress_bar.setValue(page_idx)
        self.progress_label.setText(
            f"Индексация: {source} — стр. {page_idx}/{total_pages}, чанков: {chunks_added}"
        )

    def _on_index_finished(self, result: dict) -> None:
        self._worker = None
        source = result.get("source", "")
        chunks = result.get("chunks_added", 0)
        self.status_label.setText(
            f"✓ {source} — добавлено чанков: {chunks}"
        )
        self._set_busy(False)
        # Refresh table to show new doc, then continue queue
        self._refresh_stats()
        self._index_next()

    def _on_index_error(self, msg: str) -> None:
        self._worker = None
        self._set_busy(False)
        self.status_label.setText(f"Ошибка индексации: {msg}")
        QMessageBox.critical(self, "Ошибка индексации", msg)
        # Continue with remaining files despite error
        self._index_next()

    def _on_stop(self) -> None:
        self._stop_flag[0] = True
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Останавливаем индексацию…")
        # Clear pending queue too so we don't start the next file
        self._pending_paths = []

    # ------------------------------------------------------------------
    # Remove document
    # ------------------------------------------------------------------

    def _on_remove(self) -> None:
        source = self._selected_source()
        if not source:
            return

        # Ask whether to also delete the PDF file
        remove_file_cb = QCheckBox("Также удалить файл из библиотеки")
        msg = QMessageBox(self)
        msg.setWindowTitle("Удалить из индекса")
        msg.setText(
            f"Удалить «{source}» из векторного индекса?\n\n"
            "Это не влияет на возможность повторной индексации."
        )
        msg.setCheckBox(remove_file_cb)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)

        if msg.exec() != QMessageBox.Yes:
            return

        from core.indexer import remove_document
        removed = remove_document(self._db_path, source)
        self.status_label.setText(f"Удалено {removed} чанков для «{source}».")

        # Optionally delete the PDF
        if remove_file_cb.isChecked():
            pdf_file = library_dir() / source
            if pdf_file.exists():
                try:
                    pdf_file.unlink()
                    self.status_label.setText(
                        f"Удалено {removed} чанков и файл «{source}» из библиотеки."
                    )
                except OSError as exc:
                    QMessageBox.warning(
                        self, "Не удалось удалить файл", str(exc)
                    )

        self._refresh_stats()

    # ------------------------------------------------------------------
    # Reindex document
    # ------------------------------------------------------------------

    def _on_reindex(self) -> None:
        source = self._selected_source()
        if not source:
            return

        pdf_file = library_dir() / source
        if not pdf_file.exists():
            QMessageBox.warning(
                self,
                "Файл не найден",
                f"PDF «{source}» не найден в библиотеке:\n{pdf_file}\n\n"
                "Сначала добавьте файл через кнопку «Добавить документ…».",
            )
            return

        answer = QMessageBox.question(
            self,
            "Переиндексировать",
            f"Удалить и переиндексировать «{source}»?\n"
            "Текущие чанки будут удалены и добавлены заново.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return

        # Remove existing chunks first
        from core.indexer import remove_document
        remove_document(self._db_path, source)

        # Re-index
        self._pending_paths = [pdf_file]
        self._index_next()


    # ------------------------------------------------------------------
    # Qt close — wait for worker
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._stop_flag[0] = True
            self._worker.wait(3000)
        super().closeEvent(event)
