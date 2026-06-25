"""OCR Dialog — in-app OCR management panel."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, QEvent
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTableWidget, QTableWidgetItem, QComboBox,
    QGroupBox, QWidget, QSizePolicy, QHeaderView, QMessageBox,
    QApplication,
)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

CACHE_PATH = _REPO / "scripts" / "ocr_cache.jsonl"


def ocr_available() -> bool:
    """Return True if OCR can be used in the current runtime environment.

    OCR requires:
    - Running from source (not a frozen PyInstaller EXE).
    - scripts/ocr_databooks.py present next to the source tree.
    - paddleocr importable (i.e. installed in the active venv).

    In a frozen build sys.executable is the EXE itself, paddle is not bundled,
    and scripts/ is not part of the package — so OCR is unavailable there.
    """
    if getattr(sys, "frozen", False):
        return False
    script = _REPO / "scripts" / "ocr_databooks.py"
    if not script.exists():
        return False
    try:
        import importlib.util
        if importlib.util.find_spec("paddleocr") is None:
            return False
    except Exception:
        return False
    return True


def _get_library_docs() -> list[str]:
    """Return sorted list of PDF filenames currently in the library directory."""
    try:
        from windows.app_paths import library_dir
        lib = library_dir()
        if lib.is_dir():
            return sorted(p.name for p in lib.glob("*.pdf"))
    except Exception:
        pass
    return []


class OcrWorker(QThread):
    """Background thread for running OCR on a single document.

    Spawns scripts/ocr_databooks.py as a *subprocess* with --progress-json so
    that PaddlePaddle is never imported inside the GUI process.  This avoids a
    Windows DLL conflict when paddle is loaded after torch (OSError: shm.dll).

    The subprocess emits one JSON line per page to stdout.  We read those lines
    here and re-emit the existing Qt progress / finished / error signals so the
    rest of the UI code is unchanged.

    Stop: setting stop_flag[0] = True causes us to terminate the child process.
    """

    progress = Signal(dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, doc_path: Path, cache_path: Path, stop_flag: list) -> None:
        super().__init__()
        self._doc_path = doc_path
        self._cache_path = cache_path
        self._stop_flag = stop_flag  # mutable list [False]; Stop sets [0]=True
        self._proc: Optional[subprocess.Popen] = None

    def _check_stop(self) -> bool:
        return bool(self._stop_flag[0])

    def run(self) -> None:
        # Build the subprocess command using the same interpreter as the app.
        script = str(_REPO / "scripts" / "ocr_databooks.py")
        cmd = [
            sys.executable,
            script,
            "--doc", str(self._doc_path),
            "--cache", str(self._cache_path),
            "--progress-json",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO)
        # Ensure stdout is not buffered inside the child process.
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # line-buffered
            )
        except Exception as exc:
            self.error.emit(f"Failed to start OCR subprocess: {exc}")
            return

        summary: Optional[dict] = None
        stderr_lines: list = []

        try:
            for line in self._proc.stdout:  # type: ignore[union-attr]
                # Check stop flag — terminate child if requested.
                if self._check_stop():
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                    # Build a minimal stopped summary from the last progress event.
                    if summary is None:
                        summary = {
                            'doc': self._doc_path.name,
                            'total_pages': 0,
                            'ocr_count': 0,
                            'native_count': 0,
                            'skipped_cached': 0,
                            'elapsed_s': 0.0,
                            'stopped': True,
                        }
                    else:
                        summary['stopped'] = True
                    self.finished.emit(summary)
                    return

                line = line.rstrip('\n')
                if not line:
                    continue

                if line.startswith('__SUMMARY__ '):
                    # Final summary line from the subprocess.
                    try:
                        summary = json.loads(line[len('__SUMMARY__ '):])
                    except json.JSONDecodeError:
                        pass
                    continue

                # Try to parse as a progress JSON line.
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        # Keep the latest event as a partial summary in case of
                        # early termination.
                        summary = {
                            'doc': event.get('doc', self._doc_path.name),
                            'total_pages': event.get('total_pages', 0),
                            'ocr_count': event.get('ocr_count', 0),
                            'native_count': event.get('native_count', 0),
                            'skipped_cached': 0,
                            'elapsed_s': event.get('elapsed_s', 0.0),
                            'stopped': False,
                        }
                        self.progress.emit(event)
                except json.JSONDecodeError:
                    # Not a JSON line — might be a stray print; ignore.
                    pass

            # Drain stderr for error reporting.
            if self._proc.stderr:
                for ln in self._proc.stderr:
                    stderr_lines.append(ln.rstrip('\n'))

            self._proc.wait()
            rc = self._proc.returncode

        except Exception as exc:
            self.error.emit(f"OCR subprocess error: {exc}")
            return
        finally:
            self._proc = None

        if rc != 0 and summary is None:
            err_msg = '\n'.join(stderr_lines[-20:]) or f"OCR subprocess exited with code {rc}"
            self.error.emit(err_msg)
            return

        if summary is None:
            # Process finished without emitting a summary — treat as error.
            self.error.emit("OCR subprocess produced no output")
            return

        self.finished.emit(summary)


class ReindexWorker(QThread):
    """Background thread for re-indexing a document into the vector store."""

    finished = Signal(str)
    error = Signal(str)

    def __init__(self, pdf_path: Path, ocr_cache: Path) -> None:
        super().__init__()
        self._pdf_path = pdf_path
        self._ocr_cache = ocr_cache

    def run(self) -> None:
        try:
            from core.indexer import DocumentIndexPipeline, PdfTextExtractor, Chunker
            pipeline = DocumentIndexPipeline(
                extractor=PdfTextExtractor(),
                chunker=Chunker(),
                db_path=_REPO / "assets" / "db" / "engineer.db",
            )
            pipeline.index_pdf_with_ocr(self._pdf_path, self._ocr_cache)
            self.finished.emit(str(self._pdf_path.name))
        except Exception as exc:
            self.error.emit(str(exc))


class OCRDialog(QDialog):
    """In-app OCR management panel."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("OCR — Управление индексацией")
        self.setMinimumSize(900, 680)

        self._ocr_worker: Optional[OcrWorker] = None
        self._reindex_worker: Optional[ReindexWorker] = None
        self._stop_flag: list = [False]
        self._stats_cache: list = []
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

        # ---- Section 1: document selector ----
        doc_group = QGroupBox("Выберите документ")
        doc_v = QVBoxLayout(doc_group)
        doc_v.setSpacing(6)

        self.doc_combo = QComboBox()
        self.doc_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        doc_v.addWidget(self.doc_combo)
        root.addWidget(doc_group)

        # ---- Section 2: OCR progress ----
        self._prog_group = QGroupBox("Прогресс OCR")
        prog_group = self._prog_group
        prog_v = QVBoxLayout(prog_group)
        prog_v.setSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        prog_v.addWidget(self.progress_bar)

        self.page_label = QLabel("Страница — / —")
        prog_v.addWidget(self.page_label)

        self.stats_label = QLabel("Нативных: — | OCR: — | Прошло: — | ETA: —")
        prog_v.addWidget(self.stats_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        prog_v.addWidget(self.status_label)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.run_btn = QPushButton("Запустить OCR")
        self.run_btn.clicked.connect(self._on_run_ocr)
        btn_row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_btn)

        self.reindex_btn = QPushButton("Переиндексировать")
        self.reindex_btn.setEnabled(False)
        self.reindex_btn.clicked.connect(self._on_reindex)
        btn_row.addWidget(self.reindex_btn)

        btn_row.addStretch()
        prog_v.addLayout(btn_row)
        root.addWidget(prog_group)

        # ---- Section 3: all-docs status table ----
        table_group = QGroupBox("Состояние всех документов")
        table_v = QVBoxLayout(table_group)
        table_v.setSpacing(6)

        self.stats_table = QTableWidget(0, 6)
        self.stats_table.setHorizontalHeaderLabels([
            "Документ", "Стр.", "Кэш", "Нативных", "OCR", "% Готово"
        ])
        self.stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            self.stats_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.stats_table.verticalHeader().setVisible(False)
        table_v.addWidget(self.stats_table)

        refresh_btn = QPushButton("Обновить")
        refresh_btn.clicked.connect(self._refresh_stats)
        table_v.addWidget(refresh_btn)

        root.addWidget(table_group, stretch=1)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def showEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._shown_once:
            self._shown_once = True
            from windows.anim import fade_in
            fade_in(self, duration=200)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        """Inherit the active application theme (no per-dialog override)."""
        app = QApplication.instance()
        if app is not None and not app.styleSheet():
            # Standalone testing fallback — no app stylesheet yet
            try:
                from design.tokens import DARK
                self.setStyleSheet(DARK.as_stylesheet())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Stats and combo population
    # ------------------------------------------------------------------

    def _refresh_stats(self) -> None:
        """Load stats from cache and update table + combobox."""
        from windows.app_paths import library_dir
        docs_dir = library_dir()

        try:
            from scripts.ocr_databooks import ocr_stats
            stats = ocr_stats(docs_dir, CACHE_PATH)
        except Exception:
            stats = []

        # Merge with all PDFs currently in the library
        lib_docs = _get_library_docs()
        stats_by_name = {s['filename']: s for s in stats}
        merged = []
        for name in lib_docs:
            if name in stats_by_name:
                merged.append(stats_by_name[name])
            else:
                merged.append({
                    'filename': name,
                    'total_pages': 0,
                    'cached_pages': 0,
                    'native_pages': 0,
                    'ocr_pages': 0,
                    'percent_done': 0.0,
                    'is_scan_heavy': False,
                })
        self._stats_cache = merged

        # Populate table
        self.stats_table.setRowCount(len(merged))
        for row, s in enumerate(merged):
            self.stats_table.setItem(row, 0, QTableWidgetItem(s['filename']))
            self.stats_table.setItem(row, 1, QTableWidgetItem(str(s['total_pages'])))
            self.stats_table.setItem(row, 2, QTableWidgetItem(str(s['cached_pages'])))
            self.stats_table.setItem(row, 3, QTableWidgetItem(str(s['native_pages'])))
            self.stats_table.setItem(row, 4, QTableWidgetItem(str(s['ocr_pages'])))
            pct = f"{s['percent_done']:.1f}%"
            self.stats_table.setItem(row, 5, QTableWidgetItem(pct))

        # Repopulate combobox preserving current selection
        current_text = self.doc_combo.currentText()
        self.doc_combo.blockSignals(True)
        self.doc_combo.clear()
        for s in merged:
            label = (
                f"{s['filename']} "
                f"[{s['percent_done']:.0f}% кэш, {s['cached_pages']}/{s['total_pages']} стр.]"
            )
            self.doc_combo.addItem(label)
        self.doc_combo.blockSignals(False)

        # Restore previous selection if possible
        for i in range(self.doc_combo.count()):
            if self.doc_combo.itemText(i) == current_text:
                self.doc_combo.setCurrentIndex(i)
                break

        self._update_reindex_btn()

    def _current_stat(self) -> Optional[dict]:
        idx = self.doc_combo.currentIndex()
        if 0 <= idx < len(self._stats_cache):
            return self._stats_cache[idx]
        return None

    def _update_reindex_btn(self) -> None:
        stat = self._current_stat()
        has_cache = stat is not None and stat['cached_pages'] > 0
        self.reindex_btn.setEnabled(has_cache and self._ocr_worker is None)

    # ------------------------------------------------------------------
    # OCR actions
    # ------------------------------------------------------------------

    def _on_run_ocr(self) -> None:
        stat = self._current_stat()
        if stat is None:
            return

        from windows.app_paths import library_dir
        doc_path = library_dir() / stat['filename']
        if not doc_path.exists():
            QMessageBox.warning(
                self, "Файл не найден",
                f"PDF не найден:\n{doc_path}"
            )
            return

        self._stop_flag = [False]
        self._ocr_worker = OcrWorker(doc_path, CACHE_PATH, self._stop_flag)
        self._ocr_worker.progress.connect(self._on_progress)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.error.connect(self._on_ocr_error)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.reindex_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.page_label.setText("Страница 0 / ?")
        self.stats_label.setText("Нативных: 0 | OCR: 0 | Прошло: 0s | ETA: —")
        self.status_label.setText(f"Запущен OCR: {stat['filename']}")

        self._ocr_worker.start()
        # Subtle fade-in on the progress group to draw attention
        from windows.anim import fade_in
        fade_in(self._prog_group, duration=200)

    def _on_stop(self) -> None:
        self._stop_flag[0] = True
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Останавливаем OCR...")

    def _on_progress(self, event: dict) -> None:
        total = event.get('total_pages', 0)
        done = event.get('done_count', 0)
        elapsed = event.get('elapsed_s', 0.0)
        native = event.get('native_count', 0)
        ocr = event.get('ocr_count', 0)

        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(done)
        self.page_label.setText(f"Страница {done} / {total}")

        if done > 0 and elapsed > 0:
            eta = elapsed / done * (total - done)
            eta_str = f"{eta:.0f}s"
        else:
            eta_str = "—"

        self.stats_label.setText(
            f"Нативных: {native} | OCR: {ocr} | Прошло: {elapsed:.0f}s | ETA: {eta_str}"
        )

    def _on_ocr_finished(self, summary: dict) -> None:
        self._ocr_worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if summary.get('stopped'):
            self.status_label.setText(
                f"OCR остановлен. Обработано: {summary.get('ocr_count', 0) + summary.get('native_count', 0)} стр."
            )
        else:
            self.status_label.setText(
                f"OCR завершён: {summary.get('doc', '')} — "
                f"{summary.get('ocr_count', 0)} OCR, {summary.get('native_count', 0)} нативных, "
                f"за {summary.get('elapsed_s', 0):.1f}s"
            )

        self._refresh_stats()
        self._update_reindex_btn()

    def _on_ocr_error(self, msg: str) -> None:
        self._ocr_worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Ошибка OCR: {msg}")
        QMessageBox.critical(self, "Ошибка OCR", msg)

    # ------------------------------------------------------------------
    # Re-index actions
    # ------------------------------------------------------------------

    def _on_reindex(self) -> None:
        stat = self._current_stat()
        if stat is None:
            return

        from windows.app_paths import library_dir
        pdf_path = library_dir() / stat['filename']
        self._reindex_worker = ReindexWorker(pdf_path, CACHE_PATH)
        self._reindex_worker.finished.connect(self._on_reindex_finished)
        self._reindex_worker.error.connect(self._on_reindex_error)

        self.reindex_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.status_label.setText(f"Переиндексирование: {stat['filename']}...")
        self._reindex_worker.start()

    def _on_reindex_finished(self, doc_name: str) -> None:
        self._reindex_worker = None
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Переиндексирование завершено: {doc_name}")
        self._update_reindex_btn()

    def _on_reindex_error(self, msg: str) -> None:
        self._reindex_worker = None
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Ошибка переиндексирования: {msg}")
        QMessageBox.critical(self, "Ошибка переиндексирования", msg)

    # ------------------------------------------------------------------
    # Convenience classmethod
    # ------------------------------------------------------------------

    @classmethod
    def open_dialog(cls, parent=None) -> "OCRDialog":
        """Open an OCRDialog and return it (caller should call exec())."""
        dlg = cls(parent)
        return dlg
