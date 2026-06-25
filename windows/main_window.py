"""Windows Desktop UI: PySide6 RAG companion.

Layout: single window, left sidebar (sources list + history controls),
right area split vertically:
- top: chat log with bookmark/filter bar
- bottom: SINGLE unified ask input (replaces the old dual search_edit + chat_input)

Phase 1 changes:
- Min size lowered to 960x640 (was 1200x800)
- Unified single ask input (chat_input at bottom is the one true entry point)
- search_edit / search_btn kept as shims pointing to chat_input / send_btn for
  backward-compat with existing tests
- ThemeManager wired at startup (default DARK; persists across runs)
- "Вид -> Тема" menu for live dark/light switching without restart
- app_paths.ensure_seeded() + app_paths.db_path() used for default DB

Phase 2 changes:
- "Файл → Библиотека и индекс…" opens LibraryDialog
- Status bar shows "Документов: N · Чанков: M" (refreshed on startup + dialog close)
- Retriever pipeline reset helper after library changes

Phase 3 changes:
- Source links in chat_log are clickable (source: custom URL scheme)
- anchorClicked handler opens PageViewer (windows/pdf_render + windows/page_viewer)
- Sidebar results_list double-click opens PageViewer
- PageViewer: zoom, prev/next, open-original, OCR high-DPI default, off-thread rendering
"""

import html
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QTextBrowser, QSplitter, QVBoxLayout, QWidget,
)

# Allow running from repo root without install
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from core.query import RAGQueryPipeline  # noqa: E402
from design.tokens import DribbbleDarkQt  # noqa: E402  (backward-compat import kept)
from windows.app_paths import db_path as _default_db_path, ensure_seeded  # noqa: E402
from windows.settings_dialog import SettingsDialog, load_settings, save_settings  # noqa: E402
from windows.theme import ThemeManager  # noqa: E402
from windows.anim import fade_in, fade_out  # noqa: E402


def _markdownish_to_html(text: str) -> str:
    """Lightweight conversion: bold, bullet lists, newlines."""
    text = html.escape(text)
    text = re.sub(r"\*(.+?)\*", r"<b>\1</b>", text)
    lines = text.split("\n")
    out: list[str] = []
    in_list = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("* ") or stripped.startswith("- "):
            bullet_text = html.escape(stripped[2:])
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{bullet_text}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(line)
    if in_list:
        out.append("</ul>")
    return "<br>".join(out)


class ChatHistory:
    """Persistent chat history as JSON lines."""

    def __init__(self, path: Path, max_entries: int = 200) -> None:
        self.path = path
        self.max_entries = max_entries
        self.entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.entries = []

    def add(self, query: str, answer: str, sources: list[dict]) -> None:
        self.entries.append({"query": query, "answer": answer, "sources": sources, "bookmarked": False})
        self._prune()
        self.save()

    def toggle_bookmark(self, idx: int) -> bool:
        if 0 <= idx < len(self.entries):
            entry = self.entries[idx]
            entry["bookmarked"] = not entry.get("bookmarked", False)
            self.save()
            return entry["bookmarked"]
        return False

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        self.entries = []
        self.save()

    def prune(self) -> None:
        """Remove oldest non-bookmarked entries if over limit."""
        while len(self.entries) > self.max_entries:
            for i, e in enumerate(self.entries):
                if not e.get("bookmarked", False):
                    self.entries.pop(i)
                    break
            else:
                self.entries.pop(0)

    def _prune(self) -> None:
        self.prune()

    def format_html(self, bookmarks_only: bool = False, search_term: str = "") -> str:
        parts = []
        for i, e in enumerate(self.entries):
            if bookmarks_only and not e.get("bookmarked", False):
                continue
            if search_term:
                q = e.get("query", "").lower()
                a = e.get("answer", "").lower()
                st = search_term.lower()
                if st not in q and st not in a:
                    continue
            star = "★" if e.get("bookmarked") else "☆"
            parts.append(
                f'<hr><div style="color:#7A7A7A;font-size:12px;margin-bottom:4px;">'
                f'<a href="bookmark://{i}" style="text-decoration:none;color:#FFD700;font-size:14px;">{star}</a> '
                f'Вопрос: {html.escape(e["query"])}</div>'
            )
            parts.append(f'<div style="margin-bottom:8px;">{_markdownish_to_html(e["answer"])}</div>')
            parts.append('<div style="color:#7A7A7A;font-size:11px;">Источники:</div>')
            for src in e.get("sources", []):
                # Build a source: link so user can click to open PDF page viewer
                # Format: source:{url-encoded filename}|{page}|{ocr_flag}
                section = src.get("section", "")
                is_ocr = "1" if section.startswith("[OCR] ") else "0"
                src_href = f"source:{quote(src['source'])}|{src['page']}|{is_ocr}"
                parts.append(
                    f'<div style="font-size:11px;margin-left:8px;">'
                    f'• <a href="{src_href}" style="color:#F5C518;text-decoration:underline;">'
                    f'{html.escape(src["source"])} стр.{src["page"]}</a>'
                    f' — <span style="color:#7A7A7A;">{html.escape(section)}</span></div>'
                )
        return "".join(parts)


class QueryWorker(QThread):
    """Off-thread RAG query so UI stays responsive."""

    result_ready = Signal(dict)
    error = Signal(str)

    def __init__(self, pipeline: RAGQueryPipeline, query: str) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.query = query

    def run(self) -> None:
        try:
            result = self.pipeline.ask(self.query)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class CompanionWindow(QMainWindow):
    def __init__(self, pipeline: RAGQueryPipeline | None = None) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.worker: QueryWorker | None = None
        self._meta_pulse_going: bool = False
        self.history = ChatHistory(Path.home() / ".engineer-companion" / "history.json")
        self.setWindowTitle("Engineer Companion")
        self.setMinimumSize(960, 640)   # Phase 1: reduced from 1200x800
        self.settings = load_settings()

        # Theme manager — persists preference; defaults to DARK
        self._theme_manager = ThemeManager()

        self._setup_menu()
        self._build_ui()
        self._apply_tokens()
        self._apply_startup_appearance()
        self._restore_history()
        # Phase 2: populate index status bar on startup
        self._refresh_index_status()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("Файл")
        export_action = file_menu.addAction("Экспорт в Markdown...")
        export_action.triggered.connect(self._on_export_chat)
        file_menu.addSeparator()
        settings_action = file_menu.addAction("Настройки...")
        settings_action.triggered.connect(self._on_open_settings)
        settings_action.setShortcut("Ctrl+,")
        file_menu.addSeparator()
        library_action = file_menu.addAction("Библиотека и индекс...")
        library_action.triggered.connect(self._on_open_library)
        library_action.setShortcut("Ctrl+L")
        file_menu.addSeparator()
        ocr_action = file_menu.addAction("OCR и индексация...")
        from windows.ocr_dialog import ocr_available
        if ocr_available():
            ocr_action.triggered.connect(self._on_open_ocr)
        else:
            ocr_action.setEnabled(False)
            ocr_action.setToolTip(
                "OCR доступен только при запуске из исходников (.venv-win). "
                "В собранном EXE-файле OCR отключён — запустите из источника для индексации."
            )

        # View menu — theme switcher
        view_menu = menubar.addMenu("Вид")
        theme_menu = view_menu.addMenu("Тема")
        dark_action = theme_menu.addAction("Тёмная (Dark)")
        dark_action.triggered.connect(lambda: self._on_set_theme("dark"))
        light_action = theme_menu.addAction("Светлая (Light)")
        light_action.triggered.connect(lambda: self._on_set_theme("light"))

    # ------------------------------------------------------------------
    # Theme switching
    # ------------------------------------------------------------------

    def _on_set_theme(self, name: str) -> None:
        """Switch theme with a subtle crossfade of the central widget.

        Falls back to an immediate switch if the central widget isn't visible
        (e.g. headless tests, or called before the window is shown) so that the
        stylesheet is always applied synchronously in that case.
        """
        app = QApplication.instance()
        central = self.centralWidget()
        if central is not None and central.isVisible():
            def _do_switch() -> None:
                self._theme_manager.set_theme(name, app)
                fade_in(central, duration=200)
            fade_out(central, duration=120, on_done=_do_switch)
        else:
            self._theme_manager.set_theme(name, app)

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def _on_open_library(self) -> None:
        from windows.library_dialog import LibraryDialog
        dlg = LibraryDialog(self, db_path=_default_db_path())
        dlg.exec()
        # Refresh status bar + reset pipeline retriever after any library changes
        self._refresh_index_status()
        self._reset_pipeline_retriever()

    def _rebuild_pipeline(self) -> None:
        """Rebuild RAGQueryPipeline from current self.settings."""
        from windows.settings_dialog import DEFAULT_RERANK_MODEL
        db = self.settings.get("db_path") or ""
        llm_path = self.settings.get("llm_model_path") or ""
        if db and not Path(db).exists():
            db = ""
        if llm_path and not Path(llm_path).exists():
            llm_path = ""

        rerank_enabled = self.settings.get("rerank_enabled", True)
        rerank_model = DEFAULT_RERANK_MODEL if rerank_enabled else ""

        if llm_path:
            try:
                self.pipeline = RAGQueryPipeline(
                    db_path=Path(db) if db else _default_db_path(),
                    llm_model_path=Path(llm_path),
                    top_k=self.settings.get("top_k", 8),
                    rerank_top_k=self.settings.get("rerank_top_k", 5),
                    rerank_model=rerank_model,
                    max_tokens=self.settings.get("max_tokens", 512),
                    temperature=self.settings.get("temperature", 0.3),
                    llm_n_ctx=self.settings.get("n_ctx", 2048),
                    llm_n_threads=self.settings.get("n_threads", 2),
                )
            except Exception as exc:
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить модель:\n{exc}")
                self.pipeline = None
        else:
            self.pipeline = None

    def _on_open_ocr(self) -> None:
        from windows.ocr_dialog import OCRDialog
        dlg = OCRDialog(self)
        dlg.exec()

    def _on_open_settings(self) -> None:
        from windows.settings_dialog import apply_font_scale, apply_density
        dlg = SettingsDialog(self, current=self.settings, theme_manager=self._theme_manager)
        if dlg.exec():
            self.settings = dlg.get_settings()
            self._rebuild_pipeline()
            # Apply appearance settings live
            app = QApplication.instance()
            if app is not None:
                apply_font_scale(app, self.settings.get("font_scale", 100))
                apply_density(app, self.settings.get("density", "comfortable"))

    def _on_export_chat(self) -> None:
        if not self.history.entries:
            QMessageBox.information(self, "Экспорт", "Нет записей для экспорта.")
            return
        bookmarks_only = self.bookmarks_cb.isChecked() if hasattr(self, "bookmarks_cb") else False
        search_term = self.history_search.text().strip() if hasattr(self, "history_search") else ""
        entries = self.history.entries
        if bookmarks_only or search_term:
            entries = [
                e for e in entries
                if (not bookmarks_only or e.get("bookmarked", False))
                and (not search_term or search_term.lower() in e.get("query", "").lower()
                     or search_term.lower() in e.get("answer", "").lower())
            ]
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить чат как Markdown", "engineer-companion-chat.md",
            "Markdown (*.md);;JSON (*.json);;Все файлы (*)"
        )
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        else:
            lines = ["# Engineer Companion — Экспорт чата\n"]
            for e in entries:
                star = "★" if e.get("bookmarked") else " "
                lines.append(f"## {star} Вопрос: {e['query']}\n")
                lines.append(f"{e['answer']}\n")
                for src in e.get("sources", []):
                    lines.append(f"- {src['source']}, стр.{src['page']} — {src['section']}\n")
                lines.append("\n")
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        QMessageBox.information(self, "Экспорт", f"Сохранено: {path} ({len(entries)} записей)")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ---- Left sidebar: sources panel + history controls ----
        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(8)

        sources_label = QLabel("Источники")
        sources_label.setObjectName("muted")
        left_v.addWidget(sources_label)

        self.results_list = QListWidget()
        self.results_list.setSpacing(4)
        self.results_list.itemClicked.connect(self._on_result_clicked)
        self.results_list.itemDoubleClicked.connect(self._on_result_double_clicked)
        left_v.addWidget(self.results_list, stretch=1)

        self.meta_label = QLabel()
        self.meta_label.setObjectName("muted")
        left_v.addWidget(self.meta_label)

        self.clear_btn = QPushButton("🗑 Очистить историю")
        self.clear_btn.clicked.connect(self._on_clear_history)
        left_v.addWidget(self.clear_btn)

        self.prune_btn = QPushButton("✂️ Удалить старые (>200)")
        self.prune_btn.clicked.connect(self._on_prune_history)
        left_v.addWidget(self.prune_btn)

        # ---- Right area: chat log + unified ask input ----
        right = QSplitter(Qt.Vertical)

        # Top: chat log with filter bar
        chat_top = QWidget()
        chat_top_v = QVBoxLayout(chat_top)
        chat_top_v.setContentsMargins(0, 0, 0, 0)
        chat_top_v.setSpacing(4)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        self.bookmarks_cb = QCheckBox("★ Только избранное")
        self.bookmarks_cb.toggled.connect(self._refresh_chat_log)
        filter_row.addWidget(self.bookmarks_cb)

        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("Поиск по истории...")
        self.history_search.textChanged.connect(self._refresh_chat_log)
        filter_row.addWidget(self.history_search, stretch=1)
        chat_top_v.addLayout(filter_row)

        self.chat_log = QTextBrowser()
        self.chat_log.setOpenExternalLinks(False)
        self.chat_log.setOpenLinks(False)  # suppress navigation; anchorClicked handles all links
        self.chat_log.anchorClicked.connect(self._on_anchor_clicked)
        self.chat_log.setHtml(
            '<div style="color:#7A7A7A;">'
            "Здесь появится ответ эксперта...<br>"
            "Введите вопрос внизу и нажмите Отправить."
            "</div>"
        )
        chat_top_v.addWidget(self.chat_log, stretch=1)
        right.addWidget(chat_top)

        # Bottom: UNIFIED single ask input
        bottom = QWidget()
        bottom_h = QHBoxLayout(bottom)
        bottom_h.setContentsMargins(0, 0, 0, 0)
        bottom_h.setSpacing(8)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Введите вопрос по TrueBeam / VitalBeam...")
        self.chat_input.returnPressed.connect(self._on_search)

        self.send_btn = QPushButton("Отправить")
        self.send_btn.setShortcut("Ctrl+Return")
        self.send_btn.clicked.connect(self._on_search)

        bottom_h.addWidget(self.chat_input, stretch=1)
        bottom_h.addWidget(self.send_btn)
        right.addWidget(bottom)
        right.setSizes([560, 80])

        outer_splitter = QSplitter(Qt.Horizontal)
        outer_splitter.addWidget(left)
        outer_splitter.addWidget(right)
        outer_splitter.setSizes([300, 660])
        root.addWidget(outer_splitter)

        # ---- Phase 2: index status indicator (status bar) ----
        self.index_status_label = QLabel("Документов: — · Чанков: —")
        self.index_status_label.setObjectName("muted")
        self.statusBar().addPermanentWidget(self.index_status_label)

        # ---- Backward-compat shims ----
        # Tests reference window.search_edit and window.search_btn;
        # wire them to the unified widgets so nothing breaks.
        self.search_edit = self.chat_input
        self.search_btn = self.send_btn

    # ------------------------------------------------------------------
    # Theme / stylesheet
    # ------------------------------------------------------------------

    def _apply_tokens(self) -> None:
        """Apply active theme via ThemeManager to the QApplication."""
        app = QApplication.instance()
        if app is not None:
            self._theme_manager.apply(app)
        else:
            from design.tokens import as_stylesheet
            self.setStyleSheet(as_stylesheet(self._theme_manager.get_theme_obj()))

    def _apply_startup_appearance(self) -> None:
        """Apply font scale and density from persisted settings at startup."""
        from windows.settings_dialog import apply_font_scale, apply_density
        app = QApplication.instance()
        if app is not None:
            apply_font_scale(app, self.settings.get("font_scale", 100))
            apply_density(app, self.settings.get("density", "comfortable"))

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _restore_history(self) -> None:
        if self.history.entries:
            self.chat_log.setHtml(self.history.format_html())

    # ------------------------------------------------------------------
    # Query flow
    # ------------------------------------------------------------------

    def _start_meta_pulse(self) -> None:
        """Subtle repeating opacity pulse on meta_label while query is running."""
        self._meta_pulse_anim = fade_in(
            self.meta_label, duration=600, start_value=0.4, end_value=1.0
        )
        self._meta_pulse_going = True

        def _pulse_again() -> None:
            if not self._meta_pulse_going:
                return
            self._meta_pulse_anim = fade_out(
                self.meta_label, duration=600, start_value=1.0, end_value=0.4,
                on_done=_pulse_in,
            )

        def _pulse_in() -> None:
            if not self._meta_pulse_going:
                return
            self._meta_pulse_anim = fade_in(
                self.meta_label, duration=600, start_value=0.4, end_value=1.0
            )
            self._meta_pulse_anim.finished.connect(_pulse_again)

        self._meta_pulse_anim.finished.connect(_pulse_again)

    def _stop_meta_pulse(self) -> None:
        """Stop pulse and restore meta_label to full opacity."""
        self._meta_pulse_going = False
        # Restore opacity immediately without animation jitter
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        effect = self.meta_label.graphicsEffect()
        if isinstance(effect, QGraphicsOpacityEffect):
            effect.setOpacity(1.0)

    def _on_search(self) -> None:
        text = self.chat_input.text().strip()
        if not text:
            return
        if self.pipeline is None:
            self.chat_log.append("[Ошибка: LLM pipeline не инициализирован]")
            return

        self.send_btn.setEnabled(False)
        self.chat_input.clear()
        self.results_list.clear()
        self.meta_label.setText("Ищем...")
        self._start_meta_pulse()

        self.worker = QueryWorker(self.pipeline, text)
        self._current_query = text
        self.worker.result_ready.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def _on_result(self, result: dict[str, Any]) -> None:
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        self._stop_meta_pulse()

        self.results_list.clear()
        for src in sources:
            item_text = f"{src['source']}    стр.{src['page']}    [{src['section'][:40]}]"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, json.dumps(src))
            self.results_list.addItem(item)

        self.meta_label.setText(f"Найдено источников: {len(sources)}")

        source_meta = [{"source": s["source"], "page": s["page"], "section": s["section"]} for s in sources]
        self.history.add(self._current_query, answer, source_meta)
        # Update HTML immediately (keeps tests deterministic), then fade in
        # the chat log so the new answer appears with a subtle enter animation.
        # (Enter-only: no fade-out before setHtml — avoids async timing issues.)
        self._update_chat_log_html()

    def _update_chat_log_html(self) -> None:
        """Set chat log HTML then fade in — enter-only animation."""
        self.chat_log.setHtml(self.history.format_html())
        fade_in(self.chat_log, duration=200)

    def _on_error(self, msg: str) -> None:
        self._stop_meta_pulse()
        self.chat_log.append(f'[Ошибка: {html.escape(msg)}]')
        self.meta_label.setText("Ошибка")

    def _cleanup_worker(self) -> None:
        self.send_btn.setEnabled(True)
        self.worker = None

    # ------------------------------------------------------------------
    # Phase 2: index status helpers
    # ------------------------------------------------------------------

    def _refresh_index_status(self) -> None:
        """Update the status bar label with current DB chunk counts."""
        try:
            from core.indexer import index_stats
            stats = index_stats(_default_db_path())
            n_docs = len(stats["per_doc"])
            n_chunks = stats["total_chunks"]
            self.index_status_label.setText(
                f"Документов: {n_docs} · Чанков: {n_chunks}"
            )
        except Exception:
            self.index_status_label.setText("Документов: — · Чанков: —")

    def _reset_pipeline_retriever(self) -> None:
        """Force a fresh LanceDB table handle on the next query.

        If the pipeline caches a table object (lancedb reads are lazy),
        clearing the cached handle ensures new chunks are visible immediately.
        """
        if self.pipeline is None:
            return
        try:
            retriever = getattr(self.pipeline, "retriever", None)
            if retriever is not None:
                # Clear the cached table so it is re-opened on the next search
                if hasattr(retriever, "_table"):
                    retriever._table = None
                if hasattr(retriever, "_db"):
                    retriever._db = None
        except Exception:
            pass

    def _on_result_clicked(self, item: QListWidgetItem) -> None:
        data = json.loads(item.data(Qt.UserRole))
        self.chat_log.append(
            f'<div style="color:#B2BAB6;font-size:12px;">'
            f'[Выбран источник] {html.escape(data["source"])} '
            f'стр.{data["page"]} — {html.escape(data["section"])}'
            f'</div>'
        )

    def _on_result_double_clicked(self, item: QListWidgetItem) -> None:
        """Open PDF page viewer for the double-clicked source in the sidebar."""
        data = json.loads(item.data(Qt.UserRole))
        filename = data.get("source", "")
        page = data.get("page", 1)
        section = data.get("section", "")
        is_ocr = section.startswith("[OCR] ")
        self._open_page_viewer(filename, page, is_ocr)

    def _open_page_viewer(self, filename: str, page: int, is_ocr: bool = False) -> None:
        """Resolve the PDF, then open PageViewer dialog."""
        from windows.pdf_render import resolve_pdf
        from windows.page_viewer import PageViewer
        pdf_path = resolve_pdf(filename)
        viewer = PageViewer(
            pdf_path=pdf_path,
            page_number=page,
            source_name=filename,
            parent=self,
            is_ocr=is_ocr,
        )
        viewer.exec()

    def _on_anchor_clicked(self, url: QUrl) -> None:
        if url.scheme() == "bookmark":
            idx = int(url.host())
            self.history.toggle_bookmark(idx)
            self.chat_log.setHtml(self.history.format_html())
        elif url.scheme() == "source":
            # Parse: source:{encoded_filename}|{page}|{ocr_flag}
            # QUrl.path() decodes percent-encoding and preserves pipe separators
            path = url.path()
            parts = path.split("|")
            if len(parts) >= 2:
                filename = parts[0]
                try:
                    page = int(parts[1])
                except ValueError:
                    page = 1
                is_ocr = len(parts) >= 3 and parts[2] == "1"
                self._open_page_viewer(filename, page, is_ocr)

    def _on_clear_history(self) -> None:
        self.history.clear()
        self.chat_log.setHtml(
            '<div style="color:#7A7A7A;">'
            "История очищена.<br>"
            "Введите вопрос внизу и нажмите Отправить."
            "</div>"
        )

    def _on_prune_history(self) -> None:
        old = len(self.history.entries)
        self.history.prune()
        new = len(self.history.entries)
        self.chat_log.append(
            f'<div style="color:#7A7A7A;">Удалено {old - new} старых записей. Осталось: {new}</div>'
        )

    def _refresh_chat_log(self) -> None:
        bookmarks_only = self.bookmarks_cb.isChecked()
        search_term = self.history_search.text().strip()
        content = self.history.format_html(bookmarks_only=bookmarks_only, search_term=search_term)
        if content:
            self.chat_log.setHtml(content)
        else:
            self.chat_log.setHtml('<div style="color:#7A7A7A;">Нет записей, соответствующих фильтру.</div>')


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=Path, default=None,
                    help="Override DB path (default: app_paths.db_path())")
    ap.add_argument("--llm-path", type=Path,
                    default=_REPO / "assets" / "models" / "gemma-3-4b-it-Q4_K_M.gguf")
    ap.add_argument("--no-llm", action="store_true", help="Run without LLM (search-only mode)")
    args = ap.parse_args()

    # Seed writable data dirs on first frozen run; no-op in dev
    ensure_seeded()

    effective_db = args.db_path or _default_db_path()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pipeline = None
    if not args.no_llm and args.llm_path.exists():
        pipeline = RAGQueryPipeline(db_path=effective_db, llm_model_path=args.llm_path)

    w = CompanionWindow(pipeline=pipeline)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
