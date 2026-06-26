"""Windows Desktop UI: PySide6 RAG companion — chat-centric redesign.

Layout: left rail (conversations list + navigation) + center thread view +
bottom composer. History persisted as ConversationStore (conversations.json).
Phase 5 changes:
- Left rail: per-conversation history, favorites, library, settings
- Center: bubble-based chat thread (user right, assistant left)
- Assistant bubbles: streaming token updates, source cards, star/copy actions
- ConversationStore: multi-conversation JSON persistence
- QueryWorker: passes history to pipeline for multi-turn context
- Compat shims: chat_log, results_list, meta_label for backward-compat tests
"""

import html
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

# Allow running from repo root without install
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from core.query import RAGQueryPipeline  # noqa: E402
from design.tokens import DribbbleDarkQt  # noqa: E402  (backward-compat import kept)
from windows.app_paths import db_path as _default_db_path, ensure_seeded, data_root  # noqa: E402
from windows.settings_dialog import SettingsDialog, load_settings  # noqa: E402
from windows.theme import ThemeManager  # noqa: E402
from windows.anim import fade_in, fade_out  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Compat shims
# ---------------------------------------------------------------------------

class _StreamingChatLogCompat:
    """Shim so tests that check window.chat_log.toHtml() still work."""
    def __init__(self):
        self._html = ""

    def toHtml(self):
        return self._html

    def toPlainText(self):
        return re.sub(r'<[^>]+>', '', self._html)

    def setHtml(self, h):
        self._html = h

    def append(self, t):
        self._html += t


class _ResultsListCompat:
    """Shim for tests that check results_list.count()."""
    def __init__(self):
        self._items = []

    def count(self):
        return len(self._items)

    def clear(self):
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def item(self, idx):
        return self._items[idx] if 0 <= idx < len(self._items) else None


# ---------------------------------------------------------------------------
# ChatHistory (legacy — kept for export/tests)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ConversationStore
# ---------------------------------------------------------------------------

class ConversationStore:
    def __init__(self):
        self._path = data_root() / "conversations.json"
        self.conversations: list[dict] = []
        self._load()

    def _load(self):
        try:
            if self._path.exists():
                with self._path.open("r", encoding="utf-8") as f:
                    self.conversations = json.load(f)
        except Exception:
            self.conversations = []

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        tmp.replace(self._path)

    def new_conversation(self) -> dict:
        conv = {
            "id": str(uuid.uuid4()),
            "title": "Новый диалог",
            "created": datetime.now().isoformat(),
            "turns": [],
        }
        self.conversations.insert(0, conv)
        self.save()
        return conv

    def get_all(self) -> list[dict]:
        return sorted(self.conversations, key=lambda c: c.get("created", ""), reverse=True)

    def get_conversation(self, conv_id: str) -> dict | None:
        for c in self.conversations:
            if c["id"] == conv_id:
                return c
        return None

    def update_turn_star(self, conv_id: str, turn_idx: int, starred: bool) -> None:
        conv = self.get_conversation(conv_id)
        if conv and 0 <= turn_idx < len(conv["turns"]):
            conv["turns"][turn_idx]["starred"] = starred
            self.save()


# ---------------------------------------------------------------------------
# QueryWorker
# ---------------------------------------------------------------------------

class QueryWorker(QThread):
    """Off-thread RAG query so UI stays responsive."""

    result_ready = Signal(dict)
    event_received = Signal(dict)
    error = Signal(str)

    def __init__(self, pipeline, query: str, history=None) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.query = query
        self.history = history

    def run(self) -> None:
        try:
            if hasattr(self.pipeline, "ask_streaming"):
                try:
                    result = self.pipeline.ask_streaming(self.query, self._emit_event, history=self.history)
                except TypeError:
                    # Pipeline doesn't support history kwarg — fall back
                    result = self.pipeline.ask_streaming(self.query, self._emit_event)
            else:
                try:
                    result = self.pipeline.ask(self.query, history=self.history)
                except TypeError:
                    # Pipeline doesn't support history kwarg — fall back
                    result = self.pipeline.ask(self.query)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

    def _emit_event(self, event: dict) -> None:
        self.event_received.emit(event)


# ---------------------------------------------------------------------------
# CompanionWindow
# ---------------------------------------------------------------------------

class CompanionWindow(QMainWindow):
    def __init__(self, pipeline=None) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.worker: QueryWorker | None = None
        self._dot_pulse_going: bool = False
        self._meta_pulse_going: bool = False
        self._streaming_answer: str = ""
        self._current_query: str = ""
        self._favorites_mode: bool = False
        self._current_conv: dict | None = None
        self._current_bubble_label: QLabel | None = None
        self._current_bubble_container: QWidget | None = None
        self._current_bubble_sources_layout = None
        self._current_star_btn: QPushButton | None = None
        self._current_copy_btn: QPushButton | None = None
        self._current_turn_idx: int = 0

        # Compat shims
        self.chat_log = _StreamingChatLogCompat()
        self.results_list = _ResultsListCompat()
        self.meta_label = QLabel()
        self.meta_label.setObjectName("muted")

        self.settings = load_settings()
        self._theme_manager = ThemeManager()
        self.conv_store = ConversationStore()

        # Legacy history for export/tests
        self.history = ChatHistory(Path.home() / ".engineer-companion" / "history.json")

        self.setWindowTitle("Engineer Companion")
        self.setMinimumSize(960, 640)

        self._setup_menu()
        self._build_ui()
        self._apply_tokens()
        self._apply_startup_appearance()
        self._refresh_index_status()

        # Load most recent conversation or start new
        convs = self.conv_store.get_all()
        if convs:
            self._load_conversation(convs[0]["id"])
        else:
            self._start_new_conversation()
        self._refresh_conv_list()

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
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Left rail ----
        rail = QWidget()
        rail.setObjectName("rail")
        rail.setFixedWidth(170)
        rail_v = QVBoxLayout(rail)
        rail_v.setContentsMargins(8, 8, 8, 8)
        rail_v.setSpacing(4)

        new_chat_btn = QPushButton("Новый чат")
        new_chat_btn.setProperty("primary", True)
        new_chat_btn.clicked.connect(self._start_new_conversation)
        rail_v.addWidget(new_chat_btn)

        conv_label = QLabel("Диалоги")
        conv_label.setObjectName("muted")
        rail_v.addWidget(conv_label)

        self.conv_list = QListWidget()
        self.conv_list.itemClicked.connect(self._on_conv_item_clicked)
        rail_v.addWidget(self.conv_list, stretch=1)

        self.favorites_btn = QPushButton("★ Избранное")
        self.favorites_btn.setObjectName("railBtn")
        self.favorites_btn.clicked.connect(self._on_favorites_clicked)
        rail_v.addWidget(self.favorites_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("muted")
        rail_v.addWidget(sep)

        self.library_btn = QPushButton("Библиотека")
        self.library_btn.setObjectName("railBtn")
        self.library_btn.clicked.connect(self._on_open_library)
        rail_v.addWidget(self.library_btn)

        self.settings_btn = QPushButton("Настройки")
        self.settings_btn.setObjectName("railBtn")
        self.settings_btn.clicked.connect(self._on_open_settings)
        rail_v.addWidget(self.settings_btn)

        root.addWidget(rail)

        # ---- Center area ----
        center = QWidget()
        center.setObjectName("center")
        center_v = QVBoxLayout(center)
        center_v.setContentsMargins(0, 0, 0, 0)
        center_v.setSpacing(0)

        # Thread scroll area
        self.thread_scroll = QScrollArea()
        self.thread_scroll.setWidgetResizable(True)
        self.thread_scroll.setObjectName("threadScroll")

        self.thread_container = QWidget()
        self.thread_layout = QVBoxLayout(self.thread_container)
        self.thread_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.thread_layout.setSpacing(12)
        self.thread_layout.setContentsMargins(16, 16, 16, 16)

        self.thread_scroll.setWidget(self.thread_container)
        center_v.addWidget(self.thread_scroll, stretch=1)

        # Status row
        self.status_row = QWidget()
        self.status_row.hide()
        status_h = QHBoxLayout(self.status_row)
        status_h.setContentsMargins(16, 4, 16, 4)
        status_h.setSpacing(6)

        self._status_dot = QLabel("●")
        self._status_dot.setObjectName("statusDot")
        self._status_dot.setFixedWidth(14)
        self._status_dot.hide()

        self._status_label = QLabel("")
        self._status_label.setObjectName("muted")
        self._status_label.hide()

        status_h.addWidget(self._status_dot)
        status_h.addWidget(self._status_label, stretch=1)
        center_v.addWidget(self.status_row)

        # Composer
        composer = QWidget()
        composer.setObjectName("composer")
        composer_h = QHBoxLayout(composer)
        composer_h.setContentsMargins(12, 8, 12, 12)
        composer_h.setSpacing(8)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Уточните или задайте следующий вопрос…")
        self.chat_input.returnPressed.connect(self._on_search)

        self.send_btn = QPushButton("Отправить")
        self.send_btn.setProperty("primary", True)
        self.send_btn.setShortcut("Ctrl+Return")
        self.send_btn.clicked.connect(self._on_search)

        composer_h.addWidget(self.chat_input, stretch=1)
        composer_h.addWidget(self.send_btn)
        center_v.addWidget(composer)

        root.addWidget(center, stretch=1)

        # Backward compat aliases
        self.search_edit = self.chat_input
        self.search_btn = self.send_btn

        # Status bar
        self.index_status_label = QLabel("Документов: — · Чанков: —")
        self.index_status_label.setObjectName("muted")
        self.statusBar().addPermanentWidget(self.index_status_label)

    # ------------------------------------------------------------------
    # Theme / stylesheet
    # ------------------------------------------------------------------

    def _apply_tokens(self) -> None:
        app = QApplication.instance()
        t = self._theme_manager.get_theme_obj()
        if app is not None:
            self._theme_manager.apply(app)
        else:
            from design.tokens import as_stylesheet
            self.setStyleSheet(as_stylesheet(t))

        extra_qss = f"""
QWidget#rail {{
    background-color: {t.bg_surface};
    border-right: 1px solid {t.border};
    border-radius: 0;
}}
QWidget#composer {{
    background-color: {t.bg_elevated};
    border-top: 1px solid {t.border};
}}
QPushButton#railBtn {{
    background-color: transparent;
    color: {t.text_secondary};
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    text-align: left;
    font-size: 13px;
}}
QPushButton#railBtn:hover {{
    background-color: {t.bg_row_hover};
    color: {t.text_primary};
}}
QPushButton#sourceCard {{
    background-color: {t.bg_elevated};
    color: {t.text_secondary};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 6px 10px;
    text-align: left;
    font-size: 12px;
}}
QPushButton#sourceCard:hover {{
    border-color: {t.accent_primary};
    color: {t.text_primary};
}}
QLabel#assistantBubble {{
    background-color: {t.bg_surface};
    border: 1px solid {t.border};
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 14px;
    line-height: 1.5;
}}
"""
        if app is not None:
            current = app.styleSheet()
            app.setStyleSheet(current + extra_qss)

    def _on_set_theme(self, name: str) -> None:
        app = QApplication.instance()
        central = self.centralWidget()
        if central is not None and central.isVisible():
            def _do_switch() -> None:
                self._theme_manager.set_theme(name, app)
                fade_in(central, duration=200)
            fade_out(central, duration=120, on_done=_do_switch)
        else:
            self._theme_manager.set_theme(name, app)

    def _apply_startup_appearance(self) -> None:
        from windows.settings_dialog import apply_font_scale, apply_density
        app = QApplication.instance()
        if app is not None:
            apply_font_scale(app, self.settings.get("font_scale", 100))
            apply_density(app, self.settings.get("density", "comfortable"))

    # ------------------------------------------------------------------
    # Bubble construction
    # ------------------------------------------------------------------

    def _make_user_bubble(self, text: str) -> QWidget:
        t = self._theme_manager.get_theme_obj()
        outer = QWidget()
        outer_layout = QHBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addStretch()

        bubble = QLabel()
        bubble.setText(html.escape(text).replace("\n", "<br>"))
        bubble.setTextFormat(Qt.TextFormat.RichText)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(560)
        bubble.setObjectName("userBubble")
        bubble.setStyleSheet(
            f"background: {t.accent_primary}; color: {t.accent_ink}; "
            f"border-radius: 12px; padding: 10px 14px;"
        )
        outer_layout.addWidget(bubble)
        return outer

    def _make_assistant_bubble(self):
        """Returns (container, text_label, sources_layout, star_btn, copy_btn)."""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        text_label = QLabel()
        text_label.setObjectName("assistantBubble")
        text_label.setTextFormat(Qt.TextFormat.RichText)
        text_label.setWordWrap(True)
        text_label.setOpenExternalLinks(False)
        text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(text_label)

        sources_layout = QVBoxLayout()
        sources_layout.setSpacing(4)
        v.addLayout(sources_layout)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 4, 0, 0)
        action_row.setSpacing(8)

        star_btn = QPushButton("☆ В избранное")
        star_btn.setObjectName("railBtn")
        star_btn.setCheckable(True)

        copy_btn = QPushButton("Копировать")
        copy_btn.setObjectName("railBtn")

        action_row.addWidget(star_btn)
        action_row.addWidget(copy_btn)
        action_row.addStretch()
        v.addLayout(action_row)

        return container, text_label, sources_layout, star_btn, copy_btn

    def _make_source_card(self, src: dict) -> QPushButton:
        filename = src.get("source", "")
        page = src.get("page", 1)
        section = src.get("section", "")
        is_ocr = section.startswith("[OCR] ")
        icon = "[OCR]" if is_ocr else "[PDF]"
        short_name = filename[:35] + ("..." if len(filename) > 35 else "")
        card_text = f"{icon} {short_name}  стр. {page}"
        if section:
            card_text += f" · {section[:30]}"

        btn = QPushButton(card_text)
        btn.setObjectName("sourceCard")
        btn.clicked.connect(lambda checked, s=src, o=is_ocr: self._open_page_viewer(s["source"], s["page"], o))
        btn.setToolTip(f"{filename}\nСтр. {page}\n{section}")
        return btn

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def _clear_thread(self) -> None:
        while self.thread_layout.count():
            item = self.thread_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_bubble_to_thread(self, widget: QWidget) -> None:
        self.thread_layout.addWidget(widget)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        sb = self.thread_scroll.verticalScrollBar()
        if sb:
            QTimer.singleShot(50, lambda: sb.setValue(sb.maximum()))

    # ------------------------------------------------------------------
    # Query flow
    # ------------------------------------------------------------------

    def _on_search(self) -> None:
        text = self.chat_input.text().strip()
        if not text:
            return

        self._current_query = text
        self.chat_input.clear()
        self.send_btn.setEnabled(False)

        if self.pipeline is None:
            # Add user bubble
            user_bub = self._make_user_bubble(text)
            self._add_bubble_to_thread(user_bub)
            # Add error bubble
            container, lbl, slayout, star_btn, copy_btn = self._make_assistant_bubble()
            lbl.setText('<span style="color: red;">[Ошибка: LLM pipeline не инициализирован]</span>')
            self._add_bubble_to_thread(container)
            # Update chat_log compat
            self.chat_log.setHtml('[Ошибка: LLM pipeline не инициализирован]')
            self.send_btn.setEnabled(True)
            return

        # Build history from current conversation (last 6 turns = 3 Q&A pairs)
        history = []
        if self._current_conv:
            turns = self._current_conv.get("turns", [])[-6:]
            history = [{"role": t["role"], "content": t["content"]} for t in turns]

        # Add user bubble to thread
        user_bub = self._make_user_bubble(text)
        self._add_bubble_to_thread(user_bub)

        # Add turn to conversation
        if self._current_conv is not None:
            self._current_conv["turns"].append({"role": "user", "content": text, "sources": [], "starred": False})

        # Create placeholder assistant bubble
        container, lbl, slayout, star_btn, copy_btn = self._make_assistant_bubble()
        lbl.setText('<span style="color: #888;">...</span>')
        self._add_bubble_to_thread(container)
        self._current_bubble_label = lbl
        self._current_bubble_container = container
        self._current_bubble_sources_layout = slayout
        self._current_star_btn = star_btn
        self._current_copy_btn = copy_btn
        self._current_turn_idx = len(self._current_conv["turns"]) if self._current_conv else 0

        self._start_status_indicator("Инициализация…")
        self._start_meta_pulse()
        self._streaming_answer = ""
        self.chat_log.setHtml("")

        self.worker = QueryWorker(self.pipeline, text, history=history)
        self.worker.result_ready.connect(self._on_result)
        self.worker.event_received.connect(self._on_pipeline_event)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def _on_pipeline_event(self, event: dict) -> None:
        stage = event.get("stage", "")
        if stage == "embed":
            self._set_status("Поиск по документации…")
        elif stage == "search":
            found = event.get("found", 0)
            self._set_status(f"Найдено фрагментов: {found}")
        elif stage == "rerank":
            n = event.get("from", 0)
            m = event.get("to", 0)
            self._set_status(f"Реранжирование: {n} → {m}")
        elif stage == "prompt":
            k = event.get("sources", 0)
            self._set_status(f"Контекст: {k} источников")
        elif stage == "generate_start":
            self._set_status("Генерация ответа…")
            self._streaming_answer = ""
            self.chat_log.setHtml("")
        elif stage == "token":
            chunk = event.get("text", "")
            self._streaming_answer += chunk
            escaped = html.escape(self._streaming_answer)
            display_html = escaped.replace("\n", "<br>") + '<span style="color:#888;">▌</span>'
            if self._current_bubble_label is not None:
                self._current_bubble_label.setText(display_html)
            self.chat_log.setHtml(display_html)
        elif stage == "error":
            self._set_status(f"Ошибка: {event.get('message', '')}")

    def _on_result(self, result: dict[str, Any]) -> None:
        self._stop_status_indicator()
        self._stop_meta_pulse()
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # Finalize bubble text
        if self._current_bubble_label is not None:
            display_html = _markdownish_to_html(answer)
            self._current_bubble_label.setText(display_html)
            self.chat_log.setHtml(display_html)

        # Add source cards
        if self._current_bubble_sources_layout is not None:
            for src in sources:
                card = self._make_source_card(src)
                self._current_bubble_sources_layout.addWidget(card)

        # Update results_list compat
        self.results_list.clear()
        for src in sources:
            class _Item:
                def __init__(self, text): self._text = text
                def text(self): return self._text
            item_text = f"{src['source']}    стр.{src['page']}    [{src.get('section', '')[:40]}]"
            self.results_list.addItem(_Item(item_text))

        # Add assistant turn to conversation
        if self._current_conv is not None:
            turn_idx = len(self._current_conv["turns"])
            self._current_conv["turns"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "starred": False,
            })
            conv_id = self._current_conv["id"]

            # Wire star button
            if self._current_star_btn is not None:
                star_btn = self._current_star_btn
                def _on_star(checked, cid=conv_id, tidx=turn_idx, btn=star_btn):
                    self._toggle_star(cid, tidx, btn)
                star_btn.clicked.connect(_on_star)

            # Wire copy button
            if self._current_copy_btn is not None:
                copy_btn = self._current_copy_btn
                ans_copy = answer
                copy_btn.clicked.connect(lambda checked, a=ans_copy: QApplication.clipboard().setText(a))

            # Update title if first user turn
            user_turns = [t for t in self._current_conv["turns"] if t["role"] == "user"]
            if len(user_turns) == 1:
                self._current_conv["title"] = self._current_query[:60]

            self.conv_store.save()
            self._refresh_conv_list()

        # Legacy history entry
        source_meta = [{"source": s["source"], "page": s["page"], "section": s["section"]} for s in sources]
        self.history.add(self._current_query, answer, source_meta)

        if self._current_bubble_container is not None:
            fade_in(self._current_bubble_container)
        self._scroll_to_bottom()

    def _on_error(self, msg: str) -> None:
        self._stop_status_indicator()
        self._stop_meta_pulse()
        err_html = f'<span style="color: red;">[Ошибка: {html.escape(msg)}]</span>'
        if self._current_bubble_label is not None:
            self._current_bubble_label.setText(err_html)
        self.chat_log.setHtml(err_html)
        self.send_btn.setEnabled(True)

    def _cleanup_worker(self) -> None:
        self.send_btn.setEnabled(True)
        self.worker = None

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def _start_new_conversation(self) -> None:
        self._current_conv = self.conv_store.new_conversation()
        self._clear_thread()
        self._favorites_mode = False
        self._refresh_conv_list()

    def _load_conversation(self, conv_id: str) -> None:
        conv = self.conv_store.get_conversation(conv_id)
        if conv is None:
            return
        self._current_conv = conv
        self._clear_thread()
        for turn in conv.get("turns", []):
            role = turn.get("role", "")
            content = turn.get("content", "")
            sources = turn.get("sources", [])
            if role == "user":
                bub = self._make_user_bubble(content)
                self.thread_layout.addWidget(bub)
            elif role == "assistant":
                container, lbl, slayout, star_btn, copy_btn = self._make_assistant_bubble()
                lbl.setText(_markdownish_to_html(content))
                for src in sources:
                    card = self._make_source_card(src)
                    slayout.addWidget(card)
                starred = turn.get("starred", False)
                if starred:
                    star_btn.setText("★ В избранном")
                    star_btn.setChecked(True)
                self.thread_layout.addWidget(container)
        self._scroll_to_bottom()

    def _refresh_conv_list(self) -> None:
        self.conv_list.blockSignals(True)
        self.conv_list.clear()
        convs = self.conv_store.get_all()
        current_id = self._current_conv["id"] if self._current_conv else None
        for conv in convs:
            item = QListWidgetItem(conv.get("title", "Новый диалог"))
            item.setData(Qt.ItemDataRole.UserRole, conv["id"])
            self.conv_list.addItem(item)
            if conv["id"] == current_id:
                self.conv_list.setCurrentItem(item)
        self.conv_list.blockSignals(False)

    def _on_conv_item_clicked(self, item: QListWidgetItem) -> None:
        conv_id = item.data(Qt.ItemDataRole.UserRole)
        if conv_id and (self._current_conv is None or conv_id != self._current_conv["id"]):
            self._load_conversation(conv_id)

    def _on_favorites_clicked(self) -> None:
        self._favorites_mode = not self._favorites_mode
        if self._favorites_mode:
            self._show_favorites()
        else:
            if self._current_conv:
                self._load_conversation(self._current_conv["id"])

    def _show_favorites(self) -> None:
        self._clear_thread()
        for conv in self.conv_store.get_all():
            shown_heading = False
            for turn in conv.get("turns", []):
                if turn.get("role") == "assistant" and turn.get("starred", False):
                    if not shown_heading:
                        heading = QLabel(conv.get("title", ""))
                        heading.setObjectName("muted")
                        self.thread_layout.addWidget(heading)
                        shown_heading = True
                    container, lbl, slayout, star_btn, copy_btn = self._make_assistant_bubble()
                    lbl.setText(_markdownish_to_html(turn.get("content", "")))
                    for src in turn.get("sources", []):
                        card = self._make_source_card(src)
                        slayout.addWidget(card)
                    star_btn.setText("★ В избранном")
                    star_btn.setChecked(True)
                    self.thread_layout.addWidget(container)
        self._scroll_to_bottom()

    def _toggle_star(self, conv_id: str, turn_idx: int, star_btn: QPushButton) -> None:
        conv = self.conv_store.get_conversation(conv_id)
        if conv and 0 <= turn_idx < len(conv["turns"]):
            current = conv["turns"][turn_idx].get("starred", False)
            new_state = not current
            self.conv_store.update_turn_star(conv_id, turn_idx, new_state)
            if star_btn is not None:
                if new_state:
                    star_btn.setText("★ В избранном")
                    star_btn.setChecked(True)
                else:
                    star_btn.setText("☆ В избранное")
                    star_btn.setChecked(False)

    # ------------------------------------------------------------------
    # Status indicator
    # ------------------------------------------------------------------

    def _start_status_indicator(self, text: str) -> None:
        self._status_label.setText(text)
        self.status_row.show()
        self._status_label.show()
        self._status_dot.show()
        self._dot_pulse_going = True
        self._status_dot.setStyleSheet("color: #4CAF50;")
        self._pulse_dot()

    def _pulse_dot(self) -> None:
        if not getattr(self, "_dot_pulse_going", False):
            return
        self._dot_anim = fade_out(
            self._status_dot, duration=500, start_value=1.0, end_value=0.2,
            on_done=self._pulse_dot_in,
        )

    def _pulse_dot_in(self) -> None:
        if not getattr(self, "_dot_pulse_going", False):
            return
        self._dot_anim = fade_in(
            self._status_dot, duration=500, start_value=0.2, end_value=1.0,
        )
        self._dot_anim.finished.connect(self._pulse_dot)

    def _stop_status_indicator(self) -> None:
        self._dot_pulse_going = False
        self.status_row.hide()
        self._status_label.hide()
        self._status_dot.hide()

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

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
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        effect = self.meta_label.graphicsEffect()
        if isinstance(effect, QGraphicsOpacityEffect):
            effect.setOpacity(1.0)

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def _on_open_library(self) -> None:
        from windows.library_dialog import LibraryDialog
        dlg = LibraryDialog(self, db_path=_default_db_path())
        dlg.exec()
        self._refresh_index_status()
        self._reset_pipeline_retriever()

    def _rebuild_pipeline(self) -> None:
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
            app = QApplication.instance()
            if app is not None:
                apply_font_scale(app, self.settings.get("font_scale", 100))
                apply_density(app, self.settings.get("density", "comfortable"))

    def _on_export_chat(self) -> None:
        if not self._current_conv or not self._current_conv.get("turns"):
            QMessageBox.information(self, "Экспорт", "Нет записей для экспорта.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить диалог", "engineer-companion-chat.md",
            "Markdown (*.md);;JSON (*.json);;Все файлы (*)"
        )
        if not path:
            return
        ext = Path(path).suffix.lower()
        turns = self._current_conv.get("turns", [])
        if ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(turns, f, ensure_ascii=False, indent=2)
        else:
            lines = [f"# Engineer Companion — {self._current_conv.get('title', 'Диалог')}\n\n"]
            for turn in turns:
                role = turn.get("role", "")
                content = turn.get("content", "")
                star = "★ " if turn.get("starred") else ""
                if role == "user":
                    lines.append(f"## {star}Вопрос:\n{content}\n\n")
                elif role == "assistant":
                    lines.append(f"### Ответ:\n{content}\n")
                    for src in turn.get("sources", []):
                        lines.append(f"- {src['source']}, стр.{src['page']} — {src.get('section', '')}\n")
                    lines.append("\n")
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        QMessageBox.information(self, "Экспорт", f"Сохранено: {path}")

    # ------------------------------------------------------------------
    # Index status
    # ------------------------------------------------------------------

    def _refresh_index_status(self) -> None:
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
        if self.pipeline is None:
            return
        try:
            retriever = getattr(self.pipeline, "retriever", None)
            if retriever is not None:
                if hasattr(retriever, "_table"):
                    retriever._table = None
                if hasattr(retriever, "_db"):
                    retriever._db = None
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Page viewer
    # ------------------------------------------------------------------

    def _open_page_viewer(self, filename: str, page: int, is_ocr: bool = False) -> None:
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

    def _on_anchor_clicked(self, url: "QUrl") -> None:
        """Handle anchor clicks in the chat log (legacy compat + source: URL scheme)."""
        if url.scheme() == "bookmark":
            idx = int(url.host())
            self.history.toggle_bookmark(idx)
        elif url.scheme() == "source":
            # Parse: source:{encoded_filename}|{page}|{ocr_flag}
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

    # ------------------------------------------------------------------
    # Legacy compat
    # ------------------------------------------------------------------

    def _restore_history(self) -> None:
        pass  # History restored via ConversationStore


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=Path, default=None,
                    help="Override DB path (default: app_paths.db_path())")
    ap.add_argument("--llm-path", type=Path,
                    default=_REPO / "assets" / "models" / "gemma-3-4b-it-Q4_K_M.gguf")
    ap.add_argument("--no-llm", action="store_true", help="Run without LLM (search-only mode)")
    args = ap.parse_args()

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
