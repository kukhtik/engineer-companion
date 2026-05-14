"""Windows Desktop UI: PySide6 RAG companion.

Layout: single window, left sidebar (sources list), right area split vertically:
- top: search bar + results list
- bottom: chat panel (markdown-ish display)

Design: impeccable-ui dribbble-dark tokens applied via Qt Stylesheet.
"""

import html
import json
import re
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMenuBar, QMessageBox,
    QPushButton, QTextBrowser, QSplitter, QVBoxLayout, QWidget,
)

# Allow running from repo root without install
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from core.query import RAGQueryPipeline  # noqa: E402
from design.tokens import DribbbleDarkQt  # noqa: E402
from windows.settings_dialog import SettingsDialog, load_settings, save_settings  # noqa: E402


def _markdownish_to_html(text: str) -> str:
    """Lightweight conversion: bold, bullet lists, newlines."""
    # Escape HTML entities
    text = html.escape(text)
    # Bold: *text* -> <b>text</b>
    text = re.sub(r"\*(.+?)\*", r"<b>\1</b>", text)
    # Bullet lists: lines starting with "* " or "- " -> <ul><li>
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
            # Find oldest non-bookmarked
            for i, e in enumerate(self.entries):
                if not e.get("bookmarked", False):
                    self.entries.pop(i)
                    break
            else:
                # All bookmarked — remove oldest anyway
                self.entries.pop(0)

    def _prune(self) -> None:
        """Auto-prune without explicit call in add()."""
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
                parts.append(
                    f'<div style="color:#7A7A7A;font-size:11px;margin-left:8px;">'
                    f'• {html.escape(src["source"])} стр.{src["page"]} — {html.escape(src["section"])}</div>'
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
        self.history = ChatHistory(Path.home() / ".engineer-companion" / "history.json")
        self.setWindowTitle("Engineer Companion")
        self.setMinimumSize(1200, 800)
        self.settings = load_settings()
        self._setup_menu()
        self._build_ui()
        self._apply_tokens()
        self._restore_history()

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Файл")
        export_action = file_menu.addAction("Экспорт в Markdown...")
        export_action.triggered.connect(self._on_export_chat)
        file_menu.addSeparator()
        settings_action = file_menu.addAction("Настройки...")
        settings_action.triggered.connect(self._on_open_settings)
        settings_action.setShortcut("Ctrl+,")

    def _on_open_settings(self) -> None:
        dlg = SettingsDialog(self, current=self.settings)
        if dlg.exec():
            self.settings = dlg.get_settings()
            # Rebuild pipeline if settings changed
            db_path = self.settings.get("db_path")
            llm_path = self.settings.get("llm_model_path")
            if db_path:
                p = Path(db_path)
                if not p.exists():
                    db_path = ""
            if llm_path:
                p = Path(llm_path)
                if not p.exists():
                    llm_path = ""

            from core.query import RAGQueryPipeline
            if llm_path:
                try:
                    self.pipeline = RAGQueryPipeline(
                        db_path=Path(db_path) if db_path else _REPO / "assets" / "db" / "engineer.db",
                        llm_model_path=Path(llm_path) if llm_path else None,
                        top_k=self.settings.get("top_k", 8),
                        max_tokens=self.settings.get("max_tokens", 512),
                        temperature=self.settings.get("temperature", 0.3),
                        llm_n_ctx=2048,
                        llm_n_threads=2,
                    )
                except Exception as exc:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить модель:\n{exc}")
                    self.pipeline = None
            else:
                self.pipeline = None

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
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить чат как Markdown", "engineer-companion-chat.md",
            "Markdown (*.md);;JSON (*.json);;Все файлы (*)"
        )
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext == ".json":
            import json
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

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ---- Left sidebar: search bar + results list ----
        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(8)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Введите вопрос по TrueBeam / VitalBeam...")
        self.search_edit.returnPressed.connect(self._on_search)
        left_v.addWidget(self.search_edit)

        self.search_btn = QPushButton("Найти")
        self.search_btn.clicked.connect(self._on_search)
        left_v.addWidget(self.search_btn)

        self.results_list = QListWidget()
        self.results_list.setSpacing(4)
        self.results_list.itemClicked.connect(self._on_result_clicked)
        left_v.addWidget(self.results_list, stretch=1)

        self.meta_label = QLabel()
        self.meta_label.setObjectName("muted")

        self.clear_btn = QPushButton("🗑 Очистить историю")
        self.clear_btn.clicked.connect(self._on_clear_history)
        left_v.addWidget(self.meta_label)
        left_v.addWidget(self.clear_btn)

        # ---- Prune button ----
        self.prune_btn = QPushButton("✂️ Удалить старые (>200)")
        self.prune_btn.clicked.connect(self._on_prune_history)
        left_v.addWidget(self.prune_btn)

        # ---- Right area: chat splitter ----
        right = QSplitter(Qt.Vertical)

        # Top: chat log + controls
        chat_top = QWidget()
        chat_top_v = QVBoxLayout(chat_top)
        chat_top_v.setContentsMargins(0, 0, 0, 0)
        chat_top_v.setSpacing(4)

        # Filters row
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
        self.chat_log.anchorClicked.connect(self._on_anchor_clicked)
        self.chat_log.setHtml(
            '<div style="color:#7A7A7A;">'
            "Здесь появится ответ эксперта...<br>"
            "Введите вопрос слева и нажмите Найти."
            "</div>"
        )
        chat_top_v.addWidget(self.chat_log, stretch=1)
        right.addWidget(chat_top)

        bottom = QWidget()
        bottom_h = QHBoxLayout(bottom)
        bottom_h.setContentsMargins(0, 0, 0, 0)
        bottom_h.setSpacing(8)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Уточняющий вопрос...")
        self.chat_input.returnPressed.connect(self._on_search)

        self.send_btn = QPushButton("Отправить")
        self.send_btn.setShortcut("Ctrl+Return")
        self.send_btn.clicked.connect(self._on_search)

        bottom_h.addWidget(self.chat_input, stretch=1)
        bottom_h.addWidget(self.send_btn)
        right.addWidget(bottom)
        right.setSizes([600, 80])

        outer_splitter = QSplitter(Qt.Horizontal)
        outer_splitter.addWidget(left)
        outer_splitter.addWidget(right)
        outer_splitter.setSizes([420, 780])
        root.addWidget(outer_splitter)

    def _apply_tokens(self) -> None:
        tokens = DribbbleDarkQt()
        self.setStyleSheet(tokens.as_stylesheet())

    def _restore_history(self) -> None:
        if self.history.entries:
            self.chat_log.setHtml(self.history.format_html())

    def _on_search(self) -> None:
        text = self.search_edit.text().strip() or self.chat_input.text().strip()
        if not text:
            return
        if self.pipeline is None:
            self.chat_log.append("[Ошибка: LLM pipeline не инициализирован]")
            return

        self.search_btn.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.search_edit.clear()
        self.chat_input.clear()
        self.results_list.clear()
        self.meta_label.setText("Ищем...")

        self.worker = QueryWorker(self.pipeline, text)
        self._current_query = text
        self.worker.result_ready.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def _on_result(self, result: dict[str, Any]) -> None:
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        self.results_list.clear()
        for src in sources:
            item_text = f"{src['source']}    стр.{src['page']}    [{src['section'][:40]}]"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, json.dumps(src))
            self.results_list.addItem(item)

        self.meta_label.setText(f"Найдено источников: {len(sources)}")

        source_meta = [{"source": s["source"], "page": s["page"], "section": s["section"]} for s in sources]
        self.history.add(self._current_query, answer, source_meta)
        self.chat_log.setHtml(self.history.format_html())

    def _on_error(self, msg: str) -> None:
        self.chat_log.append(f'[Ошибка: {html.escape(msg)}]')
        self.meta_label.setText("Ошибка")

    def _cleanup_worker(self) -> None:
        self.search_btn.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.worker = None

    def _on_result_clicked(self, item: QListWidgetItem) -> None:
        data = json.loads(item.data(Qt.UserRole))
        self.chat_log.append(
            f'<div style="color:#B2BAB6;font-size:12px;">'
            f'[Выбран источник] {html.escape(data["source"])} '
            f'стр.{data["page"]} — {html.escape(data["section"])}'
            f'</div>'
        )

    def _on_anchor_clicked(self, url: QUrl) -> None:
        if url.scheme() == "bookmark":
            idx = int(url.host())
            self.history.toggle_bookmark(idx)
            self.chat_log.setHtml(self.history.format_html())

    def _on_clear_history(self) -> None:
        self.history.clear()
        self.chat_log.setHtml(
            '<div style="color:#7A7A7A;">'
            "История очищена.<br>"
            "Введите вопрос слева и нажмите Найти."
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
        html = self.history.format_html(bookmarks_only=bookmarks_only, search_term=search_term)
        if html:
            self.chat_log.setHtml(html)
        else:
            self.chat_log.setHtml('<div style="color:#7A7A7A;">Нет записей, соответствующих фильтру.</div>')


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=Path, default=_REPO / "assets" / "db" / "engineer.db")
    ap.add_argument("--llm-path", type=Path, default=_REPO / "assets" / "models" / "gemma-3-4b-it-Q4_K_M.gguf")
    ap.add_argument("--no-llm", action="store_true", help="Run without LLM (search-only mode)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pipeline = None
    if not args.no_llm and args.llm_path.exists():
        pipeline = RAGQueryPipeline(db_path=args.db_path, llm_model_path=args.llm_path)

    w = CompanionWindow(pipeline=pipeline)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
