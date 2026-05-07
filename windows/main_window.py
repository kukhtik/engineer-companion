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

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QTextBrowser,
    QSplitter, QVBoxLayout, QWidget,
)

# Allow running from repo root without install
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from core.query import RAGQueryPipeline  # noqa: E402
from design.tokens import DribbbleDarkQt  # noqa: E402


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

    def __init__(self, path: Path) -> None:
        self.path = path
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
        self.entries.append({"query": query, "answer": answer, "sources": sources})
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def format_html(self) -> str:
        parts = []
        for e in self.entries:
            parts.append(f'<hr><div style="color:#7A7A7A;font-size:12px;margin-bottom:4px;">Вопрос: {html.escape(e["query"])}</div>')
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
        self._build_ui()
        self._apply_tokens()
        self._restore_history()

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
        left_v.addWidget(self.meta_label)

        # ---- Right area: chat splitter ----
        right = QSplitter(Qt.Vertical)

        self.chat_log = QTextBrowser()
        self.chat_log.setOpenExternalLinks(False)
        self.chat_log.setHtml(
            '<div style="color:#7A7A7A;">'
            "Здесь появится ответ эксперта...<br>"
            "Введите вопрос слева и нажмите Найти."
            "</div>"
        )
        right.addWidget(self.chat_log)

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

        self.chat_log.append("<hr>")
        self.chat_log.append(f'<div style="color:#7A7A7A;font-size:12px;margin-bottom:4px;">Ответ:</div>')
        self.chat_log.append(_markdownish_to_html(answer))

        for src in sources:
            item_text = f"{src['source']}    стр.{src['page']}    [{src['section'][:40]}]"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, json.dumps(src))
            self.results_list.addItem(item)

        self.meta_label.setText(f"Найдено источников: {len(sources)}")

        source_meta = [{"source": s["source"], "page": s["page"], "section": s["section"]} for s in sources]
        self.history.add(self._current_query, answer, source_meta)

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
