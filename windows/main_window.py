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
Phase C changes:
- _ActivityPanel: rich activity strip driven by pipeline events
- QTextEdit assistant bubble for proper text wrapping and height
"""

import html
import inspect
import json
import logging
import logging.handlers
import re
import socket
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer
from PySide6.QtGui import QTextOption
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

# Allow running from repo root without install
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from core.query import RAGQueryPipeline  # noqa: E402
from design.tokens import DribbbleDarkQt  # noqa: E402  (backward-compat import kept)
from windows.app_paths import db_path as _default_db_path, ensure_seeded, data_root  # noqa: E402
from windows.settings_dialog import SettingsDialog, load_settings  # noqa: E402
from windows.theme import ThemeManager  # noqa: E402
from windows.anim import fade_in  # noqa: E402

import structlog  # noqa: E402

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches single tags like [ИСТОЧНИК 1] and compound tags like [ИСТОЧНИК 1, ИСТОЧНИК 2]
# which Gemma sometimes emits instead of separate tags.
_SOURCE_TAG_RE = re.compile(r"\[ИСТОЧНИК\s*\d+(?:,\s*ИСТОЧНИК\s*\d+)*\]", re.IGNORECASE)


def _strip_source_tags(text: str) -> str:
    """Remove inline [ИСТОЧНИК N] citation tags from answer text.

    The authoritative source references are shown as cards below the bubble;
    the inline tags are noise in the displayed answer.
    """
    cleaned = _SOURCE_TAG_RE.sub("", text)
    # Collapse multiple spaces that may remain after tag removal.
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


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
        # Do NOT save yet — only persist once the conversation has actual turns.
        return conv

    def get_or_create_empty(self) -> dict:
        """Return the most-recent empty conversation, creating one if needed.

        An 'empty' conversation has no turns and still has the default title.
        This prevents "Новый диалог" spam in the left rail.
        """
        convs = self.get_all()
        for c in convs:
            if not c.get("turns") and c.get("title") == "Новый диалог":
                return c
        return self.new_conversation()

    def prune_empty(self) -> None:
        """Remove all empty/untitled conversations except the most recent one."""
        empties = [c for c in self.conversations
                   if not c.get("turns") and c.get("title") == "Новый диалог"]
        # Keep the newest empty (index 0 after get_all sort), delete the rest
        for c in empties[1:]:
            self.conversations.remove(c)
        if empties[1:]:
            self.save()

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

    def delete(self, conv_id: str) -> bool:
        """Remove a conversation by id, persist atomically. Returns True if found+deleted."""
        before = len(self.conversations)
        self.conversations = [c for c in self.conversations if c["id"] != conv_id]
        if len(self.conversations) < before:
            self.save()
            return True
        return False


# ---------------------------------------------------------------------------
# QueryWorker
# ---------------------------------------------------------------------------

class QueryWorker(QThread):
    """Off-thread RAG query so UI stays responsive."""

    result_ready = Signal(dict)
    event_received = Signal(dict)
    error = Signal(str)
    backend_used = Signal(str)  # "local" | "cloud"

    def __init__(
        self,
        pipeline,
        query: str,
        history=None,
        backend: str = "local",
        cloud_model: str | None = None,
        cloud_client=None,
        usage_tracker=None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.query = query
        self.history = history
        self.backend = backend
        self.cloud_model = cloud_model
        self.cloud_client = cloud_client
        self.usage_tracker = usage_tracker

    @staticmethod
    def _accepted_kwargs(callable_obj) -> set[str] | None:
        """Return the set of kwarg names ``callable_obj`` accepts, or None if
        it accepts arbitrary kwargs (**kwargs) / its signature can't be
        determined via introspection (conservative: caller should then send
        no optional kwargs).

        Unwraps a Mock's ``side_effect`` (if set to a plain function) so
        tests that stub ``pipeline.ask_streaming.side_effect = fn`` are
        introspected against the real ``fn``, not the generic Mock call
        signature.
        """
        target = callable_obj
        side_effect = getattr(callable_obj, "side_effect", None)
        if callable(side_effect) and not isinstance(side_effect, type):
            target = side_effect
        try:
            params = inspect.signature(target).parameters
        except (TypeError, ValueError):
            return set()
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return None  # accepts **kwargs — forward everything
        return set(params)

    def run(self) -> None:
        try:
            if hasattr(self.pipeline, "ask_streaming"):
                # Introspect the target signature ONCE, before making any
                # call, instead of catch-and-retry on TypeError. The old
                # pattern wrapped the whole (side-effecting!) call in
                # try/except TypeError and re-invoked ask_streaming with
                # fewer kwargs on ANY TypeError — including one raised deep
                # inside the call after real work had already happened
                # (e.g. a cloud request already completed and usage already
                # recorded). That masked the true error, silently repeated
                # a network call, and — since the fallback omits backend=
                # entirely — retried with backend="local" by default.
                accepted = self._accepted_kwargs(self.pipeline.ask_streaming)

                def _wants(name: str) -> bool:
                    return accepted is None or name in accepted

                kwargs: dict[str, Any] = {}
                if _wants("history"):
                    kwargs["history"] = self.history
                if _wants("backend"):
                    kwargs["backend"] = self.backend
                if _wants("cloud_model"):
                    kwargs["cloud_model"] = self.cloud_model
                if _wants("cloud_client"):
                    kwargs["cloud_client"] = self.cloud_client
                if _wants("usage_tracker"):
                    kwargs["usage_tracker"] = self.usage_tracker
                result = self.pipeline.ask_streaming(self.query, self._emit_event, **kwargs)
            else:
                accepted = self._accepted_kwargs(self.pipeline.ask)
                if accepted is None or "history" in accepted:
                    result = self.pipeline.ask(self.query, history=self.history)
                else:
                    result = self.pipeline.ask(self.query)
            self.backend_used.emit(self.backend)
            self.result_ready.emit(result)
        except Exception as exc:
            logger.error("query_worker_failed", error=str(exc), traceback=traceback.format_exc())
            self.error.emit(str(exc))

    def _emit_event(self, event: dict) -> None:
        try:
            self.event_received.emit(event)
        except Exception:
            # Never let a UI-signal marshaling hiccup abort an in-flight
            # generation (local or cloud) — log it and keep streaming.
            logger.error("event_emit_failed", stage=event.get("stage"), traceback=traceback.format_exc())


# ---------------------------------------------------------------------------
# _ActivityPanel
# ---------------------------------------------------------------------------

class _ActivityPanel(QWidget):
    """Rich activity strip shown while a RAG query is in-flight.

    Driven by pipeline events — every displayed number comes from a real event field.
    """

    STAGES = [
        ("embed",          "Векторизация запроса"),
        ("search",         "Поиск · найдено {found} фрагментов"),
        ("rerank",         "Реранжирование {from_n}→{to_n}"),
        ("prompt",         "Сборка контекста · {sources_k} источников"),
        ("generate_start", "Генерация ответа"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stage_states: dict = {}  # stage_key -> "pending"|"active"|"done"
        self._found_count: int = 0
        self._rerank_from: int = 0
        self._rerank_to: int = 0
        self._sources_k: int = 0
        self._token_count: int = 0
        self._gen_start_time: float = 0.0
        self._has_rerank: bool = False
        self._anim_refs: list = []
        self._gen_timer: QTimer | None = None

        self._build_ui()
        self.hide()

    def _build_ui(self):
        root_v = QVBoxLayout(self)
        root_v.setContentsMargins(8, 6, 8, 6)
        root_v.setSpacing(4)

        # Stage strip frame
        self._strip_frame = QFrame()
        self._strip_frame.setObjectName("activityStrip")
        self._strip_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        strip_v = QVBoxLayout(self._strip_frame)
        strip_v.setContentsMargins(8, 6, 8, 6)
        strip_v.setSpacing(3)

        self._stage_rows: dict = {}

        for key, label_tpl in self.STAGES:
            row_w = QWidget()
            row_h = QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(6)

            icon_lbl = QLabel("○")
            icon_lbl.setFixedWidth(14)
            icon_lbl.setObjectName("stageIcon")

            text_lbl = QLabel(label_tpl.replace("{found}", "0")
                                        .replace("{from_n}", "0")
                                        .replace("{to_n}", "0")
                                        .replace("{sources_k}", "0"))
            text_lbl.setObjectName("stageLbl")

            # Rerank gets an extra mini-viz container
            rerank_viz = None
            if key == "rerank":
                rerank_viz = QWidget()
                QHBoxLayout(rerank_viz).setContentsMargins(0, 0, 0, 0)

            row_h.addWidget(icon_lbl)
            row_h.addWidget(text_lbl)
            if rerank_viz:
                row_h.addWidget(rerank_viz)
            row_h.addStretch()

            strip_v.addWidget(row_w)
            self._stage_rows[key] = (icon_lbl, text_lbl, row_w, rerank_viz)

        root_v.addWidget(self._strip_frame)

        # Metrics row (visible only after generate_start)
        self._metrics_row = QWidget()
        met_h = QHBoxLayout(self._metrics_row)
        met_h.setContentsMargins(0, 0, 0, 0)
        met_h.setSpacing(12)

        self._elapsed_label = QLabel("0s")
        self._elapsed_label.setObjectName("muted")
        self._tokens_label = QLabel("0 токенов")
        self._tokens_label.setObjectName("muted")
        self._tps_label = QLabel("0 т/с")
        self._tps_label.setObjectName("muted")

        met_h.addWidget(self._elapsed_label)
        met_h.addWidget(self._tokens_label)
        met_h.addWidget(self._tps_label)
        met_h.addStretch()
        self._metrics_row.hide()
        root_v.addWidget(self._metrics_row)

        # Summary label (shown on done, replaces strip)
        self._summary_label = QLabel("")
        self._summary_label.setObjectName("muted")
        self._summary_label.setWordWrap(True)
        self._summary_label.hide()
        root_v.addWidget(self._summary_label)

    def reset(self):
        """Call before starting a new query."""
        self._found_count = 0
        self._rerank_from = 0
        self._rerank_to = 0
        self._sources_k = 0
        self._token_count = 0
        self._gen_start_time = 0.0
        self._has_rerank = False
        self._stage_states = {key: "pending" for key, _ in self.STAGES}

        self._strip_frame.show()
        self._metrics_row.hide()
        self._summary_label.hide()

        for key, _ in self.STAGES:
            icon_lbl, text_lbl, row_w, _ = self._stage_rows[key]
            icon_lbl.setText("○")
            # Reset rerank row text
            if key == "rerank":
                text_lbl.setText("Реранжирование 0→0")
                row_w.hide()
            elif key == "search":
                text_lbl.setText("Поиск · найдено 0 фрагментов")
                row_w.show()
            elif key == "prompt":
                text_lbl.setText("Сборка контекста · 0 источников")
                row_w.show()
            else:
                row_w.show()

        self._set_stage_style("embed", "active")

        if self._gen_timer:
            self._gen_timer.stop()
            self._gen_timer = None

        self.show()

    def _set_stage_style(self, stage_key: str, state: str):
        """Apply pending/active/done styling to a stage row."""
        if stage_key not in self._stage_rows:
            return
        icon_lbl, text_lbl, row_w, _ = self._stage_rows[stage_key]
        self._stage_states[stage_key] = state

        if state == "pending":
            icon_lbl.setText("○")
            icon_lbl.setStyleSheet("color: #666;")
            text_lbl.setStyleSheet("color: #666;")
        elif state == "active":
            icon_lbl.setText("●")
            icon_lbl.setStyleSheet("color: #F5C518; font-weight: bold;")
            text_lbl.setStyleSheet("color: #F5C518; font-weight: bold;")
        elif state == "done":
            icon_lbl.setText("✓")
            icon_lbl.setStyleSheet("color: #4CAF50;")
            text_lbl.setStyleSheet("color: #4CAF50;")
        elif state == "error":
            icon_lbl.setText("✗")
            icon_lbl.setStyleSheet("color: #f44336;")
            text_lbl.setStyleSheet("color: #f44336;")

    def on_embed(self):
        self._set_stage_style("embed", "active")

    def on_search(self, found: int):
        self._found_count = found
        self._set_stage_style("embed", "done")
        self._set_stage_style("search", "active")
        icon_lbl, text_lbl, row_w, _ = self._stage_rows["search"]
        text_lbl.setText(f"Поиск · найдено {found} фрагментов")

    def on_rerank(self, from_n: int, to_n: int):
        self._rerank_from = from_n
        self._rerank_to = to_n
        self._has_rerank = True
        self._set_stage_style("search", "done")

        icon_lbl, text_lbl, row_w, rerank_viz = self._stage_rows["rerank"]
        text_lbl.setText(f"Реранжирование {from_n}→{to_n} фрагментов")
        row_w.show()
        self._set_stage_style("rerank", "active")

        # Build mini-viz: from_n small colored squares, first to_n are accent
        if rerank_viz is not None:
            lay = rerank_viz.layout()
            # Clear existing squares
            while lay.count():
                item = lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            # Cap at 20 squares max for visual clarity
            show_n = min(from_n, 20)
            ratio = to_n / from_n if from_n > 0 else 0
            keep_n = round(show_n * ratio)
            for i in range(show_n):
                sq = QFrame()
                sq.setFixedSize(10, 10)
                sq.setFrameStyle(QFrame.Shape.Box)
                if i < keep_n:
                    sq.setStyleSheet("background: #F5C518; border: none; border-radius: 2px;")
                else:
                    sq.setStyleSheet("background: #444; border: none; border-radius: 2px;")
                lay.addWidget(sq)

        self._set_stage_style("rerank", "done")

    def on_prompt(self, sources: int):
        self._sources_k = sources
        if not self._has_rerank:
            self._set_stage_style("search", "done")
        else:
            self._set_stage_style("rerank", "done")
        self._set_stage_style("prompt", "active")
        icon_lbl, text_lbl, row_w, _ = self._stage_rows["prompt"]
        text_lbl.setText(f"Сборка контекста · {sources} источников")

    def on_generate_start(self):
        self._set_stage_style("prompt", "done")
        self._set_stage_style("generate_start", "active")
        self._gen_start_time = time.monotonic()
        self._token_count = 0
        self._metrics_row.show()
        self._gen_timer = QTimer(self)
        self._gen_timer.setInterval(500)
        self._gen_timer.timeout.connect(self._tick_metrics)
        self._gen_timer.start()

    def on_token(self):
        self._token_count += 1

    def _tick_metrics(self):
        elapsed = time.monotonic() - self._gen_start_time
        tps = self._token_count / elapsed if elapsed > 0 else 0.0
        self._elapsed_label.setText(f"{elapsed:.0f}s")
        self._tokens_label.setText(f"{self._token_count} токенов")
        self._tps_label.setText(f"{tps:.1f} т/с")

    def on_done(self) -> str:
        """Collapse strip to summary line briefly, then hide the panel."""
        if self._gen_timer:
            self._gen_timer.stop()
            self._gen_timer = None

        elapsed = time.monotonic() - self._gen_start_time if self._gen_start_time > 0 else 0.0
        self._set_stage_style("generate_start", "done")

        # Build summary
        parts = [f"✓ найдено {self._found_count}"]
        if self._has_rerank:
            parts.append(f"реранж {self._rerank_from}→{self._rerank_to}")
        parts.append(f"{self._sources_k} источников")
        parts.append(f"{self._token_count} токенов за {elapsed:.0f}s")
        summary = " · ".join(parts)

        self._strip_frame.hide()
        self._metrics_row.hide()
        self._summary_label.setText(summary)
        self._summary_label.show()

        # Hide the entire panel after a short display so it's invisible at idle
        QTimer.singleShot(2500, self._hide_after_done)
        return summary

    def _hide_after_done(self) -> None:
        """Hide the panel and clear the summary — returns to idle state."""
        self._summary_label.hide()
        self.hide()

    def on_error(self, msg: str):
        if self._gen_timer:
            self._gen_timer.stop()
            self._gen_timer = None
        # Mark current active stage as error
        for key, _ in self.STAGES:
            if self._stage_states.get(key) == "active":
                self._set_stage_style(key, "error")
                icon_lbl, text_lbl, row_w, _ = self._stage_rows[key]
                text_lbl.setText(f"Ошибка: {msg[:60]}")
                break
        # Hide panel after a short display
        QTimer.singleShot(3000, self.hide)


# ---------------------------------------------------------------------------
# _CloudUsageIndicator
# ---------------------------------------------------------------------------

class _CloudUsageIndicator(QWidget):
    """Status-bar widget showing Ollama Cloud local usage counters.

    Visible only when the user has an API key or is in cloud/auto mode.
    Refreshed by calling .refresh(tracker).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel("")
        self._label.setObjectName("muted")
        self._label.setToolTip(
            "Локальная оценка — Ollama не предоставляет точные остатки квоты.\n"
            "Подробнее: ollama.com"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)
        self.hide()

    def refresh(self, tracker) -> None:
        """Update label text from tracker.snapshot()."""
        if tracker is None:
            self.hide()
            return
        snap = tracker.snapshot()
        sess = snap.get("session", {})
        week = snap.get("week", {})
        rate_until = snap.get("rate_limited_until")

        if rate_until and time.time() < rate_until:
            from datetime import datetime
            reset_str = datetime.fromtimestamp(rate_until).strftime("%H:%M")
            self._label.setText(f"Облако · лимит исчерпан, сброс ~{reset_str}")
            self._label.setStyleSheet("color: #f44336;")
        else:
            sess_n = sess.get("requests", 0)
            week_n = week.get("requests", 0)
            self._label.setText(f"Облако · сессия 5ч: {sess_n} · неделя 7д: {week_n}")
            self._label.setStyleSheet("")

        self.show()


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
        self._current_bubble_label: QTextEdit | None = None
        self._current_bubble_container: QWidget | None = None
        self._current_bubble_sources_layout = None
        self._current_star_btn: QPushButton | None = None
        self._current_copy_btn: QPushButton | None = None
        self._current_turn_idx: int = 0
        self._activity_panel: _ActivityPanel | None = None
        self._anim_refs: list = []

        # Cloud state
        self._cloud_client = None
        self._cloud_usage_tracker = None
        self._online_cache: tuple[float, bool] | None = None  # (checked_at, result)
        self._backend_label: QLabel | None = None
        self._cloud_indicator: _CloudUsageIndicator | None = None

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
        self._rebuild_cloud_clients()

        # Prune stale empty/untitled conversations accumulated from previous runs
        self.conv_store.prune_empty()

        # Load most recent conversation or start new.
        # If the most recent conversation is empty/untitled, just switch to it
        # (don't create another blank one on top of it).
        convs = self.conv_store.get_all()
        if convs and convs[0].get("turns"):
            # Most recent has turns — load it normally
            self._load_conversation(convs[0]["id"])
        elif convs:
            # Most recent is empty — reuse it as current without persisting again
            self._current_conv = convs[0]
            self._clear_thread()
        else:
            # No conversations at all — create the first one
            self._current_conv = self.conv_store.new_conversation()
            self._clear_thread()
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
        self.conv_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self._on_conv_context_menu)
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

        # Status row (kept for backward-compat with tests)
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

        # Activity panel (Phase C)
        self._activity_panel = _ActivityPanel()
        center_v.addWidget(self._activity_panel)

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

        # Backend indicator label (shown after each query in activity area)
        self._backend_label = QLabel("")
        self._backend_label.setObjectName("muted")
        self._backend_label.hide()
        center_v.addWidget(self._backend_label)

        # Status bar
        self.index_status_label = QLabel("Документов: — · Чанков: —")
        self.index_status_label.setObjectName("muted")
        self.statusBar().addPermanentWidget(self.index_status_label)

        # Cloud usage indicator in status bar
        self._cloud_indicator = _CloudUsageIndicator()
        self.statusBar().addWidget(self._cloud_indicator)

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
QTextEdit#assistantBubble {{
    background-color: {t.bg_surface};
    border: 1px solid {t.border};
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 14px;
    line-height: 1.5;
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
        """Switch theme instantly — no opacity-effect crossfade on the central
        widget (that grabs the whole widget tree as a pixmap while children
        are painting, causing QPainter "Painter not active" floods)."""
        app = QApplication.instance()
        self._theme_manager.set_theme(name, app)
        # Reapply per-window extra QSS so rail/composer/bubble colours update
        self._apply_tokens()

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
        """Returns (container, text_edit, sources_layout, star_btn, copy_btn)."""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        text_edit = QTextEdit()
        text_edit.setObjectName("assistantBubble")
        text_edit.setReadOnly(True)
        text_edit.setFrameStyle(QFrame.Shape.NoFrame)
        text_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        text_edit.setMinimumHeight(40)
        text_edit.setMaximumWidth(560)
        # Resize to content
        text_edit.document().contentsChanged.connect(
            lambda te=text_edit: te.setFixedHeight(
                min(400, max(40, int(te.document().size().height()) + 16))
            )
        )
        v.addWidget(text_edit)

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

        return container, text_edit, sources_layout, star_btn, copy_btn

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
        # Re-entrancy guard: chat_input.returnPressed and send_btn's
        # "Ctrl+Return" shortcut can both fire for a single keypress. Without
        # this guard, a second call would overwrite self.worker while the
        # first QueryWorker (a QThread) is still running in the background,
        # orphaning it mid-flight — Qt then destroys the still-running
        # QThread's Python wrapper, which corrupts Shiboken's internal
        # weak-reference bookkeeping and surfaces as a confusing
        # "cannot create weak reference to 'NoneType' object" error.
        if self.worker is not None and self.worker.isRunning():
            return

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
            lbl.setHtml('<span style="color: red;">[Ошибка: LLM pipeline не инициализирован]</span>')
            self._add_bubble_to_thread(container)
            # Update chat_log compat
            self.chat_log.setHtml('[Ошибка: LLM pipeline не инициализирован]')
            self.send_btn.setEnabled(True)
            self.chat_input.setEnabled(True)
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
        lbl.setHtml('<span style="color: #888;">...</span>')
        self._add_bubble_to_thread(container)
        self._current_bubble_label = lbl
        self._current_bubble_container = container
        self._current_bubble_sources_layout = slayout
        self._current_star_btn = star_btn
        self._current_copy_btn = copy_btn
        self._current_turn_idx = len(self._current_conv["turns"]) if self._current_conv else 0

        # Reset and show activity panel
        if self._activity_panel is not None:
            self._activity_panel.reset()

        self.chat_input.setEnabled(False)
        self._start_status_indicator("Инициализация…")
        self._start_meta_pulse()
        self._streaming_answer = ""
        self.chat_log.setHtml("")

        backend = self._decide_backend()
        cloud_model = self.settings.get("cloud_model", "gpt-oss:120b-cloud") or "gpt-oss:120b-cloud"

        self.worker = QueryWorker(
            self.pipeline,
            text,
            history=history,
            backend=backend,
            cloud_model=cloud_model,
            cloud_client=self._cloud_client,
            usage_tracker=self._cloud_usage_tracker,
        )
        self.worker.result_ready.connect(self._on_result)
        self.worker.event_received.connect(self._on_pipeline_event)
        self.worker.error.connect(self._on_error)
        self.worker.backend_used.connect(self._on_backend_used)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def _on_backend_used(self, backend: str) -> None:
        """Update backend label after a query completes."""
        if self._backend_label is None:
            return
        if backend == "cloud":
            cloud_model = self.settings.get("cloud_model", "")
            self._backend_label.setText(f"Облако: {cloud_model}")
        else:
            self._backend_label.setText("Локально: Gemma")
        self._backend_label.show()
        # Refresh usage indicator
        self._refresh_cloud_indicator()

    def _on_pipeline_event(self, event: dict) -> None:
        stage = event.get("stage", "")
        panel = self._activity_panel

        if stage == "embed":
            self._set_status("Поиск по документации…")
            if panel:
                panel.on_embed()
        elif stage == "search":
            found = event.get("found", 0)
            self._set_status(f"Найдено фрагментов: {found}")
            if panel:
                panel.on_search(found)
        elif stage == "rerank":
            n = event.get("from", 0)
            m = event.get("to", 0)
            self._set_status(f"Реранжирование: {n} → {m}")
            if panel:
                panel.on_rerank(n, m)
        elif stage == "prompt":
            k = event.get("sources", 0)
            self._set_status(f"Контекст: {k} источников")
            if panel:
                panel.on_prompt(k)
        elif stage == "generate_start":
            self._set_status("Генерация ответа…")
            self._streaming_answer = ""
            self.chat_log.setHtml("")
            if panel:
                panel.on_generate_start()
        elif stage == "token":
            chunk = event.get("text", "")
            self._streaming_answer += chunk
            if self._current_bubble_label is not None:
                # _current_bubble_label is now a QTextEdit
                te = self._current_bubble_label
                display_streaming = _strip_source_tags(self._streaming_answer)
                te.setPlainText(display_streaming + "▌")
            self.chat_log.setHtml(html.escape(_strip_source_tags(self._streaming_answer)) + "▌")
            if panel:
                panel.on_token()
        elif stage == "error":
            msg = event.get("message", "")
            self._set_status(f"Ошибка: {msg}")
            if panel:
                panel.on_error(msg)
            # Show error in the current bubble
            err_html = f'<span style="color: #f44336;">[Ошибка: {html.escape(msg)}]</span>'
            if self._current_bubble_label is not None:
                self._current_bubble_label.setHtml(err_html)
            self.chat_log.setHtml(err_html)
            self._stop_status_indicator()
            self._stop_meta_pulse()
            self.send_btn.setEnabled(True)
            self.chat_input.setEnabled(True)

    def _on_result(self, result: dict[str, Any]) -> None:
        self._stop_status_indicator()
        self._stop_meta_pulse()
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # Collapse activity panel to summary
        if self._activity_panel is not None:
            self._activity_panel.on_done()

        # Finalize bubble text (remove caret, strip [ИСТОЧНИК N] tags)
        display_answer = _strip_source_tags(answer)
        if self._current_bubble_label is not None:
            te = self._current_bubble_label
            te.setPlainText(display_answer)
            display_html = _markdownish_to_html(display_answer)
            self.chat_log.setHtml(display_html)

        # Add source cards with staggered reveal.
        # We use a simple show() after a delay instead of QGraphicsOpacityEffect
        # fade-in, which avoids stacking multiple effects inside the scroll area
        # and triggering QPainter "Painter not active" errors.
        if self._current_bubble_sources_layout is not None:
            for i, src in enumerate(sources):
                card = self._make_source_card(src)
                self._current_bubble_sources_layout.addWidget(card)
                card.hide()
                delay = i * 80
                QTimer.singleShot(delay, lambda c=card: c.show())

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
            # Defer the fade-in by one event-loop tick so the widget is fully
            # laid out before the painter is activated — prevents QPainter
            # "Painter not active" when the bubble is added while the scroll
            # area is mid-repaint.
            c = self._current_bubble_container
            QTimer.singleShot(0, lambda: fade_in(c, duration=180))
        self._scroll_to_bottom()
        self.chat_input.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._stop_status_indicator()
        self._stop_meta_pulse()
        err_html = f'<span style="color: red;">[Ошибка: {html.escape(msg)}]</span>'
        if self._current_bubble_label is not None:
            self._current_bubble_label.setHtml(err_html)
        self.chat_log.setHtml(err_html)
        if self._activity_panel is not None:
            self._activity_panel.on_error(msg)
        self.send_btn.setEnabled(True)
        self.chat_input.setEnabled(True)

    def _cleanup_worker(self) -> None:
        self.send_btn.setEnabled(True)
        self.chat_input.setEnabled(True)
        self.worker = None

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def _start_new_conversation(self) -> None:
        # Reuse the current conversation if it's already empty — don't create
        # a new one just because the user clicked "Новый чат" again.
        if (self._current_conv is not None
                and not self._current_conv.get("turns")
                and self._current_conv.get("title") == "Новый диалог"):
            self._clear_thread()
            self._favorites_mode = False
            self._refresh_conv_list()
            return
        self._current_conv = self.conv_store.get_or_create_empty()
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
                lbl.setHtml(_markdownish_to_html(content))
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

    def _on_conv_context_menu(self, pos) -> None:
        """Right-click context menu on the conversation list — offers 'Удалить чат'."""
        item = self.conv_list.itemAt(pos)
        if item is None:
            return
        conv_id = item.data(Qt.ItemDataRole.UserRole)
        if not conv_id:
            return

        menu = QMenu(self)
        delete_action = menu.addAction("Удалить чат")
        action = menu.exec(self.conv_list.mapToGlobal(pos))
        if action != delete_action:
            return

        # Confirm before deleting
        reply = QMessageBox.question(
            self,
            "Удалить диалог",
            "Удалить этот диалог? Отменить действие невозможно.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        is_current = (self._current_conv is not None and self._current_conv["id"] == conv_id)
        self.conv_store.delete(conv_id)

        if is_current:
            # Switch to the most recent remaining conversation, or start fresh
            remaining = self.conv_store.get_all()
            if remaining:
                self._current_conv = remaining[0]
                self._clear_thread()
                self._load_conversation(remaining[0]["id"])
            else:
                self._current_conv = self.conv_store.new_conversation()
                self._clear_thread()

        self._refresh_conv_list()

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
                    lbl.setHtml(_markdownish_to_html(turn.get("content", "")))
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
        # Use a QTimer-driven stylesheet pulse — no QGraphicsOpacityEffect,
        # so no QPainter conflicts with sibling widgets.
        self._dot_pulse_bright = True
        self._dot_pulse_timer = QTimer(self)
        self._dot_pulse_timer.setInterval(500)
        self._dot_pulse_timer.timeout.connect(self._pulse_dot_tick)
        self._status_dot.setStyleSheet("color: #4CAF50;")
        self._dot_pulse_timer.start()

    def _pulse_dot_tick(self) -> None:
        """Toggle the dot between bright and dim via stylesheet — no opacity effect."""
        if not getattr(self, "_dot_pulse_going", False):
            return
        self._dot_pulse_bright = not getattr(self, "_dot_pulse_bright", True)
        if self._dot_pulse_bright:
            self._status_dot.setStyleSheet("color: #4CAF50;")
        else:
            self._status_dot.setStyleSheet("color: rgba(76, 175, 80, 60);")

    def _stop_status_indicator(self) -> None:
        self._dot_pulse_going = False
        timer = getattr(self, "_dot_pulse_timer", None)
        if timer is not None:
            timer.stop()
            self._dot_pulse_timer = None
        self.status_row.hide()
        self._status_label.hide()
        self._status_dot.hide()

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _start_meta_pulse(self) -> None:
        """Subtle repeating opacity pulse on meta_label while query is running.

        Uses a QTimer toggling stylesheet alpha — no QGraphicsOpacityEffect,
        so no QPainter conflicts with other animated widgets.
        """
        self._meta_pulse_going = True
        self._meta_pulse_bright = True
        self._meta_pulse_timer = QTimer(self)
        self._meta_pulse_timer.setInterval(600)
        self._meta_pulse_timer.timeout.connect(self._meta_pulse_tick)
        self.meta_label.setStyleSheet("color: rgba(255,255,255,255);")
        self._meta_pulse_timer.start()

    def _meta_pulse_tick(self) -> None:
        """Toggle meta_label between bright and dim via stylesheet."""
        if not self._meta_pulse_going:
            return
        self._meta_pulse_bright = not getattr(self, "_meta_pulse_bright", True)
        if self._meta_pulse_bright:
            self.meta_label.setStyleSheet("color: rgba(255,255,255,255);")
        else:
            self.meta_label.setStyleSheet("color: rgba(255,255,255,100);")

    def _stop_meta_pulse(self) -> None:
        """Stop pulse and restore meta_label to full opacity."""
        self._meta_pulse_going = False
        timer = getattr(self, "_meta_pulse_timer", None)
        if timer is not None:
            timer.stop()
            self._meta_pulse_timer = None
        self.meta_label.setStyleSheet("")

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def _on_open_library(self) -> None:
        from windows.library_dialog import LibraryDialog
        dlg = LibraryDialog(self, db_path=_default_db_path())
        dlg.exec()
        self._refresh_index_status()
        self._reset_pipeline_retriever()

    # ------------------------------------------------------------------
    # Cloud client management
    # ------------------------------------------------------------------

    def _rebuild_cloud_clients(self) -> None:
        """Build/rebuild OllamaCloudClient and CloudUsageTracker from current settings."""
        api_key = self.settings.get("ollama_api_key", "").strip()
        gen_mode = self.settings.get("gen_mode", "local")

        # Always maintain a usage tracker (it's cheap; reads from disk lazily)
        try:
            from core.cloud_usage import CloudUsageTracker
            self._cloud_usage_tracker = CloudUsageTracker()
        except Exception:
            self._cloud_usage_tracker = None

        # Build cloud client only when key is present
        if api_key:
            try:
                from core.cloud import OllamaCloudClient
                self._cloud_client = OllamaCloudClient(api_key)
            except Exception:
                self._cloud_client = None
        else:
            self._cloud_client = None

        # Update indicator visibility: hide for pure-local users with no key
        if self._cloud_indicator is not None:
            if not api_key and gen_mode == "local":
                self._cloud_indicator.hide()
            else:
                self._cloud_indicator.refresh(self._cloud_usage_tracker)

    def _is_online(self) -> bool:
        """Check connectivity to ollama.com (cached ~30s, non-blocking on worker thread)."""
        now = time.monotonic()
        if self._online_cache is not None:
            checked_at, result = self._online_cache
            if now - checked_at < 30.0:
                return result
        try:
            s = socket.create_connection(("ollama.com", 443), timeout=1.5)
            s.close()
            result = True
        except OSError:
            result = False
        self._online_cache = (now, result)
        return result

    def _decide_backend(self) -> str:
        """Return 'local' or 'cloud' based on gen_mode setting and current state."""
        gen_mode = self.settings.get("gen_mode", "local")
        api_key = self.settings.get("ollama_api_key", "").strip()

        if gen_mode == "local":
            return "local"

        if gen_mode == "cloud":
            # Always attempt cloud (pipeline will emit error event if it fails)
            return "cloud"

        # auto: cloud if key + online + not rate-limited, else local
        if not api_key:
            return "local"
        if self._cloud_usage_tracker is not None and self._cloud_usage_tracker.is_rate_limited():
            return "local"
        if not self._is_online():
            return "local"
        return "cloud"

    def _refresh_cloud_indicator(self) -> None:
        if self._cloud_indicator is not None:
            api_key = self.settings.get("ollama_api_key", "").strip()
            gen_mode = self.settings.get("gen_mode", "local")
            if not api_key and gen_mode == "local":
                self._cloud_indicator.hide()
            else:
                self._cloud_indicator.refresh(self._cloud_usage_tracker)

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

        # Build the pipeline whenever retrieval is possible, even without a
        # local GGUF — cloud-only generation (backend="cloud") needs only
        # retrieval; RAGQueryPipeline handles llm_model_path=None gracefully.
        try:
            self.pipeline = RAGQueryPipeline(
                db_path=Path(db) if db else _default_db_path(),
                llm_model_path=Path(llm_path) if llm_path else None,
                top_k=self.settings.get("top_k", 8),
                rerank_top_k=self.settings.get("rerank_top_k", 5),
                rerank_model=rerank_model,
                max_tokens=self.settings.get("max_tokens", 512),
                temperature=self.settings.get("temperature", 0.3),
                llm_n_ctx=self.settings.get("n_ctx", 2048),
                llm_n_threads=self.settings.get("n_threads", 2),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка", f"Не удалось инициализировать пайплайн:\n{exc}")
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
            self._rebuild_cloud_clients()
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


def _configure_logging() -> None:
    """Route structlog + uncaught exceptions to a rotating log file.

    The app normally runs via run_app.bat -> pythonw.exe (no console), so
    without this, logger.error(...) calls and any exception that escapes a
    Qt slot are silently discarded — making a crash impossible to diagnose
    after the fact. Never let logging setup itself break the app.
    """
    try:
        log_dir = data_root() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "engineer_companion.log"

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_log_level,
                structlog.processors.format_exc_info,
                structlog.processors.KeyValueRenderer(key_order=["event"]),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        def _excepthook(exc_type, exc_value, exc_tb):
            logging.getLogger("uncaught").error(
                "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
            )
            sys.__excepthook__(exc_type, exc_value, exc_tb)

        sys.excepthook = _excepthook
    except Exception:
        pass


def main() -> None:
    _configure_logging()
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
