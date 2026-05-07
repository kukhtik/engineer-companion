"""Android MVP: Kivy UI for offline RAG companion.

Single-screen app: search bar, scrollable results, chat area.
Uses design tokens (dribbble-dark CSS variant) via kv styling.

Updated: asset resolver, thread safety, Chaquopy-compatible.
"""

import sys
from pathlib import Path

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.properties import StringProperty
from kivy.clock import Clock
from kivy.core.window import Window

# Allow imports from project root
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# Shared asset resolver
from android.assets_loader import resolve_db_path, resolve_model_path  # type: ignore

# Attempt to import core pipeline; gracefully degrade if deps missing
try:
    from core.query import RAGQueryPipeline  # type: ignore
except Exception:
    RAGQueryPipeline = None  # type: ignore


class ResultCard(BoxLayout):
    """Single search result row."""
    source_text = StringProperty("")
    page_text = StringProperty("")
    section_text = StringProperty("")


class ChatMessage(BoxLayout):
    """Chat bubble-like label."""
    content = StringProperty("")


class RootWidget(BoxLayout):
    orientation = "vertical"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = None
        Window.clearcolor = (0.04, 0.04, 0.04, 1)  # near-black bg
        self._build_ui()
        Clock.schedule_once(lambda dt: self._lazy_init_pipeline(), 0)

    def _build_ui(self):
        # Search row
        search_row = BoxLayout(size_hint_y=None, height=60, padding=8, spacing=8)
        self.search_input = TextInput(
            multiline=False,
            hint_text="Вопрос по TrueBeam / VitalBeam...",
            font_size="16sp",
            background_color=(0.08, 0.08, 0.08, 1),
            foreground_color=(0.95, 0.95, 0.96, 1),
            cursor_color=(0.24, 0.72, 0.71, 1),  # accent-cyan
        )
        self.search_btn = Button(
            text="Найти",
            font_size="14sp",
            background_color=(0.13, 0.13, 0.13, 1),
            color=(0.95, 0.95, 0.96, 1),
        )
        self.search_btn.bind(on_press=self._on_search)
        search_row.add_widget(self.search_input)
        search_row.add_widget(self.search_btn)
        self.add_widget(search_row)

        # Results list (scrollable)
        results_scroll = ScrollView()
        self.results_list = BoxLayout(orientation="vertical", size_hint_y=None)
        self.results_list.bind(minimum_height=self.results_list.setter("height"))
        results_scroll.add_widget(self.results_list)
        self.add_widget(results_scroll)

        # Chat / answer area
        chat_scroll = ScrollView(size_hint_y=0.5)
        self.chat_area = BoxLayout(orientation="vertical", size_hint_y=None)
        self.chat_area.bind(minimum_height=self.chat_area.setter("height"))
        chat_scroll.add_widget(self.chat_area)
        self.add_widget(chat_scroll)

        # Status label
        self.status_label = Label(
            text="Offline RAG — документы локальны",
            font_size="12sp",
            color=(0.48, 0.48, 0.48, 1),  # text-muted
            size_hint_y=None,
            height=30,
        )
        self.add_widget(self.status_label)

    def _lazy_init_pipeline(self, *_):
        db_path = resolve_db_path()
        llm_path = resolve_model_path()

        if RAGQueryPipeline is None:
            self.status_label.text = "Ошибка: core.query недоступен"
            return
        if not db_path.exists():
            self.status_label.text = f"Ошибка: база не найдена ({db_path})"
            return

        try:
            self.pipeline = RAGQueryPipeline(
                db_path=db_path,
                llm_model_path=llm_path,
            )
            model_status = "LLM загружена" if llm_path and llm_path.exists() else "только поиск (без LLM)"
            self.status_label.text = f"Pipeline OK — {model_status}"
        except Exception as exc:
            self.status_label.text = f"Ошибка инициализации: {exc}"

    def _on_search(self, instance):
        query = self.search_input.text.strip()
        if not query:
            return
        self.status_label.text = "Ищем..."
        self.search_btn.disabled = True
        self.results_list.clear_widgets()

        if self.pipeline is None:
            self._add_chat_message("[Ошибка: pipeline не загружен]")
            self.status_label.text = "Ошибка pipeline"
            self.search_btn.disabled = False
            return

        # Off-thread via Clock so UI doesn't freeze
        Clock.schedule_once(lambda dt: self._do_query(query), 0)

    def _do_query(self, query):
        try:
            result = self.pipeline.ask(query)
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            # UI updates must happen on main thread
            def _update_ui(*_):
                self._add_chat_message(f"Ответ:\n{answer}")
                for src in sources:
                    card = ResultCard()
                    card.source_text = src["source"]
                    card.page_text = f"стр.{src['page']}"
                    card.section_text = src["section"][:60]
                    self.results_list.add_widget(card)
                self.status_label.text = f"Источников: {len(sources)}"
                self.search_btn.disabled = False
            Clock.schedule_once(_update_ui, 0)

        except Exception as exc:
            def _show_err(*_):
                self._add_chat_message(f"[Ошибка: {exc}]")
                self.status_label.text = "Ошибка"
                self.search_btn.disabled = False
            Clock.schedule_once(_show_err, 0)

    def _add_chat_message(self, text: str):
        msg = ChatMessage(content=text)
        self.chat_area.add_widget(msg)


class EngineerCompanionApp(App):
    def build(self):
        return RootWidget()


if __name__ == "__main__":
    EngineerCompanionApp().run()
