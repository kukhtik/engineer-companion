# Engineer Companion

Offline RAG-компаньон для сервисного инженера оборудования Varian TrueBeam/VitalBeam.

## Правила
- НЕ писать PLANNING.md
- План из git+tests, т.е. каждая фича — сначала тест, потом код
- Все паттерны UI — из impeccable-ui (dribbble-dark palette)
- Behavioral тесты: имитируем пользователя, ассерт на состояние виджетов
- Android/Windows — один Python-core (llama-cpp + lancedb), две обёртки
- Проектная документация только здесь, не на десктопе

## Структура
- `core/` — RAG pipeline, indexer, vector store, LLM inference (Python)
- `windows/` — PySide6 GUI
- `android/` — Android wrapper (Kivy/Chaquopy или native)
- `tests/` — behavioral UI tests + unit tests
- `assets/models/` — GGUF модели
- `assets/db/` — LanceDB векторная база
- `design/` — UI токены, QML/CSS, скриншоты референсов
- `docs/` — архитектурные заметки

## Stack
- Core: Python 3.11, llama-cpp-python, lancedb, sentence-transformers
- Windows: PySide6, QtTest
- Android: Kivy (python-for-android) или Chaquopy + Jetpack Compose
- Embeddings: multilingual-e5-small (~400MB)
- LLM: Qwen 2.5 Instruct 7B Q4_K_M (~4.5GB) или Gemma 3 4B IT
- Vector DB: LanceDB file-based

## Constraints
- Всё работает offline: модели + embeddings + DB — локально
- Samsung S25: <=6GB RAM для модели, NPU если доступен через MediaPipe
- Первый MVP только Windows, Android — после стабилизации core

## Текущий статус (2026-05-07)
### Сделано
- `core/indexer.py` — PDF → chunks → embeddings (CPU) → LanceDB, 12 PDF (~4000 стр), 23757 чанков, 45 МБ
- `core/query.py` — Retriever (cosine search), PromptBuilder, RAGQueryPipeline
- `windows/main_window.py` — PySide6 GUI: search bar, results list, chat panel, QueryWorker (QThread)
- `design/tokens.py` — impeccable-ui dark palette, Qt stylesheet
- `tests/test_behavioral_windows.py` — 9 behavioral UI tests, все проходят (headless)
- `.venv` создан через `uv venv .venv --python 3.11`, PySide6 установлен

### Блокер
- `llama-cpp-python` не установлен ни в system Python, ни в `.venv`
- `sentence-transformers` в `.venv` не работает (нужен torch, но `.venv` пуст — system-site-packages не подцепил system dist-packages)
- End-to-end с LLM не проверен

### Следующие шаги
1. Установить `llama-cpp-python` + `sentence-transformers` + `torch` CPU в `.venv`
2. End-to-end: `RAGQueryPipeline.ask()` с Gemma 3 4B и реальной DB
3. Core unit tests: `test_indexer_pdf_chunking`, `test_retriever_cosine_search`, `test_prompt_builder`
4. GUI: rich text ответов (QTextBrowser), история чата, избранное/закладки
5. Entry point: `python -m engineer_companion` или `run.sh`
6. Git commit с тегом `v0.1-mvp`
