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
