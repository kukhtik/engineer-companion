# Engineer Companion

Офлайн RAG-помощник для инженеров сервиса Varian TrueBeam/VitalBeam. Индексирует 12 PDF-руководств в векторной базе LanceDB, использует локальную LLM Gemma-3-4B для генерации ответов на русском языке с цитированием источников. Работает полностью без интернета: все модели упакованы в `assets/models/`.

---

## Архитектура

```
engineer-companion/
├── core/           # Общая логика: индексация PDF, чанкинг, эмбеддинги, запросы RAG
│   ├── indexer.py  # Индексация + фильтрация мусорных чанков (is_low_quality_chunk)
│   └── query.py    # RAGQueryPipeline: retrieval → rerank → генерация
├── windows/        # Desktop UI на PySide6
│   └── main_window.py
├── android/        # Заготовка Android (нефункциональна, см. ниже)
├── assets/
│   ├── db/         # LanceDB индекс (engineer.db), резерв (engineer.db.bak)
│   ├── models/
│   │   ├── embedder/   # multilingual-e5-small (локальные веса, ~120 МБ)
│   │   └── reranker/   # mmarco-mMiniLMv2-L12-H384-v1 (~90 МБ)
│   └── llm/        # gemma-3-4b-it-Q4_K_M.gguf (~2.5 ГБ)
├── scripts/
│   └── build_windows.py  # PyInstaller сборка EXE
├── tests/
├── engineer_companion.py  # Точка входа
└── requirements-windows.txt
```

**Модели:**
| Роль | Модель |
|------|--------|
| Эмбеддер | `multilingual-e5-small` |
| Реранкер | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| LLM | `gemma-3-4b-it-Q4_K_M.gguf` (llama.cpp, Q4_K_M) |

Веса моделей не хранятся в git (~985 МБ суммарно), но включаются в EXE при сборке через PyInstaller.

---

## Установка (Windows)

### Требования

- Python 3.12 (нативный Windows)
- CPU с поддержкой **AVX2** (тест: `python -c "import cpuinfo; print(cpuinfo.get_cpu_info()['flags'])"`)

### Создать venv и установить зависимости

```powershell
python -m venv .venv-win
.venv-win\Scripts\activate
pip install -r requirements-windows.txt
```

### КРИТИЧНО: AVX2-сборка llama-cpp-python

Стандартный wheel llama-cpp-python требует AVX-512 и **падает** с `STATUS_ILLEGAL_INSTRUCTION` на большинстве потребительских CPU (в т.ч. Intel i5/i7 8–10 поколений). Устанавливать строго так:

```
pip install "llama-cpp-python==0.3.19" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

torch CPU-only:

```
pip install torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

Точные версии всех пакетов зафиксированы в `requirements-windows.txt`.

---

## Запуск GUI

```powershell
.venv-win\Scripts\python.exe engineer_companion.py
```

---

## Сборка EXE

```powershell
.venv-win\Scripts\python.exe scripts/build_windows.py
```

EXE окажется в `dist/`. Скрипт автоматически упаковывает `assets/` (модели, индекс) и нативные DLL llama_cpp. Собирать на Windows с AVX2 CPU.

> **Внимание:** EXE в текущем `dist/` устарел — собран до изменений с реранкером и фильтрацией чанков. Пересобрать перед релизом.

---

## Запуск тестов

```powershell
$env:QT_QPA_PLATFORM="offscreen"
$env:PYTHONPATH="E:\engineer-companion"
.venv-win\Scripts\python.exe -m pytest tests\ -v
```

Ожидаемый результат: 67 passed, 1 xfail. Тест e2e может дать flaky `exit 5` (Windows threading) — не регрессия, нужен чистый перезапуск.

---

## Переиндексация

```powershell
.venv-win\Scripts\python.exe -m core.indexer
```

Низкокачественные чанки (мусор из сканов, обрывки) автоматически отфильтровываются (`drop_low_quality=True`). Текущий индекс: **11 629 чанков** (было 24 508 до фильтрации).

---

## Известные ограничения

- **Два Field Service Databook (Vol1/Vol2) частично сканированные** (~50–60% страниц): PyMuPDF вытаскивает мусор, полезного текста мало. OCR (PaddleOCR) не запущен — оценочно ~6 000 ценных чанков с таблицами и спецификациями не проиндексированы.
- **Android не функционален:** SearchViewModel возвращает заглушку, нет интеграции с LLM/эмбеддерами, APK никогда не собирался. Заморожен как shell.
- **Встроенные цитаты в тексте** иногда неточны по номерам страниц/документов (4B модель). Список источников под ответом надёжнее.

---

## Текущее состояние

См. [STATUS.md](STATUS.md).
