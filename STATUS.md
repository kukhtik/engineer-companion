# Состояние проекта Engineer Companion — на 2026-06-24

## Что это

Офлайн RAG-помощник для инженеров сервиса Varian TrueBeam/VitalBeam: поиск по 12 PDF-руководствам в векторной базе LanceDB с генерацией ответов через локальную LLM Gemma-3-4B, ответы на русском языке с цитированием источников.

---

## Окружение (ВАЖНО)

- Канонический venv: `.venv-win` (Python 3.12, нативный Windows). Старые `.venv`, `.venv.bak.*`, `.venv.wsl_bak` — это Linux/WSL venv, **не использовать под Windows**.
- CPU Intel i5-8300H поддерживает только **AVX2** (нет AVX-512). llama-cpp-python ставить ТОЛЬКО AVX2-сборкой:
  ```
  pip install "llama-cpp-python==0.3.19" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
  ```
  Иначе крэш `STATUS_ILLEGAL_INSTRUCTION` на ядре q4_K_8x8 repack.
- torch CPU-only:
  ```
  pip install torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
  ```
- Полный список пакетов с точными версиями: `requirements-windows.txt`.
- Запуск тестов:
  ```powershell
  $env:QT_QPA_PLATFORM="offscreen"
  $env:PYTHONPATH="E:\engineer-companion"
  .venv-win\Scripts\python.exe -m pytest tests\ -v
  ```

---

## Что работает (проверено)

- **Ядро:** индексация PDF (PyMuPDF), чанкинг с фильтрацией мусора, эмбеддинги (multilingual-e5-small), векторный поиск LanceDB.
- **RAG end-to-end на Windows:** ретрив + кросс-энкодер реранкинг + генерация Gemma-3-4B. Ответы на русском с цитированием. Проверено практическими вопросами (Absolute Dose Calibration, interlock reset).
- **Windows GUI (PySide6):** настройки, закладки, история с поиском, экспорт MD/JSON. Запуск: `python engineer_companion.py`.
- **CrossEncoder реранкер включён по умолчанию:** модель `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, top_k=20 → реранк до 5. `max_tokens=512`.
- **Офлайн-портабельность:** embedder (`multilingual-e5-small`) и reranker (`mmarco-mMiniLMv2-L12-H384-v1`) собраны в `assets/models/embedder` и `assets/models/reranker`, загружаются с `local_files_only=True`. Веса (~985 МБ) в `.gitignore`, но включены в EXE через PyInstaller (директория `assets/` — data dir).
- **Фильтрация мусорных чанков:** функция `is_low_quality_chunk` отсеивает некачественные блоки при индексации (`drop_low_quality=True`). После переиндексации: 24 508 → **11 629 чанков**, мусор 44% → **5,5%**. Чистые документы сохранили 92–97% чанков. Исключение: два сканированных TrueBeam Field Service Databook сохранили 29%/40% (текст из сканов).
- **In-app OCR (Файл → OCR и индексация...):** панель управления OCR с живым прогрессом и общей статистикой. OCR запускается как изолированный subprocess (paddle никогда не импортируется в процессе GUI — обходит Windows DLL-конфликт paddle-перед-torch). Кэш результатов: `scripts/ocr_cache.jsonl` (JSONL, возобновляемый). OCR доступен **только при запуске из исходников** (`.venv-win`); в собранном EXE пункт меню виден, но отключён с подсказкой.
- **Стек PaddleOCR:** paddleocr 2.8.1 / paddlepaddle 2.6.2 / opencv-contrib-python 4.10.0.84 / numpy 1.26.4 — сосуществуют с RAG-стеком в `.venv-win`. Paddle изолирован в subprocess, чтобы избежать конфликта DLL (shm.dll) при загрузке paddle после torch.
- **Резервная копия старого индекса:** `assets/db/engineer.db.bak`.
- **Тесты: 85 зелёных, 1 xfailed** (`.venv-win`). `tests/conftest.py` предварительно прогревает pyarrow, чтобы избежать сбоя pytest-qt из-за порядка загрузки DLL.

---

## OCR: что сделано и что нет

OCR — это операция индексации **по выбору пользователя**, не автоматическая. Полный прогон занимает ~8–13 ч для обоих Databook суммарно; процесс возобновляем через кэш. До запуска OCR + переиндексации индекс содержит только фильтрованный (не-OCR) контент — Field Service Databook дают 29–40% чанков. После OCR + реиндексации (~6 000 дополнительных чанков) качество поиска по этим документам значительно улучшится.

---

## Состояние EXE (dist/)

EXE в `dist/` **УСТАРЕЛ**: собран до добавления OCR-функции, реранкера и фильтрации чанков. Для финального релиза: одна команда:
```
python scripts/build_windows.py
```
Пересобранный EXE включает офлайн-модели, реранкер, фильтрованный индекс и поисковый интерфейс. Пункт меню OCR в EXE отключён (OCR — инструмент разработчика/сопровождения, не конечного пользователя).

---

## Открытые решения (не сделано)

### Android
- `SearchViewModel.search()` возвращает заглушку `"[Shell mode...]"`.
- Нет классов PythonAdapter/LlamaEngine.
- `build.gradle` без Chaquopy/CMake/NDK; нет prebuilt `libllama.so` (`jniLibs` пуст).
- APK ни разу не собирался.
- Android не запускает OCR: телефон потребляет уже улучшенный индекс, собранный на десктопе.
- Решение **не принято**: завершить Android или заморозить как Windows-only v1.

### Прочее
- Нет git-тега для текущего рабочего состояния.

---

## Известные риски и ограничения

- DLL-конфликт `llama_cpp` + `torch` на Windows (обойдён через subprocess в `scripts/build_windows.py` и OCR-subprocess в `windows/ocr_dialog.py`).
- 4B модель иногда ошибается в атрибуции страниц/документов во встроенных цитатах; ориентироваться на список источников под ответом.
- Два Field Service Databook дают лишь 29–40% чанков до OCR.
