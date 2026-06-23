# Состояние проекта Engineer Companion — на 2026-06-23

## Что это

Офлайн RAG-помощник для инженеров сервиса Varian TrueBeam/VitalBeam: поиск по 12 PDF-руководствам в векторной базе LanceDB с генерацией ответов через локальную LLM Gemma-3-4B, с результатами на русском языке и цитированием источников.

## Окружение (ВАЖНО)

- Канонический venv: `.venv-win` (Python 3.12, нативный Windows). Старые `.venv`, `.venv.bak.*`, `.venv.wsl_bak` — это Linux/WSL venv, **не использовать под Windows**.
- CPU Intel i5-8300H поддерживает только **AVX2**. llama-cpp-python ставить ТОЛЬКО AVX2-сборкой:
  ```
  pip install "llama-cpp-python==0.3.19" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
  ```
  Иначе крэш `STATUS_ILLEGAL_INSTRUCTION` на ядре q4_K_8x8 repack.
- torch CPU-only:
  ```
  pip install torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
  ```
- Запуск тестов:
  ```powershell
  $env:QT_QPA_PLATFORM="offscreen"
  $env:PYTHONPATH="E:\engineer-companion"
  .venv-win\Scripts\python.exe -m pytest tests\ -v
  ```
- Полный список пакетов с точными версиями: `requirements-windows.txt`.

## Что работает (проверено)

- **Ядро:** индексация PDF (PyMuPDF), чанкинг, эмбеддинги (multilingual-e5-small), векторный поиск LanceDB.
- **RAG end-to-end на Windows:** ретрив + генерация Gemma-3-4B, ответы на русском с цитированием источников. Проверено практическими вопросами (Absolute Dose Calibration, interlock reset).
- **Windows GUI (PySide6):** настройки, закладки, история с поиском, экспорт MD/JSON.
- **Тесты: 63/68 зелёных.** (test_core 20/20, test_behavioral_windows 25/25, test_android 13/13, test_e2e 5/10 — 5 падали ТОЛЬКО из-за загрузки модели в сломанном окружении; на `.venv-win` модель грузится.)

## Что НЕ доделано

- **Android:** `SearchViewModel.search()` — заглушка (`"[Shell mode...]"`). Нет классов PythonAdapter/LlamaEngine. `build.gradle` без Chaquopy/CMake/NDK. Нет prebuilt `libllama.so` (`jniLibs` пуст). APK ни разу не собирался. Два рассинхронных плана (MediaPipe vs JNI). `scripts/build_android.py` — старый Kivy/buildozer.
- **Windows EXE** в `dist/` собран в старом окружении (вероятно с AVX-512 wheel) — **НЕ перепроверен** на этом CPU, скорее всего падает. Пересобрать на `.venv-win`.
- Нет README, нет git-тега.

## Качество ответов RAG (наблюдения)

- Генерация заземлена на источники, без галлюцинаций.
- **Дефекты:** иногда неточная атрибуция страниц/документов в цитатах; в ответ попадают нерелевантные извлечённые чанки; recall слабоват когда ответ размазан по корпусу.
- В корпусе есть **сканированные страницы** (напр. TrueBeam 3.0 Field Service Databook стр.656, IEC Functional Performance Characteristics стр.64) — PyMuPDF извлекает из них мусор, который засоряет индекс. Кандидат на OCR-очистку (узко, только сканы) либо фильтрацию мусорных чанков.

## Известные риски

- DLL-конфликт `llama_cpp` + `torch` на Windows (обойдён проверкой зависимостей в subprocess в `scripts/build_windows.py`).
- `max_tokens` по умолчанию обрезает длинные ответы (поднимать до ~512).
