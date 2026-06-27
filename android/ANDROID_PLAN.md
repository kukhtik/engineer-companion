# Android — план реализации (сводный, заменяет PLAN.md)

## Статус реализации (готово)

Реализовано и собрано (`assembleDebug` → `app-debug.apk` ~564 MB, ожидается ~350 MB после FP16):

- **Хостовые артефакты:** ONNX-экспорт эмбеддера (e5-small-v2, ~100 MB), `tokenizer.onnx` с паритетом косинуса = 1.0 по отношению к HF-токенизатору, плоский бинарный индекс (`chunks_index.bin`, 11 629 чанков из `engineer.db`).
- **Сборочная система:** `app/build.gradle` — ONNX Runtime Android, двойная тема (Dribbble Dark + светлая), `externalNativeBuild` / CMake, `abiFilters "arm64-v8a"`.
- **Compose UI:** чат-центричный интерфейс — боковой DrawerRail, пузыри сообщений, карточки источников, стадии активности, избранное, удаление чатов, потоковый вывод, две темы.
- **OnnxRetriever:** реальный on-device ретрив (ONNX Runtime), фильтр мусорных чанков, top-K косинусный поиск.
- **llama.cpp b4600:** кросс-компиляция для arm64 через NDK/CMake (все `.so`-библиотеки включены в APK).
- **LlamaEngine + JNI:** обёртка с корутин-Mutex, потоковая генерация токенов.
- **PromptBuilder:** шаблон на русском (порт из `core/query.py`).
- **FullRagEngine:** полный пайплайн retrieve → prompt → generate на `Dispatchers.IO`.
- **SetupScreen:** SAF-пикер для импорта GGUF пользователем.

**Остаток / TODO:**
- On-device верификация на Samsung S25: установка APK, push GGUF (2.4 GB), проверка загрузки `OrtxPackage` native lib, паритет токенизатора на устройстве, замер скорости генерации.
- Доставка модели в продакшене: APK ~350 MB включает `model.onnx` FP16 (224 MB, сокращение с 448 MB FP32 — экономия ~215 MB); для Google Play всё равно нужна Play Asset Delivery или загрузка по требованию.
- Опциональное Vulkan-ускорение для LLM (Adreno 750: ожидаемо ~25–40 tok/s vs ~8–15 tok/s CPU).
- Сохранение темы через DataStore (сейчас не персистируется).
- PDF/page viewer на Android (тап на источник сейчас показывает TODO-тост).

---

## Тест на устройстве (S25)

1. **Собрать APK:**
   ```
   cd android
   .\gradlew.bat :app:assembleDebug
   ```
   Результат: `app\build\outputs\apk\debug\app-debug.apk`

2. **Установить на устройство:**
   ```
   adb install -r app\build\outputs\apk\debug\app-debug.apk
   ```

3. **Доставить GGUF-модель (2.4 GB) — два варианта:**
   - *Через SetupScreen (надёжный путь):*
     ```
     adb push assets\models\gemma-3-4b-it-Q4_K_M.gguf /sdcard/Download/
     ```
     Затем в приложении открыть SetupScreen и выбрать файл через SAF-пикер.
   - *Напрямую в файлы приложения (требует `adb root` или `run-as`):*
     ```
     adb push assets\models\gemma-3-4b-it-Q4_K_M.gguf \
         /data/data/com.varian.engcomp/files/models/gemma-3-4b-it-Q4_K_M.gguf
     ```

4. **Запустить и смотреть logcat:**
   ```
   adb logcat | findstr "engineer_companion EngineerCompanion ort llama"
   ```
   Искать: `UnsatisfiedLinkError` (`.so` не загрузилась), ошибки загрузки модели, проблемы токенизатора.

---

> Статус: ПЛАН, код не написан. Этот документ сводит два расходившихся подхода
> (MediaPipe в `PLAN.md` против JNI/llama.cpp) в один выбранный путь.
> Цель: довести Android-клиент от нерабочей Compose-заглушки до APK на Samsung S25.

## Текущее состояние (проверено по коду)
- Compose-оболочка компилируется: `MainActivity.kt`, `SearchViewModel.kt` (заглушка —
  `search()` возвращает `"[Shell mode...]"`), `AnswerPanel.kt`, `SourcesList.kt`,
  `DribbbleDark.kt`, дата-классы.
- `android/chaquopy/chaquopy_bridge.py` и `android/assets_loader.py` — есть.
- `native-lib.cpp` — JNI-заглушка на **устаревшем** API llama.cpp (`llama_eval`,
  `llama_sample`). `CMakeLists.txt` линкует несуществующий prebuilt `libllama.so`
  (каталог `jniLibs/arm64-v8a/` отсутствует).
- `app/build.gradle` — минимальная оболочка: нет Chaquopy, нет CMake, нет MediaPipe.
- `scripts/build_android.py` — старый Kivy/buildozer (мёртвый код).
- `PythonAdapter.kt` / `LlamaEngine.kt` — НЕ существуют. `IMPLEMENTATION_PLAN.md` — НЕ существует
  (есть только `PLAN.md` с MediaPipe).

## Ключевые решения
1. **LLM-инференс → JNI + llama.cpp (сборка из исходников через CMake FetchContent).**
   GGUF уже есть и совпадает с десктопом; у llama.cpp первоклассная поддержка Gemma 3.
   MediaPipe отклонён: поддержка Gemma 3 в стабильном релизе не подтверждена, конвертация
   в `.task` — чёрный ящик с риском. CPU-only: ~8–15 tok/s; Vulkan (Adreno 750): ~25–40 tok/s.
2. **Ретрив → ONNX Runtime Mobile + плоский бинарный индекс в Kotlin. Без Chaquopy/LanceDB/torch
   на Android.** Причина: LanceDB (Rust) и torch не кросс-компилируются под Android через
   Chaquopy чисто; это главный блокер. Эмбеддер e5-small экспортируется в ONNX (~100 МБ),
   косинусный поиск — в Kotlin. Reranker на Android **выключаем** (470 МБ + второй проход).
3. **Доставка GGUF (2.4 ГБ):** не в APK (превышает лимиты). Для тестов — `adb push`; для
   продакшена — загрузка при первом запуске / выбор файла пользователем.

## Размеры на устройстве (выбранный путь)
ONNX-эмбеддер ~100 МБ; индекс ~21 МБ; базовый APK ~30 МБ; GGUF 2.4 ГБ (отдельно). Итого ~2.6 ГБ.

## Шаги (в порядке зависимостей)
- **Фаза 0 — окружение (0.5 дн):** Android Studio, SDK 34, NDK r27c, CMake 3.22+.
- **Фаза 1 — сборочная система (1 дн):** переписать `app/build.gradle` (externalNativeBuild/CMake,
  `abiFilters "arm64-v8a"`, onnxruntime-android, coroutines, gson; без Chaquopy/MediaPipe);
  починить `scripts/build_android.py` → вызов `gradlew assembleDebug`.
- **Фаза 2 — ONNX-экспорт + плоский индекс (1 дн, на хосте):**
  `optimum-cli export onnx --model .../embedder ...`; скрипт `scripts/build_android_index.py`
  сериализует чанки+векторы из `engineer.db` в `chunks_index.bin`.
- **Фаза 3 — кросс-компиляция llama.cpp + переписать `native-lib.cpp` (2 дн):** CMake FetchContent
  на пин-тег llama.cpp; новый API (`llama_decode`, `llama_sampler_chain_*`, `llama_batch`).
- **Фаза 4 — Kotlin-движки (1.5 дн):** `LlamaEngine.kt` (JNI + coroutine Mutex),
  `OnnxRetriever.kt` (токенизация через `onnxruntime-extensions-android`, эмбеддинг, косинус, top-K).
- **Фаза 5 — копирование ассетов (0.5 дн):** `AssetCopier.kt` (assets → filesDir на первом запуске).
- **Фаза 6 — обвязка (1 дн):** переписать `SearchViewModel.search()` (retrieve→prompt→generate
  на `Dispatchers.IO`); `PromptBuilder.kt` (порт из `core/query.py`); `EngineerCompanionApp.kt`.
- **Фаза 7 — first-launch UX (0.5 дн):** `SetupScreen.kt` (загрузка/выбор GGUF).
- **Фаза 8 — `scripts/build_android.py` (0.25 дн).**
- **Фаза 9 — сборка/установка/тест на S25 (2 дн).**

## Главные риски
- **Сборка llama.cpp через FetchContent + совместимость NDK (HIGH)** — пинить тег, тестировать кросс-компиляцию отдельно.
- **Точность ONNX-токенизатора (MEDIUM)** — должна совпасть с HF/sentence-transformers, иначе
  эмбеддинги не сойдутся с индексом; использовать `onnxruntime-extensions-android`, проверять на парах (запрос→ожидаемый top).
- **Доставка GGUF 2.4 ГБ (HIGH)** — для тестов `adb push` в `files/models/`.
- **API churn llama.cpp (MEDIUM)** — пин одной версии.
- **Память при одновременной загрузке (MEDIUM)** — ленивая загрузка GGUF, `n_ctx=2048`.
- **JNI + потоки (MEDIUM)** — `llama_decode` не потокобезопасен; сериализовать через Mutex.

## Оценка и фазирование
**Итого ~10 дней** + 3–5 дней буфера на NDK/токенизатор.
Минимальный вертикальный срез:
- **Срез A (только ретрив, дни 1–4):** фазы 0,1,2,4(OnnxRetriever),5; `search()` показывает top-5 в `SourcesList`, без LLM. Снимает главный риск (токенизатор) до возни с NDK.
- **Срез B (полный пайплайн, дни 4–8):** фазы 3,4(LlamaEngine),6,7; GGUF через `adb push`; end-to-end.

## Критичные файлы
`android/app/build.gradle`, `android/app/src/main/cpp/CMakeLists.txt`,
`android/app/src/main/cpp/native-lib.cpp`,
`android/app/src/main/java/com/varian/engcomp/viewmodel/SearchViewModel.kt`,
`scripts/build_android.py`.

## OCR на Android

**Принцип: Android не запускает OCR — он потребляет готовый индекс.**

OCR многосотстраничных сканированных PDF (Field Service Databook Vol1/Vol2)
с помощью PaddleOCR/paddlepaddle практически неосуществим на телефоне:
paddle для Android огромен, не входит в выбранный стек (ONNX/JNI) и официально
не поддерживается под Android. OCR — это операция **времени индексации**, которая
выполняется на десктопе однократно (8–13 ч на оба Databook) и сохраняет результат
в кэш `scripts/ocr_cache.jsonl` и в `assets/db/engineer.db`.

Android-клиент просто **использует** уже улучшенный индекс (`engineer.db`),
собранный после OCR на десктопе. Телефон получает тот же файл базы (доставляется
как asset или синхронизируется вручную) и выполняет ONNX-ретрив + LLM-инференс
поверх него — без какого-либо OCR на устройстве.

**Если потребуется «запустить OCR с телефона» (вне scope v1):** реалистичный путь —
companion-режим: телефон отправляет запрос на сопряжённый десктоп/сервер, тот
запускает OCR + переиндексацию и синхронизирует обновлённый `engineer.db` обратно.
Встроенный OCR на устройстве не рассматривается.
