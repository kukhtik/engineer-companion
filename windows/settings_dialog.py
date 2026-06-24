"""Settings dialog for Engineer Companion — Phase 4: tabbed QTabWidget.

Persists to ~/.engineer-companion/settings.json.
Four tabs (Russian labels):
  1. Библиотека и индекс  — paths, index stats, library button
  2. Качество поиска       — rerank, top_k, rerank_top_k, temperature, max_tokens, n_ctx, n_threads
  3. Внешний вид           — theme, font_scale, density
  4. Управление моделями   — LLM path + presence, embedder/reranker info, check button
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


SETTINGS_DIR = Path.home() / ".engineer-companion"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    # Tab 1
    "db_path": "",
    # Tab 2
    "top_k": 8,
    "rerank_enabled": True,
    "rerank_top_k": 5,
    "temperature": 0.3,
    "max_tokens": 512,
    "n_ctx": 2048,
    "n_threads": 2,
    # Tab 3
    "theme": "dark",
    "font_scale": 100,   # percent: 90–140
    "density": "comfortable",  # "comfortable" | "compact"
    # Tab 4
    "llm_model_path": "",
}

DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def load_settings() -> dict[str, Any]:
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Merge: fill missing keys with defaults (backward compat)
            result = dict(DEFAULT_SETTINGS)
            result.update(data)
            return result
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def _path_status(p: Path | str) -> str:
    if not p:
        return "не указан"
    path = Path(p)
    if path.exists():
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            return f"есть ({size_mb:.1f} МБ)"
        except OSError:
            return "есть"
    return "НЕ НАЙДЕНА"


class SettingsDialog(QDialog):
    """Tabbed settings dialog with four sections."""

    def __init__(
        self,
        parent=None,
        current: dict[str, Any] | None = None,
        theme_manager=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(560)
        self.setMinimumHeight(420)

        self.current = dict(DEFAULT_SETTINGS)
        if current:
            self.current.update(current)

        self._theme_manager = theme_manager

        self._build_ui()
        self._populate()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_library(), "Библиотека и индекс")
        self.tabs.addTab(self._build_tab_quality(), "Качество поиска")
        self.tabs.addTab(self._build_tab_appearance(), "Внешний вид")
        self.tabs.addTab(self._build_tab_models(), "Модели")
        layout.addWidget(self.tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ---- Tab 1: Библиотека и индекс ----

    def _build_tab_library(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(10)
        v.setContentsMargins(12, 12, 12, 12)

        from windows.app_paths import library_dir, db_path as _db_path, ocr_cache_path
        lib = library_dir()
        ocr = ocr_cache_path()

        info_group = QGroupBox("Пути данных (только чтение)")
        info_form = QFormLayout(info_group)
        info_form.setSpacing(6)

        def _ro_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setObjectName("muted")
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            return lbl

        info_form.addRow("Библиотека PDF:", _ro_label(str(lib)))
        self._default_db = str(_db_path())
        info_form.addRow("База данных:", _ro_label(self._default_db))
        info_form.addRow("OCR кэш:", _ro_label(str(ocr)))
        v.addWidget(info_group)

        # Index stats
        stats_group = QGroupBox("Статистика индекса")
        stats_v = QVBoxLayout(stats_group)
        self.stats_label = QLabel("Загрузка...")
        self.stats_label.setObjectName("muted")
        stats_v.addWidget(self.stats_label)
        refresh_stats_btn = QPushButton("Обновить статистику")
        refresh_stats_btn.clicked.connect(self._refresh_stats)
        stats_v.addWidget(refresh_stats_btn)
        v.addWidget(stats_group)

        # Library button
        lib_btn = QPushButton("Открыть библиотеку…")
        lib_btn.clicked.connect(self._open_library)
        v.addWidget(lib_btn)

        # DB path override
        db_group = QGroupBox("Переопределить путь к базе данных (для опытных пользователей)")
        db_h = QHBoxLayout(db_group)
        self.db_path_edit = QLineEdit()
        self.db_path_edit.setPlaceholderText(f"По умолчанию: {self._default_db}")
        db_browse = QPushButton("Обзор")
        db_browse.clicked.connect(self._browse_db)
        db_h.addWidget(self.db_path_edit, stretch=1)
        db_h.addWidget(db_browse)
        v.addWidget(db_group)

        v.addStretch()
        return w

    # ---- Tab 2: Качество поиска ----

    def _build_tab_quality(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(10)
        v.setContentsMargins(12, 12, 12, 12)

        retrieval_group = QGroupBox("Поиск")
        form = QFormLayout(retrieval_group)
        form.setSpacing(8)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 50)
        form.addRow("Top K (кандидаты):", self.top_k_spin)

        self.rerank_cb = QCheckBox("Включить реранкинг (cross-encoder)")
        form.addRow("", self.rerank_cb)

        self.rerank_top_k_spin = QSpinBox()
        self.rerank_top_k_spin.setRange(1, 20)
        form.addRow("Rerank Top K:", self.rerank_top_k_spin)

        self.rerank_cb.toggled.connect(self.rerank_top_k_spin.setEnabled)
        v.addWidget(retrieval_group)

        gen_group = QGroupBox("Генерация")
        gen_form = QFormLayout(gen_group)
        gen_form.setSpacing(8)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 1.0)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setDecimals(2)
        gen_form.addRow("Temperature:", self.temperature_spin)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 4096)
        self.max_tokens_spin.setSingleStep(64)
        gen_form.addRow("Max tokens:", self.max_tokens_spin)

        self.n_ctx_spin = QSpinBox()
        self.n_ctx_spin.setRange(512, 8192)
        self.n_ctx_spin.setSingleStep(512)
        gen_form.addRow("Контекст LLM (n_ctx):", self.n_ctx_spin)

        self.n_threads_spin = QSpinBox()
        self.n_threads_spin.setRange(1, 16)
        gen_form.addRow("Потоки CPU (n_threads):", self.n_threads_spin)

        v.addWidget(gen_group)
        v.addStretch()
        return w

    # ---- Tab 3: Внешний вид ----

    def _build_tab_appearance(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(10)
        v.setContentsMargins(12, 12, 12, 12)

        theme_group = QGroupBox("Тема")
        theme_form = QFormLayout(theme_group)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Тёмная (Dark)", "dark")
        self.theme_combo.addItem("Светлая (Light)", "light")
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed_live)
        theme_form.addRow("Тема:", self.theme_combo)
        v.addWidget(theme_group)

        font_group = QGroupBox("Масштаб шрифта")
        font_form = QFormLayout(font_group)
        font_row = QHBoxLayout()
        self.font_scale_spin = QSpinBox()
        self.font_scale_spin.setRange(90, 140)
        self.font_scale_spin.setSingleStep(5)
        self.font_scale_spin.setSuffix(" %")
        self.font_scale_spin.valueChanged.connect(self._on_font_scale_changed_live)
        font_row.addWidget(self.font_scale_spin)
        font_row.addStretch()
        font_form.addRow("Размер (%%):", font_row)
        hint = QLabel("90% — компактно, 100% — стандарт, 140% — крупно")
        hint.setObjectName("muted")
        font_form.addRow("", hint)
        v.addWidget(font_group)

        density_group = QGroupBox("Плотность интерфейса")
        density_form = QFormLayout(density_group)
        self.density_combo = QComboBox()
        self.density_combo.addItem("Комфортная (Comfortable)", "comfortable")
        self.density_combo.addItem("Компактная (Compact)", "compact")
        density_form.addRow("Плотность:", self.density_combo)
        v.addWidget(density_group)

        v.addStretch()
        return w

    # ---- Tab 4: Управление моделями ----

    def _build_tab_models(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(10)
        v.setContentsMargins(12, 12, 12, 12)

        llm_group = QGroupBox("LLM (GGUF)")
        llm_v = QVBoxLayout(llm_group)
        llm_row = QHBoxLayout()
        self.llm_path_edit = QLineEdit()
        self.llm_path_edit.setPlaceholderText("Путь к GGUF модели")
        llm_browse = QPushButton("Обзор")
        llm_browse.clicked.connect(self._browse_llm)
        llm_row.addWidget(self.llm_path_edit, stretch=1)
        llm_row.addWidget(llm_browse)
        llm_v.addLayout(llm_row)
        self.llm_status_label = QLabel()
        self.llm_status_label.setObjectName("muted")
        llm_v.addWidget(self.llm_status_label)
        v.addWidget(llm_group)

        bundled_group = QGroupBox("Встроенные модели")
        bundled_v = QVBoxLayout(bundled_group)

        try:
            from android.assets_loader import resolve_bundled_model
            emb = resolve_bundled_model("embedder")
            rer = resolve_bundled_model("reranker")
        except ImportError:
            emb = None
            rer = None

        emb_lbl = QLabel(f"Embedder: {emb or 'скачивается автоматически'} — {_path_status(emb) if emb else 'автозагрузка'}")
        emb_lbl.setObjectName("muted")
        emb_lbl.setWordWrap(True)
        rer_lbl = QLabel(f"Reranker: {rer or DEFAULT_RERANK_MODEL} — {_path_status(rer) if rer else 'автозагрузка'}")
        rer_lbl.setObjectName("muted")
        rer_lbl.setWordWrap(True)

        bundled_v.addWidget(emb_lbl)
        bundled_v.addWidget(rer_lbl)
        note = QLabel("Встроенные модели загружаются автоматически и не требуют ручной настройки.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        bundled_v.addWidget(note)
        v.addWidget(bundled_group)

        check_btn = QPushButton("Проверить модели")
        check_btn.clicked.connect(self._check_models)
        v.addWidget(check_btn)

        v.addStretch()
        return w

    # ------------------------------------------------------------------
    # Populate from current settings
    # ------------------------------------------------------------------

    def _populate(self) -> None:
        s = self.current

        # Tab 1
        self.db_path_edit.setText(s.get("db_path", ""))

        # Tab 2
        self.top_k_spin.setValue(s.get("top_k", 8))
        rerank_enabled = s.get("rerank_enabled", True)
        self.rerank_cb.setChecked(rerank_enabled)
        self.rerank_top_k_spin.setValue(s.get("rerank_top_k", 5))
        self.rerank_top_k_spin.setEnabled(rerank_enabled)
        self.temperature_spin.setValue(s.get("temperature", 0.3))
        self.max_tokens_spin.setValue(s.get("max_tokens", 512))
        self.n_ctx_spin.setValue(s.get("n_ctx", 2048))
        self.n_threads_spin.setValue(s.get("n_threads", 2))

        # Tab 3
        theme_name = s.get("theme", "dark")
        idx = self.theme_combo.findData(theme_name)
        if idx >= 0:
            # Block signal during populate so we don't trigger live change
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentIndex(idx)
            self.theme_combo.blockSignals(False)
        self.font_scale_spin.blockSignals(True)
        self.font_scale_spin.setValue(s.get("font_scale", 100))
        self.font_scale_spin.blockSignals(False)
        density = s.get("density", "comfortable")
        di = self.density_combo.findData(density)
        if di >= 0:
            self.density_combo.setCurrentIndex(di)

        # Tab 4
        self.llm_path_edit.setText(s.get("llm_model_path", ""))
        self._update_llm_status()

        # Index stats (async-safe: just call directly, it's fast)
        self._refresh_stats()

    # ------------------------------------------------------------------
    # Live preview handlers (appearance tab)
    # ------------------------------------------------------------------

    def _on_theme_changed_live(self, index: int) -> None:
        name = self.theme_combo.currentData()
        if self._theme_manager is not None and name:
            app = QApplication.instance()
            self._theme_manager.set_theme(name, app)

    def _on_font_scale_changed_live(self, value: int) -> None:
        app = QApplication.instance()
        if app is not None:
            apply_font_scale(app, value)

    # ------------------------------------------------------------------
    # Tab 1 helpers
    # ------------------------------------------------------------------

    def _refresh_stats(self) -> None:
        try:
            from core.indexer import index_stats
            from windows.app_paths import db_path as _db_path
            db = self.db_path_edit.text().strip() or self._default_db
            stats = index_stats(Path(db))
            n_docs = len(stats["per_doc"])
            n_chunks = stats["total_chunks"]
            self.stats_label.setText(f"Документов: {n_docs} · Чанков: {n_chunks}")
        except Exception as exc:
            self.stats_label.setText(f"Ошибка: {exc}")

    def _open_library(self) -> None:
        from windows.library_dialog import LibraryDialog
        from windows.app_paths import db_path as _db_path
        db = self.db_path_edit.text().strip() or self._default_db
        dlg = LibraryDialog(self, db_path=Path(db))
        dlg.exec()
        self._refresh_stats()

    def _browse_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите базу данных", "",
            "База данных (engineer.db);;Все файлы (*)"
        )
        if path:
            self.db_path_edit.setText(path)
            self._refresh_stats()

    # ------------------------------------------------------------------
    # Tab 4 helpers
    # ------------------------------------------------------------------

    def _browse_llm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите GGUF модель", "",
            "GGUF (*.gguf);;Все файлы (*)"
        )
        if path:
            self.llm_path_edit.setText(path)
            self._update_llm_status()

    def _update_llm_status(self) -> None:
        llm = self.llm_path_edit.text().strip()
        self.llm_status_label.setText(f"Статус: {_path_status(llm) if llm else 'не указана'}")

    def _check_models(self) -> None:
        lines = []

        llm = self.llm_path_edit.text().strip()
        lines.append(f"LLM: {_path_status(llm) if llm else 'не указана'}")

        try:
            from android.assets_loader import resolve_bundled_model
            emb = resolve_bundled_model("embedder")
            lines.append(f"Embedder: {_path_status(emb) if emb else 'автозагрузка (не локальная)'}")
            rer = resolve_bundled_model("reranker")
            lines.append(f"Reranker: {_path_status(rer) if rer else 'автозагрузка (не локальная)'}")
        except ImportError:
            lines.append("Embedder: автозагрузка (не локальная)")
            lines.append("Reranker: автозагрузка (не локальная)")

        QMessageBox.information(self, "Проверка моделей", "\n".join(lines))

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        rerank_enabled = self.rerank_cb.isChecked()
        self.current = {
            # Tab 1
            "db_path": self.db_path_edit.text().strip(),
            # Tab 2
            "top_k": self.top_k_spin.value(),
            "rerank_enabled": rerank_enabled,
            "rerank_top_k": self.rerank_top_k_spin.value(),
            "temperature": self.temperature_spin.value(),
            "max_tokens": self.max_tokens_spin.value(),
            "n_ctx": self.n_ctx_spin.value(),
            "n_threads": self.n_threads_spin.value(),
            # Tab 3
            "theme": self.theme_combo.currentData() or "dark",
            "font_scale": self.font_scale_spin.value(),
            "density": self.density_combo.currentData() or "comfortable",
            # Tab 4
            "llm_model_path": self.llm_path_edit.text().strip(),
        }
        save_settings(self.current)
        self.accept()

    def get_settings(self) -> dict[str, Any]:
        return self.current


# ---------------------------------------------------------------------------
# Helpers used by main_window at startup and on settings change
# ---------------------------------------------------------------------------

def apply_font_scale(app: QApplication, scale_pct: int) -> None:
    """Adjust the application base font size based on scale percent (100 = default)."""
    base_pt = 10  # default point size
    new_pt = max(8, int(base_pt * scale_pct / 100))
    font = app.font()
    font.setPointSize(new_pt)
    app.setFont(font)


def apply_density(app: QApplication, density: str) -> None:
    """Apply density variant by injecting extra QSS onto top of application stylesheet."""
    if density == "compact":
        extra = """
QListWidget::item { padding-top: 2px; padding-bottom: 2px; }
QTableWidget::item { padding: 2px 4px; }
QFormLayout { spacing: 4px; }
"""
    else:  # comfortable (default)
        extra = """
QListWidget::item { padding-top: 6px; padding-bottom: 6px; }
QTableWidget::item { padding: 4px 6px; }
"""
    current = app.styleSheet()
    # Remove previous density block if present
    marker_start = "/* __density_start__ */"
    marker_end = "/* __density_end__ */"
    if marker_start in current:
        start = current.index(marker_start)
        end = current.index(marker_end) + len(marker_end)
        current = current[:start] + current[end:]
    inject = f"{marker_start}{extra}{marker_end}"
    app.setStyleSheet(current + inject)
