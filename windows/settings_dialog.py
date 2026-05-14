"""Settings dialog for Engineer Companion.

Persists to ~/.engineer-companion/settings.json.
Controls DB path, LLM model path, temperature, max_tokens, top_k.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QLabel,
)


SETTINGS_DIR = Path.home() / ".engineer-companion"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "db_path": "",
    "llm_model_path": "",
    "temperature": 0.3,
    "max_tokens": 512,
    "top_k": 8,
}


def load_settings() -> dict[str, Any]:
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


class SettingsDialog(QDialog):
    """Settings dialog with path pickers and spinboxes."""

    def __init__(self, parent=None, current: dict[str, Any] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(500)

        self.current = current or load_settings()

        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setSpacing(12)

        # DB path
        db_row = QHBoxLayout()
        self.db_path_edit = QLineEdit()
        self.db_path_edit.setPlaceholderText("Путь к LanceDB (engineer.db)")
        db_browse = QPushButton("Обзор")
        db_browse.clicked.connect(self._browse_db)
        db_row.addWidget(self.db_path_edit, stretch=1)
        db_row.addWidget(db_browse)
        form.addRow("База данных:", db_row)

        # LLM model path
        llm_row = QHBoxLayout()
        self.llm_path_edit = QLineEdit()
        self.llm_path_edit.setPlaceholderText("Путь к GGUF модели")
        llm_browse = QPushButton("Обзор")
        llm_browse.clicked.connect(self._browse_llm)
        llm_row.addWidget(self.llm_path_edit, stretch=1)
        llm_row.addWidget(llm_browse)
        form.addRow("Модель LLM:", llm_row)

        # Temperature
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 1.0)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setDecimals(2)
        form.addRow("Temperature:", self.temperature_spin)

        # Max tokens
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 4096)
        self.max_tokens_spin.setSingleStep(64)
        form.addRow("Max tokens:", self.max_tokens_spin)

        # Top K
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 20)
        form.addRow("Top K релевантных чанков:", self.top_k_spin)

        layout.addLayout(form)
        layout.addSpacing(10)

        # DB info label
        self.info_label = QLabel()
        self.info_label.setObjectName("muted")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        layout.addSpacing(10)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self) -> None:
        self.db_path_edit.setText(self.current.get("db_path", ""))
        self.llm_path_edit.setText(self.current.get("llm_model_path", ""))
        self.temperature_spin.setValue(self.current.get("temperature", 0.3))
        self.max_tokens_spin.setValue(self.current.get("max_tokens", 512))
        self.top_k_spin.setValue(self.current.get("top_k", 8))
        self._update_info()

    def _update_info(self) -> None:
        db = self.db_path_edit.text()
        parts = []
        if db:
            p = Path(db)
            parts.append(f"DB: {'есть' if p.exists() else 'НЕ НАЙДЕНА'}")
        llm = self.llm_path_edit.text()
        if llm:
            p = Path(llm)
            parts.append(f"Модель: {'есть' if p.exists() else 'НЕ НАЙДЕНА'}")
        if parts:
            self.info_label.setText(" | ".join(parts))
        else:
            self.info_label.setText("Пути не указаны — приложение будет работать в режиме поиска без LLM.")

    def _browse_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите базу данных (engineer.db)", "",
            "База данных (engineer.db);;Все файлы (*)"
        )
        if path:
            self.db_path_edit.setText(path)
            self._update_info()

    def _browse_llm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите GGUF модель", "",
            "GGUF (*.gguf);;Все файлы (*)"
        )
        if path:
            self.llm_path_edit.setText(path)
            self._update_info()

    def _on_accept(self) -> None:
        self.current = {
            "db_path": self.db_path_edit.text().strip(),
            "llm_model_path": self.llm_path_edit.text().strip(),
            "temperature": self.temperature_spin.value(),
            "max_tokens": self.max_tokens_spin.value(),
            "top_k": self.top_k_spin.value(),
        }
        save_settings(self.current)
        self.accept()

    def get_settings(self) -> dict[str, Any]:
        return self.current


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dlg = SettingsDialog()
    if dlg.exec():
        print("Saved:", dlg.get_settings())
    sys.exit(0)
