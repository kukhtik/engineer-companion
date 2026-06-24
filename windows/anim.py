"""Reusable PySide6 animation helpers.

Provides a small set of QPropertyAnimation wrappers to be used app-wide.
All helpers are safe to call under QT_QPA_PLATFORM=offscreen (headless) —
they construct and start the animations without relying on a visible display.

Usage::

    from windows.anim import fade_in, fade_out, slide_in
    fade_in(my_widget, duration=200)
    fade_out(my_widget, on_done=my_widget.hide)
    slide_in(my_widget, from_dx=-40)
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
)
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_opacity_effect(widget: QWidget) -> QGraphicsOpacityEffect:
    """Attach a QGraphicsOpacityEffect to *widget* if it doesn't have one yet."""
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
    return effect  # type: ignore[return-value]


def _run_once(anim: QPropertyAnimation, widget: QWidget) -> QPropertyAnimation:
    """Keep a reference to *anim* on *widget* so it isn't garbage-collected."""
    widget._anim_ref = anim  # type: ignore[attr-defined]
    return anim


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fade_in(
    widget: QWidget,
    duration: int = 180,
    *,
    start_value: float = 0.0,
    end_value: float = 1.0,
) -> QPropertyAnimation:
    """Fade *widget* in from transparent to opaque.

    Returns the running QPropertyAnimation (caller may connect signals).
    """
    effect = _ensure_opacity_effect(widget)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(start_value)
    anim.setEndValue(end_value)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    widget.show()
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    return _run_once(anim, widget)


def fade_out(
    widget: QWidget,
    duration: int = 160,
    *,
    start_value: float = 1.0,
    end_value: float = 0.0,
    on_done: Callable[[], None] | None = None,
) -> QPropertyAnimation:
    """Fade *widget* out.  Calls *on_done* when the animation finishes.

    A common pattern is ``on_done=widget.hide``.
    """
    effect = _ensure_opacity_effect(widget)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(start_value)
    anim.setEndValue(end_value)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    if on_done is not None:
        anim.finished.connect(on_done)
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    return _run_once(anim, widget)


def slide_in(
    widget: QWidget,
    from_dx: int = 0,
    from_dy: int = 0,
    duration: int = 220,
) -> QPropertyAnimation:
    """Slide *widget* into its natural position from an offset.

    ``from_dx`` / ``from_dy`` are pixel offsets added to the widget's current
    pos; negative dx slides from the left, negative dy from the top.

    The widget's final position is unchanged — only the animation start differs.
    """
    widget.show()
    # Ensure geometry is calculated before we read pos()
    widget.update()

    natural_pos = widget.pos()
    start_pos = QPoint(natural_pos.x() + from_dx, natural_pos.y() + from_dy)

    anim = QPropertyAnimation(widget, b"pos", widget)
    anim.setDuration(duration)
    anim.setStartValue(start_pos)
    anim.setEndValue(natural_pos)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    return _run_once(anim, widget)


def animate_geometry(
    widget: QWidget,
    start_rect: QRect,
    end_rect: QRect,
    duration: int = 200,
    *,
    on_done: Callable[[], None] | None = None,
) -> QPropertyAnimation:
    """Animate *widget* geometry from *start_rect* to *end_rect*.

    Useful for collapsible panels, expanding cards, etc.
    """
    anim = QPropertyAnimation(widget, b"geometry", widget)
    anim.setDuration(duration)
    anim.setStartValue(start_rect)
    anim.setEndValue(end_rect)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    if on_done is not None:
        anim.finished.connect(on_done)
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    return _run_once(anim, widget)
