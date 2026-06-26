"""Reusable PySide6 animation helpers.

Provides a small set of QPropertyAnimation wrappers to be used app-wide.
All helpers are safe to call under QT_QPA_PLATFORM=offscreen (headless) —
they construct and start the animations without relying on a visible display.

NOTE: fade_in / fade_out deliberately avoid QGraphicsOpacityEffect when the
widget already has an effect applied by a concurrent animation, to prevent the
QPainter "Painter not active" flood that occurs when multiple effects paint the
same widget simultaneously.  Callers that need continuous pulsing should use
the stylesheet-based helpers in this module instead of chaining fade_in/fade_out.

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
    """Attach a QGraphicsOpacityEffect to *widget* if it doesn't have one yet.

    IMPORTANT: If a QPropertyAnimation is already running on the existing
    effect, we stop it first so we never have two painters on the same device.
    """
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)

    # If a previous animation is still running, stop it cleanly
    old_anim = getattr(widget, "_anim_ref", None)
    if old_anim is not None:
        try:
            old_anim.stop()
        except RuntimeError:
            pass  # C++ object already deleted
        widget._anim_ref = None  # type: ignore[attr-defined]

    return effect  # type: ignore[return-value]


def _run_once(anim: QPropertyAnimation, widget: QWidget) -> QPropertyAnimation:
    """Keep a reference to *anim* on *widget* so it isn't garbage-collected."""
    widget._anim_ref = anim  # type: ignore[attr-defined]
    return anim


def _remove_effect_on_finish(anim: QPropertyAnimation, widget: QWidget) -> None:
    """Connect *anim*.finished to remove the graphics effect from *widget*.

    Removing the effect when the animation is done prevents stale painters
    from remaining attached to the widget tree.
    """
    def _cleanup():
        try:
            if widget and not widget.isVisible() is False:  # widget still alive check
                widget._anim_ref = None  # type: ignore[attr-defined]
                # Only remove effect if nothing else re-attached one
                effect = widget.graphicsEffect()
                if isinstance(effect, QGraphicsOpacityEffect):
                    widget.setGraphicsEffect(None)
        except RuntimeError:
            pass  # C++ object deleted

    anim.finished.connect(_cleanup)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fade_in(
    widget: QWidget,
    duration: int = 180,
    *,
    start_value: float = 0.0,
    end_value: float = 1.0,
    remove_effect_on_done: bool = True,
) -> QPropertyAnimation:
    """Fade *widget* in from transparent to opaque.

    Returns the running QPropertyAnimation (caller may connect signals).
    The graphics effect is removed when the animation finishes (so no stale
    painter references remain).
    """
    effect = _ensure_opacity_effect(widget)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(start_value)
    anim.setEndValue(end_value)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    widget.show()
    if remove_effect_on_done:
        _remove_effect_on_finish(anim, widget)
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    return _run_once(anim, widget)


def fade_out(
    widget: QWidget,
    duration: int = 160,
    *,
    start_value: float = 1.0,
    end_value: float = 0.0,
    on_done: Callable[[], None] | None = None,
    remove_effect_on_done: bool = True,
) -> QPropertyAnimation:
    """Fade *widget* out.  Calls *on_done* when the animation finishes.

    A common pattern is ``on_done=widget.hide``.
    The graphics effect is removed after *on_done* so no stale painters remain.
    """
    effect = _ensure_opacity_effect(widget)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(start_value)
    anim.setEndValue(end_value)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    if on_done is not None:
        anim.finished.connect(on_done)
    if remove_effect_on_done:
        _remove_effect_on_finish(anim, widget)
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
