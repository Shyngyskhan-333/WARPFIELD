from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Sequence

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

import mediapipe as mp

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QPropertyAnimation,
    QEasingCurve, QTimer, QSize, pyqtProperty, QSequentialAnimationGroup,
    QParallelAnimationGroup, QAbstractAnimation,
)
from PyQt6.QtGui import (
    QFont, QImage, QPixmap, QFontDatabase, QIcon, QColor, QPainter,
    QPen, QBrush, QLinearGradient, QRadialGradient,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QScrollArea,
    QFileDialog,
    QCheckBox,
    QSizePolicy,
    QGraphicsOpacityEffect,
    QFrame,
    QSpacerItem,
)

APP_NAME = "WARPFIELD"

# ---------------------------------------------------------------------------
# Stylesheet — Vercel / Higgsfield: sharp, monochromatic, brutally minimal
# Pure black base, razor-thin borders, no radii bloat, clinical precision.
# ---------------------------------------------------------------------------
DARK_STYLESHEET = """
/* ── Base ── */
QMainWindow, QWidget {
    background-color: #080808;
    color: #EDEDED;
    font-size: 12px;
    font-family: 'Outfit';
}

/* ── Group boxes ── */
QGroupBox {
    background-color: transparent;
    border: none;
    margin-top: 16px;
    padding: 0px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 0px;
    top: 0px;
    color: #888;
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ── Labels ── */
QLabel {
    color: rgba(237, 237, 237, 0.55);
    font-size: 12px;
    background: transparent;
}
QLabel#video_label {
    background-color: #050505;
    border: 1px solid #1a1a1a;
    border-radius: 4px;
}
QLabel#status_label {
    color: #555;
    font-size: 11px;
    font-weight: 500;
    padding: 4px 0;
    letter-spacing: 0.5px;
}
QLabel#section_label {
    color: #444;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 2px;
}
QLabel#value_label {
    color: #EDEDED;
    font-size: 11px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    padding: 2px 8px;
    min-width: 38px;
}
QLabel#panel_header {
    color: #EDEDED;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 0px;
    background: transparent;
}
QLabel#panel_subtitle {
    color: #333;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    background: transparent;
}
QLabel#app_wordmark {
    color: #EDEDED;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 4px;
    text-transform: uppercase;
    background: transparent;
}
QLabel#app_tagline {
    color: #2a2a2a;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    background: transparent;
}
QLabel#fps_label {
    color: #333;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    background: transparent;
}

/* ── Combo boxes ── */
QComboBox {
    background-color: #0e0e0e;
    border: 1px solid #222;
    border-radius: 3px;
    padding: 7px 12px;
    color: #EDEDED;
    min-height: 30px;
    font-weight: 500;
    font-size: 12px;
    selection-background-color: transparent;
}
QComboBox:hover {
    border-color: #333;
    background-color: #111;
}
QComboBox:focus {
    border-color: #555;
    outline: none;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
    subcontrol-origin: padding;
    subcontrol-position: top right;
}
QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 4px solid #444;
}
QComboBox QAbstractItemView {
    background-color: #111;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    selection-background-color: #1e1e1e;
    selection-color: #EDEDED;
    color: #aaa;
    outline: none;
    padding: 2px;
}
QComboBox QAbstractItemView::item {
    min-height: 30px;
    padding: 4px 12px;
    border-radius: 2px;
}
QComboBox QAbstractItemView::item:hover {
    background-color: #1a1a1a;
    color: #EDEDED;
}

/* ── Sliders ── */
QSlider::groove:horizontal {
    height: 2px;
    background: #1e1e1e;
    border-radius: 1px;
}
QSlider::handle:horizontal {
    background: #EDEDED;
    border: none;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
}
QSlider::sub-page:horizontal {
    background: #444;
    border-radius: 1px;
}

/* ── Toggle switch (checkbox) ── */
QCheckBox {
    spacing: 10px;
    color: #888;
    font-weight: 500;
    font-size: 12px;
}
QCheckBox::indicator {
    width: 36px;
    height: 20px;
    border-radius: 10px;
    border: 1px solid #2a2a2a;
    background-color: #111;
}
QCheckBox::indicator:checked {
    background-color: #EDEDED;
    border-color: #EDEDED;
}
QCheckBox::indicator:hover {
    border-color: #444;
}
QCheckBox::indicator:checked:hover {
    background-color: #ffffff;
}

/* ── Scrollbar ── */
QScrollBar:vertical {
    background: transparent;
    width: 3px;
    border-radius: 1px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #1e1e1e;
    border-radius: 1px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #333;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

/* ── Browse button ── */
QPushButton#browse_btn {
    background: #141414;
    color: #888;
    border: 1px solid #222;
    border-radius: 3px;
    padding: 8px 14px;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
QPushButton#browse_btn:hover {
    background: #1a1a1a;
    border-color: #333;
    color: #EDEDED;
}
QPushButton#browse_btn:pressed {
    background: #0e0e0e;
}

/* ── Frame separator ── */
QFrame#separator {
    background: #141414;
    max-height: 1px;
    min-height: 1px;
    border: none;
}

/* ── Top nav bar ── */
QWidget#navbar {
    background: #080808;
    border-bottom: 1px solid #141414;
}

/* ── Camera badge ── */
QPushButton#cam_badge {
    background: #0e0e0e;
    color: #444;
    border: 1px solid #1e1e1e;
    border-radius: 3px;
    padding: 5px 12px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
QPushButton#cam_badge:hover {
    background: #141414;
    border-color: #2a2a2a;
    color: #888;
}
"""


# ---------------------------------------------------------------------------
# Pulsing Dot — Live status indicator (monochrome edition)
# ---------------------------------------------------------------------------
class PulsingDot(QWidget):
    """Animated status dot that pulses gently."""

    def __init__(self, color: str = "#333333", size: int = 8, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._dot_size = size
        self._pulse_scale = 1.0
        self.setFixedSize(size + 10, size + 10)

        self._pulse_group = QSequentialAnimationGroup()
        anim_up = QPropertyAnimation(self, b"pulse_scale")
        anim_up.setDuration(900)
        anim_up.setStartValue(0.7)
        anim_up.setEndValue(1.3)
        anim_up.setEasingCurve(QEasingCurve.Type.InOutSine)
        anim_down = QPropertyAnimation(self, b"pulse_scale")
        anim_down.setDuration(900)
        anim_down.setStartValue(1.3)
        anim_down.setEndValue(0.7)
        anim_down.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse_group.addAnimation(anim_up)
        self._pulse_group.addAnimation(anim_down)
        self._pulse_group.setLoopCount(-1)
        self._pulse_group.start()

    def get_pulse_scale(self) -> float:
        return self._pulse_scale

    def set_pulse_scale(self, val: float) -> None:
        self._pulse_scale = val
        self.update()

    pulse_scale = pyqtProperty(float, get_pulse_scale, set_pulse_scale)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width() / 2
        cy = self.height() / 2
        radius = (self._dot_size / 2) * self._pulse_scale

        # Outer glow ring — very subtle
        glow = QColor(self._color)
        glow.setAlphaF(0.12)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(int(cx - radius * 2.0), int(cy - radius * 2.0),
                            int(radius * 4.0), int(radius * 4.0))

        # Core
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(int(cx - radius), int(cy - radius),
                            int(radius * 2), int(radius * 2))
        painter.end()


# ---------------------------------------------------------------------------
# Collapsible Box — Vercel-style: flat, borderless toggle, no chrome
# ---------------------------------------------------------------------------
class CollapsibleBox(QWidget):
    def __init__(self, title="", index: int = 0, parent=None):
        super().__init__(parent)
        self._title_text = title
        self._index = index

        # Plane index badge (01, 02, …)
        badge_num = f"{index + 1:02d}"

        self.toggle_button = QPushButton(f"{badge_num}  {title.upper()}")
        self.toggle_button.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding: 13px 16px;
                font-weight: 700;
                font-size: 11px;
                letter-spacing: 1.8px;
                background-color: transparent;
                border: none;
                border-top: 1px solid #141414;
                border-radius: 0px;
                color: #333;
            }}
            QPushButton:checked {{
                color: #EDEDED;
                border-top: 1px solid #2a2a2a;
                background: #0d0d0d;
            }}
            QPushButton:hover:!checked {{
                color: #666;
                background: #0a0a0a;
            }}
        """)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)

        self.content_area = QScrollArea()
        self.content_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #0a0a0a;
                border-bottom: 1px solid #141414;
            }
        """)
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setWidgetResizable(True)

        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout(self.content_widget)
        self.main_layout.setContentsMargins(16, 14, 16, 16)
        self.main_layout.setSpacing(12)
        self.content_area.setWidget(self.content_widget)

        self._opacity_effect = QGraphicsOpacityEffect(self.content_widget)
        self._opacity_effect.setOpacity(0.0)
        self.content_widget.setGraphicsEffect(self._opacity_effect)

        self.toggle_animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.toggle_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggle_animation.setDuration(280)

        self._fade_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_animation.setDuration(200)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.toggle_button.clicked.connect(self.on_pressed)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)
        self.setLayout(layout)

    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        content_height = self.main_layout.sizeHint().height() + 32

        self.toggle_animation.setStartValue(self.content_area.maximumHeight())
        self.toggle_animation.setEndValue(content_height if checked else 0)
        self.toggle_animation.start()

        if checked:
            self._fade_animation.setStartValue(0.0)
            self._fade_animation.setEndValue(1.0)
            QTimer.singleShot(80, self._fade_animation.start)
        else:
            self._fade_animation.setStartValue(1.0)
            self._fade_animation.setEndValue(0.0)
            self._fade_animation.start()

    def addWidget(self, widget):
        self.main_layout.addWidget(widget)


# ---------------------------------------------------------------------------
# Camera Picker Dialog
# ---------------------------------------------------------------------------
class CameraPickerWidget(QWidget):
    """Inline camera selector — probes indices 0–4 and lists available ones."""

    camera_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        lbl = QLabel("CAMERA")
        lbl.setObjectName("section_label")

        self._combo = QComboBox()
        self._combo.setMinimumWidth(140)
        self._populate()
        self._combo.currentIndexChanged.connect(self._on_changed)

        layout.addWidget(lbl)
        layout.addWidget(self._combo, stretch=1)

    def _populate(self):
        self._combo.blockSignals(True)
        self._combo.clear()
        found = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                found.append(i)
                cap.release()
        if not found:
            found = [0]  # fallback
        for idx in found:
            self._combo.addItem(f"Camera {idx}", userData=idx)
        self._combo.blockSignals(False)

    def _on_changed(self, _):
        idx = self._combo.currentData()
        if idx is not None:
            self.camera_changed.emit(int(idx))

    def current_index(self) -> int:
        data = self._combo.currentData()
        return int(data) if data is not None else 0


# ---------------------------------------------------------------------------
# Background Processing Thread
# ---------------------------------------------------------------------------
class VideoProcessorThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    hand_count_changed = pyqtSignal(int)
    fps_updated = pyqtSignal(float)

    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    TWO_HAND_QUADS: list[tuple[tuple[str, int], ...]] = [
        (("left", 8),  ("right", 8),  ("right", 4),  ("left", 4)),
        (("left", 12), ("right", 12), ("right", 8),  ("left", 8)),
        (("left", 16), ("right", 16), ("right", 12), ("left", 12)),
        (("left", 20), ("right", 20), ("right", 16), ("left", 16)),
    ]

    SINGLE_HAND_LANDMARKS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, THUMB_TIP)

    ANCHOR_COLORS = (
        (255, 255, 255),
        (180, 180, 180),
        (120, 120, 120),
        ( 80,  80,  80),
    )

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self._run_flag = True
        self._camera_index = camera_index
        self._pending_camera_index: int | None = None

        self.config: dict = {
            "show_anchors": True,
            "planes": {
                0: {"texture": "Static Image", "path": "", "effect": "None", "opacity": 1.0},
                1: {"texture": "Static Image", "path": "", "effect": "None", "opacity": 1.0},
                2: {"texture": "Static Image", "path": "", "effect": "None", "opacity": 1.0},
                3: {"texture": "Static Image", "path": "", "effect": "None", "opacity": 1.0},
            }
        }

        self._video_caps = {}
        self._video_caps_path = {}
        self._custom_images = {}
        self._custom_images_path = {}

        self._quadrilaterals: list[list[tuple[int, int]]] = []
        self._last_hand_count = -1

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self._selfie_segmentation = self._mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._overlay_image = self._load_or_create_placeholder("placeholder.jpg")
        self._grid_image: np.ndarray | None = None
        self._anim_frame: int = 0
        self._ema_smoothers = {}

    def update_config(self, plane_idx: int, config: dict) -> None:
        if plane_idx == -1:
            self.config.update(config)
        else:
            self.config["planes"][plane_idx].update(config)

    def set_camera(self, index: int) -> None:
        self._pending_camera_index = index

    def stop(self) -> None:
        self._run_flag = False
        self.wait()

    def run(self) -> None:
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {self._camera_index}.")
            return

        prev_time = time.perf_counter()

        while self._run_flag:
            # Hot-swap camera without restarting thread
            if self._pending_camera_index is not None and self._pending_camera_index != self._camera_index:
                new_cap = cv2.VideoCapture(self._pending_camera_index, cv2.CAP_DSHOW)
                if new_cap.isOpened():
                    cap.release()
                    cap = new_cap
                    self._camera_index = self._pending_camera_index
                    self._ema_smoothers.clear()
                else:
                    new_cap.release()
                self._pending_camera_index = None

            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self._hands.process(rgb)
            rgb.flags.writeable = True

            show_anchors = self.config.get("show_anchors", True)
            if results.multi_hand_landmarks and show_anchors:
                for hand_lm in results.multi_hand_landmarks:
                    self._mp_drawing.draw_landmarks(
                        frame,
                        hand_lm,
                        self._mp_hands.HAND_CONNECTIONS,
                        self._mp_styles.get_default_hand_landmarks_style(),
                        self._mp_styles.get_default_hand_connections_style(),
                    )

            self._cached_seg_masks = {}
            self._update_quadrilaterals(results, w, h)
            self._notify_hand_count(results)
            self._blend_overlays(frame, results)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now
            self.fps_updated.emit(fps)

            self.frame_ready.emit(frame)

        cap.release()
        for c in self._video_caps.values():
            c.release()
        self._selfie_segmentation.close()

    def _smooth_point(self, key: str, pt: tuple[int, int], alpha: float = 0.6) -> tuple[int, int]:
        if key not in self._ema_smoothers:
            self._ema_smoothers[key] = pt
            return pt
        ex, ey = self._ema_smoothers[key]
        nx = alpha * pt[0] + (1 - alpha) * ex
        ny = alpha * pt[1] + (1 - alpha) * ey
        self._ema_smoothers[key] = (nx, ny)
        return int(nx), int(ny)

    @staticmethod
    def _expand_quad(points: list[tuple[int, int]], scale: float = 1.25) -> list[tuple[int, int]]:
        if len(points) != 4:
            return points
        cx = sum(p[0] for p in points) / 4.0
        cy = sum(p[1] for p in points) / 4.0
        return [(int(cx + (p[0] - cx) * scale), int(cy + (p[1] - cy) * scale)) for p in points]

    def _update_quadrilaterals(self, results, frame_w: int, frame_h: int) -> None:
        self._quadrilaterals = []
        handed = self._get_handed_landmarks(results)

        if "Left" in handed and "Right" in handed:
            for q_idx, quad_def in enumerate(self.TWO_HAND_QUADS):
                pts = []
                for side, lid in quad_def:
                    px = self._lm_to_px(handed[side.capitalize()].landmark[lid], frame_w, frame_h)
                    key = f"two_{q_idx}_{side}_{lid}"
                    pts.append(self._smooth_point(key, px))
                self._quadrilaterals.append(self._expand_quad(pts))
        elif len(handed) == 1:
            lm = next(iter(handed.values()))
            side = next(iter(handed.keys()))
            pts = []
            for lid in self.SINGLE_HAND_LANDMARKS:
                px = self._lm_to_px(lm.landmark[lid], frame_w, frame_h)
                key = f"single_{side}_{lid}"
                pts.append(self._smooth_point(key, px))
            self._quadrilaterals.append(self._expand_quad(pts))

    def _notify_hand_count(self, results) -> None:
        count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        if count != self._last_hand_count:
            self._last_hand_count = count
            self.hand_count_changed.emit(count)

    @staticmethod
    def _get_handed_landmarks(results) -> dict:
        out: dict = {}
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return out
        for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handed.classification[0].label
            out[label] = lm
        return out

    @staticmethod
    def _lm_to_px(landmark, w: int, h: int) -> tuple[int, int]:
        x = int(min(max(landmark.x * w, 0), w - 1))
        y = int(min(max(landmark.y * h, 0), h - 1))
        return x, y

    def _get_texture(self, plane_idx: int, frame: np.ndarray) -> np.ndarray:
        pcfg = self.config["planes"].get(plane_idx, {})
        mode = pcfg.get("texture", "Static Image")
        path = pcfg.get("path", "")

        if mode == "Grid Pattern":
            return self._grid_texture()
        elif mode == "Custom Image":
            if plane_idx not in self._custom_images or self._custom_images_path.get(plane_idx) != path:
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.resize(img, (400, 400))
                        self._custom_images[plane_idx] = img
                        self._custom_images_path[plane_idx] = path
            return self._custom_images.get(plane_idx, self._overlay_image.copy())
        elif mode == "Custom Video":
            if plane_idx not in self._video_caps or self._video_caps_path.get(plane_idx) != path:
                if plane_idx in self._video_caps:
                    self._video_caps[plane_idx].release()
                if os.path.exists(path):
                    self._video_caps[plane_idx] = cv2.VideoCapture(path)
                    self._video_caps_path[plane_idx] = path
            cap = self._video_caps.get(plane_idx)
            if cap and cap.isOpened():
                ret, vframe = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, vframe = cap.read()
                if ret:
                    vframe = cv2.resize(vframe, (400, 400))
                    return vframe
            return self._animated_texture()

        return self._overlay_image.copy()

    def _grid_texture(self) -> np.ndarray:
        if self._grid_image is None:
            g = np.zeros((400, 400, 3), dtype=np.uint8)
            g[0:200, 0:200] = (255, 200, 200)
            g[200:, 200:] = (200, 255, 200)
            g[0:200, 200:] = (200, 200, 255)
            g[200:, 0:200] = (255, 255, 200)
            for i in range(0, 400, 40):
                cv2.line(g, (i, 0), (i, 400), (0, 0, 0), 1)
                cv2.line(g, (0, i), (400, i), (0, 0, 0), 1)
            self._grid_image = g
        return self._grid_image.copy()

    def _animated_texture(self) -> np.ndarray:
        self._anim_frame += 1
        t = self._anim_frame * 0.07
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        for i, (r, c) in enumerate([(100, (0, 106, 255)), (60, (255, 255, 255)), (40, (150, 200, 255))]):
            x = int(200 + (130 - i * 30) * math.cos(t + i * 2.1))
            y = int(200 + (130 - i * 30) * math.sin(t + i * 2.1))
            cv2.circle(img, (x, y), r, c, -1)
        cv2.putText(img, "LIVE", (145, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3)
        return img

    def _apply_effect(self, texture: np.ndarray, plane_idx: int, results) -> np.ndarray:
        effect = self.config["planes"].get(plane_idx, {}).get("effect", "None")
        if effect == "None":
            return texture

        if effect == "Pixelated View":
            h, w = texture.shape[:2]
            small = cv2.resize(texture, (max(1, w // 15), max(1, h // 15)), interpolation=cv2.INTER_NEAREST)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        if effect == "Edge Detect":
            edges = cv2.Canny(texture, 80, 180)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if effect == "Distance Modulator":
            return self._distance_modulate(texture, results)

        if effect in ["Dotted Red Silhouette (White BG)", "Red Silhouette (Black BG)", "Paper Tear Silhouette (Blue BG)"]:
            tex_id = "camera" if texture.shape[0] > 400 else f"custom_{plane_idx}"
            if tex_id not in getattr(self, '_cached_seg_masks', {}):
                rgb_tex = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                rgb_tex.flags.writeable = False
                seg_results = self._selfie_segmentation.process(rgb_tex)
                rgb_tex.flags.writeable = True
                if not hasattr(self, '_cached_seg_masks'):
                    self._cached_seg_masks = {}
                self._cached_seg_masks[tex_id] = seg_results.segmentation_mask

            mask = self._cached_seg_masks.get(tex_id)
            if mask is None:
                return texture

            condition = np.stack((mask,) * 3, axis=-1) > 0.5
            h, w = texture.shape[:2]

            if effect == "Dotted Red Silhouette (White BG)":
                bg = np.full((h, w, 3), 255, dtype=np.uint8)
                fg = np.full((h, w, 3), 255, dtype=np.uint8)
                gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
                for y in range(0, h, 8):
                    for x in range(0, w, 8):
                        brightness = int(gray[y, x])
                        radius = int(4.5 * (1.0 - brightness / 255.0))
                        if radius > 0:
                            red_val = max(100, 255 - int((255 - brightness) * 0.7))
                            cv2.circle(fg, (x, y), radius, (0, 0, red_val), -1)
                return np.where(condition, fg, bg)

            elif effect == "Red Silhouette (Black BG)":
                bg = np.zeros((h, w, 3), dtype=np.uint8)
                gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
                fg = np.zeros((h, w, 3), dtype=np.uint8)
                fg[:, :, 2] = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
                return np.where(condition, fg, bg)

            elif effect == "Paper Tear Silhouette (Blue BG)":
                bg = np.full((h, w, 3), (255, 120, 50), dtype=np.uint8)
                gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
                noise = np.random.randint(220, 255, (h, w), dtype=np.uint8)
                blur = cv2.GaussianBlur(noise, (11, 11), 0)
                edges = cv2.Canny(blur, 40, 100)
                shaded_paper = cv2.convertScaleAbs(gray, alpha=0.5, beta=120)
                fg_with_noise = cv2.cvtColor(shaded_paper, cv2.COLOR_GRAY2BGR)
                fg_with_noise[edges > 0] = (150, 150, 150)
                return np.where(condition, fg_with_noise, bg)

        return texture

    def _distance_modulate(self, texture: np.ndarray, results) -> np.ndarray:
        handed = self._get_handed_landmarks(results)
        if not handed:
            return texture
        hand = next(iter(handed.values()))
        thumb = hand.landmark[self.THUMB_TIP]
        index = hand.landmark[self.INDEX_TIP]
        dist = math.hypot(thumb.x - index.x, thumb.y - index.y)
        scale = float(np.clip(dist * 6.0, 0.15, 2.5))
        hsv = cv2.cvtColor(texture, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _blend_overlays(self, frame: np.ndarray, results) -> None:
        fh, fw = frame.shape[:2]
        show_anchors = self.config.get("show_anchors", True)

        for idx, quad in enumerate(self._quadrilaterals):
            if len(quad) != 4:
                continue

            plane_idx = idx % 4
            pcfg = self.config["planes"].get(plane_idx, {})
            mode = pcfg.get("texture", "Static Image")
            opacity = float(pcfg.get("opacity", 1.0))

            if mode == "Camera Feed":
                texture = frame.copy()
                texture = self._apply_effect(texture, plane_idx, results)
                warped = texture
                alpha_mask = np.zeros((fh, fw), dtype=np.uint8)
                pts = np.array(quad, dtype=np.int32)
                cv2.fillConvexPoly(alpha_mask, pts, 255)
            else:
                texture = self._get_texture(plane_idx, frame)
                texture = self._apply_effect(texture, plane_idx, results)
                tex_h, tex_w = texture.shape[:2]
                src_corners = np.array(
                    [[0, 0], [tex_w - 1, 0], [tex_w - 1, tex_h - 1], [0, tex_h - 1]],
                    dtype=np.float32,
                )
                dst = np.array(quad, dtype=np.float32)
                M, _ = cv2.findHomography(src_corners, dst)
                if M is None:
                    continue
                warped = cv2.warpPerspective(texture, M, (fw, fh))
                alpha_mask = cv2.warpPerspective(
                    np.full((tex_h, tex_w), 255, dtype=np.uint8), M, (fw, fh)
                )

            a = (alpha_mask.astype(np.float32) / 255.0 * opacity)[..., np.newaxis]
            np.copyto(frame, (frame.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a).astype(np.uint8))

            if show_anchors:
                color = self.ANCHOR_COLORS[idx % len(self.ANCHOR_COLORS)]
                for pt in quad:
                    cx, cy = int(pt[0]), int(pt[1])
                    # Clean square anchor markers instead of circles
                    half = 5
                    cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), color, -1)
                    cv2.rectangle(frame, (cx - half - 2, cy - half - 2), (cx + half + 2, cy + half + 2), (40, 40, 40), 1)

    @staticmethod
    def _load_or_create_placeholder(path: str) -> np.ndarray:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[:200, :200] = (255, 150, 100)
        img[:200, 200:] = (100, 200, 255)
        img[200:, :200] = (200, 100, 255)
        img[200:, 200:] = (100, 255, 150)
        cv2.circle(img, (200, 200), 90, (255, 255, 255), 4)
        cv2.putText(img, "WARP", (105, 215), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (255, 255, 255), 5)
        cv2.imwrite(path, img)
        return img.copy()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1140, 740)

        # ── Top navbar ──
        navbar = self._build_navbar()

        # ── Video feed ──
        self._video_label = QLabel()
        self._video_label.setObjectName("video_label")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(640, 480)

        # ── Status bar ──
        self._pulsing_dot = PulsingDot("#333333", 7)
        self._status_label = QLabel("Waiting for hands")
        self._status_label.setObjectName("status_label")
        self._fps_label = QLabel("— FPS")
        self._fps_label.setObjectName("fps_label")

        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(4, 4, 4, 4)
        status_layout.setSpacing(8)
        status_layout.addWidget(self._pulsing_dot)
        status_layout.addWidget(self._status_label)
        status_layout.addStretch()
        status_layout.addWidget(self._fps_label)

        status_widget = QWidget()
        status_widget.setLayout(status_layout)
        status_widget.setStyleSheet("""
            QWidget {
                background: #0a0a0a;
                border: 1px solid #141414;
                border-radius: 3px;
            }
        """)

        video_col = QVBoxLayout()
        video_col.setSpacing(8)
        video_col.addWidget(self._video_label, stretch=1)
        video_col.addWidget(status_widget)

        # ── Control panel ──
        panel = self._build_control_panel()
        panel.setFixedWidth(300)

        # ── Content area ──
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)
        content_layout.addLayout(video_col, stretch=1)
        content_layout.addWidget(panel)

        content_widget = QWidget()
        content_widget.setLayout(content_layout)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(navbar)
        root_layout.addWidget(content_widget, stretch=1)

        container = QWidget()
        container.setLayout(root_layout)
        self.setCentralWidget(container)

        # ── Kick off first camera from picker ──
        initial_cam = self._cam_picker.current_index()
        self._thread = VideoProcessorThread(camera_index=initial_cam)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.hand_count_changed.connect(self._on_hand_count)
        self._thread.fps_updated.connect(self._on_fps)
        self._thread.start()

        self._cam_picker.camera_changed.connect(self._on_camera_changed)

    def _build_navbar(self) -> QWidget:
        navbar = QWidget()
        navbar.setObjectName("navbar")
        navbar.setFixedHeight(48)

        layout = QHBoxLayout(navbar)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(16)

        wordmark = QLabel(APP_NAME)
        wordmark.setObjectName("app_wordmark")

        tagline = QLabel("AR HAND TRACKER")
        tagline.setObjectName("app_tagline")

        # Build camera picker here so it's accessible in __init__
        self._cam_picker = CameraPickerWidget()

        layout.addWidget(wordmark)
        layout.addSpacing(12)
        layout.addWidget(tagline)
        layout.addStretch()
        layout.addWidget(self._cam_picker)

        return navbar

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("""
            QWidget {
                background: #0a0a0a;
                border: 1px solid #141414;
                border-radius: 4px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Panel header ──
        header_widget = QWidget()
        header_widget.setStyleSheet("""
            QWidget {
                background: transparent;
                border: none;
                border-bottom: 1px solid #141414;
                border-radius: 0px;
            }
        """)
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setSpacing(4)

        header_label = QLabel("PLANES")
        header_label.setObjectName("panel_header")
        subtitle_label = QLabel("CONFIGURE AR OVERLAYS")
        subtitle_label.setObjectName("panel_subtitle")

        header_layout.addWidget(header_label)
        header_layout.addWidget(subtitle_label)
        layout.addWidget(header_widget)

        # ── Global settings ──
        global_widget = QWidget()
        global_widget.setStyleSheet("""
            QWidget {
                background: transparent;
                border: none;
                border-bottom: 1px solid #141414;
                border-radius: 0px;
            }
        """)
        global_layout = QHBoxLayout(global_widget)
        global_layout.setContentsMargins(16, 12, 16, 12)
        global_layout.setSpacing(12)

        self._anchors_check = QCheckBox("Show Anchors")
        self._anchors_check.setChecked(True)
        self._anchors_check.toggled.connect(self._push_global_config)
        global_layout.addWidget(self._anchors_check)
        global_layout.addStretch()
        layout.addWidget(global_widget)

        # ── Planes ──
        planes_scroll = QScrollArea()
        planes_scroll.setWidgetResizable(True)
        planes_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        planes_container = QWidget()
        planes_container.setStyleSheet("QWidget { background: transparent; border: none; }")
        planes_layout = QVBoxLayout(planes_container)
        planes_layout.setContentsMargins(0, 0, 0, 0)
        planes_layout.setSpacing(0)

        plane_names = ["Primary", "Secondary", "Tertiary", "Quaternary"]
        self.plane_controls = []
        for i in range(4):
            box = CollapsibleBox(plane_names[i], index=i)
            controls = self._build_plane_controls(i)
            for widget in controls["widgets"]:
                box.addWidget(widget)
            self.plane_controls.append(controls)
            planes_layout.addWidget(box)

            if i == 0:
                box.toggle_button.setChecked(True)
                box.on_pressed()

        planes_layout.addStretch()
        planes_scroll.setWidget(planes_container)
        layout.addWidget(planes_scroll, stretch=1)

        return panel

    def _build_plane_controls(self, plane_idx: int) -> dict:
        widgets = []

        tex_lbl = QLabel("TEXTURE SOURCE")
        tex_lbl.setObjectName("section_label")
        tex_combo = QComboBox()
        tex_combo.addItems(["Static Image", "Grid Pattern", "Custom Image", "Custom Video", "Camera Feed"])
        widgets.extend([tex_lbl, tex_combo])

        browse_btn = QPushButton("↑  Browse File")
        browse_btn.setObjectName("browse_btn")
        browse_btn.setVisible(False)
        path_label = QLabel("No file selected")
        path_label.setWordWrap(True)
        path_label.setStyleSheet("""
            font-size: 10px;
            color: #333;
            background: #0e0e0e;
            border: 1px solid #1a1a1a;
            border-radius: 3px;
            padding: 6px 10px;
            letter-spacing: 0.5px;
        """)
        path_label.setVisible(False)
        widgets.extend([browse_btn, path_label])

        sep1 = QFrame()
        sep1.setObjectName("separator")
        sep1.setFrameShape(QFrame.Shape.HLine)
        widgets.append(sep1)

        eff_lbl = QLabel("FILTER EFFECT")
        eff_lbl.setObjectName("section_label")
        eff_combo = QComboBox()
        eff_combo.addItems([
            "None",
            "Distance Modulator",
            "Edge Detect",
            "Pixelated View",
            "Dotted Red Silhouette (White BG)",
            "Red Silhouette (Black BG)",
            "Paper Tear Silhouette (Blue BG)",
        ])
        widgets.extend([eff_lbl, eff_combo])

        sep2 = QFrame()
        sep2.setObjectName("separator")
        sep2.setFrameShape(QFrame.Shape.HLine)
        widgets.append(sep2)

        op_lbl = QLabel("OPACITY")
        op_lbl.setObjectName("section_label")
        op_slider = QSlider(Qt.Orientation.Horizontal)
        op_slider.setRange(0, 100)
        op_slider.setValue(100)
        op_val = QLabel("100%")
        op_val.setObjectName("value_label")
        op_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        op_val.setFixedWidth(42)

        op_layout = QHBoxLayout()
        op_layout.setSpacing(10)
        op_layout.setContentsMargins(0, 0, 0, 0)
        op_layout.addWidget(op_slider, stretch=1)
        op_layout.addWidget(op_val)
        op_widget = QWidget()
        op_widget.setLayout(op_layout)
        op_widget.setContentsMargins(0, 0, 0, 0)
        widgets.extend([op_lbl, op_widget])

        controls = {
            "tex_combo": tex_combo,
            "browse_btn": browse_btn,
            "path_label": path_label,
            "eff_combo": eff_combo,
            "op_slider": op_slider,
            "op_val": op_val,
            "widgets": widgets,
            "current_path": "",
        }

        def on_tex_changed(text):
            is_custom = text in ["Custom Image", "Custom Video"]
            browse_btn.setVisible(is_custom)
            path_label.setVisible(is_custom)
            self._push_plane_config(plane_idx)

        def on_browse():
            mode = tex_combo.currentText()
            filter_str = "Images (*.png *.jpg *.jpeg *.webp)" if mode == "Custom Image" else "Videos (*.mp4 *.avi *.mov *.mkv)"
            path, _ = QFileDialog.getOpenFileName(self, f"Select {mode}", "", filter_str)
            if path:
                controls["current_path"] = path
                path_label.setText(os.path.basename(path))
                self._push_plane_config(plane_idx)

        tex_combo.currentTextChanged.connect(on_tex_changed)
        browse_btn.clicked.connect(on_browse)
        eff_combo.currentTextChanged.connect(lambda _: self._push_plane_config(plane_idx))

        def on_op_changed(v):
            op_val.setText(f"{v}%")
            self._push_plane_config(plane_idx)

        op_slider.valueChanged.connect(on_op_changed)

        return controls

    def _push_global_config(self):
        self._thread.update_config(-1, {
            "show_anchors": self._anchors_check.isChecked()
        })

    def _push_plane_config(self, idx: int):
        c = self.plane_controls[idx]
        self._thread.update_config(idx, {
            "texture": c["tex_combo"].currentText(),
            "path": c["current_path"],
            "effect": c["eff_combo"].currentText(),
            "opacity": c["op_slider"].value() / 100.0,
        })

    @pyqtSlot(int)
    def _on_camera_changed(self, index: int) -> None:
        self._thread.set_camera(index)

    @pyqtSlot(np.ndarray)
    def _on_frame(self, frame: np.ndarray) -> None:
        self._video_label.setPixmap(self._cv_to_pixmap(frame))

    @pyqtSlot(int)
    def _on_hand_count(self, count: int) -> None:
        states = {
            0: ("#2a2a2a", "Waiting for hands"),
            1: ("#888888", "1 hand detected"),
            2: ("#EDEDED", "2 hands — AR active"),
        }
        color, text = states.get(count, ("#2a2a2a", ""))
        self._pulsing_dot.set_color(color)
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color}; font-size: 11px; font-weight: 600;"
            f" letter-spacing: 0.3px; background: transparent;"
        )

    @pyqtSlot(float)
    def _on_fps(self, fps: float) -> None:
        self._fps_label.setText(f"{fps:.0f} FPS")

    def closeEvent(self, event) -> None:
        self._thread.stop()
        event.accept()

    def _cv_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        scaled = qt_img.scaled(
            self._video_label.width(),
            self._video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QPixmap.fromImage(scaled)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    font_path = os.path.join(os.path.dirname(__file__), "Outfit.ttf")
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            family = QFontDatabase.applicationFontFamilies(font_id)[0]
            app.setFont(QFont(family, 10))

    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
