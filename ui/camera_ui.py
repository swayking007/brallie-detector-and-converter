"""
============================================================
BrailleVisionAI — Camera UI Module
============================================================

Purpose:
    Provides all camera-related UI components and backend logic
    for Phase B of BrailleVisionAI.

    Responsibilities:
      • OpenCV webcam capture (threaded, non-blocking)
      • Real-time FPS calculation
      • Snapshot capture with timestamp filenames
      • Uploaded image saving pipeline
      • Streamlit UI helper functions for camera sections

    Architecture note:
      This module is intentionally decoupled from detection/translation.
      In Phase C, detect_braille.py will receive frames via the
      `get_current_frame()` interface exposed here.

Author: BrailleVisionAI Team
Phase:  B — Camera System & Image Input
============================================================
"""

import cv2
import time
import threading
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# ============================================================
# FOLDER CONSTANTS
# ============================================================

# Root of the project (one level up from ui/)
ROOT_DIR = Path(__file__).resolve().parent.parent

CAPTURED_FRAMES_DIR  = ROOT_DIR / "datasets" / "captured_frames"
SAMPLE_IMAGES_DIR    = ROOT_DIR / "datasets" / "sample_test_images"

# Ensure directories exist on import
CAPTURED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# WEBCAM MANAGER (Thread-safe)
# ============================================================

class WebcamManager:
    """
    Thread-safe OpenCV webcam manager.

    Runs video capture in a background thread so the Streamlit
    main thread is never blocked waiting for frames.

    Usage:
        cam = WebcamManager()
        cam.start()
        frame, fps = cam.get_frame()
        cam.stop()

    Phase C Hook:
        Pass `frame` from get_frame() directly into BrailleDetector.detect(frame).
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize webcam manager.

        Args:
            camera_index: OpenCV camera device index (0 = primary webcam).
            width:        Requested capture width in pixels.
            height:       Requested capture height in pixels.
        """
        self.camera_index  = camera_index
        self.width         = width
        self.height        = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray]     = None
        self._lock  = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # FPS tracking
        self._fps: float         = 0.0
        self._fps_start: float   = time.time()
        self._fps_counter: int   = 0

        # Error tracking
        self.error_message: Optional[str] = None

    # ----------------------------------------------------------
    # Public Interface
    # ----------------------------------------------------------

    def start(self) -> bool:
        """
        Open the webcam and start the capture thread.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        if self._running:
            return True  # Already running

        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        if not self._cap.isOpened():
            self.error_message = (
                f"❌ Could not open camera at index {self.camera_index}. "
                "Check that your webcam is connected and not used by another app."
            )
            return False

        self.error_message = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None
        self._frame = None

    def is_running(self) -> bool:
        """Return True if webcam is actively capturing."""
        return self._running

    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the latest captured frame and current FPS.

        Returns:
            (frame, fps) — frame is BGR numpy array or None if not ready.

        Phase C Hook:
            This is the primary integration point. Pass `frame` to
            the detection pipeline:
                cells = detector.detect(frame)
        """
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
        return frame, self._fps

    def get_resolution(self) -> Tuple[int, int]:
        """Return current (width, height) of the capture stream."""
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    # ----------------------------------------------------------
    # Internal capture loop (runs in background thread)
    # ----------------------------------------------------------

    def _capture_loop(self):
        """
        Continuously read frames from the webcam.
        This runs in a daemon thread — it stops when the main app exits.
        """
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()

            if not ret:
                # Skip bad frames instead of crashing
                time.sleep(0.01)
                continue

            # Update FPS every second
            self._fps_counter += 1
            elapsed = time.time() - self._fps_start
            if elapsed >= 1.0:
                self._fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_start = time.time()

            # Thread-safe frame update
            with self._lock:
                self._frame = frame

            # Tiny sleep prevents CPU pegging at 100%
            time.sleep(0.01)

        # Cleanup on exit
        if self._cap and self._cap.isOpened():
            self._cap.release()


# ============================================================
# SNAPSHOT UTILITIES
# ============================================================

def save_snapshot(frame: np.ndarray) -> Tuple[bool, str]:
    """
    Save a BGR OpenCV frame as a timestamped JPEG snapshot.

    Args:
        frame: BGR numpy array from WebcamManager.get_frame().

    Returns:
        (success, file_path_or_error_message)
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
        filename  = f"snapshot_{timestamp}.jpg"
        filepath  = CAPTURED_FRAMES_DIR / filename
        cv2.imwrite(str(filepath), frame)
        return True, str(filepath)
    except Exception as e:
        return False, f"Failed to save snapshot: {e}"


def save_uploaded_image(uploaded_file) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Save a Streamlit uploaded file to the sample_test_images directory.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        (success, file_path_or_error, pil_image_or_None)
    """
    try:
        # Build a timestamped filename preserving the original extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem      = Path(uploaded_file.name).stem
        suffix    = Path(uploaded_file.name).suffix.lower()
        filename  = f"{timestamp}_{stem}{suffix}"
        filepath  = SAMPLE_IMAGES_DIR / filename

        # Read and save via PIL to ensure valid image
        img = Image.open(uploaded_file)
        img.save(str(filepath))

        return True, str(filepath), img

    except Exception as e:
        return False, f"Failed to save image: {e}", None


def list_saved_snapshots(limit: int = 10) -> list:
    """
    Return a list of the most recent snapshot file paths.

    Args:
        limit: Maximum number of recent files to return.

    Returns:
        List of Path objects, newest first.
    """
    files = sorted(
        CAPTURED_FRAMES_DIR.glob("snapshot_*.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return files[:limit]


def list_saved_uploads(limit: int = 10) -> list:
    """
    Return a list of the most recent uploaded image file paths.

    Args:
        limit: Maximum number of recent files to return.

    Returns:
        List of Path objects, newest first.
    """
    files = sorted(
        [p for p in SAMPLE_IMAGES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return files[:limit]


# ============================================================
# IMAGE PROCESSING UTILITIES
# (Phase C hook — these will feed into the detection pipeline)
# ============================================================

def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV BGR frame to RGB for Streamlit display.

    Args:
        frame: BGR numpy array.

    Returns:
        RGB numpy array.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def add_fps_overlay(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Draw an FPS counter overlay onto a frame.

    Args:
        frame: BGR numpy array.
        fps:   Current frames-per-second value.

    Returns:
        Frame with FPS text drawn in the top-left corner.
    """
    overlay = frame.copy()

    # Draw semi-transparent background pill
    cv2.rectangle(overlay, (8, 8), (160, 42), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw FPS text
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 100),
        2,
        cv2.LINE_AA,
    )
    return frame


def add_status_overlay(frame: np.ndarray, label: str, color=(59, 130, 246)) -> np.ndarray:
    """
    Add a status label overlay at the bottom of the frame.

    Args:
        frame: BGR numpy array.
        label: Status text to display.
        color: BGR color tuple.

    Returns:
        Frame with status bar drawn at the bottom.
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame


def resize_for_display(frame: np.ndarray, max_width: int = 800) -> np.ndarray:
    """
    Resize a frame to fit within max_width while preserving aspect ratio.

    Args:
        frame:     Input frame.
        max_width: Maximum display width.

    Returns:
        Resized frame.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale  = max_width / w
    new_w  = max_width
    new_h  = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ============================================================
# STREAMLIT UI RENDERING HELPERS
# ============================================================

def render_camera_controls() -> Tuple[bool, bool, bool]:
    """
    Render Start / Stop / Snapshot control buttons in a row.

    Returns:
        (start_clicked, stop_clicked, snapshot_clicked)
    """
    c1, c2, c3 = st.columns(3)
    start_btn    = c1.button("▶️ Start Camera",    use_container_width=True, type="primary")
    stop_btn     = c2.button("⏹️ Stop Camera",     use_container_width=True)
    snapshot_btn = c3.button("📸 Capture Snapshot", use_container_width=True)
    return start_btn, stop_btn, snapshot_btn


def render_fps_badge(fps: float):
    """Render a styled FPS badge in the Streamlit UI."""
    color = "#22c55e" if fps >= 20 else "#f59e0b" if fps >= 10 else "#ef4444"
    st.markdown(
        f"""
        <div style="
            display:inline-flex; align-items:center; gap:8px;
            background:#161b22; border:1px solid {color}44;
            padding:6px 14px; border-radius:999px; margin-bottom:8px;
        ">
            <div style="
                width:10px; height:10px; border-radius:50%;
                background:{color}; box-shadow:0 0 6px {color};
            "></div>
            <span style="color:{color}; font-weight:600; font-size:0.9rem;">
                {fps:.1f} FPS
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_camera_status_badge(is_running: bool):
    """Render a LIVE / OFFLINE status pill."""
    if is_running:
        st.markdown(
            """
            <div style="
                display:inline-flex; align-items:center; gap:8px;
                background:#052e16; border:1px solid #16a34a44;
                padding:5px 14px; border-radius:999px; margin-bottom:8px;
            ">
                <div style="
                    width:9px; height:9px; border-radius:50%;
                    background:#22c55e;
                    animation: pulse 1.5s infinite;
                "></div>
                <span style="color:#22c55e; font-weight:600; font-size:0.85rem;">● LIVE</span>
            </div>
            <style>
                @keyframes pulse {
                    0%   { box-shadow: 0 0 0 0 rgba(34,197,94,.6); }
                    70%  { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
                    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                display:inline-flex; align-items:center; gap:8px;
                background:#1c1917; border:1px solid #57534e44;
                padding:5px 14px; border-radius:999px; margin-bottom:8px;
            ">
                <span style="color:#78716c; font-weight:600; font-size:0.85rem;">⏸ OFFLINE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_snapshot_gallery(limit: int = 6):
    """
    Render a thumbnail gallery of the most recent captured snapshots.

    Args:
        limit: Number of recent snapshots to display.
    """
    snapshots = list_saved_snapshots(limit=limit)
    if not snapshots:
        st.caption("No snapshots captured yet.")
        return

    cols = st.columns(min(len(snapshots), 3))
    for idx, path in enumerate(snapshots):
        col = cols[idx % 3]
        try:
            img = Image.open(path)
            col.image(img, caption=path.name, use_container_width=True)
        except Exception:
            col.warning(f"Could not load {path.name}")


def render_upload_gallery(limit: int = 6):
    """
    Render a thumbnail gallery of recently uploaded images.

    Args:
        limit: Number of recent uploads to display.
    """
    uploads = list_saved_uploads(limit=limit)
    if not uploads:
        st.caption("No images uploaded yet.")
        return

    cols = st.columns(min(len(uploads), 3))
    for idx, path in enumerate(uploads):
        col = cols[idx % 3]
        try:
            img = Image.open(path)
            col.image(img, caption=path.name, use_container_width=True)
        except Exception:
            col.warning(f"Could not load {path.name}")
