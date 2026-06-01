"""
============================================================
BrailleVisionAI - Streamlit UI Components Module
============================================================

Purpose:
    Reusable UI component functions for the Streamlit interface.
    These are imported and called from app.py to keep the
    main entry point clean and modular.

    In Phase A, all components are placeholder stubs.
    They will be wired to real pipeline data in later phases.

Author: BrailleVisionAI Team
Phase:  A — Foundation (Stubs), Phase F — Full Implementation
============================================================
"""

import streamlit as st
import numpy as np
from typing import Optional, List


# ============================================================
# LAYOUT COMPONENTS
# ============================================================

def render_header():
    """
    Render the main application header/hero banner.
    Displays the project title, subtitle, and phase badge.

    TODO (Phase F): Add real-time status indicator (model loaded / active).
    """
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1a1a2e, #0f3460);
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 1.5rem;
        ">
            <h1 style="color: white; margin: 0;">👁️ BrailleVisionAI</h1>
            <p style="color: #a0c4ff; margin-top: 0.5rem;">
                Real-time AI Braille to English translation
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav() -> str:
    """
    Render the sidebar navigation menu.

    Returns:
        str: The currently selected navigation page name.

    TODO (Phase F): Connect pages to actual module views.
    """
    with st.sidebar:
        st.markdown("## 👁️ BrailleVisionAI")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["🏠 Home", "📷 Camera", "🖼️ Upload", "📊 Logs", "⚙️ Settings"],
            label_visibility="collapsed",
        )
    return page


def render_stats_bar(
    model_loaded: bool = False,
    camera_active: bool = False,
    chars_detected: int = 0,
    phase: str = "A",
):
    """
    Render the top-row statistics cards.

    Args:
        model_loaded:    Whether the detection model is loaded.
        camera_active:   Whether the webcam is streaming.
        chars_detected:  Number of Braille characters detected so far.
        phase:           Current development phase label.
    """
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "✅ Loaded" if model_loaded else "⏸️ Not Loaded")
    c2.metric("Camera", "🟢 Active" if camera_active else "🔴 Inactive")
    c3.metric("Characters", chars_detected)
    c4.metric("Phase", phase)


# ============================================================
# INPUT COMPONENTS
# ============================================================

def render_image_uploader():
    """
    Render the image upload widget.

    Returns:
        Uploaded file object, or None if no file uploaded.

    TODO (Phase B): Pass the uploaded image to the preprocessing pipeline.
    """
    st.markdown("### 🖼️ Upload Braille Image")
    uploaded = st.file_uploader(
        "Choose a JPG, PNG, or BMP file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a clear photo of embossed or printed Braille text.",
    )
    return uploaded


def render_camera_feed():
    """
    Render the live webcam feed placeholder.

    TODO (Phase F):
        - Use cv2.VideoCapture to capture frames
        - Display frames using st.image() in a loop
        - Run preprocessing + detection on each frame
    """
    st.markdown("### 🎥 Live Camera Feed")
    st.info("📷 Live camera integration is planned for **Phase F**.")
    st.markdown(
        """
        <div style="
            border: 2px dashed #3b82f6;
            border-radius: 12px;
            padding: 4rem;
            text-align: center;
            color: #6b7280;
        ">
            <div style="font-size: 3rem;">🎥</div>
            <p>Webcam feed will appear here in Phase F</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_process_button(disabled: bool = True) -> bool:
    """
    Render the main "Detect & Translate" action button.

    Args:
        disabled: Whether the button is disabled (True until Phase C+D ready).

    Returns:
        True if the button was clicked, False otherwise.
    """
    return st.button(
        "🔍 Detect & Translate Braille",
        use_container_width=True,
        type="primary",
        disabled=disabled,
        help="Available after Phase C & D integration.",
    )


# ============================================================
# OUTPUT COMPONENTS
# ============================================================

def render_translation_output(text: str = "", confidence: float = 0.0):
    """
    Render the English translation output section.

    Args:
        text:       Translated English text to display.
        confidence: Detection confidence score (0.0 to 1.0).

    TODO (Phase D): Wire to real translator output.
    """
    st.markdown("### 📝 English Translation")
    st.text_area(
        "Translation",
        value=text or "",
        placeholder="Translated text will appear here...",
        height=120,
        label_visibility="collapsed",
        disabled=(text == ""),
    )
    st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")


def render_voice_controls(disabled: bool = True):
    """
    Render the text-to-speech control buttons.

    Args:
        disabled: Whether controls are disabled (True until Phase E ready).

    TODO (Phase E): Connect to VoiceAssistant.speak() and VoiceAssistant.stop().
    """
    st.markdown("### 🔊 Voice Output")
    col1, col2 = st.columns(2)
    col1.button("▶️ Speak", use_container_width=True, disabled=disabled)
    col2.button("⏹️ Stop", use_container_width=True, disabled=disabled)


def render_braille_cells_visualization(cells: Optional[List] = None):
    """
    Render a visual representation of detected Braille cells and dot patterns.

    Args:
        cells: List of BrailleCell objects to visualize.

    TODO (Phase C): Render actual detected dot patterns as a grid visualization.
    """
    st.markdown("### 🔤 Detected Braille Cells")
    if not cells:
        st.markdown(
            """
            <div style="
                border: 2px dashed #374151;
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                color: #6b7280;
            ">
                <div style="font-size: 2rem;">⬜</div>
                <p>Braille cell visualization will appear here after detection</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # TODO (Phase C): Render detected cells as a visual grid
        st.write(f"Detected {len(cells)} Braille cells.")
