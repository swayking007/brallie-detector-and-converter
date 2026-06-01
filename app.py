"""
BrailleVisionAI — Main Application
Restored full functionality with tabbed navigation and cleaned interface.
"""

import time
import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.set_page_config(
    page_title="BrailleVisionAI",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Imports ──────────────────────────────────────────────────
try:
    from ui.camera_ui import WebcamManager, bgr_to_rgb, resize_for_display, save_uploaded_image
    from preprocessing.quality_analyzer import analyze_frame, analyze_pil_image
    from detection.inference import get_detector, run_detection, run_cell_extraction
    from nlp.sentence_reconstructor import SentenceReconstructor
    PIPELINE_OK = True
except ImportError as e:
    PIPELINE_OK = False
    st.error(f"Failed to load core modules: {e}")

try:
    from speech.tts_engine import speak, stop, set_rate, set_volume
    TTS_OK = True
except ImportError:
    TTS_OK = False

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* Dark modern theme overrides */
[data-testid="stAppViewContainer"] { background-color: #0b0f19; color: #e2e8f0; }
.card { background-color: #151b2b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.4); border: 1px solid #1e293b; }
.card h5 { margin-top: 0; color: #f8fafc; font-weight: 600; border-bottom: 1px solid #1e293b; padding-bottom: 0.8rem; margin-bottom: 1rem; }
.trans-box { background-color: #0f172a; border-radius: 8px; padding: 1.5rem; font-size: 2.2rem; font-weight: 700; color: #38bdf8; min-height: 140px; margin-bottom: 1rem; border: 1px solid #38bdf844; word-break: break-word; letter-spacing: 0.05em; }
.guidance-item { font-size: 0.95rem; margin: 0.8rem 0; display: flex; align-items: center; background: #0f172a; padding: 0.8rem 1rem; border-radius: 8px; border: 1px solid #1e293b; }
.guidance-icon { margin-right: 0.8rem; font-size: 1.3rem; }
.text-success { color: #22c55e; font-weight: 500; }
.text-warning { color: #facc15; font-weight: 500; }
.text-error { color: #ef4444; font-weight: 500; }
.badge { display: inline-block; padding: 0.4rem 1rem; border-radius: 20px; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.badge-high { background-color: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
.badge-med { background-color: rgba(250,204,21,0.15); color: #fde047; border: 1px solid rgba(250,204,21,0.4); }
.badge-low { background-color: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
.badge-none { background-color: rgba(148,163,184,0.15); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.4); }
header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Setup ──────────────────────────────────────
if "cam_manager" not in st.session_state:
    st.session_state.cam_manager = WebcamManager(camera_index=0) if PIPELINE_OK else None
if "nlp_engine" not in st.session_state:
    st.session_state.nlp_engine = SentenceReconstructor() if PIPELINE_OK else None
if "webcam_active" not in st.session_state: st.session_state.webcam_active = False
if "detection_sensitivity" not in st.session_state: st.session_state.detection_sensitivity = "Balanced"

if "thresh_blur_clear" not in st.session_state: st.session_state.thresh_blur_clear = 120.0
if "thresh_blur_slight" not in st.session_state: st.session_state.thresh_blur_slight = 60.0
if "thresh_dark" not in st.session_state: st.session_state.thresh_dark = 60
if "thresh_over" not in st.session_state: st.session_state.thresh_over = 210
if "thresh_tilt" not in st.session_state: st.session_state.thresh_tilt = 8.0
if "thresh_close" not in st.session_state: st.session_state.thresh_close = 0.18
if "thresh_far" not in st.session_state: st.session_state.thresh_far = 0.03

if "voice_enabled" not in st.session_state: st.session_state.voice_enabled = False
if "voice_speed" not in st.session_state: st.session_state.voice_speed = 200
if "voice_volume" not in st.session_state: st.session_state.voice_volume = 1.0

if "latest_text" not in st.session_state: st.session_state.latest_text = ""
if "analytics_data" not in st.session_state: st.session_state.analytics_data = []

# ─── Helpers ──────────────────────────────────────────────────
def _get_quality_kwargs():
    return dict(
        blur_clear=st.session_state.thresh_blur_clear,
        blur_slight=st.session_state.thresh_blur_slight,
        dark_threshold=st.session_state.thresh_dark,
        over_threshold=st.session_state.thresh_over,
        tilt_threshold=st.session_state.thresh_tilt,
        close_threshold=st.session_state.thresh_close,
        far_threshold=st.session_state.thresh_far,
    )

def draw_clean_overlay(frame, cells):
    clean_frame = frame.copy()
    for cell in cells:
        if cell.dots:
            xs = [d.x for d in cell.dots]
            ys = [d.y for d in cell.dots]
            x1, y1 = max(0, min(xs) - 8), max(0, min(ys) - 8)
            x2, y2 = min(frame.shape[1], max(xs) + 8), min(frame.shape[0], max(ys) + 8)
            cv2.rectangle(clean_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 150), 1)
    return clean_frame

# ─── TOP NAVBAR ───────────────────────────────────────────────
st.markdown("<h2 style='text-align:center; margin-bottom: 20px;'>🦾 BrailleVisionAI</h2>", unsafe_allow_html=True)

tab_live, tab_upload, tab_thresh, tab_analytics, tab_about = st.tabs([
    "📷 Live Camera", 
    "🖼 Upload Image", 
    "⚙ Thresholds", 
    "📊 Analytics", 
    "ℹ About"
])

def create_main_layout():
    left_col, right_col = st.columns([0.65, 0.35], gap="large")
    with left_col:
        camera_image_ph = st.empty()
        conf_badge_ph = st.empty()
    
    with right_col:
        st.markdown("<div class='card'><h5>📄 Translation Output</h5>", unsafe_allow_html=True)
        trans_output_ph = st.empty()
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.session_state.voice_enabled = st.checkbox("🔊 Enable speech", value=st.session_state.voice_enabled, key=f"v_cb_{time.time()}")
        with c2:
            speak_btn_ph = st.empty()
            
        with st.expander("Voice Settings"):
            st.session_state.voice_speed = st.slider("Speech Speed", 100, 300, st.session_state.voice_speed, key=f"v_sp_{time.time()}")
            st.session_state.voice_volume = st.slider("Volume", 0.0, 1.0, st.session_state.voice_volume, key=f"v_vol_{time.time()}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h5>📷 Capture Guidance</h5>", unsafe_allow_html=True)
        guidance_ph = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h5>📊 Detection Summary</h5>", unsafe_allow_html=True)
        summary_ph = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔧 Advanced Analysis"):
        debug_ph = st.empty()

    return camera_image_ph, conf_badge_ph, trans_output_ph, speak_btn_ph, guidance_ph, summary_ph, debug_ph

def process_frame(frame, camera_image_ph, conf_badge_ph, trans_output_ph, speak_btn_ph, guidance_ph, summary_ph, debug_ph, is_live=False):
    t0 = time.time()
    report = analyze_frame(frame, **_get_quality_kwargs())
    
    items = []
    if report.brightness.is_ok: items.append("<div class='guidance-item'><span class='guidance-icon text-success'>✅</span> <span class='text-success'>Lighting good</span></div>")
    else: items.append("<div class='guidance-item'><span class='guidance-icon text-warning'>⚠️</span> <span class='text-warning'>Adjust lighting</span></div>")
    
    if report.blur.is_ok: items.append("<div class='guidance-item'><span class='guidance-icon text-success'>✅</span> <span class='text-success'>Image in focus</span></div>")
    else: items.append("<div class='guidance-item'><span class='guidance-icon text-error'>🔴</span> <span class='text-error'>Reduce blur</span></div>")
    
    if report.visibility.is_ok: items.append("<div class='guidance-item'><span class='guidance-icon text-success'>✅</span> <span class='text-success'>Good distance</span></div>")
    else:
        dt = "Move closer" if getattr(report.visibility.status, "name", "") == "TOO_FAR" else "Move further"
        items.append(f"<div class='guidance-item'><span class='guidance-icon text-warning'>⚠️</span> <span class='text-warning'>{dt}</span></div>")
    
    if report.alignment.is_ok: items.append("<div class='guidance-item'><span class='guidance-icon text-success'>✅</span> <span class='text-success'>Proper alignment</span></div>")
    else: items.append("<div class='guidance-item'><span class='guidance-icon text-warning'>⚠️</span> <span class='text-warning'>Straighten page</span></div>")
    
    guidance_ph.markdown("".join(items), unsafe_allow_html=True)

    smart_text = ""
    avg_conf = 0.0
    cells = []
    det_result = None

    if report.overall_ok and PIPELINE_OK:
        detector = get_detector()
        det_result = run_detection(frame, detector)
        if det_result.is_braille or det_result.is_uncertain:
            spacing = det_result.avg_spacing if det_result.avg_spacing > 0 else 15.0
            demo = (st.session_state.detection_sensitivity == "Strict")
            _, cells, _, t_result = run_cell_extraction(frame, spacing, demo_mode=demo, debug_mode=False)
            
            raw_text = t_result.full_text if t_result else "".join(c.translated_char for c in cells)
            if raw_text.strip():
                rec = st.session_state.nlp_engine
                nlp_res = rec.reconstruct(raw_text)
                smart_text = nlp_res.smart_text
                st.session_state.latest_text = smart_text
            if cells:
                avg_conf = sum(c.confidence for c in cells) / len(cells) * 100

    process_time = (time.time() - t0) * 1000
    display_frame = draw_clean_overlay(frame, cells)
    display_frame = resize_for_display(display_frame, max_width=1000)
    
    camera_image_ph.image(bgr_to_rgb(display_frame), use_container_width=True, channels="RGB")
    
    if cells:
        if avg_conf >= 80: badge = "<div class='badge badge-high'>🟢 High confidence</div>"
        elif avg_conf >= 50: badge = "<div class='badge badge-med'>🟡 Medium confidence</div>"
        else: badge = "<div class='badge badge-low'>🔴 Low confidence</div>"
    else:
        badge = "<div class='badge badge-none'>⚪ Scanning</div>"
    conf_badge_ph.markdown(f"<div style='float:right; margin-top:-45px; position:relative; z-index:10;'>{badge}</div>", unsafe_allow_html=True)
    
    placeholder = "<span style='color:#475569;'>Waiting for Braille...</span>"
    trans_output_ph.markdown(f"<div class='trans-box'>{smart_text if smart_text else placeholder}</div>", unsafe_allow_html=True)
    
    dot_c = sum(len(c.dots) for c in cells) if cells else 0
    summary_ph.markdown(f"""
    <div style='font-size: 1.1rem; color: #cbd5e1;'>
        Braille cells detected: <b style='color:#f8fafc; font-size:1.3rem;'>{len(cells)}</b><br><br>
        Dots detected: <b style='color:#f8fafc; font-size:1.3rem;'>{dot_c}</b><br><br>
        Detection confidence: <b style='color:#f8fafc; font-size:1.3rem;'>{avg_conf:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

    if speak_btn_ph.button("🔊 Speak Output", key=f"spk_{time.time()}" if not is_live else "spk_live"):
        if TTS_OK and st.session_state.voice_enabled and st.session_state.latest_text:
            set_rate(st.session_state.voice_speed)
            set_volume(st.session_state.voice_volume)
            speak(st.session_state.latest_text)

    if det_result:
        dbg = getattr(det_result, 'heuristic_result', None)
        if dbg:
            if cells:
                st.session_state.analytics_data.append({
                    "time": time.strftime("%H:%M:%S"),
                    "cells": len(cells),
                    "conf": avg_conf,
                    "dots": dot_c
                })
            
            debug_ph.markdown(f"""
            <div style='font-family:monospace; color:#94a3b8; display:flex; justify-content:space-between;'>
                <div>
                    <b>Heuristic Breakdown:</b><br>
                    Raw contours: {dbg.raw_contour_count}<br>
                    Accepted dots: {dbg.dot_count}<br>
                    Rejected tiny: {getattr(dbg, 'rejected_tiny', 0)}<br>
                    Rejected size: {getattr(dbg, 'rejected_size', 0)}<br>
                </div>
                <div>
                    <b>Geometry & Timing:</b><br>
                    Avg cell spacing: {dbg.avg_spacing:.1f} px<br>
                    Grid valid: {dbg.grid_valid}<br>
                    Row count: {getattr(dbg, 'row_count', 'N/A')}<br>
                    Detection timing: {process_time:.1f} ms<br>
                    Phase: Phase D/E Pipeline<br>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─── TAB 1: LIVE CAMERA ───────────────────────────────────────
with tab_live:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.session_state.webcam_active = st.toggle("🎥 Start Webcam", value=st.session_state.webcam_active)
        st.session_state.detection_sensitivity = st.selectbox("Sensitivity", ["Strict", "Balanced", "Relaxed"], index=1)
    
    phs = create_main_layout()
    
    if st.session_state.webcam_active and PIPELINE_OK:
        cam = st.session_state.cam_manager
        if not cam.is_running(): cam.start()
        
        frame_counter = 0
        while st.session_state.webcam_active:
            frame, _ = cam.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            
            frame_counter += 1
            if frame_counter % 3 == 0:
                process_frame(frame, *phs, is_live=True)
            else:
                display_frame = resize_for_display(frame, max_width=1000)
                phs[0].image(bgr_to_rgb(display_frame), use_container_width=True, channels="RGB")
                
            time.sleep(0.04)
    else:
        if PIPELINE_OK:
            cam = st.session_state.cam_manager
            if cam and cam.is_running(): cam.stop()
        phs[0].markdown("""
        <div style='background-color:#0f172a; border-radius:12px; border:2px dashed #1e293b; 
             aspect-ratio:4/3; display:flex; align-items:center; justify-content:center; 
             color:#475569; font-size:1.2rem; flex-direction:column;'>
            <div style='font-size:3.5rem; margin-bottom:1rem;'>📷</div>
            <div>Camera Offline</div>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 2: UPLOAD IMAGE ──────────────────────────────────────
with tab_upload:
    uploaded_img = st.file_uploader("Choose a Braille image (JPG / PNG)", type=["jpg", "jpeg", "png"])
    phs_up = create_main_layout()
    
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        rgb = np.array(img.convert("RGB"))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        process_frame(frame, *phs_up, is_live=False)

# ─── TAB 3: THRESHOLDS ────────────────────────────────────────
with tab_thresh:
    st.markdown("### ⚙️ Quality Threshold Tuner")
    st.info("Adjust thresholds to match your specific camera and Braille material.")
    
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**🔍 Blur Thresholds**")
        st.session_state.thresh_blur_clear = float(st.slider("Clear Image (min Laplacian var)", 50, 300, int(st.session_state.thresh_blur_clear), 10))
        st.session_state.thresh_blur_slight = float(st.slider("Slight Blur boundary", 10, 150, int(st.session_state.thresh_blur_slight), 10))
        st.markdown("**📏 Visibility Thresholds**")
        st.session_state.thresh_close = st.slider("Too Close (edge density)", 0.10, 0.40, st.session_state.thresh_close, 0.01)
        st.session_state.thresh_far = st.slider("Too Far (edge density)", 0.005, 0.10, st.session_state.thresh_far, 0.005)
    with tc2:
        st.markdown("**💡 Brightness Thresholds**")
        st.session_state.thresh_dark = st.slider("Too Dark", 10, 120, st.session_state.thresh_dark, 5)
        st.session_state.thresh_over = st.slider("Overexposed", 150, 254, st.session_state.thresh_over, 5)
        st.markdown("**📐 Alignment Threshold**")
        st.session_state.thresh_tilt = st.slider("Max tilt angle (degrees)", 1.0, 20.0, st.session_state.thresh_tilt, 0.5)

# ─── TAB 4: ANALYTICS ─────────────────────────────────────────
with tab_analytics:
    st.markdown("### 📊 Session Analytics")
    if not st.session_state.analytics_data:
        st.info("No analytics data gathered yet. Run detection to see session stats.")
    else:
        st.write("Recent detection metrics (last 50 frames):")
        data = st.session_state.analytics_data[-50:]
        chart_data = {
            "Cells": [d["cells"] for d in data],
            "Confidence": [d["conf"] for d in data]
        }
        st.line_chart(chart_data)

# ─── TAB 5: ABOUT ─────────────────────────────────────────────
with tab_about:
    st.markdown("### 📚 About BrailleVisionAI")
    st.markdown("""
    **BrailleVisionAI** is an AI-powered accessibility assistant that reads
    real embossed/handwritten Braille using a camera and converts it to English text and speech.

    **Phase Pipeline:**
    * **A/B**: Camera System & Image Input
    * **C**: Smart Quality Analyzer & AI Guidance
    * **D/E**: Braille Presence & Dot/Cell Extraction
    * **F**: Braille Translation & Speech Output
    * **G**: Smart NLP Text Reconstruction
    * **H**: Demo Stabilization & Calibration Profiles

    **Tech Stack:** Python · Streamlit · OpenCV · NumPy · pyttsx3 · YOLOv8
    """)
