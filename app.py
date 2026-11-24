"""
Streamlit Web Application: Multilingual Language Detection and Automatic Reply System Using OpenCV and OCR.
Provides image upload, live webcam snapshot, sample gallery, OpenCV pipeline inspection,
multilingual language identification, and automatic conversational reply generation.
"""

import os
import sys
import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from core.ocr_engine import MultilingualOCREngine
from core.language_detector import MultilingualLanguageDetector
from core.reply_generator import MultilingualReplyGenerator
from core.image_processor import OpenCVImageProcessor

# Page Configuration
st.set_page_config(
    page_title="Multilingual Language Detection & Auto-Reply | OpenCV & OCR",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #312E81 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        max-width: 850px;
        line-height: 1.5;
    }
    .result-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .reply-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #86EFAC;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.1);
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_ocr_engine(profile: str, use_gpu: bool):
    return MultilingualOCREngine(default_profile=profile, use_gpu=use_gpu)


@st.cache_resource
def get_language_detector():
    return MultilingualLanguageDetector()


@st.cache_resource
def get_reply_generator(api_key: str = None):
    return MultilingualReplyGenerator(api_key=api_key)


def render_tts_audio_widget(text: str, lang_code: str):
    """
    Renders browser Web Speech API button allowing the user to hear the reply spoken aloud.
    """
    tts_lang_map = {
        "en": "en-US",
        "hi": "hi-IN",
        "fr": "fr-FR",
        "es": "es-ES",
        "de": "de-DE",
        "it": "it-IT",
        "mr": "mr-IN"
    }
    target_locale = tts_lang_map.get(lang_code, "en-US")
    clean_js_text = text.replace('"', '\\"').replace("'", "\\'")

    html_code = f"""
    <div style="margin-top: 10px;">
        <button onclick="speakReply()" style="
            background-color: #059669;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            🔊 Listen to Reply ({target_locale})
        </button>
        <script>
            function speakReply() {{
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    var msg = new SpeechSynthesisUtterance("{clean_js_text}");
                    msg.lang = "{target_locale}";
                    msg.rate = 0.95;
                    window.speechSynthesis.speak(msg);
                }} else {{
                    alert("Speech synthesis is not supported in this browser.");
                }}
            }}
        </script>
    </div>
    """
    st.components.v1.html(html_code, height=55)


def main():
    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">
            <span>🌐</span> Multilingual Language Detection & Auto-Reply
        </div>
        <div class="hero-subtitle">
            Combines <b>OpenCV</b> computer vision image enhancement, <b>Optical Character Recognition (OCR)</b>, 
            multilingual language identification, and <b>context-aware automatic reply generation</b> across English, Hindi, French, Spanish, German, and more.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ System Configuration")

        st.subheader("1. OCR Recognition Script")
        profile_options = {
            "latin": "European / Latin (English, French, Spanish, German)",
            "hindi": "Hindi / Devanagari (हिन्दी + English)"
        }
        selected_profile = st.selectbox(
            "Select Script Profile",
            options=list(profile_options.keys()),
            format_func=lambda x: profile_options[x],
            index=0
        )

        st.subheader("2. OpenCV Preprocessing")
        enable_denoise = st.checkbox("Bilateral Denoising (Edge-Preserving)", value=True)
        enable_clahe = st.checkbox("Contrast Enhancement (CLAHE)", value=True)
        enable_thresh = st.checkbox("Document Binarization (Otsu)", value=False)
        conf_thresh = st.slider("Minimum OCR Confidence", 0.05, 0.90, 0.20, 0.05)

        st.subheader("3. Response Engine Mode")
        reply_mode = st.radio("Auto-Reply Generator", ["Built-in Conversational Rules", "Gemini AI Model (LLM)"])
        gemini_api_key = None
        if reply_mode == "Gemini AI Model (LLM)":
            gemini_api_key = st.text_input("Gemini API Key (optional)", type="password")

        use_gpu = st.checkbox("Enable GPU Acceleration (CUDA)", value=False)

        st.divider()
        st.markdown("""
        **Supported Primary Languages:**
        - 🇬🇧 **English** (`en`)
        - 🇮🇳 **Hindi** (हिन्दी / `hi`)
        - 🇫🇷 **French** (Français / `fr`)
        - 🇪🇸 **Spanish** (Español / `es`)
        - 🇩🇪 **German** (Deutsch / `de`)
        - 🇮🇹 **Italian** (Italiano / `it`)
        - 🇮🇳 **Marathi** (मराठी / `mr`)
        """)

    engine = get_ocr_engine(selected_profile, use_gpu)
    lang_detector = get_language_detector()
    reply_gen = get_reply_generator(gemini_api_key)

    # Main Tabs
    tab_upload, tab_camera, tab_samples, tab_text = st.tabs([
        "📁 Upload Image / Document",
        "📷 Live Camera Snapshot",
        "🧪 Multilingual Samples Gallery",
        "✍️ Direct Text & Auto-Reply Tester"
    ])

    input_bgr = None
    input_source_desc = ""

    # TAB 1: Upload Image
    with tab_upload:
        st.write("Upload any document, book snippet, screenshot, or sign containing printed text.")
        uploaded = st.file_uploader("Select Image File", type=["png", "jpg", "jpeg", "webp", "bmp"])
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            input_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            input_source_desc = f"Uploaded File: {uploaded.name}"

    # TAB 2: Camera Snapshot
    with tab_camera:
        st.write("Hold a printed paper, card, or phone screen in front of your webcam.")
        cam_shot = st.camera_input("Take a photo")
        if cam_shot:
            pil_img = Image.open(cam_shot).convert("RGB")
            input_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            input_source_desc = "Webcam Live Capture"

    # TAB 3: Pre-Generated Sample Gallery
    with tab_samples:
        st.write("Test instantly using verified high-resolution sample cards covering all project target languages:")
        samples_dir = os.path.join(os.path.dirname(__file__), "sample_images")
        sample_files = []
        if os.path.exists(samples_dir):
            sample_files = [f for f in os.listdir(samples_dir) if f.endswith(".png")]

        if sample_files:
            sample_labels = {
                "english_sample.png": "🇬🇧 English Query: 'Hello, how are you?...'",
                "hindi_sample.png": "🇮🇳 Hindi Query: 'नमस्ते, आप कैसे हैं?...'",
                "french_sample.png": "🇫🇷 French Query: 'Bonjour, comment allez-vous ?...'",
                "spanish_sample.png": "🇪🇸 Spanish Query: '¡Hola! ¿Cómo estás?...'",
                "german_sample.png": "🇩🇪 German Query: 'Guten Tag, wie geht es Ihnen?...'",
            }
            chosen_sample = st.selectbox(
                "Choose a preloaded multilingual sample:",
                sample_files,
                format_func=lambda x: sample_labels.get(x, x)
            )

            # Auto-recommend profile based on sample
            if "hindi" in chosen_sample and selected_profile != "hindi":
                st.info("💡 Tip: For the Hindi sample, ensure 'Hindi / Devanagari' is selected in the sidebar.")

            if st.button("🚀 Load and Analyze This Sample", type="primary"):
                s_path = os.path.join(samples_dir, chosen_sample)
                pil_img = Image.open(s_path).convert("RGB")
                input_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                input_source_desc = f"Sample: {chosen_sample}"
        else:
            st.warning("Sample images not found. Run `python generate_samples.py`.")

    # TAB 4: Direct Text Tester
    with tab_text:
        st.subheader("Real-Time NLP & Auto-Reply Testing")
        st.write("Test language identification and system responses directly without an image:")
        user_input_text = st.text_area(
            "Enter text in English, Hindi, French, Spanish, German, etc.:",
            value="Bonjour, comment allez-vous ?",
            height=100
        )
        if st.button("🔍 Detect Language & Generate Reply"):
            with st.spinner("Analyzing text..."):
                t_lang = lang_detector.identify_language(user_input_text)
                t_reply = reply_gen.generate_reply(
                    user_input_text,
                    lang_code=t_lang["lang_code"],
                    lang_name=t_lang["language_name"],
                    use_ai=(reply_mode == "Gemini AI Model (LLM)")
                )

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown(f"""
                <div class="result-card">
                    <span style="font-size: 0.85rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Detected Language</span>
                    <h2 style="margin: 0.4rem 0 0.8rem 0; color: #0F172A;">{t_lang['flag']} {t_lang['language_name']}</h2>
                    <p><b>ISO Code:</b> <code>{t_lang['lang_code']}</code></p>
                    <p><b>Script Family:</b> {t_lang['script']}</p>
                    <p><b>Confidence:</b> {t_lang['confidence']*100:.1f}%</p>
                    <p><b>Detection Method:</b> {t_lang['method']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_t2:
                st.markdown(f"""
                <div class="reply-card">
                    <span style="font-size: 0.85rem; color: #166534; font-weight: 600; text-transform: uppercase;">System Auto-Reply</span>
                    <h3 style="margin: 0.4rem 0 0.8rem 0; color: #14532D;">"{t_reply['reply_text']}"</h3>
                    <p><b>Intent:</b> <code>{t_reply['intent']}</code></p>
                    <p><b>Engine:</b> {t_reply['mode']}</p>
                </div>
                """, unsafe_allow_html=True)
                render_tts_audio_widget(t_reply['reply_text'], t_lang['lang_code'])

    # Processing Image if input provided
    if input_bgr is not None:
        st.divider()
        st.subheader(f"📊 Analysis Results ({input_source_desc})")

        with st.spinner("Processing image with OpenCV, running OCR, detecting language, and generating reply..."):
            # Execute pipeline
            result = engine.process_image(
                input_bgr,
                profile=selected_profile,
                conf_threshold=conf_thresh,
                apply_preprocessing=True,
                use_ai_replies=(reply_mode == "Gemini AI Model (LLM)")
            )

        lang_meta = result["primary_language"]
        reply_meta = result["system_reply"]
        full_text = result["full_extracted_text"]
        detections = result["detections"]
        stages = result["preprocessing_stages"]

        # KPI Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Detected Language", f"{lang_meta.get('flag', '')} {lang_meta.get('language_name', 'Unknown')}")
        with m2:
            st.metric("Language Confidence", f"{lang_meta.get('confidence', 0.0) * 100:.1f}%")
        with m3:
            st.metric("Text Blocks Found", len(detections))
        with m4:
            st.metric("Conversational Intent", reply_meta.get("intent", "None").title())

        # Main Output Cards
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase; margin-bottom: 0.5rem;">
                    📝 Extracted OCR Text
                </div>
                <div style="font-size: 1.15rem; color: #1E293B; line-height: 1.6; font-weight: 500; min-height: 80px; background: #F8FAFC; padding: 1rem; border-radius: 8px; border: 1px solid #E2E8F0;">
                    {full_text if full_text else "<i>No readable text detected.</i>"}
                </div>
                <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #64748B;">
                    <b>Script:</b> {lang_meta.get('script', 'Unknown')} &nbsp;|&nbsp; 
                    <b>Method:</b> {lang_meta.get('method', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_right:
            st.markdown(f"""
            <div class="reply-card">
                <div style="font-size: 0.85rem; color: #166534; font-weight: 700; text-transform: uppercase; margin-bottom: 0.5rem;">
                    🤖 System Auto-Reply in Detected Language
                </div>
                <div style="font-size: 1.25rem; color: #14532D; line-height: 1.5; font-weight: 600; min-height: 80px; background: #FFFFFF; padding: 1rem; border-radius: 8px; border: 1px solid #BBF7D0;">
                    "{reply_meta.get('reply_text', '')}"
                </div>
                <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #166534;">
                    <b>Generator Engine:</b> {reply_meta.get('mode', '')} &nbsp;|&nbsp; 
                    <b>Language:</b> {lang_meta.get('language_name', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if reply_meta.get('reply_text'):
                render_tts_audio_widget(reply_meta['reply_text'], lang_meta.get('lang_code', 'en'))

        # Visual Image Comparison
        st.subheader("🖼️ Computer Vision & OCR Localization")
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.caption("Original Input Image")
            orig_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
            st.image(orig_rgb, use_container_width=True)

        with v_col2:
            st.caption("OpenCV Multilingual Annotated Output (Bounding Boxes + Language Badges)")
            annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

        # OpenCV Preprocessing Stages Expander
        with st.expander("🔬 View OpenCV Image Preprocessing Pipeline Stages"):
            st.write("Inspect how OpenCV optimizes contrast and denoises the image prior to OCR extraction:")
            s_cols = st.columns(3)
            with s_cols[0]:
                if "grayscale" in stages:
                    st.caption("1. Grayscale Conversion")
                    st.image(stages["grayscale"], use_container_width=True)
            with s_cols[1]:
                if "denoised" in stages:
                    st.caption("2. Bilateral Denoising")
                    st.image(stages["denoised"], use_container_width=True)
            with s_cols[2]:
                if "enhanced" in stages:
                    st.caption("3. Contrast Enhancement (CLAHE)")
                    st.image(stages["enhanced"], use_container_width=True)

        # Detailed Breakdown Table
        st.subheader("📋 Detailed Line-by-Line Breakdown")
        if detections:
            table_rows = []
            for idx, item in enumerate(detections, 1):
                table_rows.append({
                    "#": idx,
                    "Extracted Text": item["text"],
                    "Language": f"{item['flag']} {item['language']}",
                    "OCR Conf": f"{item['ocr_confidence']*100:.1f}%",
                    "Lang Conf": f"{item['confidence']*100:.1f}%",
                    "Intent": item["intent"],
                    "Line System Reply": item["reply"]
                })
            st.dataframe(table_rows, use_container_width=True)
        else:
            st.info("No text blocks met the confidence threshold.")

        # Export Report
        st.subheader("📥 Export Results")
        export_data = {
            "source": input_source_desc,
            "detected_language": lang_meta,
            "extracted_text": full_text,
            "system_reply": reply_meta,
            "line_detections": detections
        }
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name="language_detection_report.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
