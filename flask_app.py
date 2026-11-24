"""
Flask Web Application Server:
AI-Based Language Detection System Using OpenCV & Hybrid Script/NLP Models.
Provides:
- Real-time webcam video feed with bounding box and language annotations
- Drag-and-drop document and image scan upload
- Interactive OpenCV Preprocessing Inspection Studio (Grayscale, Bilateral, CLAHE, Otsu, Contours, Deskew)
- Visual Script CNN (Approach A) + NLP Language ID (Approach B) + Hybrid Pipeline
- Instant English translation of foreign text
- Text-to-Speech (TTS) audio playback
- CSV and JSON report exports
- Voice microphone assistant with native accent replies
"""

import os
import sys
import io
import csv
import json
import base64
import time
import tempfile
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_file
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    gTTS = None
    HAS_GTTS = False

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from core.ocr_engine import MultilingualOCREngine
from core.language_detector import MultilingualLanguageDetector, INDIAN_LANGUAGES_22
from core.reply_generator import MultilingualReplyGenerator
from core.image_processor import OpenCVImageProcessor
from core.script_cnn import HybridScriptClassifier

app = Flask(__name__)

# Initialize singletons
processor = OpenCVImageProcessor()
lang_detector = MultilingualLanguageDetector()
reply_gen = MultilingualReplyGenerator()
script_classifier = HybridScriptClassifier()
ocr_engine = MultilingualOCREngine(default_profile="latin", use_gpu=False)

# Camera state & last analysis cache
camera = None
latest_frame = None
last_analysis_result = None
active_feed_source = "hindi"
feed_images_cache = {}

# ==============================================================================
# ⚙️ COMPREHENSIVE CONFIGURABLE SYSTEM PARAMETERS & VISION TUNER
# ==============================================================================
DEFAULT_PARAMETERS = {
    # 1. OCR & Confidence
    "conf_threshold": 0.20,
    "profile": "hindi",
    "word_segmentation": True,
    "detection_mode": "deep_ocr",

    # 2. OpenCV Vision Preprocessing
    "apply_preprocessing": True,
    "max_dim": 1600,
    "denoise": True,
    "denoise_method": "bilateral",
    "bilateral_d": 9,
    "bilateral_sigma_color": 75.0,
    "bilateral_sigma_space": 75.0,
    "clahe": True,
    "clahe_clip_limit": 2.5,
    "clahe_tile_grid": 8,
    "binarize": False,
    "binarize_method": "otsu",
    "adaptive_block_size": 19,
    "adaptive_c": 9,

    # 3. Video & Keyframe Acquisition
    "video_sampling_step": 5,
    "recording_duration": 3.0,
    "min_sharpness_score": 50.0,

    # 4. Output, Speech & AI Reply
    "target_lang": "hi",
    "enable_translation": True,
    "use_ai_replies": False,
    "tts_speed": 1.0,
}

PARAMETER_METADATA = {
    "conf_threshold": {"type": "float", "min": 0.05, "max": 0.95, "step": 0.05, "label": "OCR Confidence Threshold", "category": "OCR & Models"},
    "profile": {"type": "select", "options": ["hindi", "tamil", "telugu", "bengali", "latin", "auto"], "label": "OCR Profile", "category": "OCR & Models"},
    "word_segmentation": {"type": "bool", "label": "Word-by-Word Sub-Bounding Boxes", "category": "OCR & Models"},
    "apply_preprocessing": {"type": "bool", "label": "OpenCV Image Preprocessing", "category": "OpenCV Vision"},
    "max_dim": {"type": "select", "options": [800, 1280, 1600, 0], "label": "Max Resize Dimension (px)", "category": "OpenCV Vision"},
    "denoise": {"type": "bool", "label": "Denoising Filter", "category": "OpenCV Vision"},
    "denoise_method": {"type": "select", "options": ["bilateral", "gaussian", "median", "none"], "label": "Denoise Algorithm", "category": "OpenCV Vision"},
    "bilateral_d": {"type": "int", "min": 3, "max": 15, "step": 2, "label": "Bilateral Diameter (d)", "category": "OpenCV Vision"},
    "bilateral_sigma_color": {"type": "float", "min": 25.0, "max": 150.0, "step": 5.0, "label": "Bilateral Sigma Color", "category": "OpenCV Vision"},
    "clahe": {"type": "bool", "label": "CLAHE Contrast Enhancement", "category": "OpenCV Vision"},
    "clahe_clip_limit": {"type": "float", "min": 1.0, "max": 8.0, "step": 0.5, "label": "CLAHE Clip Limit", "category": "OpenCV Vision"},
    "clahe_tile_grid": {"type": "select", "options": [4, 8, 16], "label": "CLAHE Tile Grid Size", "category": "OpenCV Vision"},
    "binarize": {"type": "bool", "label": "Binarization Before OCR", "category": "OpenCV Vision"},
    "binarize_method": {"type": "select", "options": ["otsu", "adaptive"], "label": "Binarization Mode", "category": "OpenCV Vision"},
    "video_sampling_step": {"type": "int", "min": 1, "max": 20, "step": 1, "label": "Video Frame Sampling Step", "category": "Video & Keyframe"},
    "recording_duration": {"type": "float", "min": 1.5, "max": 6.0, "step": 0.5, "label": "Camera Recording Duration (s)", "category": "Video & Keyframe"},
    "target_lang": {"type": "string", "label": "Target Spoken Language", "category": "Output & Speech"},
    "enable_translation": {"type": "bool", "label": "Auto-Translation", "category": "Output & Speech"},
    "tts_speed": {"type": "select", "options": [0.75, 1.0, 1.25, 1.5], "label": "Voice Speed", "category": "Output & Speech"}
}

PARAMETER_PRESETS = {
    "optimal_default": {
        "name": "Optimal Default (Balanced)",
        "description": "Balanced contrast and edge-preserving filter suited for general Indian signboards and street text.",
        "params": {
            "conf_threshold": 0.20,
            "apply_preprocessing": True,
            "max_dim": 1600,
            "denoise": True,
            "denoise_method": "bilateral",
            "bilateral_d": 9,
            "bilateral_sigma_color": 75.0,
            "clahe": True,
            "clahe_clip_limit": 2.5,
            "clahe_tile_grid": 8,
            "binarize": False,
            "binarize_method": "otsu",
            "word_segmentation": True,
            "video_sampling_step": 5,
            "recording_duration": 3.0
        }
    },
    "low_light": {
        "name": "Low Light / Night Signs",
        "description": "High CLAHE contrast enhancement and increased bilateral smoothing for dark, noisy captures.",
        "params": {
            "conf_threshold": 0.15,
            "apply_preprocessing": True,
            "max_dim": 1600,
            "denoise": True,
            "denoise_method": "bilateral",
            "bilateral_d": 11,
            "bilateral_sigma_color": 90.0,
            "clahe": True,
            "clahe_clip_limit": 4.0,
            "clahe_tile_grid": 8,
            "binarize": False,
            "binarize_method": "otsu",
            "word_segmentation": True,
            "video_sampling_step": 4,
            "recording_duration": 3.5
        }
    },
    "high_speed": {
        "name": "High Speed / Real-Time",
        "description": "Downsamples image to 800px and samples every 10th video frame for fast responsiveness.",
        "params": {
            "conf_threshold": 0.25,
            "apply_preprocessing": True,
            "max_dim": 800,
            "denoise": True,
            "denoise_method": "gaussian",
            "bilateral_d": 7,
            "bilateral_sigma_color": 50.0,
            "clahe": True,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": 4,
            "binarize": False,
            "binarize_method": "otsu",
            "word_segmentation": True,
            "video_sampling_step": 10,
            "recording_duration": 2.0
        }
    },
    "document_scan": {
        "name": "Document Scan (Otsu Binarization)",
        "description": "Applies Otsu thresholding for printed black-and-white documents, book pages, and receipts.",
        "params": {
            "conf_threshold": 0.20,
            "apply_preprocessing": True,
            "max_dim": 1600,
            "denoise": True,
            "denoise_method": "gaussian",
            "bilateral_d": 7,
            "bilateral_sigma_color": 60.0,
            "clahe": True,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": 8,
            "binarize": True,
            "binarize_method": "otsu",
            "word_segmentation": True,
            "video_sampling_step": 5,
            "recording_duration": 3.0
        }
    }
}

current_parameters = DEFAULT_PARAMETERS.copy()


def get_merged_parameters(req=None):
    """Merges current global parameters with per-request query or form overrides."""
    p = current_parameters.copy()
    if not req:
        return p

    def _parse_val(val, default_val):
        if val is None:
            return None
        if isinstance(default_val, bool):
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("true", "1", "yes", "on")
        if isinstance(default_val, int):
            try:
                return int(val)
            except (ValueError, TypeError):
                return None
        if isinstance(default_val, float):
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        return str(val)

    source = {}
    if req.args:
        source.update(req.args.to_dict())
    if req.form:
        source.update(req.form.to_dict())
    json_data = req.get_json(silent=True)
    if json_data and isinstance(json_data, dict):
        source.update(json_data)

    for k, default_val in DEFAULT_PARAMETERS.items():
        if k in source:
            parsed = _parse_val(source[k], default_val)
            if parsed is not None:
                p[k] = parsed

    return p


def load_feed_image(source_name):
    """Loads and caches simulated live Indian signboard camera feed frames."""
    filename_map = {
        "hindi": "hindi_sample.png",
        "marathi": "marathi_sample.png",
        "tamil": "tamil_sample.png",
        "telugu": "telugu_sample.png",
        "bengali": "bengali_sample.png",
        "gujarati": "gujarati_sample.png",
        "punjabi": "punjabi_sample.png",
        "multilingual": "multilingual_india.png"
    }
    fname = filename_map.get(source_name, "hindi_sample.png")
    if fname not in feed_images_cache:
        path = os.path.join(os.path.dirname(__file__), "sample_images", fname)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                feed_images_cache[fname] = cv2.resize(img, (720, 480))
    return feed_images_cache.get(fname)


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    return camera


def generate_camera_frames():
    """Generates continuous MJPEG camera stream with live Indian signboard video frames."""
    global latest_frame, active_feed_source
    cap = get_camera()

    while True:
        frame = None
        # 1. Try physical hardware camera if explicitly chosen and open
        if active_feed_source == "hardware" and cap.isOpened():
            success, raw = cap.read()
            if success and raw is not None:
                frame = raw

        # 2. Virtual Indian Signboard Camera Feed (Active & ready for OCR + video recording)
        if frame is None:
            base_img = load_feed_image(active_feed_source)
            if base_img is not None:
                frame = base_img.copy()
                # Dynamic timestamp overlay simulating real live surveillance camera
                now_str = time.strftime("%H:%M:%S")
                cv2.putText(
                    frame, f"OPENCV LIVE: {active_feed_source.upper()} [{now_str}]",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 255), 2
                )
                # Subtle simulated camera scan-line
                scan_y = int((time.time() * 75) % 480)
                cv2.line(frame, (0, scan_y), (720, scan_y), (0, 165, 255), 1)
            else:
                standby = np.zeros((480, 720, 3), dtype=np.uint8)
                cv2.putText(standby, "OpenCV Camera Standby", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                frame = standby

        latest_frame = frame.copy()

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)  # ~25 FPS


def encode_image_base64(img_bgr, format=".jpg"):
    """Encodes a BGR numpy image into a Base64 data URI string."""
    if img_bgr is None:
        return ""
    success, buffer = cv2.imencode(format, img_bgr)
    if not success:
        return ""
    mime = "image/jpeg" if format == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer).decode('utf-8')}"


def serialize_analysis_result(result):
    """Packages OCR + Preprocessing + Translation output for web client JSON."""
    global last_analysis_result
    last_analysis_result = result

    # Enrich primary language with Indian cultural metadata if applicable
    p_lang = result.get("primary_language", {})
    l_code = p_lang.get("lang_code", "")
    if l_code in INDIAN_LANGUAGES_22:
        meta = INDIAN_LANGUAGES_22[l_code]
        p_lang.setdefault("family", meta.get("family", "Indo-Aryan"))
        p_lang.setdefault("region", meta.get("region", "India"))
        p_lang.setdefault("official_status", meta.get("official_status", "8th Schedule Recognized"))
        p_lang.setdefault("native_name", meta.get("native", p_lang.get("native_name", "")))
        p_lang.setdefault("english_name", meta.get("name", p_lang.get("english_name", "")))

    # Serialize OpenCV preprocessing stages for inspection studio
    stages_b64 = {}
    stages = result.get("preprocessing_stages", {})
    stage_keys = ["original", "grayscale", "denoised", "enhanced_clahe", "threshold_otsu", "morphology", "contour_detection"]
    for key in stage_keys:
        if key in stages and isinstance(stages[key], np.ndarray):
            stages_b64[key] = encode_image_base64(stages[key], format=".jpg")

    return {
        "status": "success",
        "primary_language": p_lang,
        "primary_script": result.get("primary_script", {}),
        "full_extracted_text": result["full_extracted_text"],
        "full_translation": result.get("full_translation", ""),
        "system_reply": result["system_reply"],
        "target_language": result.get("target_language", "hi"),
        "video_stats": result.get("video_stats", {}),
        "detections": result["detections"],
        "words": result.get("words", []),
        "word_count": result.get("word_count", len(result.get("words", []))),
        "region_count": result.get("region_count", len(result["detections"])),
        "skew_angle": result.get("skew_angle", 0.0),
        "annotated_image_base64": encode_image_base64(result["annotated_image"], format=".png"),
        "preprocessing_stages_base64": stages_b64,
        "parameters_used": result.get("parameters_used", current_parameters)
    }


# ==============================================================================
# 🎛️ PARAMETERS MANAGEMENT REST API
# ==============================================================================
@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    """Returns the current configurable system parameters, presets, and metadata schema."""
    return jsonify({
        "status": "success",
        "parameters": current_parameters,
        "presets": PARAMETER_PRESETS,
        "metadata": PARAMETER_METADATA
    })


@app.route('/api/parameters', methods=['POST'])
def update_parameters():
    """Dynamically updates one or more system parameters in real time."""
    global current_parameters
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({"status": "error", "message": "No parameters provided."}), 400

    updated = []
    for k, default_val in DEFAULT_PARAMETERS.items():
        if k in data:
            val = data[k]
            if isinstance(default_val, bool):
                current_parameters[k] = val if isinstance(val, bool) else (str(val).lower() in ("true", "1", "yes", "on"))
            elif isinstance(default_val, int):
                try:
                    current_parameters[k] = int(val)
                except (ValueError, TypeError):
                    continue
            elif isinstance(default_val, float):
                try:
                    current_parameters[k] = float(val)
                except (ValueError, TypeError):
                    continue
            else:
                current_parameters[k] = str(val)
            updated.append(k)

    return jsonify({
        "status": "success",
        "message": f"Successfully updated {len(updated)} parameter(s): {', '.join(updated)}",
        "parameters": current_parameters
    })


@app.route('/api/parameters/reset', methods=['POST'])
def reset_parameters():
    """Resets all system parameters back to the optimal default configuration."""
    global current_parameters
    current_parameters = DEFAULT_PARAMETERS.copy()
    return jsonify({
        "status": "success",
        "message": "All parameters have been reset to optimal defaults.",
        "parameters": current_parameters
    })


@app.route('/api/parameters/preset/<preset_name>', methods=['POST'])
def apply_preset(preset_name):
    """Applies a curated configuration preset (e.g. low_light, high_speed, document_scan)."""
    global current_parameters
    if preset_name not in PARAMETER_PRESETS:
        return jsonify({"status": "error", "message": f"Unknown preset: {preset_name}"}), 404

    preset = PARAMETER_PRESETS[preset_name]
    for k, v in preset["params"].items():
        if k in current_parameters:
            current_parameters[k] = v

    return jsonify({
        "status": "success",
        "message": f"Applied preset '{preset['name']}'.",
        "preset": preset_name,
        "parameters": current_parameters
    })


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_camera_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/analyze-frame', methods=['GET', 'POST'])
def analyze_frame():
    """Captures and analyzes the current live webcam video frame with configurable parameters."""
    global latest_frame, active_feed_source
    if latest_frame is None:
        img = load_feed_image(active_feed_source)
        if img is not None:
            latest_frame = img.copy()
    if latest_frame is None:
        cap = get_camera()
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                latest_frame = frame

    if latest_frame is None:
        return jsonify({"status": "error", "message": "No active camera frame available."}), 400

    params = get_merged_parameters(request)
    profile = params.get('profile', 'hindi')

    # Automatically adapt profile to match active feed if auto
    if not profile or profile == 'auto' or profile == 'hindi':
        if active_feed_source in ('tamil', 'telugu', 'bengali'):
            profile = active_feed_source
        else:
            profile = 'hindi'
    params['profile'] = profile

    result = ocr_engine.process_image(
        latest_frame,
        **params
    )
    return jsonify(serialize_analysis_result(result))


@app.route('/api/process-recording', methods=['POST'])
def process_recording():
    """
    Processes a recorded video clip (from client MediaRecorder or file upload):
    1. Reads video frames using OpenCV (cv2.VideoCapture) with configurable sampling step
    2. Measures Laplacian sharpness of all frames
    3. Selects the sharpest keyframe containing stable text
    4. Preprocesses and detects language using active parameters
    5. Translates & responds in user's selected target language
    """
    params = get_merged_parameters(request)
    sampling_step = params.get("video_sampling_step", 5)

    best_frame = None
    video_stats = {}

    # 1. Process uploaded video file
    video_file = request.files.get('video')
    if video_file and video_file.filename != '':
        suffix = os.path.splitext(video_file.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            video_file.save(tmp_path)

        try:
            best_frame, video_stats = processor.process_video_file(tmp_path, sampling_step=sampling_step)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # 2. Process burst frames JSON fallback if video file decode yielded no frames
    if best_frame is None and 'frames_json' in request.form:
        try:
            frames_b64 = json.loads(request.form['frames_json'])
            bgr_frames = []
            for b64 in frames_b64:
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                data = base64.b64decode(b64)
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    bgr_frames.append(img)
            if bgr_frames:
                best_frame, best_score, best_idx = processor.select_sharpest_frame(bgr_frames)
                video_stats = {
                    "total_video_frames": len(bgr_frames),
                    "sampled_frames_analyzed": len(bgr_frames),
                    "best_frame_index": best_idx,
                    "sharpness_score": best_score,
                    "source": "burst_frames"
                }
        except Exception:
            pass

    # 3. Fallback to latest live camera frame if needed
    if best_frame is None:
        global latest_frame
        if latest_frame is None:
            sample_path = os.path.join(os.path.dirname(__file__), "sample_images", "hindi_sample.png")
            if os.path.exists(sample_path):
                latest_frame = cv2.imread(sample_path)
        if latest_frame is not None:
            best_frame = latest_frame.copy()
            video_stats = {"source": "live_camera_fallback", "sharpness_score": 100.0}
        else:
            return jsonify({"status": "error", "message": "No readable video frames received."}), 400

    result = ocr_engine.process_image(
        best_frame,
        **params
    )
    result["video_stats"] = video_stats
    return jsonify(serialize_analysis_result(result))


@app.route('/api/feed-sources', methods=['GET'])
def get_feed_sources():
    """Returns available virtual and physical camera feed sources."""
    global active_feed_source
    cap = get_camera()
    has_hw = cap.isOpened()
    return jsonify({
        "status": "success",
        "active_source": active_feed_source,
        "hardware_camera_available": has_hw,
        "sources": [
            {"id": "hindi", "name": "Hindi (हिन्दी) Railway Inquiry Notice", "flag": "🇮🇳"},
            {"id": "tamil", "name": "Tamil (தமிழ்) Heritage Signboard", "flag": "🇮🇳"},
            {"id": "telugu", "name": "Telugu (తెలుగు) Civic Notice Board", "flag": "🇮🇳"},
            {"id": "bengali", "name": "Bengali (বাংলা) Kolkata Metro Info", "flag": "🇮🇳"},
            {"id": "marathi", "name": "Marathi (मराठी) Civic Bulletin", "flag": "🇮🇳"},
            {"id": "gujarati", "name": "Gujarati (ગુજરાતી) Commercial Sign", "flag": "🇮🇳"},
            {"id": "punjabi", "name": "Punjabi (ਪੰਜਾਬੀ) Highway Sign", "flag": "🇮🇳"},
            {"id": "multilingual", "name": "Trilingual Station Board", "flag": "🇮🇳"},
            {"id": "hardware", "name": "Physical Webcam (USB / Hardware)", "flag": "📷"}
        ]
    })


@app.route('/api/set-feed-source', methods=['POST'])
def set_feed_source():
    """Switches the active camera feed source for live view, scanning, and recording."""
    global active_feed_source, latest_frame
    data = request.get_json() or {}
    source = data.get("source", "hindi").lower()
    active_feed_source = source
    # Pre-warm frame immediately
    img = load_feed_image(source)
    if img is not None:
        latest_frame = img.copy()
    return jsonify({"status": "success", "active_feed_source": active_feed_source})


@app.route('/api/record-camera', methods=['POST', 'GET'])
def record_camera():
    """
    Records video frames across configurable duration from the active OpenCV camera,
    selects the sharpest keyframe using Laplacian variance, and detects language.
    """
    global latest_frame, active_feed_source
    if latest_frame is None:
        img = load_feed_image(active_feed_source)
        if img is not None:
            latest_frame = img.copy()

    params = get_merged_parameters(request)
    duration = float(params.get("recording_duration", 3.0))
    profile = params.get('profile', 'hindi')

    # Automatically map profile to match active feed if auto
    if not profile or profile == 'auto' or profile == 'hindi':
        if active_feed_source in ('tamil', 'telugu', 'bengali'):
            profile = active_feed_source
        else:
            profile = 'hindi'
    params['profile'] = profile

    frames = []
    start_time = time.time()
    while (time.time() - start_time) < duration:
        if latest_frame is not None:
            frames.append(latest_frame.copy())
        time.sleep(0.10)

    if frames:
        best_frame, best_score, best_idx = processor.select_sharpest_frame(frames)
        video_stats = {
            "total_video_frames": len(frames),
            "sampled_frames_analyzed": len(frames),
            "best_frame_index": best_idx,
            "sharpness_score": best_score,
            "source": f"opencv_camera_stream ({active_feed_source})"
        }
    elif latest_frame is not None:
        best_frame = latest_frame.copy()
        video_stats = {"source": "opencv_single_frame", "sharpness_score": 100.0}
    else:
        return jsonify({"status": "error", "message": "No active camera frame available."}), 400

    result = ocr_engine.process_image(
        best_frame,
        **params
    )
    result["video_stats"] = video_stats
    return jsonify(serialize_analysis_result(result))


@app.route('/api/upload', methods=['POST'])
def handle_upload():
    """Processes uploaded document scan or image through the full OpenCV + AI pipeline with active parameters."""
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    params = get_merged_parameters(request)

    file_bytes = np.frombuffer(file.read(), np.uint8)
    bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr_img is None:
        return jsonify({"status": "error", "message": "Failed to decode image file."}), 400

    result = ocr_engine.process_image(
        bgr_img,
        **params
    )
    return jsonify(serialize_analysis_result(result))


@app.route('/api/sample/<filename>', methods=['GET'])
def handle_sample(filename):
    """Processes one of the preloaded sample images with active parameters."""
    samples_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    file_path = os.path.join(samples_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "Sample file not found."}), 404

    params = get_merged_parameters(request)
    profile = params.get('profile', '')

    if not profile or profile == 'auto' or profile == 'latin':
        fn_lower = filename.lower()
        if "hindi" in fn_lower or "marathi" in fn_lower or "multilingual" in fn_lower or "gujarati" in fn_lower or "punjabi" in fn_lower:
            profile = 'hindi'
        elif "tamil" in fn_lower:
            profile = 'tamil'
        elif "telugu" in fn_lower:
            profile = 'telugu'
        elif "bengali" in fn_lower:
            profile = 'bengali'
        else:
            profile = 'hindi'
    params['profile'] = profile

    bgr_img = cv2.imread(file_path)
    if bgr_img is None:
        return jsonify({"status": "error", "message": "Could not read sample file."}), 500

    result = ocr_engine.process_image(
        bgr_img,
        **params
    )
    return jsonify(serialize_analysis_result(result))


@app.route('/api/samples-list', methods=['GET'])
def list_samples():
    """Returns metadata for all available test samples with focus on Indian Languages."""
    samples = [
        {
            "filename": "hindi_sample.png",
            "title": "Hindi Rail & Public Inquiry",
            "language": "Hindi (हिन्दी)",
            "flag": "🇮🇳",
            "script": "Devanagari (देवनागरी)",
            "profile": "hindi",
            "description": "Devanagari headline Shirorekha with Hindi greetings 'नमस्ते, आप कैसे हैं?' and customer inquiry."
        },
        {
            "filename": "marathi_sample.png",
            "title": "Marathi Civic Bulletin",
            "language": "Marathi (मराठी)",
            "flag": "🇮🇳",
            "script": "Devanagari (देवनागरी)",
            "profile": "hindi",
            "description": "Maharashtra notice disambiguated from Hindi via distinct Marathi markers ('नमस्कार, तुम्ही कसे आहात, आहे')."
        },
        {
            "filename": "multilingual_india.png",
            "title": "Trilingual Indian Station Board",
            "language": "Multi-Script Indic",
            "flag": "🇮🇳",
            "script": "Devanagari + English + Regional",
            "profile": "hindi",
            "description": "Standard Indian Railway multi-script station sign demonstrating localized region detection."
        },
        {
            "filename": "tamil_sample.png",
            "title": "Tamil Nadu Heritage Board",
            "language": "Tamil (தமிழ்)",
            "flag": "🇮🇳",
            "script": "Tamil (தமிழ்)",
            "profile": "tamil",
            "description": "Dravidian rounded looping glyphs with Tamil Unicode range and English translation."
        },
        {
            "filename": "telugu_sample.png",
            "title": "Andhra / Telangana Notice",
            "language": "Telugu (తెలుగు)",
            "flag": "🇮🇳",
            "script": "Telugu (తెలుగు)",
            "profile": "telugu",
            "description": "Characteristic circular loops and checkmarks characteristic of the Telugu script."
        },
        {
            "filename": "bengali_sample.png",
            "title": "Kolkata Metro Information",
            "language": "Bengali (বাংলা)",
            "flag": "🇮🇳",
            "script": "Bengali (বাংলা)",
            "profile": "bengali",
            "description": "Eastern Indo-Aryan script with prominent upper horizontal headline and acute angular triangles."
        },
        {
            "filename": "gujarati_sample.png",
            "title": "Gujarat Commercial Board",
            "language": "Gujarati (ગુજરાતી)",
            "flag": "🇮🇳",
            "script": "Gujarati (ગુજરાતી)",
            "profile": "hindi",
            "description": "Indo-Aryan script structurally derived from Devanagari but uniquely lacking the top Shirorekha headline."
        },
        {
            "filename": "punjabi_sample.png",
            "title": "Punjab Highway Sign",
            "language": "Punjabi (ਪੰਜਾਬੀ)",
            "flag": "🇮🇳",
            "script": "Gurmukhi (ਗੁਰਮੁਖੀ)",
            "profile": "hindi",
            "description": "North Indian Gurmukhi script created for Punjabi with distinct top bar and straight vertical stems."
        }
    ]
    return jsonify({"status": "success", "samples": samples})


@app.route('/api/languages-22', methods=['GET'])
def get_languages_22():
    """Returns metadata and interactive test samples for all 22 official Indian languages."""
    return jsonify({
        "status": "success",
        "count": len(INDIAN_LANGUAGES_22),
        "languages": INDIAN_LANGUAGES_22
    })


@app.route('/api/text-reply', methods=['POST'])
def direct_text():
    """Direct NLP text evaluation with response in selected Indian/International language."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    target_lang = data.get("target_lang", "hi")

    if not text:
        return jsonify({"status": "error", "message": "Text is empty."}), 400

    lang_info = lang_detector.identify_language(text)
    translation = ocr_engine.translate_text(text, source_lang=lang_info["lang_code"], target_lang=target_lang)
    reply_info = reply_gen.generate_reply(
        text,
        lang_code=lang_info["lang_code"],
        target_lang_code=target_lang,
        lang_name=lang_info["language_name"]
    )

    return jsonify({
        "status": "success",
        "language": lang_info,
        "translation": translation,
        "reply": reply_info,
        "target_language": target_lang
    })


@app.route('/api/voice-process', methods=['POST'])
def voice_process():
    """Processes spoken microphone transcript with language ID, translation, and auto-reply."""
    data = request.get_json() or {}
    transcript = data.get("transcript", "").strip()
    target_lang = data.get("target_lang", "hi")

    if not transcript:
        return jsonify({"status": "error", "message": "Voice transcript is empty."}), 400

    lang_info = lang_detector.identify_language(transcript)
    translation = ocr_engine.translate_text(transcript, source_lang=lang_info["lang_code"], target_lang=target_lang)
    reply_info = reply_gen.generate_reply(
        transcript,
        lang_code=lang_info["lang_code"],
        target_lang_code=target_lang,
        lang_name=lang_info["language_name"]
    )

    tts_locale_map = {
        "en": "en-IN",
        "hi": "hi-IN",
        "mr": "mr-IN",
        "ta": "ta-IN",
        "te": "te-IN",
        "bn": "bn-IN",
        "gu": "gu-IN",
        "kn": "kn-IN",
        "ml": "ml-IN",
        "pa": "pa-IN",
        "ur": "ur-IN",
        "as": "as-IN",
        "or": "or-IN",
        "fr": "fr-FR",
        "es": "es-ES",
        "de": "de-DE",
        "it": "it-IT"
    }

    return jsonify({
        "status": "success",
        "transcript": transcript,
        "language": lang_info,
        "translation": translation,
        "reply": reply_info,
        "target_language": target_lang,
        "tts_locale": tts_locale_map.get(target_lang, "hi-IN")
    })


@app.route('/api/tts', methods=['GET', 'POST'])
def get_tts_audio():
    """Generates MP3 audio for spoken feedback via gTTS in the selected language."""
    text = request.args.get('text') or request.form.get('text') or (request.get_json(silent=True) or {}).get('text', '')
    lang = request.args.get('lang') or request.form.get('lang') or (request.get_json(silent=True) or {}).get('lang', 'hi')

    if not text:
        return jsonify({"status": "error", "message": "No text provided for TTS"}), 400

    # Extended mapping for all Indian languages to supported gTTS voices
    tts_voice_map = {
        "hi": "hi",
        "bn": "bn",
        "ta": "ta",
        "te": "te",
        "mr": "mr",
        "gu": "gu",
        "kn": "kn",
        "ml": "ml",
        "pa": "pa",
        "ur": "ur",
        "en": "en",
        "as": "bn",    # Assamese maps to Bengali TTS voice phonetics
        "brx": "hi",   # Bodo maps to Devanagari Hindi TTS
        "doi": "hi",   # Dogri maps to Hindi TTS
        "kok": "mr",   # Konkani maps to Marathi TTS
        "mai": "hi",   # Maithili maps to Hindi TTS
        "mni": "bn",   # Manipuri maps to Bengali TTS
        "ne": "hi",    # Nepali maps to Hindi/Nepali
        "or": "bn",    # Odia maps to Bengali TTS phonetics
        "sa": "hi",    # Sanskrit maps to Hindi TTS
        "sat": "hi",   # Santali maps to Hindi
        "sd": "ur",    # Sindhi maps to Urdu TTS
        "fr": "fr",
        "es": "es",
        "de": "de",
        "it": "it"
    }
    gtts_lang = tts_voice_map.get(lang, "hi")

    if not HAS_GTTS or gTTS is None:
        return jsonify({"status": "unavailable", "message": "Server-side gTTS not installed; client uses Web Speech API"}), 200

    try:
        tts = gTTS(text=text[:350], lang=gtts_lang, slow=False)
        mp3_io = io.BytesIO()
        tts.write_to_fp(mp3_io)
        mp3_io.seek(0)
        return send_file(mp3_io, mimetype='audio/mpeg')
    except Exception as e:
        # Fallback to English TTS if specific regional voice fails
        try:
            tts = gTTS(text=text[:350], lang='en', slow=False)
            mp3_io = io.BytesIO()
            tts.write_to_fp(mp3_io)
            mp3_io.seek(0)
            return send_file(mp3_io, mimetype='audio/mpeg')
        except Exception as e2:
            return jsonify({"status": "error", "message": str(e2)}), 500


@app.route('/api/export-csv', methods=['GET'])
def export_csv():
    """Generates downloadable CSV summary of the latest detection analysis."""
    global last_analysis_result
    if not last_analysis_result or not last_analysis_result.get("detections"):
        return "No analysis data available to export.", 404

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Region_ID", "Bounding_Box_X", "Bounding_Box_Y", "Width", "Height",
        "Script", "Script_Confidence", "Detected_Language", "Language_Code",
        "Language_Confidence", "OCR_Confidence", "Extracted_Text", "English_Translation", "System_Reply"
    ])

    for i, item in enumerate(last_analysis_result["detections"]):
        rect = item.get("rect", [0, 0, 0, 0])
        writer.writerow([
            i + 1,
            rect[0], rect[1], rect[2], rect[3],
            item.get("script", "Unknown"),
            f"{item.get('script_confidence', 0.0):.2f}",
            item.get("language", "Unknown"),
            item.get("lang_code", ""),
            f"{item.get('confidence', 0.0):.2f}",
            f"{item.get('ocr_confidence', 0.0):.2f}",
            item.get("text", ""),
            item.get("translation", ""),
            item.get("reply", "")
        ])

    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8-sig')) # UTF-8 with BOM for Excel Unicode support
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"language_detection_report_{int(time.time())}.csv"
    )


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" 🚀 OPENCV & AI LANGUAGE DETECTION SYSTEM WEB APPLICATION")
    print(" Server running at: http://localhost:5000")
    print("=" * 70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
