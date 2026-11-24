"""
Live Desktop Webcam Runner for OpenCV Language Detection & Auto-Reply System.
Provides interactive camera capture, real-time visual HUD, and instant language reply.
"""

import os
import sys
import time
import cv2
import numpy as np

from core.ocr_engine import MultilingualOCREngine
from core.image_processor import OpenCVImageProcessor


def main():
    print("=" * 65)
    print(" 📹 OPENCV LIVE WEBCAM LANGUAGE DETECTION & AUTO-REPLY")
    print("=" * 65)
    print(" Initializing OCR Models and Camera Stream...")
    print(" Controls:")
    print("   [SPACE] : Capture & Analyze Current Frame")
    print("   [A]     : Toggle Auto-Scan (every ~2 seconds)")
    print("   [P]     : Switch Profile (Latin/European <-> Hindi/Devanagari)")
    print("   [S]     : Save Screenshot")
    print("   [Q]     : Quit Application")
    print("=" * 65 + "\n")

    current_profile = "latin"
    engine = MultilingualOCREngine(default_profile=current_profile, use_gpu=False)
    processor = OpenCVImageProcessor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Ensure a camera is attached and accessible.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    auto_scan = False
    last_scan_time = 0.0
    scan_interval = 2.5

    last_result = None
    fps_history = []
    status_msg = "Ready. Point camera at text and press SPACE to analyze."

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            break

        now = time.time()

        # Handle Auto-Scan
        if auto_scan and (now - last_scan_time > scan_interval):
            status_msg = f"Scanning frame ({current_profile})..."
            last_result = engine.process_image(frame, profile=current_profile, conf_threshold=0.25)
            last_scan_time = now
            status_msg = f"Detected: {last_result['primary_language']['language_name']}"

        # Display output frame
        if last_result and "annotated_image" in last_result:
            display_frame = last_result["annotated_image"].copy()
        else:
            display_frame = frame.copy()

        # Calculate FPS
        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 30.0
        fps_history.append(fps)
        if len(fps_history) > 15:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Draw Top HUD Banner
        h, w = display_frame.shape[:2]
        cv2.rectangle(display_frame, (0, 0), (w, 55), (15, 23, 42), -1)
        profile_label = "Latin (EN/FR/ES/DE)" if current_profile == "latin" else "Hindi (HI/EN)"
        mode_str = "AUTO-SCAN: ON" if auto_scan else "MANUAL: PRESS SPACE"

        hud_line1 = f"OpenCV LangDetect | Profile: {profile_label} | {mode_str} | FPS: {avg_fps:.1f}"
        cv2.putText(display_frame, hud_line1, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display_frame, status_msg, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)

        # Draw Bottom Result Dialog if available
        if last_result and last_result.get("full_extracted_text"):
            lang_info = last_result["primary_language"]
            reply_info = last_result["system_reply"]
            card_h = 100
            cv2.rectangle(display_frame, (10, h - card_h - 10), (w - 10, h - 10), (24, 24, 27), -1)
            cv2.rectangle(display_frame, (10, h - card_h - 10), (w - 10, h - 10), (99, 102, 241), 2)

            t_line = f"Detected Text: {last_result['full_extracted_text'][:80]}"
            l_line = f"Language: {lang_info['language_name']} (Conf: {int(lang_info['confidence']*100)}%)"
            r_line = f"System Reply: {reply_info['reply_text'][:90]}"

            cv2.putText(display_frame, t_line, (25, h - card_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1)
            cv2.putText(display_frame, l_line, (25, h - card_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2)
            cv2.putText(display_frame, r_line, (25, h - card_h + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (74, 222, 128), 2)

        cv2.imshow("OpenCV Multilingual Language Detection & Auto-Reply", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # ESC or q
            break
        elif key == 32:  # Spacebar
            status_msg = f"Analyzing frame with {current_profile} profile..."
            print(f"[ACTION] Processing snapshot with {current_profile} OCR profile...")
            last_result = engine.process_image(frame, profile=current_profile, conf_threshold=0.20)
            lang = last_result["primary_language"]["language_name"]
            reply = last_result["system_reply"]["reply_text"]
            status_msg = f"Detected: {lang}"
            print(f" -> Text: '{last_result['full_extracted_text']}'")
            print(f" -> Language: {lang}")
            print(f" -> Reply: '{reply}'\n")
        elif key == ord('a') or key == ord('A'):
            auto_scan = not auto_scan
            status_msg = f"Auto-Scan turned {'ON' if auto_scan else 'OFF'}"
            print(f"[ACTION] {status_msg}")
        elif key == ord('p') or key == ord('P'):
            current_profile = "hindi" if current_profile == "latin" else "latin"
            status_msg = f"Switched profile to: {current_profile.upper()}"
            print(f"[ACTION] {status_msg}")
        elif key == ord('s') or key == ord('S'):
            out_file = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(out_file, display_frame)
            status_msg = f"Saved screenshot to {out_file}"
            print(f"[ACTION] {status_msg}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
