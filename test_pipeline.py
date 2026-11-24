"""
Automated Pipeline Verification Suite.
Tests OpenCV image preprocessing, language detection, intent matching,
and multilingual auto-reply generation.
"""

import os
import cv2
import numpy as np

from core.image_processor import OpenCVImageProcessor
from core.language_detector import MultilingualLanguageDetector
from core.reply_generator import MultilingualReplyGenerator
from core.ocr_engine import MultilingualOCREngine


def test_language_detector_direct():
    print("\n--- TEST 1: Direct Language Detection & Metadata ---")
    detector = MultilingualLanguageDetector()

    test_cases = [
        ("Hello, how are you?", "en", "English"),
        ("नमस्ते, आप कैसे हैं?", "hi", "Hindi"),
        ("Bonjour, comment allez-vous ?", "fr", "French"),
        ("¡Hola! ¿Cómo estás?", "es", "Spanish"),
        ("Guten Tag, wie geht es Ihnen?", "de", "German"),
    ]

    for text, expected_code, expected_name in test_cases:
        res = detector.identify_language(text)
        print(f"Input: '{text}' -> Detected: {res['language_name']} ({res['lang_code']}), Conf: {res['confidence']:.2f}, Method: {res['method']}")
        assert res["lang_code"] == expected_code, f"Expected {expected_code}, got {res['lang_code']} for '{text}'"

    print(">> [PASS] Test 1: Language Detection passed for all sample phrases!")


def test_reply_generator():
    print("\n--- TEST 2: Multilingual Auto-Reply Generation ---")
    generator = MultilingualReplyGenerator()

    cases = [
        ("Hello, how are you?", "en", "I am fine, thank you! How can I help you?"),
        ("नमस्ते, आप कैसे हैं?", "hi", "मैं ठीक हूँ, धन्यवाद! मैं आपकी क्या सहायता कर सकता हूँ?"),
        ("Bonjour, comment allez-vous ?", "fr", "Je vais bien, merci ! Comment puis-je vous aider ?"),
        ("¡Hola! ¿Cómo estás?", "es", "¡Estoy bien, muchas gracias! ¿Cómo puedo ayudarte hoy?"),
    ]

    for text, lang_code, expected_reply in cases:
        reply_info = generator.generate_reply(text, lang_code=lang_code)
        print(f"\nLanguage: {lang_code}\nInput: '{text}'\nIntent: {reply_info['intent']}\nSystem Reply: '{reply_info['reply_text']}'")
        assert reply_info["reply_text"] == expected_reply, f"Expected '{expected_reply}', got '{reply_info['reply_text']}'"

    print(">> [PASS] Test 2: Auto-reply generation matches Section 6 specification exactly!")


def test_opencv_preprocessing():
    print("\n--- TEST 3: OpenCV Image Preprocessing Pipeline ---")
    processor = OpenCVImageProcessor()

    # Synthetic test frame
    dummy_img = np.full((300, 500, 3), 200, dtype=np.uint8)
    cv2.putText(dummy_img, "Test OpenCV Preprocess", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)

    processed, stages = processor.preprocess_pipeline(dummy_img, denoise=True, clahe=True, binarize=False)
    assert processed is not None, "Processed image should not be None"
    assert "grayscale" in stages, "Stages should include grayscale"
    assert "denoised" in stages, "Stages should include denoised"
    assert "enhanced" in stages, "Stages should include enhanced"

    # Test annotation drawing
    detections = [{
        "bbox": [(50, 100), (450, 200)],
        "text": "Hello world",
        "language": "English",
        "lang_code": "en",
        "confidence": 0.95,
        "reply": "I am fine, thank you!"
    }]
    annotated = processor.draw_annotations(dummy_img, detections)
    assert annotated.shape == dummy_img.shape, "Annotated frame dimensions must match original"
    print(">> [PASS] Test 3: OpenCV Preprocessing and Annotation Rendering passed!")


def test_ocr_end_to_end_on_samples():
    print("\n--- TEST 4: End-to-End OCR & Auto-Reply on Real Samples ---")
    engine = MultilingualOCREngine(default_profile="latin", use_gpu=False)

    samples_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    english_sample = os.path.join(samples_dir, "english_sample.png")

    if os.path.exists(english_sample):
        bgr = cv2.imread(english_sample)
        res = engine.process_image(bgr, profile="latin", conf_threshold=0.20)
        print(f"\nEnglish Sample Extracted Text:\n  '{res['full_extracted_text']}'")
        print(f"Detected Primary Language: {res['primary_language']['language_name']}")
        print(f"System Reply: '{res['system_reply']['reply_text']}'")
        assert len(res["detections"]) > 0, "Detections should not be empty"
        assert res["primary_language"]["lang_code"] == "en", "Should detect English"

    french_sample = os.path.join(samples_dir, "french_sample.png")
    if os.path.exists(french_sample):
        bgr = cv2.imread(french_sample)
        res = engine.process_image(bgr, profile="latin", conf_threshold=0.20)
        print(f"\nFrench Sample Extracted Text:\n  '{res['full_extracted_text']}'")
        print(f"Detected Primary Language: {res['primary_language']['language_name']}")
        print(f"System Reply: '{res['system_reply']['reply_text']}'")
        assert len(res["detections"]) > 0, "Detections should not be empty"
        assert res["primary_language"]["lang_code"] == "fr", "Should detect French"

    print(">> [PASS] Test 4: End-to-end OCR and Auto-Reply pipeline passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING MULTILINGUAL OPENCV LANGUAGE DETECTION TEST SUITE")
    print("=" * 70)
    test_language_detector_direct()
    test_reply_generator()
    test_opencv_preprocessing()
    test_ocr_end_to_end_on_samples()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
    print("=" * 70)
