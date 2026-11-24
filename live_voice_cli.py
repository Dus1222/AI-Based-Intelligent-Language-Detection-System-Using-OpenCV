"""
Live Desktop Microphone Voice Assistant for Language Detection & Auto-Reply.
Captures live audio from the PC microphone, transcribes speech,
identifies the language (Hindi, English, French, Spanish, German, etc.),
generates an automatic response in the same language, and speaks it aloud.
"""

import sys
import os
import subprocess

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import speech_recognition as sr
from core.language_detector import MultilingualLanguageDetector
from core.reply_generator import MultilingualReplyGenerator


def speak_windows_tts(text: str):
    """Speaks text aloud using Windows PowerShell SAPI voice synthesis."""
    try:
        # Escape quotes for PowerShell
        safe_text = text.replace('"', '`"').replace("'", "''")
        ps_cmd = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = 0; "
            f"$synth.Speak('{safe_text}')"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=False)
    except Exception as e:
        print(f"[WARN] TTS playback failed: {e}")


def main():
    print("=" * 65)
    print(" 🎙️ LIVE MICROPHONE MULTILINGUAL ASSISTANT")
    print(" Speak in Hindi, English, French, Spanish, German, etc.")
    print("=" * 65)

    lang_detector = MultilingualLanguageDetector()
    reply_gen = MultilingualReplyGenerator()
    recognizer = sr.Recognizer()

    # Adjust for ambient noise
    with sr.Microphone() as source:
        print("\n[INFO] Calibrating microphone for background noise (1 sec)...")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        print("[READY] Microphone ready! Speak now (Press Ctrl+C to exit)...")

    # Language listening presets
    lang_codes = ["hi-IN", "en-US", "fr-FR", "es-ES", "de-DE"]

    while True:
        try:
            with sr.Microphone() as source:
                print("\n" + "-" * 55)
                print(" 🔴 LISTENING... (Say something in any language)")
                audio = recognizer.listen(source, timeout=8.0, phrase_time_limit=10.0)

            print(" ⏳ Analyzing voice speech...")

            # Transcribe with Google Speech Recognition (multilingual fallback)
            recognized_text = None
            detected_locale = None

            # Try recognizing across languages
            for locale in ["hi-IN", "en-US", "fr-FR", "es-ES"]:
                try:
                    text = recognizer.recognize_google(audio, language=locale)
                    if text and len(text.strip()) > 0:
                        recognized_text = text.strip()
                        detected_locale = locale
                        break
                except (sr.UnknownValueError, sr.RequestError):
                    continue

            if not recognized_text:
                print(" [?] Could not recognize clear speech. Please speak again.")
                continue

            # Step 1: Detect Language
            lang_info = lang_detector.identify_language(recognized_text)

            # Step 2: Generate Reply in the Same Language
            reply_info = reply_gen.generate_reply(
                recognized_text,
                lang_code=lang_info["lang_code"],
                lang_name=lang_info["language_name"]
            )

            # Output results
            print(f"\n 🗣️ You Said: \"{recognized_text}\"")
            print(f" 🌐 Detected Language: {lang_info['flag']} {lang_info['language_name']} ({int(lang_info['confidence']*100)}% conf)")
            print(f" 🤖 System Auto-Reply: \"{reply_info['reply_text']}\"")
            print(f" 🔊 Speaking response...")

            # Speak response aloud
            speak_windows_tts(reply_info['reply_text'])

        except sr.WaitTimeoutError:
            print(" [TIMEOUT] No speech heard. Listening again...")
        except KeyboardInterrupt:
            print("\n[EXIT] Exiting live voice assistant. Goodbye!")
            break
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
