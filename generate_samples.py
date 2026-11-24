"""
High-Resolution Multilingual Sample Generator.
Creates synthetic test image cards for English, Hindi, French, Spanish, and German
using Windows Unicode typography (Nirmala / Arial).
"""

import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_images")

# System fonts on Windows
FONT_DEVANAGARI = r"C:\Windows\Fonts\Nirmala.ttc"
FONT_LATIN = r"C:\Windows\Fonts\arial.ttf"

SAMPLES = [
    {
        "filename": "english_sample.png",
        "lang_name": "English",
        "font_path": FONT_LATIN,
        "title": "Document: Customer Support Query (English)",
        "lines": [
            "Hello, how are you?",
            "Welcome to the OpenCV Language Detection System.",
            "Can you help me process this document?"
        ],
        "bg_color": (248, 250, 252),
        "header_color": (15, 23, 42),
        "text_color": (30, 41, 59),
        "accent": (14, 165, 233)
    },
    {
        "filename": "hindi_sample.png",
        "lang_name": "Hindi (हिंदी)",
        "font_path": FONT_DEVANAGARI,
        "title": "दस्तावेज़: ग्राहक सहायता पूछताछ (हिन्दी)",
        "lines": [
            "नमस्ते, आप कैसे हैं?",
            "भाषा पहचान और स्वचालित उत्तर प्रणाली।",
            "कृत्रिम बुद्धिमत्ता और कंप्यूटर विज़न का उपयोग।"
        ],
        "bg_color": (255, 251, 245),
        "header_color": (67, 20, 7),
        "text_color": (67, 20, 7),
        "accent": (234, 88, 12)
    },
    {
        "filename": "french_sample.png",
        "lang_name": "French (Français)",
        "font_path": FONT_LATIN,
        "title": "Document: Demande d'assistance (Français)",
        "lines": [
            "Bonjour, comment allez-vous ?",
            "Système de détection de langue avec OpenCV et OCR.",
            "Pouvez-vous m'aider avec cette traduction s'il vous plaît ?"
        ],
        "bg_color": (245, 247, 255),
        "header_color": (30, 27, 75),
        "text_color": (30, 27, 75),
        "accent": (99, 102, 241)
    },
    {
        "filename": "spanish_sample.png",
        "lang_name": "Spanish (Español)",
        "font_path": FONT_LATIN,
        "title": "Documento: Consulta de usuario (Español)",
        "lines": [
            "¡Hola! ¿Cómo estás?",
            "Sistema inteligente de detección de idioma mediante OpenCV.",
            "Me gustaría recibir asistencia en español."
        ],
        "bg_color": (255, 247, 237),
        "header_color": (67, 20, 7),
        "text_color": (67, 20, 7),
        "accent": (234, 88, 12)
    },
    {
        "filename": "german_sample.png",
        "lang_name": "German (Deutsch)",
        "font_path": FONT_LATIN,
        "title": "Dokument: Kundenanfrage (Deutsch)",
        "lines": [
            "Guten Tag, wie geht es Ihnen?",
            "Automatische Spracherkennung mit OpenCV und OCR.",
            "Vielen Dank für Ihre freundliche Unterstützung."
        ],
        "bg_color": (254, 252, 232),
        "header_color": (66, 32, 6),
        "text_color": (66, 32, 6),
        "accent": (202, 138, 4)
    }
]


def generate_all_samples():
    """Generates all multilingual sample images."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for item in SAMPLES:
        font_file = item["font_path"] if os.path.exists(item["font_path"]) else FONT_LATIN
        try:
            font_title = ImageFont.truetype(font_file, 20)
            font_body = ImageFont.truetype(font_file, 28)
            font_sub = ImageFont.truetype(font_file, 16)
        except Exception:
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()
            font_sub = ImageFont.load_default()

        width, height = 900, 380
        img = Image.new("RGB", (width, height), color=item["bg_color"])
        draw = ImageDraw.Draw(img)

        # Header banner
        draw.rectangle([0, 0, width, 60], fill=item["header_color"])
        draw.rectangle([0, 58, width, 62], fill=item["accent"])
        draw.text((30, 18), item["title"], font=font_title, fill=(255, 255, 255))

        # Text body
        y = 100
        for idx, line in enumerate(item["lines"]):
            draw.text((45, y), line, font=font_body, fill=item["text_color"])
            y += 68

        # Footer badge
        footer_text = f"Sample Image for OpenCV Language Detection Engine | Expected Language: {item['lang_name']}"
        draw.text((45, 340), footer_text, font=font_sub, fill=(100, 116, 139))

        # Elegant border
        draw.rectangle([0, 0, width - 1, height - 1], outline=(203, 213, 225), width=2)

        out_path = os.path.join(OUTPUT_DIR, item["filename"])
        img.save(out_path, "PNG")
        print(f"[OK] Generated: {out_path}")


if __name__ == "__main__":
    generate_all_samples()
