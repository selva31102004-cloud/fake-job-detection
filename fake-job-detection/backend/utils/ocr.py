"""
OCR Module - Extract text from job posting images
Uses Tesseract OCR via pytesseract with PIL/Pillow for image processing
Falls back to easyocr if tesseract is unavailable
"""

import io
import logging
import re
from typing import Optional


logger = logging.getLogger(__name__)


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image bytes using OCR.
    Tries pytesseract first, then easyocr, then basic extraction.
    """
    text = ""

    # Try pytesseract (Tesseract OCR)
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        import pytesseract
      

        img = Image.open(io.BytesIO(image_bytes))

        # Preprocess for better OCR accuracy
        img = preprocess_image(img)

        # OCR config: prefer document text
        custom_config = "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%&*()-_+=:;/\\\""
        text = pytesseract.image_to_string(img, config=custom_config)
        logger.info(f"Tesseract OCR extracted {len(text)} characters")
        return clean_ocr_text(text)

    except ImportError:
        logger.warning("pytesseract not available, trying easyocr")
    except Exception as e:
        logger.warning(f"pytesseract failed: {e}, trying easyocr")

    # Try easyocr
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img)
        results = reader.readtext(img_array, detail=0)
        text = " ".join(results)
        logger.info(f"EasyOCR extracted {len(text)} characters")
        return clean_ocr_text(text)

    except ImportError:
        logger.warning("easyocr not available")
    except Exception as e:
        logger.warning(f"easyocr failed: {e}")

    # Fallback: Try to read image metadata or return placeholder
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        # Return image info as context
        text = f"[Image: {img.width}x{img.height}px {img.format}] Job posting image uploaded. OCR libraries not available - please install pytesseract or easyocr."
        logger.warning("Using fallback OCR response")
        return text
    except Exception as e:
        logger.error(f"Complete OCR failure: {e}")
        return "Job posting image uploaded. Unable to extract text - OCR service unavailable."


def preprocess_image(img):
    """Enhance image for better OCR accuracy"""
    try:
        from PIL import Image, ImageEnhance, ImageFilter

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize if too small
        if img.width < 300:
            scale = 300 / img.width
            img = img.resize((int(img.width * scale), int(img.height * scale)))

        # Convert to grayscale for OCR
        img = img.convert('L')

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Sharpen
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)

        return img
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
        return img


def clean_ocr_text(text: str) -> str:
    """Clean and normalize OCR output"""
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove OCR artifacts
    text = re.sub(r'[|\\]{3,}', '', text)

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    return text.strip()
