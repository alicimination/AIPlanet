"""Image OCR utilities with PaddleOCR primary and Tesseract fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import tempfile

from PIL import Image


@dataclass
class OCRResult:
    text: str
    confidence: float
    lines: List[str]
    engine: str
    error: Optional[str] = None


def extract_text_from_image(uploaded_file) -> OCRResult:
    """Extract text from image bytes.

    Returns OCRResult containing text, confidence and line breakdown.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image = Image.open(uploaded_file)
            image.save(tmp.name)
            image_path = tmp.name
    except Exception as exc:  # pragma: no cover - IO edge case
        return OCRResult(text="", confidence=0.0, lines=[], engine="none", error=str(exc))

    # Try PaddleOCR first.
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(image_path, cls=True)
        lines = []
        conf = []
        for block in result or []:
            for item in block:
                txt = item[1][0]
                score = float(item[1][1])
                lines.append(txt)
                conf.append(score)
        text = "\n".join(lines).strip()
        confidence = sum(conf) / len(conf) if conf else 0.0
        return OCRResult(text=text, confidence=confidence, lines=lines, engine="paddleocr")
    except Exception:
        pass

    # Fallback to pytesseract.
    try:
        import pytesseract

        text = pytesseract.image_to_string(Image.open(image_path))
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        confidence = 0.65 if text.strip() else 0.0
        return OCRResult(text=text.strip(), confidence=confidence, lines=lines, engine="tesseract")
    except Exception as exc:  # pragma: no cover - dependency edge case
        return OCRResult(text="", confidence=0.0, lines=[], engine="none", error=str(exc))
