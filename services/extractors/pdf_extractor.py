import logging

import fitz  # PyMuPDF

from services.gemini import vision_extract

logger = logging.getLogger(__name__)

# Pages with fewer than this many chars of extracted text
# are treated as scanned/image pages and sent to Gemini Vision.
SCANNED_PAGE_THRESHOLD = 50


async def extract_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text_pages: list[str] = []
    scanned_pages: list[tuple[int, bytes]] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if len(text) > SCANNED_PAGE_THRESHOLD:
            text_pages.append(f"--- Page {page_num + 1} ---\n{text}")
        else:
            # Render page as image for Gemini Vision
            pix = page.get_pixmap(dpi=200)
            image_bytes = pix.tobytes("png")
            scanned_pages.append((page_num + 1, image_bytes))

    doc.close()

    # Process scanned pages through Gemini Vision
    for page_num, image_bytes in scanned_pages:
        try:
            vision_text = await vision_extract(image_bytes, "image/png")
            text_pages.append(f"--- Page {page_num} (scanned/OCR) ---\n{vision_text}")
        except Exception as e:
            logger.warning(f"Vision extraction failed for page {page_num}: {e}")
            text_pages.append(f"--- Page {page_num} (scanned - extraction failed) ---")

    return "\n\n".join(text_pages)
