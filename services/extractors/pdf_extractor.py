import logging
import time

import fitz  # PyMuPDF

from services.gemini import vision_extract

logger = logging.getLogger(__name__)

# Pages with fewer than this many chars of extracted text
# are treated as scanned/image pages and sent to Gemini Vision.
SCANNED_PAGE_THRESHOLD = 50


async def extract_pdf(file_path: str) -> str:
    t0 = time.time()
    doc = fitz.open(file_path)
    total_pages = len(doc)
    text_pages: list[str] = []
    scanned_pages: list[tuple[int, bytes]] = []

    logger.info(f"PDF extraction: {total_pages} pages in {file_path.split('/')[-1]}")

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if len(text) > SCANNED_PAGE_THRESHOLD:
            text_pages.append(f"--- Page {page_num + 1} ---\n{text}")
            logger.debug(f"Page {page_num + 1}/{total_pages}: {len(text)} chars (text)")
        else:
            # Render page as image for Gemini Vision
            pix = page.get_pixmap(dpi=200)
            image_bytes = pix.tobytes("png")
            scanned_pages.append((page_num + 1, image_bytes))
            logger.info(f"Page {page_num + 1}/{total_pages}: scanned/image ({len(text)} chars text), queued for OCR")

    doc.close()

    text_time = time.time() - t0
    logger.info(
        f"PDF text extraction: {len(text_pages)} text pages + {len(scanned_pages)} scanned pages "
        f"in {text_time:.1f}s"
    )

    # Process scanned pages through Gemini Vision
    if scanned_pages:
        logger.info(f"Starting OCR for {len(scanned_pages)} scanned pages via Gemini Vision...")

    for i, (page_num, image_bytes) in enumerate(scanned_pages):
        try:
            ocr_start = time.time()
            vision_text = await vision_extract(image_bytes, "image/png")
            ocr_time = time.time() - ocr_start
            text_pages.append(f"--- Page {page_num} (scanned/OCR) ---\n{vision_text}")
            logger.info(f"OCR page {page_num} ({i+1}/{len(scanned_pages)}): {len(vision_text)} chars in {ocr_time:.1f}s")
        except Exception as e:
            logger.warning(f"Vision extraction failed for page {page_num}: {e}")
            text_pages.append(f"--- Page {page_num} (scanned - extraction failed) ---")

    total_time = time.time() - t0
    total_chars = sum(len(p) for p in text_pages)
    logger.info(f"PDF extraction complete: {total_chars} total chars from {total_pages} pages in {total_time:.1f}s")

    return "\n\n".join(text_pages)
