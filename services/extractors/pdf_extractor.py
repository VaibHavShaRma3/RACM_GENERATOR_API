import asyncio
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

    # Process scanned pages through Gemini Vision (parallel)
    if scanned_pages:
        logger.info(f"Starting parallel OCR for {len(scanned_pages)} scanned pages via Gemini Vision...")
        ocr_start = time.time()

        async def _ocr_page(page_num: int, image_bytes: bytes) -> tuple[int, str]:
            try:
                text = await vision_extract(image_bytes, "image/png")
                logger.info(f"OCR page {page_num}: {len(text)} chars")
                return (page_num, f"--- Page {page_num} (scanned/OCR) ---\n{text}")
            except Exception as e:
                logger.warning(f"Vision extraction failed for page {page_num}: {e}")
                return (page_num, f"--- Page {page_num} (scanned - extraction failed) ---")

        ocr_results = await asyncio.gather(
            *[_ocr_page(pn, ib) for pn, ib in scanned_pages]
        )
        # Sort by page number to maintain order, then append
        for _, page_text in sorted(ocr_results, key=lambda x: x[0]):
            text_pages.append(page_text)

        ocr_time = time.time() - ocr_start
        logger.info(f"Parallel OCR complete: {len(scanned_pages)} pages in {ocr_time:.1f}s")

    total_time = time.time() - t0
    total_chars = sum(len(p) for p in text_pages)
    logger.info(f"PDF extraction complete: {total_chars} total chars from {total_pages} pages in {total_time:.1f}s")

    return "\n\n".join(text_pages)
