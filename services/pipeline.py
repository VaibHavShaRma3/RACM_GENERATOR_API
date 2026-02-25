import asyncio
import json
import logging
import re

from database import update_job
from config import settings
from services.extractors.pdf_extractor import extract_pdf
from services.extractors.excel_extractor import extract_excel, extract_csv
from services.extractors.image_extractor import extract_image
from services.chunker import semantic_chunk
from services.gemini import analyze_chunk, consolidation_pass
from services.deduplicator import deduplicate_racm

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp", "webp"}


def extract_document_metadata(raw_text: str, file_name: str) -> dict:
    """Extract lightweight metadata from the document for chunk context headers."""
    preview = raw_text[:3000]

    # Extract likely title from first line or filename
    first_line = preview.split("\n")[0].strip()
    title = first_line if len(first_line) > 5 else file_name

    # Find role-like patterns (e.g., "AM – Bid & Auctions", "TPA Manager")
    role_pattern = r'(?:AM|TPA|Manager|Director|Officer|Authority|Executive|Supervisor|Analyst|Lead|Head)[\s\-–—]*[A-Za-z&\s]*'
    roles = list(set(re.findall(role_pattern, raw_text)))[:10]

    return {"title": title, "roles": roles}


async def _update_progress(job_id: str, phase: str, pct: int, msg: str):
    await update_job(job_id, phase=phase, progress_pct=pct, progress_msg=msg)


async def extract_content(file_path: str, file_type: str) -> str:
    """Route file to the appropriate extractor."""
    ft = file_type.lower()

    if ft == "pdf":
        return await extract_pdf(file_path)
    elif ft in ("xlsx", "xls"):
        return await extract_excel(file_path)
    elif ft == "csv":
        return await extract_csv(file_path)
    elif ft in IMAGE_EXTENSIONS:
        return await extract_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


async def analyze_chunks_batched(
    job_id: str,
    chunks: list[str],
    user_instructions: str | None,
) -> tuple[list[dict], list[dict]]:
    """Process chunks in batches with concurrency control and progress updates."""
    all_detailed: list[dict] = []
    all_summary: list[dict] = []
    batch_size = settings.batch_concurrency
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        completed = min(i + len(batch), total)
        pct = 15 + int((completed / total) * 65)
        await _update_progress(
            job_id, "analyzing", pct,
            f"Analyzing batch {batch_num}/{total_batches} ({completed}/{total} chunks)...",
        )

        results = await asyncio.gather(
            *[analyze_chunk(chunk, user_instructions) for chunk in batch],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Chunk analysis failed: {result}")
                continue
            if isinstance(result, dict):
                all_detailed.extend(result.get("detailed_entries", []))
                all_summary.extend(result.get("summary_entries", []))

        await update_job(job_id, completed_chunks=completed)

    return all_detailed, all_summary


async def process_job(job_id: str, file_path: str, file_type: str, prompt: str | None, file_name: str = ""):
    """Full RACM processing pipeline: extract -> chunk -> analyze -> consolidate -> dedupe."""

    # Phase 1: Extract content
    await _update_progress(job_id, "extracting", 5, "Extracting content from file...")
    raw_text = await extract_content(file_path, file_type)

    if len(raw_text.strip()) < 50:
        raise ValueError("Extracted content too short — file may be empty or unreadable.")

    # Extract document metadata for chunk context headers
    metadata = extract_document_metadata(raw_text, file_name or file_path)

    # Phase 2: Chunk
    await _update_progress(job_id, "chunking", 12, "Splitting into semantic chunks...")
    chunks = semantic_chunk(
        raw_text,
        max_size=settings.chunk_size,
        overlap_sentences=settings.chunk_overlap_sentences,
    )

    # Prepend context header with chunk numbering to each chunk
    roles_str = ', '.join(metadata['roles']) if metadata['roles'] else 'Not identified'
    for i in range(len(chunks)):
        header = (
            f"DOCUMENT: {metadata['title']}\n"
            f"CHUNK: {i + 1}/{len(chunks)}\n"
            f"KEY ROLES: {roles_str}\n"
            f"---"
        )
        chunks[i] = f"{header}\n{chunks[i]}"

    await update_job(job_id, total_chunks=len(chunks))

    # Phase 3: Analyze each chunk via Gemini
    await _update_progress(job_id, "analyzing", 15, f"Analyzing {len(chunks)} segments...")
    all_detailed, all_summary = await analyze_chunks_batched(job_id, chunks, prompt)

    if not all_detailed:
        raise ValueError("No RACM entries were generated. The file may not contain audit-relevant content.")

    # Phase 4: Consolidation pass
    await _update_progress(job_id, "consolidating", 82, "Running consolidation synthesis...")
    consolidated = await consolidation_pass(all_detailed, all_summary, prompt)

    # Phase 5: Deduplication
    await _update_progress(job_id, "deduplicating", 92, "Deduplicating entries...")
    result = deduplicate_racm(consolidated)

    # Phase 6: Store result
    await update_job(
        job_id,
        status="completed",
        phase="completed",
        progress_pct=100,
        progress_msg="Analysis complete",
        result_json=json.dumps(result, ensure_ascii=False),
    )

    return result
