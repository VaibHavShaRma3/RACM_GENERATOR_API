import asyncio
import json
import logging
import re
import time

from database import update_job
from config import settings
from services.extractors.pdf_extractor import extract_pdf
from services.extractors.excel_extractor import extract_excel, extract_csv
from services.extractors.image_extractor import extract_image
from services.chunker import semantic_chunk
from services.gemini import analyze_chunk, consolidation_pass, generate_racm_summary
from services.deduplicator import deduplicate_racm

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp", "webp"}


def _fmt_size(chars: int) -> str:
    """Format character count as human-readable size."""
    if chars < 1024:
        return f"{chars} chars"
    return f"{chars / 1024:.1f}KB"


def _fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


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


async def _update_progress(job_id: str, phase: str, pct: int, msg: str, detail: str = ""):
    fields = dict(phase=phase, progress_pct=pct, progress_msg=msg)
    if detail:
        fields["detail_msg"] = detail
    await update_job(job_id, **fields)


async def _update_detail(job_id: str, detail: str):
    """Update only the detail_msg field (lightweight, frequent updates)."""
    await update_job(job_id, detail_msg=detail)


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


def _estimate_chunk_time(chunk_chars: int) -> float:
    """Estimate Gemini analysis time based on content size.

    Observed: 14,749 chars → 175s (Gemini 2.5 Flash with thinking).
    Rate: ~12s per 1000 chars, minimum 45s per chunk (API overhead).
    """
    return max(45, chunk_chars * 12 / 1000)


async def analyze_chunks_batched(
    job_id: str,
    chunks: list[str],
    user_instructions: str | None,
    needs_consolidation: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Process chunks in batches with concurrency control, progress updates, and ETA."""
    all_detailed: list[dict] = []
    all_summary: list[dict] = []
    batch_size = settings.batch_concurrency
    total = len(chunks)
    analysis_start = time.time()

    # ETA constants
    CONSOLIDATION_ESTIMATE = 60   # seconds for consolidation pass
    SUMMARY_ESTIMATE = 15         # seconds for summary generation
    OVERHEAD_BUFFER = 5           # dedup + misc overhead

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        completed = min(i + len(batch), total)
        pct = 15 + int((completed / total) * 65)

        batch_detail = f"Sending chunk {i+1}-{min(i+batch_size, total)} of {total} to Gemini ({batch_size} concurrent)..."
        await _update_progress(
            job_id, "analyzing", pct,
            f"Analyzing batch {batch_num}/{total_batches} ({completed}/{total} chunks)...",
            detail=batch_detail,
        )

        logger.info(f"[{job_id[:8]}] Analyzing batch {batch_num}/{total_batches} (chunks {i+1}-{min(i+batch_size, total)}/{total})")
        batch_start = time.time()

        results = await asyncio.gather(
            *[analyze_chunk(chunk, user_instructions) for chunk in batch],
            return_exceptions=True,
        )

        batch_elapsed = time.time() - batch_start
        batch_detailed = 0
        batch_summary = 0
        batch_failures = 0

        for idx, result in enumerate(results):
            chunk_idx = i + idx + 1
            if isinstance(result, Exception):
                batch_failures += 1
                logger.warning(f"[{job_id[:8]}] Chunk {chunk_idx}/{total} FAILED: {type(result).__name__}: {str(result)[:200]}")
                continue
            if isinstance(result, dict):
                d = result.get("detailed_entries", [])
                s = result.get("summary_entries", [])
                batch_detailed += len(d)
                batch_summary += len(s)
                all_detailed.extend(d)
                all_summary.extend(s)
                logger.info(f"[{job_id[:8]}] Chunk {chunk_idx}/{total} → {len(d)} detailed + {len(s)} summary entries")

        elapsed_total = time.time() - analysis_start

        # Recalculate ETA based on actual performance
        chunks_done = min(i + len(batch), total)
        chunks_remaining = total - chunks_done
        if chunks_done > 0 and chunks_remaining > 0:
            # Use measured average per chunk for remaining estimate
            avg_per_chunk = elapsed_total / chunks_done
            remaining_batches = (chunks_remaining + batch_size - 1) // batch_size
            eta_remaining = avg_per_chunk * remaining_batches
        else:
            eta_remaining = 0
        eta_remaining += (CONSOLIDATION_ESTIMATE if needs_consolidation else 0) + SUMMARY_ESTIMATE + OVERHEAD_BUFFER
        await update_job(job_id, eta_seconds=max(0, int(eta_remaining)))

        detail = (
            f"Batch {batch_num}/{total_batches} done in {_fmt_duration(batch_elapsed)} → "
            f"{batch_detailed} detailed + {batch_summary} summary entries"
            f"{f' ({batch_failures} failed)' if batch_failures else ''}"
            f" | Total so far: {len(all_detailed)} entries in {_fmt_duration(elapsed_total)}"
        )
        await _update_detail(job_id, detail)

        logger.info(
            f"[{job_id[:8]}] Batch {batch_num}/{total_batches} completed in {_fmt_duration(batch_elapsed)}: "
            f"+{batch_detailed} detailed, +{batch_summary} summary"
            f"{f', {batch_failures} failures' if batch_failures else ''}"
            f" (running total: {len(all_detailed)} detailed, {len(all_summary)} summary)"
        )

        await update_job(job_id, completed_chunks=completed)

    total_elapsed = time.time() - analysis_start
    logger.info(
        f"[{job_id[:8]}] Analysis complete: {len(all_detailed)} detailed + {len(all_summary)} summary entries "
        f"from {total} chunks in {_fmt_duration(total_elapsed)}"
    )

    return all_detailed, all_summary


async def process_job(job_id: str, file_path: str, file_type: str, prompt: str | None, file_name: str = ""):
    """Full RACM processing pipeline: extract -> chunk -> analyze -> consolidate -> dedupe."""
    job_start = time.time()
    short_id = job_id[:8]

    logger.info(f"[{short_id}] {'='*60}")
    logger.info(f"[{short_id}] Starting pipeline: {file_name} ({file_type})")
    logger.info(f"[{short_id}] {'='*60}")

    # ── Phase 1: Extract content ────────────────────────────────
    phase_start = time.time()
    await _update_progress(
        job_id, "extracting", 3,
        f"Extracting content from {file_type.upper()} file...",
        detail=f"Reading {file_name}..."
    )
    logger.info(f"[{short_id}] Phase 1/5: EXTRACTING — {file_name} ({file_type})")

    raw_text = await extract_content(file_path, file_type)
    extract_time = time.time() - phase_start

    if len(raw_text.strip()) < 50:
        raise ValueError("Extracted content too short — file may be empty or unreadable.")

    logger.info(f"[{short_id}] Extraction done in {_fmt_duration(extract_time)}: {_fmt_size(len(raw_text))}")
    await _update_detail(job_id, f"Extracted {_fmt_size(len(raw_text))} in {_fmt_duration(extract_time)}")

    # Extract document metadata for chunk context headers
    metadata = extract_document_metadata(raw_text, file_name or file_path)
    logger.info(f"[{short_id}] Document title: {metadata['title'][:80]}")
    if metadata['roles']:
        logger.info(f"[{short_id}] Detected roles: {', '.join(metadata['roles'][:5])}")

    # ── Phase 2: Chunk ──────────────────────────────────────────
    phase_start = time.time()
    await _update_progress(
        job_id, "chunking", 10,
        "Splitting document into semantic chunks...",
        detail=f"Chunking {_fmt_size(len(raw_text))} with max {settings.chunk_size // 1024}KB per chunk..."
    )
    logger.info(f"[{short_id}] Phase 2/5: CHUNKING — {_fmt_size(len(raw_text))}, max_size={settings.chunk_size}")

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

    chunk_time = time.time() - phase_start
    chunk_sizes = [len(c) for c in chunks]
    avg_size = sum(chunk_sizes) // len(chunk_sizes) if chunk_sizes else 0
    min_size = min(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0

    logger.info(
        f"[{short_id}] Chunking done in {_fmt_duration(chunk_time)}: "
        f"{len(chunks)} chunks (avg {_fmt_size(avg_size)}, min {_fmt_size(min_size)}, max {_fmt_size(max_size)})"
    )
    await _update_progress(
        job_id, "chunking", 12,
        f"Created {len(chunks)} chunks",
        detail=f"Split into {len(chunks)} chunks (avg {_fmt_size(avg_size)}) in {_fmt_duration(chunk_time)}"
    )
    await update_job(job_id, total_chunks=len(chunks))

    # Calculate initial ETA estimate based on actual content size
    CONSOLIDATION_ESTIMATE = 60
    SUMMARY_ESTIMATE = 15
    OVERHEAD_BUFFER = 5
    needs_consolidation = len(chunks) > 2

    # Estimate analysis time per batch: use the largest chunk in each batch
    total_analysis_time = 0
    for b_start in range(0, len(chunks), settings.batch_concurrency):
        batch_chunks = chunks[b_start : b_start + settings.batch_concurrency]
        # Batch time = time for the slowest (largest) chunk in the batch
        batch_time = max(_estimate_chunk_time(len(c)) for c in batch_chunks)
        total_analysis_time += batch_time

    initial_eta = int(
        total_analysis_time
        + (CONSOLIDATION_ESTIMATE if needs_consolidation else 0)
        + SUMMARY_ESTIMATE + OVERHEAD_BUFFER
    )
    await update_job(job_id, eta_seconds=initial_eta)
    logger.info(f"[{short_id}] Initial ETA: {_fmt_duration(initial_eta)} (chunks={len(chunks)}, total_chars={sum(chunk_sizes)}, consolidation={'yes' if needs_consolidation else 'skip'})")

    # ── Phase 3: Analyze each chunk via Gemini ──────────────────
    phase_start = time.time()
    total_batches = (len(chunks) + settings.batch_concurrency - 1) // settings.batch_concurrency
    await _update_progress(
        job_id, "analyzing", 15,
        f"Analyzing {len(chunks)} chunks in {total_batches} batches...",
        detail=f"Starting Gemini analysis: {len(chunks)} chunks, batch size {settings.batch_concurrency}"
    )
    logger.info(f"[{short_id}] Phase 3/5: ANALYZING — {len(chunks)} chunks in {total_batches} batches (concurrency={settings.batch_concurrency})")

    all_detailed, all_summary = await analyze_chunks_batched(job_id, chunks, prompt, needs_consolidation=needs_consolidation)
    analyze_time = time.time() - phase_start

    if not all_detailed:
        raise ValueError("No RACM entries were generated. The file may not contain audit-relevant content.")

    logger.info(
        f"[{short_id}] Analysis phase done in {_fmt_duration(analyze_time)}: "
        f"{len(all_detailed)} detailed + {len(all_summary)} summary entries"
    )

    # ── Phase 4: Consolidation pass ─────────────────────────────
    phase_start = time.time()

    await update_job(job_id, eta_seconds=max(0, int(CONSOLIDATION_ESTIMATE if needs_consolidation else 0) + SUMMARY_ESTIMATE + OVERHEAD_BUFFER))

    if len(chunks) <= 2:
        # Single/dual chunk — no cross-chunk duplicates to merge, skip consolidation
        consolidated = {"detailed_entries": all_detailed, "summary_entries": all_summary}
        consolidate_time = time.time() - phase_start
        c_detailed = len(all_detailed)
        c_summary = len(all_summary)
        logger.info(f"[{short_id}] Consolidation SKIPPED: only {len(chunks)} chunk(s), no cross-chunk merging needed")
        await _update_progress(
            job_id, "consolidating", 90,
            "Consolidation skipped (single document chunk)",
            detail=f"Skipped consolidation — only {len(chunks)} chunk(s), no cross-chunk duplicates possible"
        )
    else:
        await _update_progress(
            job_id, "consolidating", 82,
            "Merging and consolidating entries...",
            detail=f"Consolidating {len(all_detailed)} detailed + {len(all_summary)} summary entries via Gemini..."
        )
        logger.info(f"[{short_id}] Phase 4/5: CONSOLIDATING — {len(all_detailed)} detailed + {len(all_summary)} summary entries")

        consolidated = await consolidation_pass(all_detailed, all_summary, prompt)
        consolidate_time = time.time() - phase_start

        c_detailed = len(consolidated.get("detailed_entries", []))
        c_summary = len(consolidated.get("summary_entries", []))
        logger.info(
            f"[{short_id}] Consolidation done in {_fmt_duration(consolidate_time)}: "
            f"{len(all_detailed)} → {c_detailed} detailed, {len(all_summary)} → {c_summary} summary"
        )
        await _update_detail(
            job_id,
            f"Consolidated {len(all_detailed)} → {c_detailed} detailed entries in {_fmt_duration(consolidate_time)}"
        )

    # ── Phase 5: Deduplication ──────────────────────────────────
    await update_job(job_id, eta_seconds=max(0, SUMMARY_ESTIMATE + OVERHEAD_BUFFER))
    phase_start = time.time()
    await _update_progress(
        job_id, "deduplicating", 92,
        "Removing duplicate entries...",
        detail=f"Deduplicating {c_detailed} detailed + {c_summary} summary entries..."
    )
    logger.info(f"[{short_id}] Phase 5/5: DEDUPLICATING — {c_detailed} detailed + {c_summary} summary entries")

    result = deduplicate_racm(consolidated)
    dedup_time = time.time() - phase_start

    final_detailed = len(result.get("detailed_entries", []))
    final_summary = len(result.get("summary_entries", []))
    removed = c_detailed - final_detailed
    logger.info(
        f"[{short_id}] Deduplication done in {_fmt_duration(dedup_time)}: "
        f"removed {removed} duplicates → {final_detailed} detailed + {final_summary} summary final"
    )
    await _update_detail(
        job_id,
        f"Removed {removed} duplicates → {final_detailed} detailed + {final_summary} summary entries"
    )

    # ── Phase 6: Summary generation ──────────────────────────────
    await update_job(job_id, eta_seconds=SUMMARY_ESTIMATE)
    phase_start = time.time()
    await _update_progress(
        job_id, "summarizing", 95,
        "Generating executive summary...",
        detail=f"Summarizing {final_detailed} entries into a narrative overview..."
    )
    logger.info(f"[{short_id}] Phase 6/6: SUMMARIZING — {final_detailed} entries")

    try:
        summary_narrative = await generate_racm_summary(
            result.get("detailed_entries", []),
            result.get("summary_entries", []),
            file_name or "Uploaded document",
        )
        summary_time = time.time() - phase_start
        logger.info(f"[{short_id}] Summary generated in {_fmt_duration(summary_time)}: {len(summary_narrative)} chars")
        await _update_detail(job_id, f"Summary generated in {_fmt_duration(summary_time)}")
    except Exception as e:
        summary_narrative = ""
        summary_time = time.time() - phase_start
        logger.warning(f"[{short_id}] Summary generation failed in {_fmt_duration(summary_time)}: {e}")
        await _update_detail(job_id, f"Summary generation failed (non-critical): {str(e)[:100]}")

    result["summary_narrative"] = summary_narrative

    # ── Phase 7: Store result ───────────────────────────────────
    total_time = time.time() - job_start
    completion_detail = (
        f"Done! {final_detailed} detailed + {final_summary} summary entries "
        f"generated in {_fmt_duration(total_time)}"
    )

    await update_job(
        job_id,
        status="completed",
        phase="completed",
        progress_pct=100,
        progress_msg="Analysis complete",
        detail_msg=completion_detail,
        eta_seconds=0,
        result_json=json.dumps(result, ensure_ascii=False),
    )

    logger.info(f"[{short_id}] {'='*60}")
    logger.info(
        f"[{short_id}] PIPELINE COMPLETE in {_fmt_duration(total_time)}: "
        f"{final_detailed} detailed + {final_summary} summary entries"
    )
    logger.info(
        f"[{short_id}] Timing breakdown: extract={_fmt_duration(extract_time)}, "
        f"chunk={_fmt_duration(chunk_time)}, analyze={_fmt_duration(analyze_time)}, "
        f"consolidate={_fmt_duration(consolidate_time)}, dedup={_fmt_duration(dedup_time)}, "
        f"summary={_fmt_duration(summary_time)}"
    )
    logger.info(f"[{short_id}] {'='*60}")

    return result
