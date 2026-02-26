import json
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from config import settings
from database import create_job, delete_job, get_job, update_job, update_result
from models import (
    DeleteResponse,
    JobCreateResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
    RACMResponse,
    UpdateResultRequest,
)
from services.queue import dispatch_job

router = APIRouter()

ALLOWED_EXTENSIONS = {
    "pdf", "xlsx", "xls", "csv",
    "png", "jpg", "jpeg", "tiff", "bmp", "webp",
}

MAX_FILE_BYTES = settings.max_file_size_mb * 1024 * 1024


def _get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@router.post("/jobs", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_analysis_job(
    file: UploadFile = File(...),
    prompt: str = Form(default=None),
):
    # Validate extension
    ext = _get_extension(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_file_size_mb}MB limit",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty",
        )

    # Save file
    job_id = str(uuid.uuid4())
    safe_name = file.filename or f"upload.{ext}"
    file_path = os.path.join(settings.upload_dir, f"{job_id}_{safe_name}")

    os.makedirs(settings.upload_dir, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

    # Create job record
    await create_job(
        job_id=job_id,
        file_name=safe_name,
        file_path=file_path,
        file_type=ext,
        prompt=prompt,
    )

    # Dispatch to background worker
    await dispatch_job(job_id)

    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.queued,
        message="Job queued for processing",
    )


@router.delete("/jobs/{job_id}", response_model=DeleteResponse)
async def cancel_or_delete_job(job_id: str):
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    job_status = job["status"]

    if job_status in ("queued", "processing"):
        # Cancel: mark as failed so pipeline stops at next checkpoint
        await update_job(
            job_id,
            status="failed",
            error_message="Cancelled by user",
        )
        # Clean up uploaded file
        file_path = job.get("file_path", "")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        return DeleteResponse(
            job_id=job_id,
            deleted=True,
            message="Job cancelled",
        )

    # completed or failed: delete the DB row + file
    file_path = await delete_job(job_id)
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
    return DeleteResponse(
        job_id=job_id,
        deleted=True,
        message="Report deleted",
    )


@router.put("/jobs/{job_id}/result")
async def save_inline_edits(job_id: str, body: UpdateResultRequest):
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Can only edit results of completed jobs",
        )

    # Rebuild result JSON preserving summary_narrative
    existing = json.loads(job["result_json"]) if job["result_json"] else {}
    existing["detailed_entries"] = body.detailed_entries
    existing["summary_entries"] = body.summary_entries
    await update_result(job_id, json.dumps(existing))

    return {"job_id": job_id, "status": "saved"}


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return JobStatusResponse(
        job_id=job["id"],
        status=job["status"],
        phase=job["phase"],
        progress_pct=job["progress_pct"],
        progress_msg=job["progress_msg"] or "",
        detail_msg=job.get("detail_msg") or "",
        eta_seconds=job.get("eta_seconds") or 0,
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        file_name=job["file_name"],
    )


@router.get("/jobs/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str):
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job["status"] == "completed":
        result_data = json.loads(job["result_json"]) if job["result_json"] else None
        result = RACMResponse(**result_data) if result_data else None
        return JobResultResponse(
            job_id=job["id"],
            status=JobStatus.completed,
            result=result,
        )

    if job["status"] == "failed":
        return JobResultResponse(
            job_id=job["id"],
            status=JobStatus.failed,
            error=job["error_message"],
        )

    # Still processing
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "job_id": job["id"],
            "status": job["status"],
            "phase": job["phase"],
            "progress_pct": job["progress_pct"],
            "message": "Job is still processing. Poll /status for progress.",
        },
    )
