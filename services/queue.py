import asyncio
import logging

from database import get_job, update_job, get_incomplete_jobs
from services.pipeline import process_job

logger = logging.getLogger(__name__)

# In-memory set of active job IDs to prevent double-dispatch
_active_jobs: set[str] = set()


async def dispatch_job(job_id: str):
    """Fire-and-forget a job into the asyncio event loop."""
    if job_id in _active_jobs:
        logger.warning(f"Job {job_id} already active, skipping dispatch")
        return
    _active_jobs.add(job_id)
    asyncio.create_task(_run_job(job_id))


async def _run_job(job_id: str):
    """Worker coroutine that processes a job and updates status in SQLite."""
    try:
        await update_job(job_id, status="processing", phase="extracting", progress_pct=2)
        job = await get_job(job_id)
        if job is None:
            logger.error(f"Job {job_id} not found in database")
            return

        await process_job(
            job_id=job_id,
            file_path=job["file_path"],
            file_type=job["file_type"],
            prompt=job["prompt"],
            file_name=job.get("file_name", ""),
        )
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        await update_job(
            job_id,
            status="failed",
            phase="failed",
            progress_msg=f"Failed: {str(e)[:500]}",
            error_message=str(e),
        )
    finally:
        _active_jobs.discard(job_id)


async def recover_incomplete_jobs():
    """Called at startup. Re-dispatches any jobs that were interrupted by a server restart."""
    incomplete = await get_incomplete_jobs()
    if incomplete:
        logger.info(f"Recovering {len(incomplete)} incomplete jobs from previous session")
    for job in incomplete:
        # Reset progress so they start fresh
        await update_job(job["id"], status="queued", phase="queued", progress_pct=0)
        await dispatch_job(job["id"])
