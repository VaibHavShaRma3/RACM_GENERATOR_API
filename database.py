import aiosqlite
from datetime import datetime, timezone

from config import settings

DB_PATH = settings.database_path


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id              TEXT PRIMARY KEY,
                status          TEXT NOT NULL DEFAULT 'queued',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                file_name       TEXT NOT NULL,
                file_path       TEXT NOT NULL,
                file_type       TEXT NOT NULL,
                prompt          TEXT,
                progress_pct    INTEGER DEFAULT 0,
                progress_msg    TEXT DEFAULT '',
                phase           TEXT DEFAULT 'queued',
                total_chunks    INTEGER DEFAULT 0,
                completed_chunks INTEGER DEFAULT 0,
                result_json     TEXT,
                error_message   TEXT
            )
        """)
        await db.commit()


async def create_job(
    job_id: str,
    file_name: str,
    file_path: str,
    file_type: str,
    prompt: str | None = None,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO jobs (id, status, created_at, updated_at, file_name, file_path, file_type, prompt)
               VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)""",
            (job_id, now, now, file_name, file_path, file_type, prompt),
        )
        await db.commit()
    return await get_job(job_id)


async def get_job(job_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)


async def update_job(job_id: str, **fields) -> None:
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values())
    values.append(job_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE jobs SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()


async def get_incomplete_jobs() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM jobs WHERE status IN ('queued', 'processing')"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
