import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth import verify_token
from config import settings
from database import init_db
from routers.jobs import router as jobs_router
from services.queue import recover_incomplete_jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    await init_db()
    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info("Recovering incomplete jobs...")
    await recover_incomplete_jobs()
    logger.info("RACM Generator API ready")
    yield
    # Shutdown
    logger.info("Shutting down")


app = FastAPI(
    title="RACM Generator API",
    description="Generates Risk and Control Matrices from SOP documents using Gemini AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    jobs_router,
    prefix="/api",
    dependencies=[Depends(verify_token)],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
