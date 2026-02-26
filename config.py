from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Auth
    api_token: str = "abc"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Processing
    chunk_size: int = 40000
    chunk_overlap_sentences: int = 3
    batch_concurrency: int = 3
    max_file_size_mb: int = 50

    # Paths
    upload_dir: str = "uploads"
    database_path: str = "racm_jobs.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
