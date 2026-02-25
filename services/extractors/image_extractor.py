import mimetypes

from services.gemini import vision_extract


async def extract_image(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "image/png"

    with open(file_path, "rb") as f:
        image_bytes = f.read()

    text = await vision_extract(image_bytes, mime_type)
    return f"--- IMAGE CONTENT ---\n{text}"
