"""POST /transcribe — convert audio/image files to text via OpenAI."""

import base64
import logging

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB (Whisper API limit)

IMAGE_SYSTEM_PROMPT = (
    "Extract the key medical/scientific information from this image. "
    "Return a concise natural language query or summary suitable for "
    "searching medical literature."
)


class TranscribeResponse(BaseModel):
    text: str
    media_type: str  # "audio" or "image"


def _transcribe_audio(file_bytes: bytes, filename: str) -> str:
    import openai

    client = openai.OpenAI()
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, file_bytes),
    )
    return result.text


def _transcribe_image(file_bytes: bytes, content_type: str) -> str:
    import openai

    b64 = base64.b64encode(file_bytes).decode()
    data_url = f"data:{content_type};base64,{b64}"

    client = openai.OpenAI()
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                ],
            },
        ],
        max_tokens=300,
    )
    return result.choices[0].message.content


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(file: UploadFile):
    content_type = file.content_type or ""

    if not (content_type.startswith("audio/") or content_type.startswith("image/")):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {content_type}. Only audio/* and image/* are accepted.",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)} bytes). Maximum is {MAX_FILE_SIZE} bytes (25MB).",
        )

    try:
        if content_type.startswith("audio/"):
            text = _transcribe_audio(file_bytes, file.filename or "audio.wav")
            media_type = "audio"
        else:
            text = _transcribe_image(file_bytes, content_type)
            media_type = "image"
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")

    return TranscribeResponse(text=text, media_type=media_type)
