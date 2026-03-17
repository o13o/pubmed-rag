"""POST /transcribe — convert audio/image/document files to text via OpenAI."""

import base64
import io
import logging

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from src.shared.prompt_loader import load_prompt

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB (Whisper API limit)
MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB
MAX_EXTRACTED_CHARS = 10_000

DOCUMENT_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_IMAGE_PROMPT = load_prompt("transcribe/image")
_DOCUMENT_PROMPT = load_prompt("transcribe/document")


class TranscribeResponse(BaseModel):
    text: str
    media_type: str  # "audio", "image", or "document"


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
            {"role": "system", "content": _IMAGE_PROMPT["system"]},
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


def _extract_pdf(file_bytes: bytes) -> str:
    import pymupdf

    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def _extract_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8")


def _extract_docx(file_bytes: bytes) -> str:
    import docx

    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def _summarize_document(raw_text: str) -> str:
    import openai

    truncated = raw_text[:MAX_EXTRACTED_CHARS]
    client = openai.OpenAI()
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _DOCUMENT_PROMPT["system"]},
            {"role": "user", "content": truncated},
        ],
        max_tokens=300,
    )
    return result.choices[0].message.content


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(file: UploadFile):
    content_type = file.content_type or ""

    is_document = content_type in DOCUMENT_CONTENT_TYPES
    is_media = content_type.startswith("audio/") or content_type.startswith("image/")

    if not (is_document or is_media):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {content_type}. Accepted: audio/*, image/*, PDF, TXT, DOCX.",
        )

    file_bytes = await file.read()

    size_limit = MAX_DOCUMENT_SIZE if is_document else MAX_FILE_SIZE
    size_label = "10MB" if is_document else "25MB"
    if len(file_bytes) > size_limit:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)} bytes). Maximum for {'documents' if is_document else 'media'} is {size_label}.",
        )

    try:
        if content_type.startswith("audio/"):
            text = _transcribe_audio(file_bytes, file.filename or "audio.wav")
            media_type = "audio"
        elif content_type.startswith("image/"):
            text = _transcribe_image(file_bytes, content_type)
            media_type = "image"
        elif content_type == "application/pdf":
            raw_text = _extract_pdf(file_bytes)
            if not raw_text.strip():
                raise HTTPException(
                    status_code=422,
                    detail="No text could be extracted from this file.",
                )
            text = _summarize_document(raw_text)
            media_type = "document"
        elif content_type == "text/plain":
            raw_text = _extract_txt(file_bytes)
            if not raw_text.strip():
                raise HTTPException(
                    status_code=422,
                    detail="No text could be extracted from this file.",
                )
            text = _summarize_document(raw_text)
            media_type = "document"
        else:
            # DOCX
            raw_text = _extract_docx(file_bytes)
            if not raw_text.strip():
                raise HTTPException(
                    status_code=422,
                    detail="No text could be extracted from this file.",
                )
            text = _summarize_document(raw_text)
            media_type = "document"
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")

    return TranscribeResponse(text=text, media_type=media_type)
