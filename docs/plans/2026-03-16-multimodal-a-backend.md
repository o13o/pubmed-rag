# Multimodal Transcription — Backend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /transcribe` endpoint that converts audio (Whisper) and images (GPT-4o-mini vision) to text.

**Architecture:** Single new route file `transcribe.py` receives multipart file upload, detects MIME type, dispatches to OpenAI Whisper (audio) or GPT-4o-mini vision (image), returns extracted text. Uses `openai` SDK directly (already installed). No changes to existing RAG pipeline.

**Tech Stack:** FastAPI, openai SDK 2.26.0, Python 3.11+

**Spec:** `capstone/docs/specs/2026-03-16-multimodal-transcribe-design.md`

**Parallel:** This plan can run in parallel with `2026-03-16-multimodal-b-frontend.md`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/src/api/routes/transcribe.py` | Create | `/transcribe` endpoint, MIME routing, OpenAI calls |
| `backend/tests/unit/test_api_transcribe.py` | Create | Unit tests for all cases |
| `backend/src/api/main.py` | Edit (line 10, 77) | Register transcribe router |

---

### Task 1: Write unit tests for `/transcribe` endpoint

**Files:**
- Create: `backend/tests/unit/test_api_transcribe.py`

- [ ] **Step 1: Create test file with all test cases**

```python
"""Tests for POST /transcribe endpoint."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("src.api.main.connections"), \
         patch("src.api.main.Collection") as mock_col, \
         patch("src.api.main.LLMClient"), \
         patch("src.api.main.MeSHDatabase"), \
         patch("src.api.main.get_reranker"):

        mock_col.return_value = MagicMock(num_entities=100)

        from src.api.main import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


@patch("src.api.routes.transcribe._transcribe_audio")
def test_transcribe_audio(mock_transcribe, client):
    mock_transcribe.return_value = "cancer treatment options"
    audio_bytes = b"\x00\x01\x02\x03" * 100
    response = client.post(
        "/transcribe",
        files={"file": ("recording.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "cancer treatment options"
    assert data["media_type"] == "audio"
    mock_transcribe.assert_called_once()


@patch("src.api.routes.transcribe._transcribe_image")
def test_transcribe_image(mock_transcribe, client):
    mock_transcribe.return_value = "chart showing survival rates"
    # Minimal valid PNG header
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    response = client.post(
        "/transcribe",
        files={"file": ("figure.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "chart showing survival rates"
    assert data["media_type"] == "image"
    mock_transcribe.assert_called_once()


def test_transcribe_rejects_unsupported_mime(client):
    response = client.post(
        "/transcribe",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert response.status_code == 415


def test_transcribe_rejects_oversized_file(client):
    big_bytes = b"\x00" * (26 * 1024 * 1024)  # 26MB > 25MB limit
    response = client.post(
        "/transcribe",
        files={"file": ("big.mp3", io.BytesIO(big_bytes), "audio/mpeg")},
    )
    assert response.status_code == 413
```

- [ ] **Step 2: Run tests — expect failure (transcribe module not found)**

Run: `cd capstone/backend && .venv/bin/python -m pytest tests/unit/test_api_transcribe.py -v`
Expected: ImportError or ModuleNotFoundError for `src.api.routes.transcribe`

- [ ] **Step 3: Commit test file**

```bash
git add capstone/backend/tests/unit/test_api_transcribe.py
git commit -m "test: add unit tests for POST /transcribe endpoint"
```

---

### Task 2: Implement `/transcribe` endpoint

**Files:**
- Create: `backend/src/api/routes/transcribe.py`

- [ ] **Step 1: Create the transcribe route module**

```python
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
```

- [ ] **Step 2: Run tests — expect failure (router not registered yet, but imports work)**

Run: `cd capstone/backend && .venv/bin/python -c "from src.api.routes.transcribe import router; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit implementation**

```bash
git add capstone/backend/src/api/routes/transcribe.py
git commit -m "feat(api): add POST /transcribe endpoint for audio and image"
```

---

### Task 3: Register router and run all tests

**Files:**
- Modify: `backend/src/api/main.py` (lines 10 and 74-77)

- [ ] **Step 1: Edit `main.py` to import and register transcribe router**

In `src/api/main.py` line 10, change:
```python
from src.api.routes import analyze, ask, health, search
```
to:
```python
from src.api.routes import analyze, ask, health, search, transcribe
```

After line 77 (`app.include_router(analyze.router)`), add:
```python
    app.include_router(transcribe.router)
```

- [ ] **Step 2: Run transcribe tests — should all pass**

Run: `cd capstone/backend && .venv/bin/python -m pytest tests/unit/test_api_transcribe.py -v`
Expected: 4 passed

- [ ] **Step 3: Run all unit tests — should still pass**

Run: `cd capstone/backend && .venv/bin/python -m pytest tests/unit/ -q`
Expected: 127 passed (123 existing + 4 new)

- [ ] **Step 4: Commit**

```bash
git add capstone/backend/src/api/main.py
git commit -m "feat(api): register /transcribe router in app"
```
