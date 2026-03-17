# Multimodal Query Support via Transcription Endpoint

**Date:** 2026-03-16
**Status:** Accepted
**Owner:** Yasuhiro Okamoto

## Problem

The project requirements (statements.md) specify "Multimodal Query Understanding" — the system should interpret queries provided through text, voice, or uploaded research documents. Currently only text input is supported.

## Decision

Add a single `POST /transcribe` endpoint that converts audio and image files to text. The frontend sends the file, receives text back, and populates the chat input. The user confirms/edits and submits via the existing `/ask` or `/search` flow. No changes to the existing RAG pipeline.

## Scope

- **Audio** (voice recordings, .mp3, .wav, .m4a, .webm) — OpenAI Whisper API
- **Image** (research figures, screenshots, .png, .jpg, .webp) — GPT-4o-mini vision
- **PDF/TXT/DOCX** — added in 2026-03-17, see `docs/specs/2026-03-17-document-upload-design.md`

## Architecture

```
Frontend                  Backend                    OpenAI
  |                         |                          |
  +-- upload file --------->|                          |
  |   (multipart/form-data) +-- detect MIME type       |
  |                         +-- audio/* -> Whisper --->|
  |                         +-- image/* -> GPT-4o-mini |
  |                         |             (vision) --->|
  |<-- { "text": "..." } --+                          |
  |                         |                          |
  +-- /ask or /search ----->|  (existing flow, no change)
```

Key property: the transcription step is **decoupled** from the RAG pipeline. The `/transcribe` endpoint is a pure text extraction service.

## Backend: `POST /transcribe`

### Route: `src/api/routes/transcribe.py`

- Accepts `multipart/form-data` with a single `file` field
- Validates MIME type (`audio/*` or `image/*`), rejects others with 415
- File size limit: 25MB (Whisper API limit)
- Routes to appropriate handler based on MIME prefix

### Response Model

```python
class TranscribeResponse(BaseModel):
    text: str
    media_type: str  # "audio" or "image"
```

### Audio Handling

- Uses OpenAI Whisper API (`POST /v1/audio/transcriptions`)
- Model: `whisper-1`
- Sends file bytes directly — no temp file needed
- Returns raw transcription text

### Image Handling

- Uses GPT-4o-mini with vision capability
- Encodes image as base64 data URL
- System prompt: "Extract the key medical/scientific information from this image. Return a concise natural language query or summary suitable for searching medical literature."
- Returns generated text description

### Error Handling

- 415 Unsupported Media Type for non-audio/non-image files
- 413 if file exceeds 25MB
- 502 if OpenAI API call fails (wraps upstream error message)

## Frontend Changes

### ChatPanel.tsx

- Add a file attachment button (paperclip icon) next to the text input
- Hidden `<input type="file" accept="audio/*,image/*">` triggered by the button
- On file select:
  1. Show a loading indicator on the button
  2. POST to `/transcribe` with `FormData`
  3. On success: populate the text input with returned `text`
  4. On error: show error message
  5. User reviews/edits text, then clicks Send (existing flow)

### api.ts

- Add `transcribeFile(file: File): Promise<{ text: string; media_type: string }>` function

## Dependencies

### Backend

- `openai` Python SDK (already available via `litellm` dependency, but used directly for Whisper)
- No new packages needed — `openai` is already in the dependency tree

### Frontend

- No new packages needed — uses native `FormData` and `fetch`

## Testing

### Unit Test: `tests/unit/test_api_transcribe.py`

- Mock OpenAI calls
- Test audio file -> returns transcription text
- Test image file -> returns description text
- Test unsupported MIME type -> 415
- Test oversized file -> 413

## Files to Create/Modify

| File | Action |
|------|--------|
| `backend/src/api/routes/transcribe.py` | Create — new endpoint |
| `backend/tests/unit/test_api_transcribe.py` | Create — unit tests |
| `backend/src/api/main.py` | Edit — register transcribe router |
| `frontend/src/lib/api.ts` | Edit — add `transcribeFile()` |
| `frontend/src/components/ChatPanel.tsx` | Edit — add file upload button + logic |
| `frontend/nginx.conf` | Edit — add `/transcribe` to proxy routes |
