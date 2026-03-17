# Document Upload (PDF/TXT/Word) — Design Spec

**Date:** 2026-03-17
**Status:** Approved

All file paths are relative to `capstone/`.

## 1. Goal

Extend the existing `/transcribe` endpoint to accept PDF, TXT, and Word (.docx) files in addition to audio and image. Extracted text is summarized by LLM into a concise medical literature search query, matching the existing image transcription pattern.

## 2. Current State

The `/transcribe` endpoint (`backend/src/api/routes/transcribe.py`) currently handles:
- `audio/*` — OpenAI Whisper API for speech-to-text
- `image/*` — GPT-4o-mini for visual content extraction

The frontend (`frontend/src/components/ChatPanel.tsx`) has a file upload button with `accept="audio/*,image/*"`. The extracted text is placed into the input field for the user to review and submit.

## 3. Approach

Add document handling to the same `/transcribe` endpoint. The flow is:

```
Upload file → content_type routing → text extraction → truncation → LLM query generation → response
```

### 3.1 Supported Content Types

| Content Type | Extension | Extraction Method |
|-------------|-----------|-------------------|
| `application/pdf` | .pdf | PyMuPDF (`pymupdf`) |
| `text/plain` | .txt | `bytes.decode("utf-8")` |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | .docx | `python-docx` |

### 3.2 Two-Stage Limits

| Limit | Value | Rationale |
|-------|-------|-----------|
| Document file size | 10 MB | PDF papers are typically 1-5 MB; prevents large scanned PDFs |
| Audio/image file size | 25 MB | Unchanged (existing behavior) |
| Extracted text truncation | 10,000 characters | Fits comfortably in GPT-4o-mini context window |

File size is checked before reading the full body. Text truncation is applied after extraction and before LLM summarization.

### 3.3 LLM Query Generation

Reuse the same pattern as `_transcribe_image`. After extracting raw text from the document, pass it to GPT-4o-mini with a system prompt:

```
Extract the key medical/scientific information from this document.
Return a concise natural language query or summary suitable for
searching medical literature.
```

This converts potentially long documents into a focused search query.

### 3.4 Response

The existing `TranscribeResponse` schema is extended:

```python
class TranscribeResponse(BaseModel):
    text: str
    media_type: str  # "audio" | "image" | "document"
```

## 4. Changes

### 4.1 Backend

**`backend/src/api/routes/transcribe.py`:**
- Add `_extract_pdf(file_bytes) -> str` using PyMuPDF
- Add `_extract_txt(file_bytes) -> str` using UTF-8 decode
- Add `_extract_docx(file_bytes) -> str` using python-docx
- Add `_summarize_document(text) -> str` that truncates to 10,000 chars then calls GPT-4o-mini
- Add `DOCUMENT_CONTENT_TYPES` set for routing
- Add `MAX_DOCUMENT_SIZE = 10 * 1024 * 1024` constant
- Update content_type check to accept document types
- Add two-stage size validation (file size first, then text truncation)

**`backend/pyproject.toml`** (or equivalent dependency file):
- Add `pymupdf` and `python-docx`

### 4.2 Frontend

**`frontend/src/components/ChatPanel.tsx`:**
- Change `accept` attribute to `"audio/*,image/*,.pdf,.txt,.doc,.docx"`
- Update tooltip text

**`frontend/src/types/index.ts`:**
- Extend `TranscribeResponse.media_type` to include `"document"`

### 4.3 Tests

**`backend/tests/unit/test_transcribe.py`** (new file):
- Test `_extract_txt` with UTF-8 bytes
- Test `_extract_pdf` with mock PyMuPDF
- Test `_extract_docx` with mock python-docx
- Test file size rejection (413) for documents > 10MB
- Test text truncation at 10,000 characters
- Test unsupported content type rejection (415)
- Test `_summarize_document` calls LLM with truncated text

## 5. Error Handling

| Scenario | HTTP Status | Detail |
|----------|-------------|--------|
| Unsupported content type | 415 | "Unsupported media type: {type}" |
| Document > 10MB | 413 | "File too large. Maximum for documents is 10MB." |
| Audio/image > 25MB | 413 | "File too large. Maximum is 25MB." |
| PDF extraction fails | 502 | "Text extraction failed: {error}" |
| LLM summarization fails | 502 | "Summarization failed: {error}" |
| Empty document (no text extracted) | 422 | "No text could be extracted from this file." |

## 6. Dependencies

| Package | Purpose | License |
|---------|---------|---------|
| `pymupdf` | PDF text extraction | AGPL-3.0 |
| `python-docx` | Word .docx text extraction | MIT |
