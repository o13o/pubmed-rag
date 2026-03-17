# Document Upload (PDF/TXT/Word) — Implementation Plan

**Goal:** Extend `/transcribe` to accept PDF, TXT, and Word files. Extract text, truncate to 10,000 chars, then summarize via LLM into a search query.

**Spec:** `capstone/docs/specs/2026-03-17-document-upload-design.md`

All file paths relative to `capstone/`.

---

## Chunk 1: Backend — Dependencies and extraction functions

### Task 1: Add dependencies

**Files:**
- Modify: `backend/pyproject.toml` (or `requirements.txt`)

- [ ] **Step 1: Add pymupdf and python-docx**

Add `pymupdf` and `python-docx` to the project dependencies.

```
pymupdf
python-docx
```

- [ ] **Step 2: Install and verify**

Run `pip install pymupdf python-docx` (or equivalent) and verify import works.

### Task 2: Add extraction functions and update endpoint

**Files:**
- Modify: `backend/src/api/routes/transcribe.py`

- [ ] **Step 1: Add constants**

Add document-specific constants at the top of `transcribe.py`:

```python
MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB
MAX_EXTRACTED_CHARS = 10_000

DOCUMENT_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

DOCUMENT_SYSTEM_PROMPT = (
    "Extract the key medical/scientific information from this document. "
    "Return a concise natural language query or summary suitable for "
    "searching medical literature."
)
```

- [ ] **Step 2: Add `_extract_pdf` function**

```python
def _extract_pdf(file_bytes: bytes) -> str:
    import pymupdf
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text
```

- [ ] **Step 3: Add `_extract_txt` function**

```python
def _extract_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8")
```

- [ ] **Step 4: Add `_extract_docx` function**

```python
def _extract_docx(file_bytes: bytes) -> str:
    import io
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)
```

- [ ] **Step 5: Add `_summarize_document` function**

Truncate to `MAX_EXTRACTED_CHARS` then call GPT-4o-mini:

```python
def _summarize_document(raw_text: str) -> str:
    import openai

    truncated = raw_text[:MAX_EXTRACTED_CHARS]
    client = openai.OpenAI()
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DOCUMENT_SYSTEM_PROMPT},
            {"role": "user", "content": truncated},
        ],
        max_tokens=300,
    )
    return result.choices[0].message.content
```

- [ ] **Step 6: Update `transcribe_endpoint` to handle documents**

Modify the endpoint to:
1. Check if content_type is in `DOCUMENT_CONTENT_TYPES`
2. Use `MAX_DOCUMENT_SIZE` for documents, `MAX_FILE_SIZE` for audio/image
3. Route to the correct extraction function
4. Check for empty extracted text (raise 422)
5. Pass extracted text through `_summarize_document`
6. Return `TranscribeResponse(text=..., media_type="document")`

Key changes to the content_type check:

```python
is_document = content_type in DOCUMENT_CONTENT_TYPES
is_media = content_type.startswith("audio/") or content_type.startswith("image/")

if not (is_document or is_media):
    raise HTTPException(status_code=415, ...)

# Size check with different limits
size_limit = MAX_DOCUMENT_SIZE if is_document else MAX_FILE_SIZE
if len(file_bytes) > size_limit:
    raise HTTPException(status_code=413, ...)
```

Document routing:

```python
if is_document:
    if content_type == "application/pdf":
        raw_text = _extract_pdf(file_bytes)
    elif content_type == "text/plain":
        raw_text = _extract_txt(file_bytes)
    else:
        raw_text = _extract_docx(file_bytes)

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from this file.")

    text = _summarize_document(raw_text)
    media_type = "document"
```

---

## Chunk 2: Backend — Tests

### Task 3: Add unit tests

**Files:**
- New: `backend/tests/unit/test_transcribe.py`

- [ ] **Step 1: Test `_extract_txt`**

```python
def test_extract_txt():
    text = _extract_txt(b"hello world")
    assert text == "hello world"
```

- [ ] **Step 2: Test `_extract_pdf` with mock**

Mock `pymupdf.open` to return a document with pages that have `get_text()` methods.

- [ ] **Step 3: Test `_extract_docx` with mock**

Mock `docx.Document` to return an object with paragraphs.

- [ ] **Step 4: Test document size limit (413)**

Call the endpoint with a document content_type and body > 10MB. Expect 413.

- [ ] **Step 5: Test text truncation**

Verify that `_summarize_document` receives at most 10,000 characters even when raw text is longer.

- [ ] **Step 6: Test empty document (422)**

Mock extraction to return empty string. Expect 422.

- [ ] **Step 7: Test unsupported content type (415)**

Upload a file with `application/zip` content type. Expect 415.

---

## Chunk 3: Frontend

### Task 4: Update file input and types

**Files:**
- Modify: `frontend/src/components/ChatPanel.tsx`
- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Extend `accept` attribute**

In `ChatPanel.tsx`, change:
```tsx
accept="audio/*,image/*"
```
to:
```tsx
accept="audio/*,image/*,.pdf,.txt,.doc,.docx"
```

- [ ] **Step 2: Update button tooltip**

Change `title="Upload audio or image"` to `title="Upload audio, image, or document"`.

- [ ] **Step 3: Extend `TranscribeResponse` type**

In `types/index.ts`, change:
```typescript
media_type: "audio" | "image";
```
to:
```typescript
media_type: "audio" | "image" | "document";
```

---

## Verification

After all chunks are complete:

- [ ] Upload a PDF file and verify it gets summarized into a search query
- [ ] Upload a .txt file and verify the same flow
- [ ] Upload a .docx file and verify the same flow
- [ ] Upload a file > 10MB and verify 413 error
- [ ] Upload an unsupported file type and verify 415 error
- [ ] Run `pytest backend/tests/unit/test_transcribe.py` and verify all pass
