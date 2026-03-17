ADR: Multimodal Input — Decoupled Transcription Endpoint

Status: Accepted
Date: 2026-03-16
Owner: Yasuhiro Okamoto

## Context

The requirements specify "Multimodal Query Understanding — The system can interpret user queries provided through text, voice, or uploaded research documents" (Key Capabilities). The system must support at least text and voice input, with image support as an additional capability.

The architectural question is whether multimodal processing should be integrated into the RAG pipeline or decoupled as a separate preprocessing step.

## Decision

Implement a **decoupled `/transcribe` endpoint** that converts non-text inputs (audio, image, documents) to text. The transcribed text is returned to the frontend, where the user reviews it before submitting as a regular text query to `/ask` or `/search`.

```
Audio/Image/Document → POST /transcribe → { text, media_type } → User reviews → POST /ask
```

The RAG pipeline (`/ask`, `/search`) remains text-only. Multimodal understanding is handled entirely at the input boundary.

## Supported Modalities

| Input Type | Processing | Model | Use Case |
|------------|-----------|-------|----------|
| Text | Direct input | — | Primary input method |
| Audio | Speech-to-text | OpenAI Whisper (`whisper-1`) | Voice recordings of research questions |
| Image | Vision-to-text | GPT-4o-mini (vision) | Screenshots of research figures, clinical notes |
| PDF | Text extraction + summarization | PyMuPDF + GPT-4o-mini | Research papers, clinical reports |
| TXT | UTF-8 decode + summarization | GPT-4o-mini | Clinical notes, plain text |
| DOCX | Text extraction + summarization | python-docx + GPT-4o-mini | Word documents, clinical protocols |

### Audio Processing

- Accepts `audio/*` MIME types (mp3, wav, m4a, etc.)
- Max file size: 25MB (Whisper API limit)
- Returns transcribed text verbatim

### Image Processing

- Accepts `image/*` MIME types (png, jpg, etc.)
- Image is base64-encoded and sent to GPT-4o-mini vision
- System prompt instructs: "Extract the key medical/scientific information from this image. Return a concise natural language query or summary suitable for searching medical literature."
- Returns an interpretive text summary, not OCR

## Why Decoupled (Not Integrated)

### Option A: Integrated (multimodal directly in `/ask`)

```
POST /ask { file: audio.mp3, ... } → transcribe → expand → search → generate
```

- Single API call for the user
- User cannot verify or correct the transcription before RAG processing
- Harder to test (file upload + RAG in one endpoint)
- Error in transcription leads to irrelevant search results with no recourse

### Option B: Decoupled (chosen)

```
POST /transcribe { file: audio.mp3 } → { text: "..." }
User reviews text → POST /ask { query: "..." }
```

- User sees and can edit the transcribed text before searching
- `/transcribe` and `/ask` are independently testable
- No changes to the RAG pipeline — it remains text-in, text-out
- Frontend can show a loading state during transcription separately from search

**Option B was chosen** because:

1. **User control.** Whisper and vision models are not perfect. Showing the transcribed text before searching lets users catch errors ("BRCA1" misheard as "burka one"). In medical contexts, transcription errors can lead to completely wrong results.
2. **Separation of concerns.** The RAG pipeline does not need to know about audio or image formats. Adding new modalities (e.g., PDF) only requires a new handler in `/transcribe`, not changes to `/ask`.
3. **Testability.** Each endpoint can be tested independently with known inputs.

## Frontend Integration

`ChatPanel.tsx`:

1. User clicks the attachment button (paperclip icon)
2. File picker opens with `accept="audio/*,image/*,.pdf,.txt,.doc,.docx"`
3. Selected file is POSTed to `/transcribe`
4. While transcribing, a spinner shows on the attachment button and the input says "Transcribing..."
5. On success, transcribed text populates the chat input field
6. User reviews, optionally edits, and presses Send
7. The text is submitted as a normal query to `/ask`

This two-step flow is explicit: the user always sees what the system "heard" before searching.

## Document Upload (PDF/TXT/DOCX)

Added in 2026-03-17. The `/transcribe` endpoint now also accepts document files:

| Input Type | Processing | Model | Use Case |
|------------|-----------|-------|----------|
| PDF | Text extraction via PyMuPDF | GPT-4o-mini (summarization) | Research papers, clinical reports |
| TXT | UTF-8 decode | GPT-4o-mini (summarization) | Clinical notes, plain text |
| DOCX | Text extraction via python-docx | GPT-4o-mini (summarization) | Word documents, clinical protocols |

Documents have separate limits from audio/image:
- **File size**: 10MB (vs 25MB for audio/image)
- **Extracted text**: Truncated to 10,000 characters before LLM summarization

The flow is the same as image: extract text → LLM generates a concise search query → user reviews in chat input → submits to `/ask`. See `docs/specs/2026-03-17-document-upload-design.md` for the full spec.

## Consequences

### Positive

- RAG pipeline stays simple and text-only
- Users can catch and correct transcription errors before searching
- New modalities can be added to `/transcribe` without changing `/ask`
- Audio and image processing use standard OpenAI APIs — no custom model training

### Trade-offs

- Two-step UX (transcribe → review → search) adds friction compared to a single "upload and search" flow
- Depends on external APIs (OpenAI Whisper, GPT-4o-mini) — adds cost and latency for multimodal queries
- Image interpretation is approximate — GPT-4o-mini may not capture all details from complex research figures
- Document upload (PDF/TXT/DOCX) uses LLM summarization which adds latency and cost for long documents
