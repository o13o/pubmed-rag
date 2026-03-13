ADR: Chunking Strategy and Embedding Model Selection for PubMed/MEDLINE Abstract–Based RAG

Status: Accepted
Date: 2026-03-13
Owner: Yasuhiro Okamoto

Context

This Proof of Concept (PoC) system is built as part of a training / learning program and implements a Retrieval-Augmented Generation (RAG) pipeline using PubMed/MEDLINE abstracts as the primary corpus.
PubMed abstracts historically followed word limits of 250 words (or 400 words under certain conditions), but modern PubMed records are generally not truncated, and abstract length is determined by the publisher.
Large-scale analysis of PubMed Central articles (≈61k papers, 2016–2021) shows:

Median abstract length: ~263 words
90% range: ~163–416 words
Most abstracts fall in the 200–300 word range, which is well suited for single-chunk retrieval.


The primary objective is understanding and validating the RAG pipeline (indexing → retrieval → ranking → generation), rather than achieving maximum possible retrieval accuracy.

Decision
1. Chunking Strategy

Use one abstract as one document / one chunk by default.
Do not apply fixed-size or sliding-window chunking.
As an exception, very long structured abstracts (e.g., Results-heavy clinical abstracts) may be split into at most two chunks (e.g., Methods and Results), but this is optional.
The indexed text shall be:
Title + Abstract (+ optional MeSH terms appended as plain text)



2. Embedding Model Selection (Initial PoC)

Adopt OpenAI text-embedding-3-small as the default embedding model.
Keep text-embedding-3-large as an optional alternative for side-by-side comparison during training or evaluation exercises.

Rationale
Why “1 Abstract = 1 Chunk”

PubMed abstracts are:

Short (typically 200–300 words)
Semantically self-contained
Written in a standardized academic style (often IMRAD or structured)


Treating the entire abstract as a single chunk:

Preserves semantic coherence
Avoids chunk-boundary artifacts
Simplifies indexing and retrieval logic


Since abstracts are no longer systematically truncated in PubMed, each record can be assumed to represent a complete semantic unit.

Why text-embedding-3-small

For a training-focused PoC, the key requirements are:

Low cost
Low latency
Predictable, stable behavior


Given the compact and domain-consistent nature of PubMed abstracts, text-embedding-3-small provides sufficient semantic quality for:

Abstract-level similarity search
Natural-language question retrieval


Using the “small” model also makes it easier to:

Iterate rapidly
Compare embedding impact later (e.g., small vs large)
Demonstrate that retrieval quality depends on more than just embedding size



Considered Alternatives
Chunking Options

Section-based chunking (Background / Methods / Results / Conclusion)

✅ Fine-grained retrieval
❌ More complex; fragile for unstructured abstracts; unnecessary for PoC


Fixed-length or sliding-window chunking

✅ Uniform processing
❌ Breaks semantic units; increases redundancy; higher complexity



Embedding Models

text-embedding-3-large

✅ Better at subtle semantic distinctions
❌ Higher cost; not required for basic PoC goals


OSS models (bge-large-en, e5-large-v2)

✅ No API dependency; strong retrieval quality
❌ Additional infrastructure and operational overhead


Biomedical-specialized embeddings (BioSentVec, PubMedBERT, SapBERT)

✅ Strong domain alignment and MeSH awareness
❌ Higher implementation and maintenance cost; overkill for training PoC



Consequences
Positive

Very simple ingestion and indexing pipeline
Clear mental model for learners (“one paper → one vector”)
Easy to demonstrate:

Query formulation effects
Role of reranking
Impact of swapping embedding models



Trade-offs

Extremely long or highly multi-topic abstracts may lose fine-grained resolution
Highest possible precision is not the main goal at this stage

Future Extensions

Add reranking (cross-encoder or LLM-based) to improve precision@1
Introduce hybrid retrieval (dense + keyword / MeSH-based filtering)
Revisit chunking when expanding to full-text articles

Implementation Notes

Indexing
Plain TextTitle: <title>Abstract: <abstract text>MeSH: <mesh_term_1>; <mesh_term_2>; ...Show more lines

Retrieval

Use natural-language queries (rather than keyword lists)
Retrieve top‑k (e.g., k=10) candidates via vector search
Apply optional reranking before generation


Evaluation

Compare small vs large using the same query set
Metrics: hit@k, MRR, qualitative answer inspection



References

PubMed / NLM abstract length policies and truncation history
Large-scale empirical analysis of PubMed Central abstract lengths
NLM documentation on structured abstracts (IMRAD-style)
