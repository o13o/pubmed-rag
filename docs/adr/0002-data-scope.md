ADR: MVP Corpus Scope — 5‑Year English Subset with Abstracts Only, Year‑Stratified Sampling, and MeSH Minimum Coverage
Status: Accepted
Date: 2026‑03‑13
Owner: Yasuhiro Okamoto
Context
For this PoC, we need a PubMed/MEDLINE‑based corpus that demonstrates latest‑evidence search with time filters, hybrid retrieval, and RAG, while staying within tight compute and storage limits. PubMed’s annual growth is high (≈1.5–1.7M new citations/year in recent fiscal years), which makes ingesting all records from many years impractical for an MVP on constrained hardware. [nlm.nih.gov]
To keep time filters meaningful and preserve a realistic “recency” experience, we prefer a recent publication window over random slices across many years. Publication‑year totals can be verified programmatically via NLM’s open dataset “PubMed total records by publication year” (CSV/JSON API), which provides year‑level counts aligned to the annual baseline; we will use this for sanity checks when computing year‑wise sampling quotas. [datadiscov...lm.nih.gov], [catalog.data.gov]
Not all PubMed records include abstracts, and focusing on abstract‑bearing records both reduces volume and improves RAG quality (there is documented evidence in NLM baseline statistics distinguishing title vs. abstract counts/lengths). [huggingface.co]
Finally, the ingestion process must be repeatable (annual baseline + optional updates) and scalable (easily dial corpus size up from 100k to 300k or to full ingestion later). NLM’s baseline and daily update distribution model supports this operational pattern. [nlm.nih.gov], [healthdata.gov], [data.virginia.gov]
Decision

Time Window: Restrict the MVP corpus to the most recent 5 publication years (e.g., 2021–2025). [datadiscov...lm.nih.gov]
Language: English‑language records only.
Content Requirement: Abstract present (non‑empty) — records without abstracts are excluded. [huggingface.co]
Target Size (Cap): N_MAX = 100,000 abstracts for the MVP due to hardware limits. The pipeline must allow increasing this cap (e.g., 300,000) or disabling sampling for full ingestion without changing logic.
Sampling Method: Year‑stratified random sampling with equal allocation (i.e., N_MAX / 5 per year).
Minimum Coverage per Year (Topic Diversity): Enforce a minimum of MIN_PER_CAT = 500 abstracts per MeSH top‑level disease category (see list below) per year. If a category/year lacks sufficient candidates, select all available and log the shortfall; do not fail the run.
MeSH Categories (10):

Neoplasms
Cardiovascular Diseases
Infectious Diseases
Nervous System Diseases
Respiratory Tract Diseases
Digestive System Diseases
Urogenital Diseases
Musculoskeletal Diseases
Nutritional and Metabolic Diseases
Immune System Diseases


Reproducibility: Use a fixed random seed and write an audit log (effective config, year‑wise population & selected counts, per‑category coverage, sampled PMIDs).
Operational Source: Prefer annual baseline files for initial loads; optionally apply daily update files after baseline for freshness, following NLM’s prescribed order and replacement semantics. [nlm.nih.gov], [healthdata.gov], [data.virginia.gov]

Rationale

Clinical relevance & demo value: A recent 5‑year window aligns with typical medical‑literature behaviors (recency bias) and makes time filters (e.g., “since 2022”) meaningful and easy to explain in demos. [nlm.nih.gov]
Volume control & quality: Limiting to English + abstracts‑only reduces size while keeping high‑quality text for embeddings and RAG. NLM baseline statistics clearly show differences between title vs. abstract availability/lengths. [huggingface.co]
Representativeness: Equal per‑year allocation prevents year imbalance; MeSH minimums prevent topic collapse caused by pure random sampling.
Scalability: A single N_MAX knob (and MIN_PER_CAT) lets us scale from 100k → 300k → all without re‑architecting.
Measurable & auditable: Publication‑year totals are available via an official NLM dataset, enabling sanity checks and transparent reporting. [datadiscov...lm.nih.gov], [catalog.data.gov]

Considered Alternatives

All records from last 5 years (no sampling): Best recall/coverage but ~7.9M new citations over 5 recent fiscal years is too heavy for the MVP hardware/time budget. [nlm.nih.gov]
Random sample from the last 10 years: Simpler, but weakens time‑filter semantics and undermines a “latest research” story.
Single specialty full coverage (e.g., oncology only): Smaller and focused, but reduces generality of the platform demo and might bias evaluation.

Consequences
Positive

Keeps time filters and “latest evidence” narratives intact.
Controls ingestion, embedding, and indexing costs for MVP.
Ensures topic diversity through MeSH minimums; reduces variance across runs via seed and audit logs.

Trade‑offs

Some rare topics may still be underrepresented within a given year if the source has very few abstracts.
Long‑term trend analysis beyond 5 years is out of scope for MVP (can be added later by raising N_MAX and/or widening the window).

Implementation Notes

Configuration (example):

years: [2021, 2022, 2023, 2024, 2025]
language: "eng"
require_abstract: true
sampling.n_max: 100000 (change to 300000 or more for scale‑up)
sampling.seed: 42
sampling.allocation: "equal_per_year"
sampling.min_coverage.enabled: true
sampling.min_coverage.per_category_per_year: 500 (tunable)
sampling.min_coverage.mesh_categories: [10 items above]


Order of operations:

Filter by publication year → 2) language=eng → 3) abstract present → 4) compute per‑year populations (optionally cross‑check with NLM year totals dataset) → 5) enforce per‑category minimums per year → 6) fill remaining per‑year quota randomly → 7) (optional) top‑up to N_MAX across years → 8) write audit artifacts. [datadiscov...lm.nih.gov], [catalog.data.gov]


Data sources & refresh:

Use annual baseline as the authoritative starting point; if freshness is needed, apply daily update files strictly after loading all baseline files, as per NLM guidance. [nlm.nih.gov], [healthdata.gov], [data.virginia.gov]


Storage planning (heads‑up): If using 1536‑dim embeddings (e.g., modern API models), raw vector size ≈ 6 KB/record (float32). With 100k records, raw vectors ≈ 0.6 GB; index + metadata typically doubles/triples this. (This is a planning note; not sourced from NLM.)

Metrics & Evaluation

Retrieval: nDCG@10, Recall@50 — reported per year and aggregated.
RAG correctness: Citation consistency (claim ↔ quoted snippet ↔ PMID) and time consistency with query filters.
Diversity checks: Per‑year MeSH coverage vs. targets; log shortfalls where categories lack sufficient abstracts.

References

Annual volume and growth: NLM MEDLINE/PubMed Production Statistics (FY2018–FY2023; shows ≈1.3–1.7M PubMed citations added per year). [nlm.nih.gov]
Publication‑year totals dataset (API): “PubMed total records by publication year” (CSV/JSON). Use for year‑wise sanity checks and reporting. [datadiscov...lm.nih.gov], [catalog.data.gov]
Abstract availability/length indicators: MEDLINE/PubMed Baseline Statistics — Misc report (title vs. abstract counts/lengths and related field stats). [huggingface.co]
Operational guidance: PubMed Baseline (2025) release note and baseline/daily update instructions (load baseline first, then ordered updates). [nlm.nih.gov], [healthdata.gov], [data.virginia.gov]

