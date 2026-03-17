# Capstone Evaluation Checklist

## 1. Filesystem & Documentation

- [ ] **Clear Folder Structure:**
  - `/requirements`: Original project requirements and specs
  - `/docs/architecture`: High-level system design diagrams
  - `/docs/data-flow`: Specific logic flows (Ingestion, Retrieval)
  - `/src`: Production implementation (structured by service/domain)
  - `/tests`: Comprehensive testing suite (API, Performance)
  - `README.md`: Clear installation and evaluation guide

- [ ] **Stakeholder PPT:** A briefing deck summarizing EDA, Design, Decisions, and Evaluation

- [ ] **Readme Consistency:** Does the code and structure actually follow the Readme?

## 2. Architecture & Design Integrity

- [ ] **Architecture vs. Data Flow Distinction:**
  - **Architecture:** Highlights software components, microservices, and caches
  - **Data Flow:** Shows logical movement (e.g., retrieval sequence). Do not confuse the two

- [ ] **Production Scale Deployment:**
  - [ ] Does the architecture discuss API Gateways, Load Balancers, and Kubernetes (K8s)?
  - [ ] Clear distinction between what is in POC vs. Production

- [ ] **Observability & MLOps:** Architecture explicitly includes monitoring, logging, and ML lifecycle layers

- [ ] **Design Decisions:**
  - [ ] ADRs included explaining Pros/Cons for major choices (e.g., `DECISIONS.md`)
  - [ ] Trade-offs explicitly highlighted (e.g., Accuracy vs. Performance)
  - [ ] Decoupling: Can you swap your Vector DB (e.g., FAISS to Milvus) without rewriting the core API?

## 3. Implementation & Code Quality

- [ ] **Production Grade Code:**
  - [ ] Zero `print()` Statements: 100% usage of structured logging (JSON preferred)
  - [ ] Clean Code: No hardcoded secrets, and no monolithic "God" files
  - [ ] Microservices Representation: Is the microservice architecture reflected in code boundaries/packages?
  - [ ] Connection Pooling: Always used for DB and downstream services (zero per-request TCP overhead)
  - [ ] Input Validation: Are API inputs/outputs validated using a schema (e.g., Pydantic)?

- [ ] **Containerization:** Working `Dockerfile` and `docker-compose.yml` for a "one-command" startup

- [ ] **Resource Management:**
  - [ ] Memory-efficient processing (streaming/generators)
  - [ ] Cold Start Optimization: Are models and indices loaded at startup, not on the first request?

- [ ] **Error Handling:** Does the system handle missing files, empty data, or API timeouts gracefully?

## 4. Testing & Validation (Accuracy)

- [ ] **API Testing:** Automated tests for both Loading (Ingestion) and Retrieving (Search)

- [ ] **Performance Measurement:** Monitoring latency (p99) and throughput for retrieval

- [ ] **Accuracy Validation:**
  - [ ] Clearly defined methodology for validating retrieval accuracy (e.g., LLM-as-Judge or custom rubrics)
  - [ ] Ground Truth Dataset: Is the data used for self-evaluation provided and documented?
  - [ ] Metrics Summary: Does documentation include a summary of evaluation results (Latency & Accuracy)?

- [ ] **ML Resiliency:**
  - [ ] Local Fallback: Working code for a local model (Flan-T5/Transformers) if OpenAI times out
  - [ ] Graceful Degradation: System returns fallback results (e.g., keyword-only) if vector indexing fails

## 5. Verification: The Reviewer's Lens (SME Checklist)

| Dimension | "No" Verdict | "Borderline" Verdict | "Yes" Verdict |
|-----------|--------------|----------------------|---------------|
| **Correctness** | Bugs in core flow / Brittle logic | Functional, but brittle | Robust, handles edge cases |
| **Architecture** | Monolithic / Spaghetti code. No clear separation of concerns | Layered but tightly coupled (e.g., logic in routers) | Modular Microservices. Clean separation of DB, AI, and API layers |
| **Design Decisions** | Ad-hoc technology choice without reasoning | Choosing "popular" techs but unable to explain trade-offs | ADRs included. Clear justification of cost vs. latency vs. complexity |
| **Performance** | High-latency LLM calls for every trivial task. No caching | Basic in-memory caching. Single-threaded ingestion | Optimized Pipelines. SLMs for latency, Parallel ingestion, distributed caching |
| **Testing** | Manual testing only. No scripts | Basic unit tests for utility functions only | Comprehensive Suite. Unit, Integration, and Load tests (locust/custom) |
| **Evaluation** | None (Manual checks) | Script-based logs | Automated Rubrics. LLM-as-Judge or IR metrics (NDCG/MAP) |
| **Scalability** | Global shared state / No pooling | Vertical scaling only | Cloud-Ready. Stateless, horizontally scalable, connection pooled |
| **Reliability** | No error handling / No fallbacks | Basic try/except | Resilient. Retry logic, Local fallbacks (T5), and circuit breakers |
| **Maintainability** | Hardcoded keys / Improper layout | Clean but lacking documentation | Self-Documenting. Clean code, logic grouping, and Dockerized setup |
| **Observation** | `print()` statements | Basic log output | Production-Ready. Structured logs & health checks |

## 6. Final Benchmarking (SME 'Yes' Grade)

- [ ] **Core Flow:** System is robust and handles "sad path" edge cases (empty data, API timeouts)

- [ ] **Architecture:** Clear separation between DB, AI, and API layers in a modular/service-oriented layout

- [ ] **Design Decisions:** ADRs provide clear justification for tech choices based on cost, latency, and complexity

- [ ] **Performance:** Ingestion is parallelized, and high-frequency tasks use optimized SLM models

- [ ] **Testing:** Includes a comprehensive suite of Unit, Integration, and Load tests

- [ ] **Evaluation:** Quality is measured using automated rubrics (LLM-as-Judge) or IR metrics

- [ ] **Scalability:** Application is stateless, horizontally scalable, and utilizes connection pooling

- [ ] **Reliability:** Implements retry logic, local model fallbacks, and circuit breakers

- [ ] **Maintainability:** Code is self-documenting, grouped logically, and fully Dockerized

- [ ] **Observation:** Production-ready structured logs and functional health check endpoints
