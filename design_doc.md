
Below is a **design document** for the benchmark-builder app you described. It’s organized around **SOLID**, a clean separation of **/UI** and **/Benchmark**, and a Streamlit workflow that goes **paper-by-paper**, starting with **3 questions per paper**, with **accept -> verify** and persistent JSON files.

---

# Design Document: RAG Benchmark Builder (Streamlit + SOLID)

## Goals

Build a semi-automated benchmark construction tool that:

* Ingests a corpus of PDF papers.
* Chunks each paper deterministically into **300-token chunks with 60-token overlap**.
* Stores chunked text to disk in a structured layout under `rag_corpus_chunked/`.
* Generates **3 candidate questions per paper**, proceeding **paper-by-paper**.
* Retrieves a **FAISS top-k** set of evidence chunks for each question during verification.
* Proposes candidate supporting chunks (gold candidates), then requires **human verification** per question.
* Automatically assigns a difficulty class (single-hop / multi-hop / etc.) with ability to edit.
* Produces a benchmark dataset ready for retrieval and RAG evaluation.
* Leaves room to later add **cross-paper synthesis** (not implemented yet, but designed for).

Non-goals (for now):

* Automatic “final” gold truth without human review.
* Cross-paper question generation.

---

## Repository Structure

```
project_root/
  app.py
  requirements.txt
  .env.example

  UI/
    pages/
      1_ingest.py
      2_question_generation.py
      3_verify_questions.py
    components/
      paper_selector.py
      chunk_viewer.py
      question_editor.py
      evidence_picker.py
      difficulty_editor.py
    state/
      session_state.py

  Benchmark/
    config.py
    domain/
      models.py
      enums.py
    ingestion/
      pdf_loader.py
      text_cleaner.py
      chunker.py
      chunk_store.py
    embedding/
      embedder.py
      vector_index.py
    generation/
      question_generator.py
      evidence_proposer.py
      difficulty_classifier.py
    verification/
      verifier.py
      audit_log.py
    persistence/
      benchmark_store.py
      schema_migrations.py
    services/
      pipeline.py
      paper_service.py
      question_service.py
      retrieval_service.py

  data/
    rag_corpus_pdf/            # raw PDFs (input)
    rag_corpus_text/           # extracted per-paper text (optional cache)
    rag_corpus_chunked/        # REQUIRED output format
    faiss_rag_index/           # chunks.faiss + metadata for retrieval
    unverified_questions.json  # accepted questions awaiting verification
    verified_questions.json    # finalized verified questions
    question_id_counter.txt    # global sequential id counter
```

Key rule: **UI never does backend logic directly.** UI calls **services** in `/Benchmark/services`.

---

## Storage Layout Requirements

### Chunk storage (required)

`rag_corpus_chunked/<paper_id>/chunk_<chunk_id>.txt`

Example:

```
rag_corpus_chunked/
  AttentionIsAllYouNeed/
    AttentionIsAllYouNeed_chunk_0000.txt
    AttentionIsAllYouNeed_chunk_0001.txt
  RAGAS_Paper/
    RAGAS_Paper_chunk_0000.txt
```

Chunk IDs should be deterministic:

* `chunk_id = <paper_id>_chunk_<4-digit index>`
* token window 300, overlap 60

### Question records (JSON)

Two main artifacts:

1. `unverified_questions.json` (accepted from generation, pending verify)
2. `verified_questions.json` (human-verified, final)

Record shape (domain model):

```json
{
  "question_id": "q_000123",
  "paper_id": "string",
  "question_text": "string",
  "source_paper_id": "string",
  "ground_truth": "",
  "golden_chunk_ids": ["..."],
  "difficulty": "single_hop|multi_hop|definition|comparison|negative",
  "date_created": "...",
  "notes": "..."
}
```

---

## Core Workflow (Paper-by-Paper)

### Step 1: Ingest + Chunk

* User selects input folder `data/rag_corpus_pdf/`.
* For each PDF:

  1. Extract text
  2. Clean text
  3. Chunk into 300/60
  4. Write chunks to `rag_corpus_chunked/<paper_id>/...`
  5. Update a corpus manifest (paper_id, file hash, chunk count, timestamps)

### Step 2: Generate 3 Questions per Paper (sequential)

* UI shows current paper (start at paper #1).
* Backend generates exactly **3 candidate questions** (initially).
* Questions are shown in blocks: question text, decline feedback, `Decline and regenerate`, `Accept`.
* `Accept` appends to `data/unverified_questions.json` and removes the item from generation display.

### Step 3: Evidence Suggestion (FAISS retrieval in Verify)

For each question:

* Verify page queries the persisted FAISS index.
* Retrieve top-k chunks ordered by relevance (k is configurable in UI, default 20).
* User checks candidate chunks for gold selection.

### Step 4: Gold Chunk Ordering at Verify

* On `Verify`, app opens a popup modal listing selected chunk ids.
* User drags selected chunks from most important to least.
* Popup includes dropdowns for each selected chunk to inspect chunk text.
* Stored `golden_chunk_ids` order matches this final drag order.

### Step 5: Difficulty Classification (auto + editable)

* Auto classifier assigns `difficulty_auto`.
* UI presents difficulty as editable dropdown.

### Step 6: Ground Truth Authoring

* Verify page includes an editable `Ground truth answer` text area.
* If at least one chunk is selected, user can click `Generate answer`.
* Generated answer uses selected chunk text + question as context and remains editable.

### Step 7: Verification Gate

User must explicitly mark each question:

* `Verify` appends to `data/verified_questions.json` (including `ground_truth`) and removes from `unverified_questions.json`.
* `Reject` removes from `unverified_questions.json`.
* `Needs revision` keeps question in `unverified_questions.json`.

---

## SOLID-Oriented Backend Design

### Domain Layer (`Benchmark/domain`)

* `Paper`, `Chunk`, `Question`, `EvidenceCandidate`, `BenchmarkRecord`
* Enums: `QuestionStatus`, `DifficultyLabel`

**Single Responsibility:** Domain objects are data + validation only.

### Ingestion Layer (`Benchmark/ingestion`)

* `PdfLoader` (extract text)
* `TextCleaner` (normalize whitespace, remove headers/footers if needed)
* `Chunker` (300/60 token chunking)
* `ChunkStore` (writes chunk text files to disk)

**Open/Closed:** Chunker can support new strategies without changing UI.

### Embedding/Retrieval (`Benchmark/embedding`)

* `Embedder` interface

  * `OpenAIEmbedder` implementation
* `VectorIndex` interface

  * `InMemoryVectorIndex` (for per-paper retrieval)
  * later: `FaissVectorIndex`

**Dependency Inversion:** services depend on interfaces, not implementations.

### Generation (`Benchmark/generation`)

* `QuestionGenerator` (3 questions per paper)
* `EvidenceProposer` (LLM chooses chunk IDs from candidate list)
* `DifficultyClassifier` (LLM or heuristic classifier)

### Services (`Benchmark/services`)

Orchestrate the pipeline:

* `PaperService`: list papers, load text, manage current paper cursor
* `QuestionService`: generate/save/edit questions
* `RetrievalService`: run FAISS top-k retrieval for verification
* `PipelineService`: end-to-end per-paper processing

**Interface Segregation:** UI calls small methods (e.g., `get_current_paper()`, `generate_questions(paper_id, n=3)`).

### Persistence (`Benchmark/persistence`)

* `BenchmarkStore` keeps in-memory records used by pipeline runtime
* `AuditLog` records append-only in-memory events:

  * edits, verification actions, overrides, timestamps
* `QuestionIdAllocator` produces global sequential ids shared across unverified + verified files
* `UnverifiedQuestionStore` and `VerifiedQuestionStore` append/remove JSON rows in `data/`

---

## Streamlit UI Design

### Page 1: Ingest

* Choose corpus folder
* Ingest status table
* “Chunk all” button
* Preview: show extracted text + sample chunks

### Page 2: Question Generation (paper-by-paper)

* Paper selector
* “Generate 3 questions”
* Auto-resizing question text areas
* Per-question controls:
  * decline feedback input
  * `Decline and regenerate` (feedback-aware regeneration)
  * `Accept` (writes to `unverified_questions.json`, hides accepted question)

### Page 3: Verify Questions (file-backed)

* Loads questions from `data/unverified_questions.json` (one question shown at a time)
* Runs FAISS top-k retrieval for the current question
* Candidate checkboxes + chunk text
* Difficulty dropdown + Notes field
* Ground truth section:
  * editable answer text area
  * `Generate answer` button (enabled when chunks are selected)
* Buttons: `Verify`, `Needs revision`, `Reject`
* `Verify` opens ordering popup and persists ordered chunk ids + ground truth into `verified_questions.json`

---

## Retrieval Specification

Verification retrieval uses:

1. Query embedding via OpenAI embedding model
2. FAISS search on prebuilt chunk index
3. Top-k ranked chunks (UI-configurable, default 20)

This keeps verification focused and consistent across papers.

---

## Models (Cheap OpenAI Choices)

* Question generation / evidence proposing / difficulty classification + ground-truth generation:

  * `gpt-4o-mini` (cheap, strong for structured labeling)
* Embeddings:

  * `text-embedding-3-small` (cheap, good baseline)

(Models are injected via config so you can swap later.)

---

## Config (Centralized)

`Benchmark/config.py`

* chunk_size_tokens = 300
* chunk_overlap_tokens = 60
* questions_per_paper = 3
* retrieval_top_k = 8
* retrieval_threshold = 0.15
* retrieval_cap = 25
* models + temperatures
* corpus and cache directory paths

---

## Future Extension: Cross-Paper Synthesis (Not Designed Yet)

We’ll reserve extension points:

* `QuestionGenerator` gains a `generate_cross_paper_questions(paper_ids: list[str])`
* `RetrievalService` supports global retrieval across papers
* UI can add a “Synthesis Mode” page later

But no implementation now.

---

## Acceptance Criteria

* [ ] Chunked files are written exactly as specified under `rag_corpus_chunked/<paper_id>/...`
* [ ] Streamlit supports sequential paper-by-paper workflow
* [ ] Exactly 3 questions generated per paper by default
* [ ] Accepted questions are persisted to `data/unverified_questions.json`
* [ ] Verified questions are persisted to `data/verified_questions.json`
* [ ] Ground truth text is persisted to `verified_questions.json`
* [ ] Difficulty label is auto-assigned and user-editable
* [ ] Verify uses FAISS top-k retrieval (default 20, configurable)
* [ ] Gold chunk order in `verified_questions.json` matches verify popup drag order

---

If you want, next I can turn this design doc into:

* a concrete **file-by-file scaffold** (empty modules + interfaces + dataclasses),
* plus the minimal Streamlit pages wired to services (no heavy logic in UI),
* plus a JSONL schema and an audit log format you can keep stable long-term.
