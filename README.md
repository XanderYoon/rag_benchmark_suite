# RAG Benchmarking Suite

Streamlit app for building a paper-grounded RAG benchmark with a human-in-the-loop workflow:

1. Ingest PDFs and chunk them
2. Generate candidate questions
3. Accept questions into an unverified queue
4. Verify questions with FAISS retrieval + chunk ordering
5. Persist verified benchmark entries to JSON

## Features

- PDF ingestion and deterministic chunking (`300` tokens, `60` overlap)
- Question generation with per-question decline/regenerate flow
- Global sequential question IDs shared across unverified + verified sets
- Verification page backed by `data/unverified_questions.json`
- FAISS top-k retrieval in verification (default `20`, configurable in UI)
- Drag-order popup for selected chunks before final verify
- Ground-truth authoring:
  - editable free-text answer
  - optional `Generate answer` from selected chunks

## Project Layout

- `app.py`: Streamlit entrypoint
- `UI/pages/1_ingest.py`: ingest/chunk UI
- `UI/pages/2_question_generation.py`: generate + accept flow
- `UI/pages/3_verify_questions.py`: verify flow
- `Benchmark/`: services, generation, persistence, verification logic
- `scripts/build_faiss_rag_index.py`: one-off FAISS index builder
- `data/`: runtime artifacts

## Requirements

Base app dependencies are in `requirements.txt`:

- `streamlit`
- `pypdf`
- `streamlit-sortables` (for drag ordering in verify popup)

For FAISS indexing + retrieval generation script:

- `numpy`
- `faiss-cpu`
- `openai`

## Setup

From repo root:

```bash
pip install -r requirements.txt
pip install numpy faiss-cpu openai
```

Set OpenAI key (needed for FAISS query embeddings and answer generation):

```bash
export OPENAI_API_KEY="your_api_key"
```

## Run the App

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## End-to-End Workflow

### 1) Prepare corpus

Put PDFs in:

- `data/rag_corpus_pdf/`
  or
- `rag_corpus_pdf/` at repo root

### 2) Ingest/chunk

In **Ingest** page:

- select corpus folder
- click `Chunk all`

Chunks are written to:

- `data/rag_corpus_chunked/<paper_id>/<paper_id>_chunk_XXXX.txt`

### 3) Build FAISS index (one-off script)

```bash
python3 scripts/build_faiss_rag_index.py --overwrite
```

Outputs:

- `data/faiss_rag_index/chunks.faiss`
- `data/faiss_rag_index/chunks_metadata.jsonl`
- `data/faiss_rag_index/index_manifest.json`

### 4) Generate and accept questions

In **Question Generation** page:

- click `Generate 3 questions for current paper`
- for each question:
  - `Decline and regenerate` with feedback, or
  - `Accept` to add it to unverified queue

Accepted questions are appended to:

- `data/unverified_questions.json`

### 5) Verify questions

In **Verify Questions** page:

- questions are loaded from `data/unverified_questions.json` (one at a time)
- select chunk evidence from FAISS top-k candidates
- optional `Generate answer` to draft ground truth
- edit ground truth and notes
- click `Verify`
  - reorder selected chunks in popup (most important -> least)
  - confirm

On confirm:

- entry appended to `data/verified_questions.json`
- question removed from `data/unverified_questions.json`

If `Reject`:

- question removed from `data/unverified_questions.json`

If `Needs revision`:

- question remains in `data/unverified_questions.json`

## Data Files

### `data/unverified_questions.json`

Each row includes:

- `question_id`
- `question_text`
- `paper_id`
- `default_difficulty`

### `data/verified_questions.json`

Each row includes:

- `question_id`
- `question_text`
- `source_paper_id`
- `ground_truth`
- `golden_chunk_ids` (ordered by popup ranking)
- `difficulty`
- `date_created`
- `notes`

### `data/question_id_counter.txt`

Persistent counter used to issue globally unique sequential IDs (e.g. `q_000001`) across unverified and verified sets.

## Notes

- Verify popup drag ordering requires `streamlit-sortables`.
- Ground truth generation falls back to text-based synthesis if OpenAI call is unavailable.
- FAISS retrieval requires index artifacts in `data/faiss_rag_index/`.
