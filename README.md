# TriageAI

TriageAI is a supervisor-led, multi-agent complaint intelligence system for financial services. It does not stop at intake or classification. The codebase implements a full complaint operating model: conversational intake, document ingestion and OCR, agent orchestration, risk and root-cause analysis, compliance checks, routing, resolution planning, live traces, and evaluation infrastructure.

This repository is meant to show depth. The architecture on the home page is not a concept graphic. It maps directly to the workflow, storage, retrieval, observability, and evaluation code in this repo.

## Why This Repo Is Different

Most complaint demos stop at "classify a complaint with an LLM."

TriageAI goes further:

- A central supervisor coordinates specialist agents instead of running one monolithic prompt.
- Uploaded evidence is processed before downstream analysis, including OCR for scanned PDFs and images.
- Complaint decisions are persisted as workflow runs and workflow steps with token, latency, and cost rollups.
- Historical complaints and company knowledge are retrievable through PostgreSQL + pgvector.
- Production complaints can be evaluated after execution, and benchmark datasets can be materialized and scored.
- The product includes admin-facing visibility: traces, analytics, evaluation views, and case detail pages.

## Architecture

The homepage architecture is the system architecture:

```mermaid
flowchart TD
    U[Complaint Intake<br/>chat or voice] --> D[Document Gate]
    D --> C[Document Consistency Check]
    C --> S[Supervisor]

    S --> CL[Classification Agent]
    S --> RK[Risk Agent]
    S --> CP[Compliance Agent]
    S --> RC[Root Cause Agent]
    S --> RT[Routing Agent]
    S --> RV[Review Agent]
    S --> RS[Resolution Agent]

    CL --> S
    RK --> S
    CP --> S
    RC --> S
    RT --> S
    RV --> S
    RS --> S

    S --> O[Persisted Outcome<br/>case status, route, severity]
    O --> T[Live Trace + Analytics + Evaluation]
```

### Supervisor-Led Agentic Flow

- `intake` normalizes and validates the case payload.
- `document_gate` waits for uploaded evidence to finish background processing.
- `check_document_consistency` compares the complaint narrative with extracted document facts.
- `supervisor` decides which specialist to run next.
- Specialists cover classification, risk, compliance, root cause, routing, review, and resolution.
- Final outputs are persisted with workflow metadata for later inspection.

The implementation lives in [app/orchestrator/workflow.py](app/orchestrator/workflow.py).

## What Exists In Code

### 1. Evidence-First Complaint Processing

This system treats uploaded evidence as a first-class input, not an attachment afterthought.

- PDF text extraction for digital documents
- OCR for scanned PDFs via `pdftoppm` + `tesseract`
- OCR for screenshots and images
- fact extraction for amounts, dates, reference numbers, and signals
- document chunking and embeddings for downstream retrieval
- document-vs-narrative contradiction checks before agent reasoning

See [app/documents/service.py](app/documents/service.py).

### 2. Specialist Agents, Not One Giant Prompt

The system has separate agents and schemas for:

- intake
- classification
- risk
- root cause
- resolution
- compliance
- review
- routing

This is backed by dedicated prompts, structured schemas, and a supervisor router across the `app/agents`, `app/prompts`, and `app/schemas` packages.

### 3. Retrieval Backed By Real Infrastructure

Retrieval is not mocked behind an in-memory demo.

- complaint similarity search is backed by PostgreSQL + pgvector
- embeddings support local HuggingFace models or OpenAI
- retrieval can surface historical complaint patterns and company context

See [app/retrieval/complaint_index.py](app/retrieval/complaint_index.py) and [app/retrieval/ingest.py](app/retrieval/ingest.py).

### 4. Observability Built Into The Workflow

The repo includes a real forensic layer for AI execution:

- workflow runs persisted to Postgres
- per-step snapshots and state diffs
- trace IDs and OpenTelemetry integration
- token, latency, and cost accounting
- version tracking for workflow, prompts, knowledge pack, and model
- admin UI for live trace inspection

See [app/observability/persistence.py](app/observability/persistence.py), [app/observability/tracing.py](app/observability/tracing.py), and [app/templates/trace.html](app/templates/trace.html).

### 5. Evaluation Infrastructure, Not Just Manual Spot Checks

This repo includes both production evaluation and benchmark-style evaluation:

- database-backed evaluation datasets
- weak-gold label generation
- rubric-based LLM judge runs
- disagreement queues for human review
- production case evaluation against real workflow outputs

See [app/evals/service.py](app/evals/service.py) and [app/evals/judge.py](app/evals/judge.py).

### 6. Product Surface Beyond The API

The repo includes a server-rendered product, not only backend endpoints.

- public-facing marketing and home pages
- end-user lodge flow with conversational intake
- optional voice intake
- admin dashboards
- analytics and evaluation pages
- team and queue views
- case detail and resolution history pages

The UI templates live under [app/templates](app/templates).

## Repository Map

```text
app/
  agents/          specialist agents, supervisor, tools, LLM helpers
  api/             FastAPI endpoints and intake integrations
  db/              ORM models and session management
  documents/       upload persistence, OCR, extraction, summaries
  evals/           benchmark runners, judge, review services
  knowledge/       company knowledge and taxonomy context
  observability/   tracing, event logging, versioning, cost tracking
  orchestrator/    LangGraph workflow, rules, retrieval gates, state
  retrieval/       embeddings, ingest, pgvector-backed indexes
  schemas/         structured outputs for agents and cases
  templates/       product UI and admin views
  ui/              server-rendered routes and page context
tests/
architecture.md    deeper knowledge-base and regulatory architecture
```

## Product Highlights

- Supervisor-led LangGraph workflow
- complaint intake via chat with optional voice mode
- document-aware complaint processing
- scanned-PDF and image OCR
- complaint similarity retrieval with pgvector
- risk, root-cause, compliance, and routing agents
- live workflow trace UI
- production evaluation reports
- benchmark dataset and judge infrastructure
- website-friendly case IDs such as `CASE00001`

## Tech Stack

- Python 3.11+
- FastAPI
- Jinja2
- SQLAlchemy
- PostgreSQL
- pgvector
- LangGraph / LangChain
- OpenTelemetry
- OpenAI or DeepSeek
- optional ElevenLabs voice output

## Running Locally

### Prerequisites

- Python 3.11+
- PostgreSQL with `pgvector`
- one LLM provider configured: OpenAI or DeepSeek
- `tesseract` and `poppler` for OCR

Recommended:

- `uv`
- Docker

### Environment

```bash
cp .env.example .env
```

Minimum variables:

```env
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/complaints
LLM_PROVIDER=openai
OPENAI_API_KEY=...
```

DeepSeek option:

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
```

Useful optional variables:

- `OPENAI_CHAT_MODEL`
- `DEEPSEEK_CHAT_MODEL`
- `EMBEDDING_PROVIDER`
- `HF_EMBEDDING_MODEL`
- `TRACE_INTAKE_TO_LANGSMITH`
- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_PROJECT`
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`

### Install

With `uv`:

```bash
uv sync
```

With `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### OCR Dependencies

macOS:

```bash
brew install tesseract poppler
```

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

### Database

```bash
docker compose up db -d
```

### Start The App

```bash
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

For a full local stack:

```bash
docker compose up --build -d
docker compose logs -f app
```

Historical workflow cost aggregates are backfilled on startup. You can also run that manually:

```bash
python3 scripts/backfill_cost_ledger.py
```

## Voice Intake

The lodge flow supports chat and optional voice mode.

- speech-to-text uses the browser Web Speech API
- spoken responses can use ElevenLabs when configured
- microphone access requires `localhost` or HTTPS

For local HTTPS:

```bash
./scripts/dev_https.sh
```

By default this serves `https://127.0.0.1:8001`. The certificate setup notes are in [scripts/dev_https.sh](scripts/dev_https.sh).

## Demo Accounts

Admin:

- email: `admin@triage.ai`
- password: `admin123`

End user:

- email: `user@triage.ai`
- password: `user123`

Team users:

- seeded automatically
- password pattern: `<local-part>123`
- reference: [Team Credentials](https://github.com/ayman-tech/Multi-Agent-Complaint-System/wiki/Team-Credentials)

## Further Reading

- [architecture.md](architecture.md)
- [repository_architecture.pdf](repository_architecture.pdf)
- [repository_architecture_detailed.pdf](repository_architecture_detailed.pdf)

## License

No license file is currently included in this repository. Treat usage and redistribution as private unless an explicit license is added.
