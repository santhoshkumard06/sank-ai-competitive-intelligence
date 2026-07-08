# Sank — AI Competitive Intelligence Analyst

> **Be10X AI Generalist Hackathon B15 · June 2026**

Sank is a multi-agent AI pipeline, orchestrated by n8n, that continuously monitors public competitor signals — changelogs, pricing pages, review sites — and delivers a daily severity-ranked intelligence briefing to your inbox. Instead of a PM spending 3–5 hours/week manually checking competitor pages, Sank's three AI agents do it automatically every morning.

**Result: 3–5 hours/week → 5-minute email, every weekday at 9am.**

---

## The Problem

Product managers and founders at SMB SaaS companies (1–200 employees) spend roughly 3–5 hours a week manually tracking what competitors are shipping — scattered across browser tabs, with no consistent record. By the time it's written up, it's stale. Enterprise tools like Klue and Crayon solve this, but they're priced as annual enterprise contracts. Sank fills that gap.

---

## Architecture

```
n8n Schedule Trigger (daily 9am)
        │
        ▼
Execute Sank Pipeline
        │
        ▼
Agent 1 — Signal Extraction
(reads unstructured changelog text → structured signal)
        │
        ▼
Semantic RAG Matching
(embeds signal → retrieves nearest roadmap item from vector store)
        │
        ▼
Agent 2 — Threat Scoring
(judges severity + reason against your specific roadmap)
        │
        ▼
Agent 3 — Briefing Writer
(ranks all signals → writes plain-English digest)
        │
        ▼
n8n Gmail Node → HTML email digest delivered to inbox
```

---

## Why This Requires AI (Not Just Rules or RSS)

A competitor shipping "one-click onboarding" and your roadmap saying "reduce setup friction" share **zero keywords** — a keyword alert or RSS reader misses the threat entirely. Sank uses **Retrieval-Augmented Generation (RAG)**:

1. **Retrieve**: embed the extracted signal, query the vector store for the nearest roadmap item by *meaning*, not words
2. **Augment**: inject the matched roadmap item into the scoring agent's context
3. **Generate**: the agent reasons about severity and produces an explanation specific to *your* roadmap

This judgment step — deciding whether a change matters for you specifically — is impossible to fake with rules.

---

## AI Stack

| Component | Technology |
|---|---|
| LLM (all 3 agents) | Google Gemini 2.5 Flash (free tier, 1,500 req/day) |
| Vector store (demo) | Local TF-IDF (scikit-learn, zero dependencies) |
| Vector store (production) | Pinecone — index pre-built, swap one constructor |
| Orchestration | n8n (schedule + execute + email delivery) |
| Retry / fallback | tenacity — exponential backoff, auto-fallback to gemini-2.5-flash-lite |

---

## Project Layout

```
src/sank/
  models.py         Data contracts (Entity, Signal, Digest — domain-agnostic)
  config.py         Loads YAML into validated typed objects
  fetch.py          Network fetch + HTML cleaning
  llm_client.py     GeminiLLMClient + AnthropicLLMClient + MockLLMClient
  vector_store.py   LocalVectorStore (TF-IDF) + PineconeVectorStore
  agents.py         ExtractionAgent · ScoringAgent · BriefingAgent
  pipeline.py       Orchestrates all agents; isolates per-source failures
  cli.py            sank run / sank validate
config/
  domains/
    competitive_intelligence.yaml   Agent prompts for competitor tracking
    creator_comparison.yaml         Same pipeline, different domain (proof of generality)
  watchlist.example.yaml            Lovable · Cursor · Replit
  reference_corpus.example.yaml     8-item sample product roadmap
n8n/
  sank_n8n_workflow.json            Importable n8n workflow (schedule → run → email)
tests/                              85 tests, 0 failures — run with: pytest
```

---

## Setup

### 1. Python pipeline

```bash
git clone <your-repo-url>
cd sank
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Add your GEMINI_API_KEY to .env (free at aistudio.google.com)
pytest          # 85 passed, 0 failed
sank validate --watchlist config/watchlist.example.yaml
sank run --watchlist config/watchlist.example.yaml \
         --reference config/reference_corpus.example.yaml \
         --domain config/domains/competitive_intelligence.yaml
```

### 2. n8n automation (daily email delivery)

```bash
npm install -g n8n
n8n start
# Open http://localhost:5678
# Import: n8n/sank_n8n_workflow.json
# Add Gmail OAuth2 credential (Settings → Credentials → New)
# Update Execute Command path if sank is not at ~/sank
# Click "Manual Test Run" → verify email arrives
# Toggle workflow Active → runs daily at 9am Mon–Fri
```

---

## Swapping in Pinecone (Production)

```python
from sank.vector_store import PineconeVectorStore

store = PineconeVectorStore(
    api_key=os.environ["PINECONE_API_KEY"],
    index_name="sank-reference",
    embed_fn=your_embed_function,   # sentence-transformers, OpenAI, Voyage — anything
    dimension=384,
)
```

Pass `store` to `run_pipeline()` instead of `LocalVectorStore()`. Nothing else changes.

---

## Adding a New Domain (1 YAML file, 0 code changes)

The word "competitor" does not appear in any Python file. The agents speak in terms of "entities" and "reference items." To point Sank at YouTube creators instead of SaaS competitors:

```bash
cp config/domains/competitive_intelligence.yaml config/domains/creator_comparison.yaml
# Edit entity_label, reference_label, and the three agent prompts
# Update watchlist.yaml to list creator URLs
# Update reference_corpus.yaml to list your content pillars
sank run --domain config/domains/creator_comparison.yaml ...
```

A working `creator_comparison.yaml` is already included.

---

## Reliability (production-honest)

Three real failures were hit during testing against the live Gemini free tier and fixed before submission:

| Issue | Fix |
|---|---|
| Gemini's "thinking" tokens consumed `max_output_tokens` budget, truncating JSON responses mid-character | `thinking_budget=0` set on every structured-output call |
| 429 rate-limit and 503 "high demand" responses under peak load | Exponential backoff, up to 6 attempts (up to 60s wait) |
| Primary model fully exhausted | Automatic fallback to `gemini-2.5-flash-lite` (separate capacity pool) |

A broken source (site blocked, rate limited) is isolated — the rest of the run completes normally.

---

## Test Suite

```bash
pytest          # 85 tests, 0 failed, ~7 seconds
pytest -v       # verbose
```

Includes regression tests for every real failure mode above, so they cannot silently return.

---

## Business Value

| Metric | Current manual process | With Sank |
|---|---|---|
| Time per week | 3–5 hours | 5 minutes (read the email) |
| Staleness | Hours to days | Same-day |
| Coverage | Whatever you remembered to check | Every configured source, every day |
| Delivery | Notes app / memory | Formatted email in your inbox at 9am |

**Market context:** Klue raised $62M, Crayon raised $45M — competitive intelligence is a proven paid category. Sank targets the SMB gap those tools don't serve.

---

## Future Roadmap

- Signal deduplication (don't re-report the same change twice)
- Slack / WhatsApp delivery via additional n8n nodes
- Historical digest archive with trend detection
- Web dashboard (Lovable / Vercel) for non-technical users
- Per-signal "counter-positioning" recommendation
