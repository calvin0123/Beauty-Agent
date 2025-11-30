# Health / Beauty Agent

Multilingual research assistant that ingests beauty YouTube content, builds a searchable knowledge base, and answers product questions with citations.

## Problem Statement
- Beauty shoppers struggle to remember which products influencers actually recommend, in which order, and why.
- Transcripts are long, often bilingual (Mandarin/English), and hard to query without rewatching an entire video.
- Goal: ingest creator content, normalize the language, index it, and expose an agent that can search, cite, and recommend products in the user’s language.

## Solution Highlights
- **YouTube extraction (`service/youtube_extraction`)** – pulls channel videos, downloads transcripts, summarizes with LLMs, chunks with overlap, and stores cached JSON for fast reuse.
- **Retrieval agent (`service/agent`)** – Clarify Agent decides whether to ingest or use cache, YouTube Agent runs multilingual search (Elasticsearch + embeddings) and returns structured product recs.
- **Evaluation service (`service/evals`)** – builds ground-truth answers, replays the agent, and scores outputs against a checklist to tune prompts + retrieval.
- **Apps & UI** – `terminal_chatbot.py` for a CLI loop and `app.py` (Streamlit) for a lightweight UI, both routed through `Makefile` targets.

See service-level READMEs for detailed scripts and run commands:

| Service | Description | README |
| --- | --- | --- |
| Data ingestion | YouTube parsing, chunking, search indexing | `service/youtube_extraction/README.md` |
| Agents | Clarify + YouTube recommendation agents | `service/agent/README.md` |
| Evals | Ground-truth + automated rubric judge | `service/evals/README.md` |

## Demo & Deliverables
- **Pitch/problem slide:** _add link or path here_
- **Demo video:** _add recording link here_
- **User research / notes:** keep in `service/youtube_extraction/Notes.md` or link above.

## Quickstart
1. Install dependencies with [uv](https://github.com/astral-sh/uv): `uv sync`
2. Export required secrets (use your own `.env` file or the list below):
   - `GOOGLE_API_KEY`, `YOUTUBE_CHANNEL_ID`
   - `OPENAI_API_KEY`
   - `PROXY_USERNAME` / `PROXY_PASSWORD` (Webshare transcripts)
   - `WEBSHARE_HTTP`, `WEBSHARE_HTTPS` (optional translation proxy)
3. Use the `Makefile` to run workflows:

| Target | Purpose |
| --- | --- |
| `make run-get-videos` | Fetch latest videos + transcripts for the configured creator. |
| `make run-terminal-app` | Launch interactive CLI loop (`terminal_chatbot.py`). |
| `make run-streamlit-app` | Start the Streamlit UI (`app.py`). |
| `make run-main` | Run `main.py` orchestration script end-to-end. |
| `make run-ground-truth` | Regenerate golden answers for evals. |
| `make run-ground-truth-evals` | Score agent responses against the checklist. |

Each target automatically prefixes commands with `uv run …`, so dependencies are isolated in the project environment.

## Data Flow
1. `run-get-videos` downloads transcripts -> `.cache/<youtuber>_*` holds bilingual chunks.
2. Clarify Agent checks cache status and (if needed) calls `YoutuberTranscriptProcessor.chunk_transcript`.
3. YouTube Agent searches Elasticsearch/MinSearch using SentenceTransformers embeddings.
4. Agents respond via CLI or Streamlit UI; logs feed into eval pipelines.
5. Evals replay stored questions, compare to ground truth, and output pass/fail metrics.

## Open Questions & Learning Notes
### Agent design (tracked for future research)
1. Should every sub-agent own specialized tools, or should a simple toolset be orchestrated centrally for easier debugging in production?
2. Retrieval freshness vs. latency: pre-built RAG indexes vs. live API calls — what works best for lean experiments?
3. Multilingual handling: translate + index once, translate at query time, or store both Chinese + English chunks to unlock cross-lingual recall? (MinSearch currently misses pure Chinese text.)

### Framework study plan
- When picking up a new agent framework (e.g., PydanticAI), start with the docs for high-level concepts, then inspect the source for classes like `EventStreamHandler`, `RunResult`, `StreamResult`, `AgentRunResult`.
- Use type hints and small instrumentation prints while running `terminal_chatbot.py` to see what attributes (e.g., `tool_name`, `args`, `ctx`) are exposed.
- Tests and example scripts (see `service/evals/src/evals/`) double as living docs for acceptable inputs/outputs.

Keep refining this README with learnings (eval metrics, deployment notes, UI screenshots) as the project matures.
