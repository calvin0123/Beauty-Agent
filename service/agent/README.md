# Agent Service

Agent stack for the Beauty/Health project: Clarify Agent, YouTube Agent, and the Orchestrator that chains them together.

## Directory Map
- `src/agent/clarify_agent.py` – cache-aware router that inspects intent, video IDs, and whether transcripts need chunking.
- `src/agent/youtube_agent.py` – multilingual retrieval + summarization agent powered by Elasticsearch + SentenceTransformers.
- `src/agent/orchestrator_agent.py` – top-level coordinator that always runs Clarify first, then triggers YouTube search when needed.
- `src/agent/__init__.py` – package marker for uv/pip installs.

## Clarify Agent
Purpose: deterministically understand the request before any heavy search work.

- **Instructions:** enforce “cache-first, decision-only” behavior; never answer user questions or fetch transcripts directly.
- **Tools exposed:**
  - `YoutuberTranscriptProcessor.chunk_transcript` – loads raw transcripts, summarizes with OpenAI, creates overlapping chunks, and caches them.
  - `has_cached_transcript` – checks `.cache/<youtuber>_<window>_<step>_Chinese` for existing chunks.
  - `list_cached_videos` – returns cached metadata so the UI can show existing videos when no video ID is provided.
- **Output schema:** `ClarifyDecision` (`user_intent`, `topic`, `video_id`, `in_cache`, optional `available_videos` list).
- **Usage:**

```python
from service.agent.src.agent.clarify_agent import create_clarify_agent

clarify = create_clarify_agent()
decision = clarify.run_sync("請幫我分析這支粉底液影片：https://youtu.be/XXXX")
decision.output.print_agent_output()
```

## YouTube Agent
Purpose: search bilingual transcript chunks and return grounded product recommendations or summaries.

- **Instructions:** detect user language (Chinese/English), filter by beauty subcategories, rebuild the index once if empty, and cite transcript snippets with timestamps.
- **Tools:** `YoutuberTranscriptSearcher.ensure_es_index` and `search_es` (Elasticsearch dense-vector search backed by `paraphrase-multilingual-MiniLM-L12-v2` embeddings).
- **Output schema:** `YoutubeSummaryOutput` containing video metadata plus per-product details (`name`, `start_time`, `recommend`, `reason`, `transcript`). Helper methods format CLI, Streamlit, or log-friendly summaries.
- **Usage:**

```python
from service.agent.src.agent.youtube_agent import create_youtube_agent

youtube_agent = create_youtube_agent()
response = youtube_agent.run_sync("Which lip oils did HeyItsMindy recommend?")
response.output.print_youtube_summary()
```

## Orchestrator Agent
Purpose: combine Clarify + YouTube agents into a single tool that understands when to ingest, when to search, and what to return.

- **Workflow rules:**
  1. **Clarify first, always.** The orchestrator’s `clarify_tool_initial` runs `clarify_agent.run()` and captures the structured decision.
  2. **Trigger search for high-priority intents.** If `user_intent` is `find_product`, `recommendation`, `compare`, or `video_question`, the orchestrator immediately calls `recommendation_tool` (wrapping `youtube_agent.run`) and may search up to two times.
  3. **Short-circuit when nothing else is needed.** If Clarify determines no YouTube analysis is required, the orchestrator stops and surfaces the ClarifyDecision for follow-up UI handling.
  4. **Tool-output only.** The agent is instructed to return pure tool responses (no free text), which keeps downstream parsing simple.
- **Usage:**

```python
from service.agent.src.agent.orchestrator_agent import create_orchestration_agent

orchestrator = create_orchestration_agent()
result = orchestrator.run_sync("Compare HeyItsMindy's favorite Sephora foundations.")

is result.output.clarify:
    result.output.clarify.print_agent_output()
if result.output.youtube:
    result.output.youtube.print_youtube_summary()

```

## Running the Agents
- `make run-terminal-app` – launches the CLI loop that now calls the Orchestrator Agent for every turn.
- `make run-main` – executes `main.py`, which also uses the Orchestrator to process scripted prompts.
- `make run-streamlit-app` – runs the Streamlit UI (currently interacts directly with the YouTube agent output).

Before any run, ensure:
1. Transcripts are ingested (`make run-get-videos`).
2. Elasticsearch is running (see command in `service/youtube_extraction/README.md`).

## Extending
- Swap creators by instantiating processors/searchers with `youtuber="<channel>"`.
- Add tools by registering new functions in the `Agent(..., tools=[...])` definitions.
- Update instruction blocks + Pydantic models if you add new beauty categories, output fields, or multi-agent flows (e.g., Sephora product crawler, price comparison agent).
