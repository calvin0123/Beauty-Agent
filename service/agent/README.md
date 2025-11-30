# Agent Service

Entry point for the Clarify and YouTube recommendation agents that power the Beauty/Health assistant.

## Directory Map
- `src/agent/clarify_agent.py` – routes user intent, decides whether cached transcripts exist, and calls chunking tools.
- `src/agent/youtube_agent.py` – multilingual retrieval + summarization agent that returns structured product recs.
- `src/agent/__init__.py` – package marker (kept minimal for uv imports).

## Clarify Agent

[Pending] Purpose: keep routing logic deterministic before any heavy inference.

- **Instructions:** force cache-first behavior, never answer user questions directly.
- **Tools exposed:**
  - `YoutuberTranscriptProcessor.chunk_transcript` – builds sliding-window chunks + summaries.
  - `has_cached_transcript` – quick filesystem check under `.cache/<youtuber>_<window>_<step>_Chinese`.
  - `list_cached_videos` – surfaces available cached items so the UI can show choices.
- **Output schema:** `ClarifyDecision` (intent, topic, video_id, cache flag, optional list of videos).
- **Usage snippet:**

```python
from service.agent.src.agent.clarify_agent import create_clarify_agent

clarify = create_clarify_agent()
result = clarify.run_sync(user_prompt="請幫我分析這支粉底液影片：https://youtu.be/XXXX")
print(result.output.dict())
```

## YouTube Agent
Purpose: run multilingual retrieval over cached transcript chunks and surface grounded product recommendations.

- **Instructions:** detect query language, filter by beauty subcategory, rebuild index once when empty, and respond only with transcript-backed evidence.
- **Tools:** `YoutuberTranscriptSearcher.ensure_es_index`, `YoutuberTranscriptSearcher.search_es` (SentenceTransformer embeddings + Elasticsearch KNN).
- **Output schema:** `YoutubeSummaryOutput` with helper methods (`format_youtube_summary`, `display_streamlit`) for CLI/UI rendering.
- **Key behaviors:**
  - Auto-translates transcript excerpts to the user’s language before summarizing.
  - Returns per-product citations with timestamps for deep links (e.g., `https://youtu.be/<id>?t=123`).

```python
from service.agent.src.agent.youtube_agent import create_youtube_agent

youtube_agent = create_youtube_agent()
response = youtube_agent.run_sync("Which lip oils did HeyItsMindy recommend?")
response.output.print_youtube_summary()
```

## Running the Agents
- **CLI loop:** `make run-terminal-app`
- **Streamlit UI:** `make run-streamlit-app`
- **Single run:** `make run-main` (calls `main.py`, which chains YouTube agents and return the streaming output using `JSONStreaming`) 

Ensure transcripts are already chunked (`make run-get-videos`) and Elasticsearch (`docker run …` command in `service/youtube_extraction/README.md`) is running before invoking the agents.

## Extending
- Change the default creator by passing `youtuber="<channel>"` when instantiating `YoutuberTranscriptSearcher` or `YoutuberTranscriptProcessor`.
- To add new tools, register them in the respective `Agent(..., tools=[...])` definitions.
- Update the instruction strings if you introduce new beauty categories or output schema fields; keep `YoutubeSummaryOutput` in sync with any downstream UI expectations.
