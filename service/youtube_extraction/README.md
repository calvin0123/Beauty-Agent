# YouTube Extraction Service

Data-ingestion toolkit that fetches channel metadata, downloads transcripts, chunks them, and builds retrieval indexes for the Beauty/Health agent.

## Workflow Overview
1. **Fetch video metadata** – `YoutubeParser.get_all_video_id` hits the YouTube Data API and stores the latest uploads under `data/<youtuber>/videos/`.
2. **Download transcripts** – `YoutubeParser.download_all_transcripts` uses `youtube-transcript-api` + Webshare proxies to save subtitles in `data/<youtuber>/transcript/`.
3. **Summarize & chunk** – `YoutuberTranscriptProcessor` (Chinese-default) or `YoutuberTranscriptProcessorEng` parses the raw `.txt` transcript, calls OpenAI for summaries, and writes overlapping chunks to `.cache/<youtuber>_<window>_<step>[_Chinese].json`.
4. **Index for retrieval** – `YoutuberTranscriptSearcher` can create MinSearch BM25 indexes or embed every chunk and push it into Elasticsearch for semantic search.
5. **Serve to agents/tools** – Agents import `YoutuberTranscriptProcessor` for chunking tools and `YoutuberTranscriptSearcher` for search tools.

## Key Scripts

| Path | Purpose | How to run |
| --- | --- | --- |
| `src/youtube_extraction/youtube_parser.py` | Download video lists + transcripts. | `make run-get-videos` (wraps `uv run … youtube_parser.py`). Configure `GOOGLE_API_KEY`, `PROXY_USERNAME`, `PROXY_PASSWORD`. |
| `src/youtube_extraction/youtube_process.py` | Defines processors, chunking helpers, and LLM summary schemas (Chinese + English variants). | Import the relevant class and call `chunk_transcript(video_id)`; invoked automatically by the Clarify Agent. |
| `src/youtube_extraction/transcripts.py` | Legacy chunker with Deep Translator support; still used for experiments that need English-only content chunks. | Run via notebook or custom script when comparing chunking strategies. |
| `src/youtube_extraction/search_tool.py` | Prototype MinSearch workflow (kept for reference). | `uv run … search_tool.py` (builds `.cache/search_tools_*`). |
| `src/youtube_extraction/youtube_search.py` | Production search class using Elasticsearch + SentenceTransformers (with optional MinSearch fallback). | Imported by `service/agent/src/agent/youtube_agent.py`. Run `ensure_es_index` once, then `search_es(query)`. |
| `src/youtube_extraction/utils.py` | Shared Pydantic schemas + translation helper that wraps `deep_translator`. | Imported where needed; no standalone entrypoint. |

## Running the Pipeline
1. **Set env vars**
   ```bash
   export GOOGLE_API_KEY=...
   export OPENAI_API_KEY=...
   export PROXY_USERNAME=...
   export PROXY_PASSWORD=...
   export WEBSHARE_HTTP=...   # optional, for translations
   export WEBSHARE_HTTPS=...
   ```
2. **Download transcripts**
   ```bash
   make run-get-videos
   ```
   Adjust defaults by editing `YoutubeParser` arguments (channel ID, youtuber name) in the script.
3. **Start Elasticsearch (optional but recommended)**
   ```bash
   docker run -d \
     --name elasticsearch \
     -m 6g \
     -p 9200:9200 \
     -p 9300:9300 \
     -e "discovery.type=single-node" \
     -e "xpack.security.enabled=false" \
     -v es9_data:/usr/share/elasticsearch/data \
     docker.elastic.co/elasticsearch/elasticsearch:9.1.1
   ```
4. **Chunk & cache** – triggered automatically the first time the Clarify Agent processes a video, or run manually:
   ```python
   from service.youtube_extraction.src.youtube_extraction.youtube_process import YoutuberTranscriptProcessor
   ytp = YoutuberTranscriptProcessor(youtuber="heyitsmindy")
   ytp.chunk_transcript(video_id="2SvN45DKWFg", window_size=15, step_size=3)
   ```
5. **Index chunks** – automatically kicked off via `YoutuberTranscriptSearcher.ensure_es_index`. To force rebuild:
   ```python
   from service.youtube_extraction.src.youtube_extraction.youtube_search import YoutuberTranscriptSearcher
   searcher = YoutuberTranscriptSearcher(youtuber="heyitsmindy")
   searcher.ensure_es_index()
   ```

## Chunking Strategy
- **Approach:** convert the transcript into `{time, text}` entries, group ~15 entries per chunk with a stride of 3 (≈5-entry overlap).
- **Why:** overlap preserves context at boundaries, while keeping each chunk small enough for embeddings + translation.
- **Language handling:** default cache stores Chinese `content` plus English metadata; optional `translate_english` can add `content_eng` for MinSearch.

## Caching & File Layout
- Video IDs List → `data/<youtuber>/videos/<youtuber>.json`
- Raw transcripts → `data/<youtuber>/transcript/<video_id>.txt`
- Chunked JSON → `.cache/<youtuber>_<window>_<step>_Chinese/<youtuber>_<video_id>.json`
- Ground truth data → `data/ground_truth/…`


## Learning
- **Tool schema errors:** avoid exposing `self`-bound methods directly as OpenAI tools (wrap them or use standalone functions) to keep function signatures JSON-serializable.
- **Large transcripts:** chunk Chinese text before sending to the model; whole transcripts may exceed model limits.
- **Translation gaps:** MinSearch cannot match Chinese text unless `content_eng` is populated or you use multilingual embeddings (preferred approach via Elasticsearch).

