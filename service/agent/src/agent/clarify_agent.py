from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.messages import FunctionToolCallEvent
from service.youtube_extraction.src.youtube_extraction.youtube_process import (
    YoutuberTranscriptProcessor,
)
from typing import Optional, List

import json
from pathlib import Path

load_dotenv()

CACHE_DATA_DIR = Path(".cache")


clarify_instruction = """

You are the Clarify Agent for a Beauty & YouTube Review Search System.

Your ONLY responsibilities are:
1. Understanding the user's message.
2. Determining whether to use cached transcript data or fetch/index new data.
3. Routing the flow correctly using the ClarifyDecision schema.

You MUST NOT:
- Answer beauty questions.
- Recommend products.
- Summarize any video.
- Pull transcript content.
You ONLY perform routing and decision-making.

====================================================
PRIMARY RULE — Cache First
====================================================

Always check if the transcript for the video is in the cache **before doing anything else**.
- Always calls `has_cached_transcript` to check.
- If the transcript exists in cache → do NOT call chunk_transcript.
- If the transcript does NOT exist in cache → always call chunk_transcript to generate and save the chunks.
- At the end, always **call `list_cached_videos(youtuber)`** to return all available videos in the cache for this YouTuber.

This step is REQUIRED in every execution.


====================================================
1. QUERY UNDERSTANDING
====================================================
From the user's message, extract:

- youtuber:
    The content creator mentioned or implied.
    If none is present, return an empty string "".

- user_intent:
    Pick one of the following:
        - "find_product"          → Asking about a product or item.
        - "recommendation"        → Asking what to buy, or for suggestions.
        - "compare_products"      → Comparing two or more products.
        - "video_question"        → Asking for info about a specific video.
        - "index_video"           → Asking to ingest/index a specific video.
    Choose the best match based on user language.

- topic:
    A product or category mentioned (e.g., "foundation", "lip gloss").
    If none is mentioned, return null.

- video_id:
    Extract the YouTube video ID from:
        - Full URLs, e.g., "https://www.youtube.com/watch?v=2SvN45DKWFg"
        - Share links, e.g., "https://youtu.be/2SvN45DKWFg"
        - Raw video IDs, e.g., "2SvN45DKWFg"
    If no video ID is found → return null.

====================================================
2. ROUTING LOGIC
====================================================

-----------------------------------------
CASE A — The user does NOT provide a video_id
-----------------------------------------
If no video_id can be extracted:
    - Set:
        video_id = null
        in_cache = false
        user_intent = "no_video_id"
        available_videos = the provided list of cached videos
    - Do NOT attempt to guess, search, or index.

-----------------------------------------
CASE B — The user DOES provide a video_id
-----------------------------------------
If video_id is provided:
    - Call `has_cached_transcript(video_id)` exactly ONCE.
    - Set:
        in_cache = True  → if transcript is already cached  
        in_cache = False → if transcript does not exist

    - If in_cache = True:
        - Do NOT index again.
        - Prepare decision output immediately.

    - If in_cache = False:
        - Mark user_intent = "index_video" if user requested indexing explicitly
        - Or keep original intent but indicate in_cache = False
        - **ALWAYS Call the method `chunk_transcript(video_id)`** to:
            - Load the raw transcript
            - Split transcript into segments
            - Generate LLM-based summary
            - Build sliding-window chunks
            - Save chunks to cache
    - Only call `chunk_transcript` **once per request*

====================================================
3. OUTPUT FORMAT
====================================================
You MUST return a JSON object exactly matching the ClarifyDecision schema:

class ClarifyDecision(BaseModel):
    youtuber: str
    user_intent: str
    topic: Optional[str]
    video_id: Optional[str]
    in_cache: bool
    available_videos: Optional[List[Videos]]

- Do NOT answer the user.
- Do NOT comment on the query.
- ONLY produce structured routing decisions.

"""


class Videos(BaseModel):
    title: str = Field(..., description="Title of the cached video.")
    summary: str = Field(..., description="Summary of the cached video.")
    url: str = Field(..., description="Youtuber URL for the cached video")


class ClarifyDecision(BaseModel):
    """
    Routing + query understanding for the Beauty YouTube system.
    """

    # Understanding the request
    youtuber: str = Field(
        ..., description="The Youtuber mentioned or inferred from the user query."
    )
    user_intent: str = Field(
        ...,
        description="The goal of the user: e.g. 'add_data', 'find_product', 'recommendation', 'compare', 'video_question'.",
    )
    topic: Optional[str] = Field(
        None, description="Product or topic extracted from the user query."
    )
    video_id: Optional[str] = Field(
        None, description="YouTube video ID if specified, else None."
    )

    # Cache and data routing
    in_cache: bool = Field(
        ..., description="Whether the transcript for this video exists in the cache."
    )
    # use_cache_only: bool = Field(..., description="If True, system should use cached data only.")
    # requires_online_search: bool = Field(..., description="If True, system must fetch the transcript online and chunk it.")
    # add_to_cache: bool = Field(..., description="If True, system must chunk, summarize, and save the transcript.")

    # When video_id is missing
    # ask_for_video: bool = Field(..., description="If True, ask the user to pick a video.")
    available_videos: Optional[List[Videos]] = Field(
        None,
        description="List of cached videos (title, summary, url) to show user when no video_id provided.",
    )

    def format_agent_output(self):
        """
        Return the full Clarify Agent output in a readable format.

        Args:
            decision (ClarifyDecision): The ClarifyDecision object returned by the age
        """
        output = ""
        output += "=== Clarify Agent Output ===\n"
        output += f"Intent       : {self.user_intent}"
        output += f"YouTuber     : {self.youtuber}"
        output += f"Video ID     : {self.video_id}"
        output += f"Topic        : {self.topic}"
        # print(f"Add Data     : {decision.add_to_cache}")
        output += f"In Cache  : {self.in_cache}\n"
        
        if self.available_videos:
            output += "Hey I can start answer your question!\nHere are Available Videos (metadata):\n"
            for idx, video in enumerate(self.available_videos, start=1):
                output += f"{idx}. Title  : {video.title}"
                output += f"   Summary: {video.summary}"
                output += f"   URL    : {video.url}\n"

        else:
            output += "No available videos to display."

        return output


    def print_agent_output(self):
        """
        Print the full Clarify Agent output in a readable format.

        Args:
            decision (ClarifyDecision): The ClarifyDecision object returned by the agent.
        """
        print("=== Clarify Agent Output ===")
        print(f"Intent       : {self.user_intent}")
        print(f"YouTuber     : {self.youtuber}")
        print(f"Video ID     : {self.video_id}")
        print(f"Topic        : {self.topic}")
        # print(f"Add Data     : {decision.add_to_cache}")
        print(f"In Cache  : {self.in_cache}\n")

        # if decision.ask_for_video:

        if self.available_videos:
            print(
                "Hey I can start answer your question!\nHere are Available Videos (metadata):\n"
            )
            for idx, video in enumerate(self.available_videos, start=1):
                print(f"{idx}. Title  : {video.title}")
                print(f"   Summary: {video.summary}")
                print(f"   URL    : {video.url}\n")
        # elif self.available_videos:
        #     print("Available Video IDs:")
        #     for vid in decision.available_videos:
        #         print(f"- {vid}")
        else:
            print("No available videos to display.")
        print("============================\n")


class NamedCallback:
    def __init__(self, agent):
        self.agent_name = agent.name

    async def print_function_calls(self, ctx, event):
        # Detect nested streams
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)


def create_clarify_agent():
    ytp = YoutuberTranscriptProcessor(youtuber="heyitsmindy")

    clarify_agent = Agent(
        name="Clarify_Agent",
        instructions=clarify_instruction,
        tools=[ytp.chunk_transcript, has_cached_transcript, list_cached_videos],
        model="gpt-4o-mini",
        output_type=ClarifyDecision,
    )

    return clarify_agent


def has_cached_transcript(
    youtuber, video_id, window_size: int = 15, step_size: int = 3
):
    """
    Check whether cached transcript chunks exist for a given YouTube video.

    The function looks for a JSON file named
        {youtuber}_{video_id}.json
    inside the directory:
        {CACHE_DATA_DIR}/{youtuber}_{window_size}_{step_size}/

    Args:
        youtuber (str): The YouTuber's name used for folder and file naming.
        video_id (str): The YouTube video ID.
        window_size (int): Sliding window size used during transcript chunking.
        step_size (int): Step size used during transcript chunking.

    Returns:
        bool: True if the cached transcript file exists, False otherwise.
    """

    cache_dir = CACHE_DATA_DIR / Path(f"{youtuber}_{window_size}_{step_size}_Chinese")
    file_path = cache_dir / f"{youtuber}_{video_id}.json"

    return file_path.exists()


def list_cached_videos(
    youtuber: str = 'heyitsmindy', window_size: int = 15, step_size: int = 3
) -> List[dict]:
    """
    List all cached video IDs for a given YouTuber.

    Args:
        youtuber (str): Name of the YouTuber.
        window_size (int): Sliding window size used during processing.
        step_size (int): Step size used during processing.

    Returns:
        List[dict]: List of title, summary, video IDs available in the cache for this YouTuber.
    """

    cache_dir = CACHE_DATA_DIR / f"{youtuber}_{window_size}_{step_size}_Chinese"

    if not cache_dir.exists():
        return []

    video_url = []
    for file in cache_dir.glob(f"{youtuber}_*.json"):
        with open(file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        summary = chunks[0]["summary"]
        title = chunks[0]["title"]
        # Extract video_id from filename
        name = file.stem  # e.g., "heyitsmindy_9QSmr_2EGXM"
        parts = name.split(f"{youtuber}_")
        if len(parts) == 2:
            video = {
                "title": title,
                "summary": summary,
                "url": f"https://www.youtube.com/watch?v={parts[1]}",
            }
            video_url.append(video)
    return video_url
