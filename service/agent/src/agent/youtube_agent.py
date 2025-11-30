from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.messages import FunctionToolCallEvent
from service.youtube_extraction.src.youtube_extraction.youtube_search import (
    YoutuberTranscriptSearcher,
)

load_dotenv()


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


beauty_instructions_v1 = """
You are a multilingual YouTube transcript search and analysis agent.

Your responsibilities:
- Detect the user's query language (Chinese or English).
- If the user asks in Chinese: return the final summary in Chinese.
- If the user asks in English: return the final summary in English.
- All internal reasoning steps (search, analysis, summarization) can be in English.
- If the video content is in another language, you must translate it before summarizing.

Your job is to:
1. Search relevant transcript segments
2. Rebuild the index once if no results are found
3. Summarize findings based only on retrieved transcript
4. Output in YoutubeSummaryOutput format

---

### WORKFLOW (Follow strictly)

### **1. Query Understanding**
- Detect user query language.
- Detect if the query contains a **specific beauty category**.  
  Supported categories:

        - Lip Care / 護唇
        - Pre-Makeup Skincare / 妝前保養
        - Foundation / 粉底
        - Concealer / 遮瑕
        - Setting / 定妝
        - Eyebrow Makeup / 眉彩
        - Contour / 修容
        - Eyeshadow / 眼影
        - Highlighter / 打亮
        - Blush / 腮紅
        - Lip Gloss / 唇彩
        - Others / 其他其他

- If the user query mentions any category (either English or Chinese):
    → Only return results related to that category.
- If no category is mentioned:
    → Return all relevant items normally.

### **2. Relevant Transcript Extraction**
- Collect all transcript segments returned.
- Do **not** include the full transcript.
- Only keep segments relevant to the query.

### **3. Product or Topic Analysis**
From the retrieved transcript segments:
- Identify all **products**, **items**, or **topics** mentioned.
- Summarize what the creator says about them.
- Include opinions, pros/cons, and reasoning expressed in the transcript.
- The summary must be **grounded strictly in the retrieved text**, not external knowledge.
- If transcript is not in the user’s language:
    - Translate before summarizing.

### **4. Multilingual Output Rules**
- If the user query is in **Chinese** → Return final output in **Chinese**.
- If the user query is in **English** → Return final output in **English**.
- If transcript is in another language → Translate into user’s language before summarizing.

### **5. Output Formatting**
Return the final response in the **YoutubeSummaryOutput** schema, which includes:
- Extracted product(s) or topic(s)
- Summary for each
- Relevant transcript segments with timestamps
- All written in the user’s language
- Respect category filtering rules

Make the final summary:
- Clear
- Concise
- Accurate
- Directly tied to transcript evidence

---
"""


beauty_instructions = """
You are a multilingual YouTube transcript search and analysis agent.

Your responsibilities:
- Detect the user's query language (Chinese or English).
- If the user asks in Chinese: return the final summary in Chinese.
- If the user asks in English: return the final summary in English.
- Internal reasoning steps may be in English.
- If the transcript is in any other language, translate it **before summarizing**.

Your job is to:
1. Search for relevant transcript segments.
2. If no segments are returned, rebuild the index once and search again.
3. Summarize based **only** on retrieved transcript segments.
4. Produce output strictly in YoutubeSummaryOutput format.

================================================================
### 1. QUERY UNDERSTANDING
================================================================
- Detect user query language.
- Identify whether the query mentions a **specific beauty category**.

Supported categories:
    - Lip Care / 護唇
    - Pre-Makeup Skincare / 妝前保養
    - Foundation / 粉底
    - Concealer / 遮瑕
    - Setting / 定妝
    - Eyebrow Makeup / 眉彩
    - Contour / 修容
    - Eyeshadow / 眼影
    - Highlighter / 打亮
    - Blush / 腮紅
    - Lip Gloss / 唇彩
    - Others / 其他

Category rule:
- If a category is detected → **Only return products related to that category.**
- If no category is detected → Return all relevant products normally.

================================================================
### 2. TRANSCRIPT RETRIEVAL (IMPORTANT)
================================================================
From the transcript search results:

- **Every transcript segment must include:**
  - The raw transcript text
  - Its associated `start_time` timestamp
- Only keep segments directly related to the query or category.


================================================================
### 3. PRODUCT / TOPIC ANALYSIS (WITH TIMESTAMPS)
================================================================
### **3. Product or Topic Analysis**
For each **product or topic** mentioned in the transcript:
1. Only consider segments that are relevant to the product/topic.
2. Extract:
    - **start_time** (use the exact value from the transcript segment)
    - Transcript excerpt that mentions the product/topic
3. Summarize the creator’s opinion, pros/cons, and reasoning **strictly using the relevant segments**.
4. Do not include unrelated segments.
5. If multiple segments mention the same product, merge them into a single summary but **keep the earliest start_time as the reference**.
    - extract the **exact `start_time`** from the transcript segment where it first appears.
    - Do not guess or default to 00:00.
    - If multiple segments mention the same product, use the **earliest start_time**.
    

Rules:
- No guessing, no external knowledge.
- If the transcript is in another language, translate it before summarizing.
- If no transcript segment refers to a product → do NOT include it.

================================================================
### 4. MULTILINGUAL OUTPUT RULES
================================================================
- If user asks in **Chinese** → respond in **Chinese**.
- If user asks in **English** → respond in **English**.
- All transcript excerpts must be translated to match the user’s language.

================================================================
### 5. OUTPUT FORMAT (STRICT)
================================================================
Return a response in the **YoutubeSummaryOutput** schema, containing:

- List of products or topics detected
- For each product:
    - name
    - category (if recognized)
    - summary (in user’s language)
    - transcript_evidence: list of objects:
        - text (translated into user’s language)
        - start_time (from transcript search)
- Make sure all information is backed by transcript excerpts.

Final summary must be:
- Clear
- Concise
- Accurate
- Based strictly on transcript evidence

"""
class ProductDetails(BaseModel):
    name: str = Field(..., description="The name of the product.")
    start_time: str = Field(
        ..., description="The start time of the product mention in the video."
    )
    recommend: bool = Field(
        ...,
        description="Whether the product is recommended or not. Only return yes or no.",
    )
    reason: str = Field(
        ...,
        description="Summary from the youtube video why this product is recommended or not.",
    )
    transcript: str = Field(
        ..., description="Original transcript from the video and what creator said"
    )


class YoutubeSummaryOutput(BaseModel):
    """
    A single, verifiable citation to a transcript snippet or video segment.
    """

    title: str = Field(..., description="The title of the YouTube video.")
    youtuber: str = Field(..., description="The name of the YouTuber.")
    url: str = Field(..., description="The URL of the YouTube video.")
    category: str = Field(..., description="The category of the video content.")
    summary: str = Field(..., description="A brief summary of the video content.")
    products: list[ProductDetails] = Field(
        ..., description="A list of recommended products mentioned in the video."
    )

    def time_to_seconds(self, time_str: str) -> int:
        parts = time_str.split(":")
        parts = [int(p) for p in parts]

        if len(parts) == 2:  # MM:SS
            minutes, seconds = parts
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid time format: {time_str}")

    def print_youtube_summary(self):
        """
        Print the YouTube video summary and all associated product details
        in a structured format.

        Args:
            result (YoutubeSummaryOutput): The output object containing video
                and product information.
        """

        print("=== Youtube Beauty Search Output ===")
        for product in self.products:
            print(f"* Product: {product.name}")
            print(f"  ** Buy: {product.recommend}")
            print(f"  ** Reason: {product.reason}")
            print(f"  ** Transcript: {product.transcript}\n")

            # Convert start_time to seconds if needed
            start_seconds = (
                self.time_to_seconds(product.start_time)
                if hasattr(product, "start_time")
                else 0
            )
            print("  [Video Reference]")
            print(f"  Youtuber: {self.youtuber}")
            print(f"  Title: {self.title}")
            print(f"  URL: {self.url}?t={start_seconds}\n")

            print("======================")

    
    def format_youtube_summary(self):
        """
        Print the YouTube video summary and all associated product details
        in a structured format.

        Args:
            result (YoutubeSummaryOutput): The output object containing video
                and product information.
        """
        output = ''
        # output += "=== Youtube Beauty Search Output ==="
        for product in self.products:
            output += f"* Product: {product.name}\n"
            output += f"  ** Buy: {product.recommend}\n"
            output += f"  ** Reason: {product.reason}\n"
            output += f"  ** Transcript: {product.transcript}\n\n"

            # Convert start_time to seconds if needed
            start_seconds = (
                self.time_to_seconds(product.start_time)
                if hasattr(product, "start_time")
                else 0
            )
            output += "  [Video Reference]\n"
            output += f"  Youtuber: {self.youtuber}\n"
            output += f"  Title: {self.title}\n"
            output += f"  URL: {self.url}?t={start_seconds}\n"

            output += "======================"
        return output

    def display_streamlit(self):
        """Display the summary in Streamlit."""
        import streamlit as st
        # st.header("🎬 YouTube Video Summary")
        # st.subheader(f"{self.title} by {self.youtuber}")
        # st.write(f"**Category:** {self.category}")
        # st.write(f"**Summary:** {self.summary}")
        # st.write(f"[Watch Video]({self.url})")

        # st.markdown("---")
        st.subheader("🛍️ Recommended Products")
        for product in self.products:
            start_seconds = (
                self.time_to_seconds(product.start_time)
                if hasattr(product, "start_time")
                else 0
            )
            st.markdown(f"**Product:** {product.name}")
            st.write(f"- **Buy:** {product.recommend}")
            st.write(f"- **Reason:** {product.reason}")
            st.write(f"- **Transcript:** {product.transcript}")
            st.write(
                f"- **Video Reference:** [{self.url}?t={start_seconds}]({self.url}?t={start_seconds})"
            )
            st.markdown("---")


def create_youtube_agent():
    yts = YoutuberTranscriptSearcher(youtuber="heyitsmindy")

    youtube_agent = Agent(
        name="Youtube_Agent",
        instructions=beauty_instructions,
        # tools=[yts.search, yts.create_index],
        tools=[
            yts.search_es,
            yts.ensure_es_index,
        ],
        model="gpt-4o-mini",
        output_type=YoutubeSummaryOutput,
    )
    # youtube_callback = NamedCallback(youtube_agent)

    return youtube_agent
