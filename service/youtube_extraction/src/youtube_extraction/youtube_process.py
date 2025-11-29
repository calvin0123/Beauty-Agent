from pathlib import Path
import json
from pydantic import BaseModel, Field
from typing import List, Any, Tuple

from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()
RAW_DATA_DIR = Path("data")
CACHE_DATA_DIR = Path(".cache")


# -------------------------
# English Version
# -------------------------
# ##### Chunk the transcript if needed
class VideoCategory(str, Enum):
    makeup = "makeup"
    skincare = "skincare"
    beauty = "beauty"


class TranscriptEng(BaseModel):
    time: str = Field(description="The timestamp of the transcript segment.")
    text: str = Field(
        description="The translated text content of the transcript segment."
    )
    original_text: str = Field(
        description="The chinese text content of the transcript segment."
    )


class YTSummaryEngResponse(BaseModel):
    # summary_zh: str = Field(description="The summary of the youtube video transcript.")
    # title_zh: str = Field(description="The title of the YouTube video in Chinese")
    summary_en: str = Field(description="English translation of the summary.")
    title_en: str = Field(description="English translation of the title.")
    category: VideoCategory = Field(
        description=(
            "Main beauty category of the video. Choose one of:\n"
            "- makeup: tutorials, reviews, product try-ons\n"
            "- skincare: routines, product reviews, ingredients\n"
            "- haircare: styling, coloring, or hair treatment tips"
        )
    )
    subcategory: str = Field(
        description=(
            "The subcategory of the video based on specific product or topic, e.g.:\n"
            "- Lip Care / 護唇\n"
            "- Pre-Makeup Skincare / 妝前保養\n"
            "- Foundation / 粉底\n"
            "- Concealer / 遮瑕\n"
            "- Setting / 定妝\n"
            "- Eyebrow Makeup / 眉彩\n"
            "- Contour / 修容\n"
            "- Eyeshadow / 眼影\n"
            "- Highlighter / 打亮\n"
            "- Blush / 腮紅\n"
            "- Lip Gloss / 唇彩\n"
            "- Others / 其他其他"
        )
    )
    transcripts: List[TranscriptEng] = Field(
        description="The list of video transcripts."
    )


class YoutuberTranscriptProcessorEng:
    """
    A utility class for managing YouTube transcript data for a specific creator.

    Responsibilities:
    1. Check whether a transcript or processed transcript chunks exist locally.
    2. Load raw transcript text from disk.
    3. Split and summarize the transcript using an LLM.
    4. Generate sliding-window transcript chunks for index-based retrieval.
    5. Cache all outputs so repeated queries for the same video are fast.

    Typical usage in an agent:
    - When the user provides a YouTube video ID or URL:
        1. Determine if the transcript chunks already exist.
        2. If cached, return the cached chunks.
        3. If not cached:
            a. Load the raw transcript
            b. Summarize the transcript using an LLM model
            c. Convert the transcript into overlapping chunks
            d. Save the results for future retrieval

    Inputs:
        youtuber (str): Name of the creator used as the directory key.

    Outputs:
        - Dictionary containing raw transcript data
        - List of dictionaries for transcript chunks
    """

    def __init__(self, youtuber):
        self.youtuber = youtuber
        self.raw_data_dir = RAW_DATA_DIR / youtuber
        self.cache_dir = CACHE_DATA_DIR / youtuber

    def load_transcript(self, video_id):
        """
        Load a raw transcript file for a given YouTube video ID.

        Args:
            video_id (str): The ID of the YouTube video.

        Returns:
            dict: A dictionary containing:
                - "youtuber": the creator's name
                - "video_id": the video ID
                - "transcript": the full transcript text
        """
        filepath = self.raw_data_dir / f"{video_id}.txt"
        with open(filepath, "r") as f:
            transcript = f.read()

        return {
            "youtuber": self.youtuber,
            "video_id": f"{video_id}",
            "transcript": transcript,
        }

    def chunk_transcript(self, video_id, window_size: int = 15, step_size: int = 3):
        """
        Create and return sliding-window transcript chunks for a YouTube video.

        This method automatically:
        - Checks whether a cached JSON file of chunks already exists.
        - If cached → loads and returns the chunks.
        - If not cached → performs full processing:
            1. Loads the raw transcript text.
            2. Splits the transcript into segments.
            3. Generates an LLM-based summary.
            4. Builds sliding-window chunks using the summary + parsed content.
            5. Saves the result to cache.

        Args:
            video_id (str): YouTube video ID.
            window_size (int): Number of transcript segments per sliding window.
            step_size (int): How far the window moves between chunks.

        Returns:
            list[dict]: A list of chunk dictionaries containing transcript segments,
                        metadata, and summary information.
        """

        cache_dir = Path(f"{self.cache_dir}_{window_size}_{step_size}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = cache_dir / f"{self.youtuber}_{video_id}.json"
        if file_path.exists():
            chunks = json.loads(file_path.read_text(encoding="utf-8"))

        else:
            video = self.load_transcript(video_id=video_id)
            print(f"Processing... {video['video_id']}")
            split_transcript, transcript_str = split_video_to_multiple_transcript(video)

            # summary
            video_summary = youtuber_summarize(
                # instructions=instructions,
                user_prompt=str(split_transcript),
                output_format=YTSummaryResponse,
            )

            # start
            chunks = sliding_window(
                video_id=video["video_id"],
                youtuber=video["youtuber"],
                parsed=video_summary.transcripts,
                video_summary=video_summary,
                window_size=window_size,
                step_size=step_size,
            )

            self.save_transcript(file_path, chunks)

        return chunks

    def save_transcript(self, file_path, chunks):
        """
        Save processed transcript chunks to a JSON file.

        Args:
            file_path (Path): The output file location.
            chunks (list): List of chunk dictionaries to save.

        Notes:
            Writes JSON with UTF-8 encoding and indentation for readability.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)


def youtuber_summarize_eng(user_prompt, output_format, model="gpt-4o-mini"):
    instructions = """
    You are a multilingual beauty content analyst.

    Your task is to analyze YouTube transcripts related to beauty, skincare, or makeup content.

    Follow these steps carefully:

    1. **Summarize** the full video transcript in **Chinese**, focusing on key talking points, tone, and product mentions.
    2. **Translate** the summary and title into **English**.
    3. **Categorize** the video into one of the following beauty categories:
    - makeup: tutorials, reviews, product try-ons
    - skincare: routines, product reviews, ingredients
    - haircare: styling, coloring, or hair treatment tips
    4. **Translate each transcript segment** (keeping both the timestamp and the translated text and the original text).

    """.strip()

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = openai_client.responses.parse(
        model=model, input=messages, text_format=output_format
    )

    return response.output[0].content[0].parsed


# -------------------------
# Helper Function
# -------------------------

from typing import Iterable, Dict
import re


def split_video_to_multiple_transcript(
    video: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], str]:
    """
    Convert a raw YouTube transcript string into structured time-stamped entries.

    Args:
        video (dict): Dictionary containing:
            - "transcript" (str): Raw transcript text with lines starting with timestamps (e.g., "0:15 Hello world").
            - Optional keys: "youtuber", "video_id" (not used in output, but can be present).

    Returns:
        Tuple:
            - List[Dict[str, str]]: Each dictionary contains:
                - "time": Timestamp string (e.g., "0:15")
                - "text": Corresponding transcript sentence
            - str: Full transcript concatenated into a single string (all sentences joined with spaces)

    Example:
        video = {"transcript": "0:00 Hi\n0:05 Welcome\n"}
        parsed, full_text = split_video_to_multiple_transcript(video)
        # parsed => [{"time": "0:00", "text": "Hi"}, {"time": "0:05", "text": "Welcome"}]
        # full_text => "Hi Welcome "
    """
    lines = video["transcript"].strip().split("\n")
    pattern = re.compile(r"(\d+:\d+)\s*(.*)")
    parsed = []

    transcript = ""
    for line in lines:
        match = pattern.match(line)
        if match:
            time_str, sentence = match.groups()
            # parsed.append({"youtuber": video["youtuber"], "video_id": video["video_id"], "time": time_str, "text": sentence.strip()})
            parsed.append({"time": time_str, "text": sentence.strip()})

            transcript += sentence.strip()
            transcript += " "
    return parsed, transcript


def sliding_window(
    video_id: str,
    youtuber: str,
    parsed: List[Dict[str, Any]],
    video_summary: Any,
    window_size: int = 15,
    step_size: int = 3,
    language: str = "Chinese",
) -> List[Dict[str, Any]]:
    """
    Create overlapping transcript chunks from a list of transcript entries.

    Each chunk contains a subset of the transcript entries, allowing for
    overlapping windows. This is commonly used for embedding and search.

    Args:
        video_id (str): YouTube video ID.
        youtuber (str): Name of the video creator.
        parsed (List[Dict]): Transcript entries, each a dict containing at least:
            - 'text' (str): Transcript sentence.
            - 'time' (str): Timestamp of the sentence (e.g., "0:15").
        video_summary (object): Object containing video summary information, with attributes:
            - title_en (str): Video title in English
            - summary_en (str): Video summary in English
            - title_zh (str): Video title in Chinese
            - summary_zh (str): Video summary in Chinese
            - category (Enum or str): Category of the video
        window_size (int, optional): Number of transcript entries per chunk. Defaults to 15.
        step_size (int, optional): Number of entries to move forward for the next chunk (overlap). Defaults to 3.
        language (str, optional): Language of the transcript entries ('Chinese' or 'English'). Defaults to 'Chinese'.

    Returns:
        List[Dict]: List of overlapping transcript chunks. Each chunk dictionary contains:
            - 'video_id': YouTube video ID
            - 'youtuber': Creator name
            - 'title': Video title in English
            - 'summary': Video summary in English
            - 'category': Video category
            - 'title_zh': Video title in Chinese (if language='Chinese')
            - 'summary_zh': Video summary in Chinese (if language='Chinese')
            - 'chunk_id': Sequential chunk number
            - 'start_time': Timestamp of first transcript entry in the chunk
            - 'end_time': Timestamp of last transcript entry in the chunk
            - 'content': Concatenated transcript text in the chunk

    """
    chunks = []
    # video_id = parsed[0]['video_id']
    # youtuber = parsed[0]['youtuber']
    total = len(parsed)
    chunk_id = 1

    for start in range(0, total, step_size):
        end = min(start + window_size, total)
        window = parsed[start:end]

        # If window too small at end, you can skip or include:
        if len(window) < 2:
            break  # stop if not enough data left

        # Combine texts from the window

        # original_text_chunk = " ".join([entry.original_text for entry in window])
        if language == "English":
            text_chunk = " ".join([entry.text for entry in window])
            chunks.append(
                {
                    "video_id": video_id,
                    "youtuber": youtuber,
                    "title": video_summary.title_en,
                    "summary": video_summary.summary_en,
                    "category": video_summary.category.value,
                    "chunk_id": chunk_id,
                    "start_time": window[0].time,
                    "end_time": window[-1].time,
                    "content": text_chunk,
                    # 'original_content': original_text_chunk
                }
            )
        else:
            text_chunk = " ".join([entry["text"] for entry in window])
            chunks.append(
                {
                    "video_id": video_id,
                    "youtuber": youtuber,
                    "title": video_summary.title_en,
                    "summary": video_summary.summary_en,
                    "category": video_summary.category.value,
                    "title_zh": video_summary.title_zh,
                    "summary_zh": video_summary.summary_zh,
                    "chunk_id": chunk_id,
                    "start_time": window[0]["time"],
                    "end_time": window[-1]["time"],
                    "content": text_chunk,
                    # 'original_content': original_text_chunk
                }
            )

        chunk_id += 1

        if end == total:  # reached end of list
            break

    return chunks


# -------------------------
# Chinese Version
# -------------------------
class VideoCategory(str, Enum):
    makeup = "makeup"
    skincare = "skincare"
    beauty = "beauty"


class Transcript(BaseModel):
    time: str = Field(description="The timestamp of the transcript segment.")
    text: str = Field(description="The text content of the transcript segment.")


class YTSummaryResponse(BaseModel):
    summary_zh: str = Field(description="The summary of the youtube video transcript.")
    title_zh: str = Field(description="The title of the youtube video title")
    summary_en: str = Field(description="English translation of the summary.")
    title_en: str = Field(description="English translation of the title.")
    category: VideoCategory = Field(
        description=(
            "Main beauty category of the video. Choose one of:\n"
            "- makeup: tutorials, reviews, product try-ons\n"
            "- skincare: routines, product reviews, ingredients\n"
            "- haircare: styling, coloring, or hair treatment tips"
        )
    )
    # transcripts: List[Transcript] = Field(description="The list of video transcripts.")


class YoutuberTranscriptProcessor:
    """
    A utility class for managing YouTube transcript data for a specific creator.

    Responsibilities:
    1. Check whether a transcript or processed transcript chunks exist locally.
    2. Load raw transcript text from disk.
    3. Split and summarize the transcript using an LLM.
    4. Generate sliding-window transcript chunks for index-based retrieval.
    5. Cache all outputs so repeated queries for the same video are fast.

    Typical usage in an agent:
    - When the user provides a YouTube video ID or URL:
        1. Determine if the transcript chunks already exist.
        2. If cached, return the cached chunks.
        3. If not cached:
            a. Load the raw transcript
            b. Summarize the transcript using an LLM model
            c. Convert the transcript into overlapping chunks
            d. Save the results for future retrieval

    Inputs:
        youtuber (str): Name of the creator used as the directory key.

    Outputs:
        - Dictionary containing raw transcript data
        - List of dictionaries for transcript chunks
    """

    def __init__(self, youtuber: str = "heyitsmindy"):
        self.youtuber = youtuber
        self.raw_data_dir = RAW_DATA_DIR / youtuber
        self.cache_dir = CACHE_DATA_DIR / youtuber

    def load_transcript(self, video_id):
        """
        Load a raw transcript file for a given YouTube video ID.

        Args:
            video_id (str): The ID of the YouTube video.

        Returns:
            dict: A dictionary containing:
                - "youtuber": the creator's name
                - "video_id": the video ID
                - "transcript": the full transcript text
        """
        filepath = self.raw_data_dir / f"{video_id}.txt"
        with open(filepath, "r") as f:
            transcript = f.read()

        return {
            "youtuber": self.youtuber,
            "video_id": f"{video_id}",
            "transcript": transcript,
        }

    def chunk_transcript(self, video_id, window_size: int = 15, step_size: int = 3):
        """
        Create and return sliding-window transcript chunks for a YouTube video.

        This method automatically:
        - Checks whether a cached JSON file of chunks already exists.
        - If cached → loads and returns the chunks.
        - If not cached → performs full processing:
            1. Loads the raw transcript text.
            2. Splits the transcript into segments.
            3. Generates an LLM-based summary.
            4. Builds sliding-window chunks using the summary + parsed content.
            5. Saves the result to cache.

        Args:
            video_id (str): YouTube video ID.
            window_size (int): Number of transcript segments per sliding window.
            step_size (int): How far the window moves between chunks.

        Returns:
            list[dict]: A list of chunk dictionaries containing transcript segments,
                        metadata, and summary information.
        """

        cache_dir = Path(f"{self.cache_dir}_{window_size}_{step_size}_Chinese")
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = cache_dir / f"{self.youtuber}_{video_id}.json"
        if file_path.exists():
            chunks = json.loads(file_path.read_text(encoding="utf-8"))

        else:
            video = self.load_transcript(video_id=video_id)
            print(f"Processing... {video['video_id']}")
            split_transcript, transcript_str = split_video_to_multiple_transcript(video)

            # summary
            video_summary = youtuber_summarize(
                # instructions=instructions,
                user_prompt=str(split_transcript),
                output_format=YTSummaryResponse,
            )

            # start
            chunks = sliding_window(
                video_id=video["video_id"],
                youtuber=video["youtuber"],
                parsed=split_transcript,
                video_summary=video_summary,
                window_size=window_size,
                step_size=step_size,
            )

            self.save_transcript(file_path, chunks)

        # return chunks

    def save_transcript(self, file_path, chunks):
        """
        Save processed transcript chunks to a JSON file.

        Args:
            file_path (Path): The output file location.
            chunks (list): List of chunk dictionaries to save.

        Notes:
            Writes JSON with UTF-8 encoding and indentation for readability.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)


def youtuber_summarize(
    user_prompt: str, output_format: str, model: str = "gpt-4o-mini"
) -> str:
    """
    Summarize and categorize a YouTube beauty-related transcript using a multilingual LLM.

    Steps performed:
    1. Summarize the transcript in Chinese, highlighting key points, tone, and product mentions.
    2. Translate the summary and video title into English.
    3. Categorize the video into a predefined beauty category (makeup, skincare, haircare).

    Args:
        user_prompt (str): User input or transcript text to summarize.
        output_format (str): The structured output format expected from the LLM (e.g., JSON or YAML).
        model (str, optional): LLM model to use (default: "gpt-4o-mini").

    Returns:
        str: The parsed output content from the LLM according to the specified output_format.

    Example:
        result = youtuber_summarize("This is the transcript...", output_format="YoutubeSummaryOutput")
    """
    instructions = """
    You are a multilingual beauty content analyst.

    Your task is to analyze YouTube transcripts related to beauty, skincare, or makeup content.

    Follow these steps carefully:

    1. **Summarize** the full video transcript in **Chinese**, focusing on key talking points, tone, and product mentions.
    2. **Translate** the summary and title into **English**.
    3. **Categorize** the video into one of the following beauty categories:
    - makeup: tutorials, reviews, product try-ons
    - skincare: routines, product reviews, ingredients
    - haircare: styling, coloring, or hair treatment tips

    """.strip()

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = openai_client.responses.parse(
        model=model, input=messages, text_format=output_format
    )

    return response.output[0].content[0].parsed
