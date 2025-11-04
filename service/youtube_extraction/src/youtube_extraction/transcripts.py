

from typing import Iterable, Dict    
import re
from deep_translator import GoogleTranslator
from youtube_extraction.utils import (youtuber_summarize, YTSummaryResponse, translate_english)


def split_video_to_multiple_transcript(video):
    """
    Convert raw transcript string into list of {time, text} entries.
    """
    lines = video["transcript"].strip().split("\n")
    pattern = re.compile(r"(\d+:\d+)\s*(.*)")
    parsed = []

    transcript = ''
    for line in lines:
        match = pattern.match(line)
        if match:
            time_str, sentence = match.groups()
            parsed.append({"youtuber": video["youtuber"], "video_id": video["video_id"], "time": time_str, "text": sentence.strip()})

            transcript += sentence.strip()
            transcript += ' '
    return parsed, transcript



def sliding_window(parsed, video_summary, window_size=10, step_size=5):
    """
    Create overlapping chunks from a list of transcript dicts.

    Args:
        parsed (list[dict]): Transcript entries with 'text', 'time', etc.
        video_summary: Object with title_en, summary_en, category.
        window_size (int): Number of transcript entries per chunk.
        step_size (int): How many entries to move forward (creates overlap).

    Returns:
        list[dict]: Overlapping transcript chunks.
    """
    chunks = []
    video_id = parsed[0]['video_id']
    youtuber = parsed[0]['youtuber']
    total = len(parsed)
    chunk_id = 1

    for start in range(0, total, step_size):
        end = min(start + window_size, total)
        window = parsed[start:end]

        # If window too small at end, you can skip or include:
        if len(window) < 2:
            break  # stop if not enough data left

        # Combine texts from the window
        text_chunk = " ".join([entry['text'] for entry in window])

        chunks.append({
            'video_id': video_id,
            'youtuber': youtuber,
            'chunk_id': chunk_id,
            'start_time': window[0]['time'],
            'end_time': window[-1]['time'],
            'content': text_chunk,
            'title': video_summary.title_en,
            'summary': video_summary.summary_en,
            'category': video_summary.category.value,
        })

        chunk_id += 1

        if end == total:  # reached end of list
            break

    return chunks


def chunk_transcripts(
    # youtuber: str,
    # video_id: str,
    transcripts: Iterable[Dict[str, str]],
    window_size: int = 10,
    step_size: int = 5,
    translate_or_not: bool = False
    ):
    """
    Chunk transcript based on character length (~400 chars).
    """
    
    results = []

    for video in transcripts:
        # provide trnascript
        print(f'Processing... {video['video_id']}')
        split_transcript, transcript_str = split_video_to_multiple_transcript(video)
        
        # summary
        video_summary = youtuber_summarize(
            # instructions=instructions,
            user_prompt=transcript_str,
            output_format=YTSummaryResponse
        )

        # start
        chunks = sliding_window(
            parsed=split_transcript,
            video_summary=video_summary,
            window_size=window_size,
            step_size=step_size
        )

        if translate_or_not:
            print('Tranlating...')
            for chunk in chunks:
                chunk['content_eng'] = translate_english(chunk['content'])

        results.extend(chunks)

    return results
