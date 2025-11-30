"""
Previous version
"""

from minsearch import Index
from typing import Any, List, Dict
from pathlib import Path
import pickle
from youtube_extraction.transcripts import chunk_transcripts
import os


class SearchTools:
    def __init__(self, index: Index, top_k: int):
        self.index = index
        # self.file_index = file_index
        self.top_k = top_k

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the index for documents matching the given query.

        Args:
            query (str): The search query string.

        Returns:
            A list of search results
        """
        return self.index.search(
            query=query,
            num_results=5,
        )

    def show_video(self):
        pass


def load_all_videos(youtuber: str = "heyitsmindyy"):
    folder_path = f"data/{youtuber}"
    parsed_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            video_id = filename.replace(".txt", "")
            with open(os.path.join(folder_path, filename), "r") as f:
                transcript = f.read()

            parsed_data.append(
                {"youtuber": youtuber, "video_id": video_id, "transcript": transcript}
            )

    return parsed_data


def prepare_search_index(parsed_data, chunk_size: int, chunk_step: int):
    needed = ["2SvN45DKWFg", "KNR7lyrTThg", "LGmXiNc-TCI", "Nt_dYGI73RI"]
    # needed = ['2SvN45DKWFg']
    sample = []
    for parsed in parsed_data:
        if parsed["video_id"] in needed:
            sample.append(parsed)
    # print(sample)
    ###

    chunks = chunk_transcripts(
        sample, window_size=chunk_size, step_size=chunk_step, translate_or_not=True
    )

    index = Index(
        text_fields=["title", "summary", "content", "content_eng"],
        keyword_fields=["youtuber", "category"],
    )

    index.fit(chunks)

    return index


def _prepare_search_tools(chunk_size: int, chunk_step: int, top_k: int):
    parsed_data = load_all_videos()

    search_index = prepare_search_index(
        parsed_data=parsed_data,
        # parsed_data=sample,
        chunk_size=chunk_size,
        chunk_step=chunk_step,
    )

    # file_index = prepare_file_index(parsed_data=parsed_data)

    return SearchTools(
        index=search_index,
        # file_index=file_index,
        top_k=top_k,
    )


def prepare_search_tools(chunk_size: int, chunk_step: int, top_k: int):
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"search_tools_{chunk_size}_{chunk_step}_{top_k}.bin"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            search_tools = pickle.load(f)
            return search_tools

    search_tools = _prepare_search_tools(
        chunk_size=chunk_size, chunk_step=chunk_step, top_k=top_k
    )

    with open(cache_file, "wb") as f:
        pickle.dump(search_tools, f)

    return search_tools


if __name__ == "__main__":
    prepare_search_tools(chunk_size=15, chunk_step=3, top_k=5)
