from minsearch import Index
from pathlib import Path
import pickle
import json
from typing import List, Dict
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from typing import Optional

# ######### Search the content in the transcript

RAW_DATA_DIR = Path("data")
CACHE_DATA_DIR = Path(".cache")
INDEX_DIR = Path(".cache")


# -------------------------
# MinSearch Classes Pre
# -------------------------
class YoutuberTranscriptSearcherPre:
    """
    A search engine wrapper for YouTube video transcripts and summaries.

    This class loads pre-processed transcript chunks stored as JSON files,
    builds or loads a MinSearch-like index, and provides text search across
    titles, summaries, and transcript content.

    Attributes:
        youtuber (str): Name of the YouTuber being indexed.
        cache_dir (Path): Directory containing JSON transcript files.
        index_path (Path): Directory storing the serialized search index.
        search_file_path (Path): Path to the pickled index file.
    """

    def __init__(self, youtuber, window_size: int = 15, step_size: int = 3):
        """
        Initialize the transcript search engine for a specific YouTuber.

        Args:
            youtuber (str): YouTuber identifier. Used for loading/storing data.
            window_size (int): Sliding window size used when chunking transcripts.
            step_size (int): Sliding window step size used when chunking transcripts.
        """
        self.youtuber = youtuber
        self.cache_dir = CACHE_DATA_DIR / f"{youtuber}_{window_size}_{step_size}"
        self.index_path = INDEX_DIR / "search_tool" / youtuber
        self.search_file_path = (
            self.index_path / f"search_tools_{window_size}_{step_size}.bin"
        )

    def search(self, query, num_results: int = 10):
        """
        Perform a search query over indexed transcripts.

        Args:
            query (str): Search query text.
            num_results (int): Maximum number of results to return.

        Returns:
            List[Dict]: Ranked search results from the index.
        """
        self.index = self.load_or_create_index()
        return self.index.search(
            query=query,
            num_results=num_results,
        )

    def load_or_create_index(self):
        """
        Load the existing search index if available.
        Otherwise, build a new one from cached JSON transcript files.

        Returns:
            Index: A fitted MinSearch/BM25 style index.

        Notes:
            - Automatically handles corrupted index files by rebuilding.
            - Ensures index consistency when new data is added.
        """
        if self.search_file_path.exists():
            try:
                with open(self.search_file_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                print("Corrupted index. Rebuilding...")

        index = self.create_index()
        self.save_index(index)
        return index

    def create_index(self):
        """
        Build a new search index from cached transcript JSON files.

        Returns:
            Index: A newly created and fitted index.

        Raises:
            ValueError: If any JSON file does not contain a list of dictionaries.

        Side Effects:
            - Reads all .json files in the cache directory.
        """
        combined = []
        print(self.cache_dir)

        for file in self.cache_dir.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

                if not isinstance(data, list):
                    raise ValueError(
                        f"{file} does not contain a list, got {type(data)}"
                    )

                combined.extend(data)

        index = Index(
            text_fields=["title", "summary", "content", "content_eng"],
            keyword_fields=["youtuber", "category"],
        )

        index.fit(combined)
        return index

    def save_index(self, index):
        """
        Save the index to disk as a pickle file.

        Args:
            index (Index): The trained index object to serialize.

        Side Effects:
            - Writes a binary .bin file to `search_file_path`.
        """
        self.search_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.search_file_path, "wb") as f:
            pickle.dump(index, f)


# -------------------------
# Elastic Search + Minsearch Classes
# -------------------------
class YoutuberTranscriptSearcher:
    """
    Unified search agent for video transcripts, supporting both Elasticsearch and MinSearch.
    """

    def __init__(
        self,
        youtuber="heyitsmindy",
        backend="elasticsearch",
        index_name="youtube_bilingual",
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir="heyitsmindy_15_3_Chinese",
    ):
        """
        Initialize the transcript searcher.

        Args:
            youtuber (str): YouTuber whose videos are being indexed/searched.
            backend (str): Backend for search ('elasticsearch' or 'minsearch').
            index_name (str): Name of the Elasticsearch index.
            model_name (str): SentenceTransformer model for embedding text.
            cache_dir (str): Directory for cached JSON transcripts (required for minsearch backend).

        Raises:
            ValueError: If backend is unsupported or cache_dir is missing for minsearch.
        """
        self.backend = backend
        self.index_name = index_name
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.youtuber = youtuber
        self.cache_dir = CACHE_DATA_DIR / f"{youtuber}_15_3_Chinese"
        # self.index_path = INDEX_DIR / "search_tool" / youtuber
        # self.search_file_path = self.index_path / f"search_tools_{window_size}_{step_size}.bin"

        if backend == "elasticsearch":
            self.es = Elasticsearch("http://localhost:9200")
            self.ensure_es_index()
        elif backend == "minsearch":
            if not self.cache_dir:
                raise ValueError("cache_dir must be provided for minsearch backend")
            self.searcher = self.load_or_create_minsearch_index()
        else:
            raise ValueError("backend must be 'elasticsearch' or 'minsearch'")

    # -------------------------
    # Elasticsearch Methods
    # -------------------------
    def ensure_es_index(self):
        """Check if ES index exists; create if missing."""
        if not self.es.indices.exists(index=self.index_name):
            print("Creating new index name")
            index_body = {
                "mappings": {
                    "properties": {
                        "video_id": {"type": "keyword"},
                        "youtuber": {"type": "keyword"},
                        "category": {"type": "keyword"},
                        "title": {"type": "text"},
                        "summary": {"type": "text"},
                        "title_zh": {"type": "text"},
                        "summary_zh": {"type": "text"},
                        "content": {"type": "text"},
                        "chunk_id": {"type": "integer"},
                        "start_time": {"type": "keyword"},
                        "end_time": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine",
                        },
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=index_body, ignore=400)

            combined = []

            for file in self.cache_dir.glob("*.json"):
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError(f"{file} does not contain a list")
                    combined.extend(data)

            self.index_chunks_es(combined)

    def check_index_es(self, video_id: str) -> List[dict]:
        """
        Check if a transcript for a specific video is cached.
        If not cached, chunk the transcript, save to cache, and index in Elasticsearch.

        Args:
            video_id (str): YouTube video ID to check and index.

        Returns:
            List[dict]: List of transcript chunks for the video.
        """
        self.search_data_path = self.cache_dir / f"{self.youtuber}_{video_id}.json"

        if self.search_data_path.exists():
            current_directory = os.getcwd()
            full_file_path = os.path.join(current_directory, self.search_data_path)
            print(full_file_path)
            with open(full_file_path, "r") as f:
                chunks = json.load(f)

            # else:
            # from service.youtube_extraction.src.youtube_extraction.youtube_process import YoutuberTranscriptProcessor
            # ytp = YoutuberTranscriptProcessor(youtuber=self.youtuber)
            # chunks = ytp.chunk_transcript(video_id=video_id)

            # with open(self.search_data_path, 'w') as f:
            #     json.dump(chunks, f, ensure_ascii=False, indent=4) # indent for pretty-printing

            # self.index_chunks_es(chunks)
            # print("Added new chunk to elasticsearch!")

            return chunks

    def index_chunks_es(self, chunks) -> List[Dict]:
        """
        Index a list of transcript chunks into Elasticsearch, adding embeddings.

        Args:
            chunks (list): List of transcript chunks (dicts) with keys 'title_zh', 'content', etc.
        """
        for chunk in tqdm(chunks):
            text_to_embed = (
                # f"{chunk['title_zh']} {chunk['summary_zh']} {chunk['content']}"
                f"{chunk['title_zh']} {chunk['content']}"
            )
            embedding = self.model.encode(text_to_embed).tolist()
            chunk["embedding"] = embedding
            self.es.index(index=self.index_name, document=chunk)
        self.es.indices.refresh(index=self.index_name)

    def search_es(self, query_text, k=5):
        """
        Perform semantic search in Elasticsearch using a query.

        Args:
            query_text (str): The query string to search for.
            k (int): Number of top results to return.

        Returns:
            List[dict]: Top-k search results with keys:
                - score (float): Similarity score.
                - video_id (str): Video ID.
                - summary (str): English summary.
                - summary_zh (str): Chinese summary.
                - title (str): Video title.
                - content (str): Chunk content.
        """
        query_emb = self.model.encode(query_text).tolist()
        response = self.es.search(
            index=self.index_name,
            knn={
                "field": "embedding",
                "query_vector": query_emb,
                "k": k,
                "num_candidates": max(50, k),
            },
        )
        results = [
            {
                "score": hit["_score"],
                "video_id": hit["_source"]["video_id"],
                # "summary": hit["_source"]["summary"],
                # "summary_zh": hit["_source"]["summary_zh"],
                "title": hit["_source"].get("title"),
                "start_time": hit["_source"].get("start_time"),
                "content": hit["_source"].get("content"),
            }
            for hit in response["hits"]["hits"]
        ]
        return results

    # -------------------------
    # MinSearch Methods
    # -------------------------
    def load_or_create_minsearch_index(self):
        search_file_path = self.cache_dir / "search_tools.bin"
        if search_file_path.exists():
            try:
                with open(search_file_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                print("Corrupted index. Rebuilding...")

        return self.create_minsearch_index(search_file_path)

    def create_minsearch_index(self, search_file_path):
        combined = []
        for file in self.cache_dir.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"{file} does not contain a list")
                combined.extend(data)

        # Here `Index` is assumed to be your MinSearch index class
        index = Index(
            text_fields=["title", "summary", "content", "content_eng"],
            keyword_fields=["youtuber", "category"],
        )
        index.fit(combined)

        search_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(search_file_path, "wb") as f:
            pickle.dump(index, f)
        return index

    def search_minsearch(self, query_text, num_results=10):
        return self.searcher.search(query=query_text, num_results=num_results)

    # -------------------------
    # Unified search interface
    # -------------------------
    def search(self, query_text, k=10):
        if self.backend == "elasticsearch":
            return self.search_es(query_text, k)
        elif self.backend == "minsearch":
            return self.search_minsearch(query_text, num_results=k)


if __name__ == "__main__":
    yts = YoutuberTranscriptSearcher()
