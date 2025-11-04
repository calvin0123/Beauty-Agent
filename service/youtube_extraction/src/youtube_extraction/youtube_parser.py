from pathlib import Path
from typing import List
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from googleapiclient.discovery import build
from tqdm.auto import tqdm
import os


class YoutubeParser:
    """
    A class to parse YouTube channels, fetch video IDs, and download transcripts.

    Attributes:
        youtube (googleapiclient.discovery.Resource): YouTube API client.
        data_dir (Path): Directory to save video IDs and transcripts.
    """

    def __init__(self, api_key: str, data_dir: str = "data") -> None:
        """
        Initialize YoutubeParser with YouTube API key and data directory.

        Args:
            api_key (str): YouTube Data API key.
            data_dir (str): Directory to save data files. Defaults to 'data'.
        """
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def get_all_video_id(self, channel_id: str) -> List[str]:
        """
        Retrieve all video IDs from a YouTube channel.

        Args:
            channel_id (str): The YouTube channel ID.

        Returns:
            List[str]: List of video IDs in the channel.
        """
        channel_response = self.youtube.channels().list(
            id=channel_id,
            part="contentDetails"
        ).execute()

        uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

        videos = []
        next_page_token = None

        while True:
            playlist_items = self.youtube.playlistItems().list(
                playlistId=uploads_playlist_id,
                part="contentDetails",
                maxResults=50,
                pageToken=next_page_token
            ).execute()

            for item in playlist_items["items"]:
                videos.append(item["contentDetails"]["videoId"])

            next_page_token = playlist_items.get("nextPageToken")
            if not next_page_token:
                break

        return videos

    def save_video_ids(self, channel_id: str, filename: str = None) -> Path:
        """
        Get video IDs from a channel and save them to a file.

        Args:
            channel_id (str): YouTube channel ID.
            filename (str, optional): Filename to save video IDs. Defaults to '{channel_id}_video_ids.txt'.

        Returns:
            Path: Path to the saved file.
        """
        video_ids = self.get_all_video_id(channel_id)
        filename = filename or f"{channel_id}_video_ids.txt"
        filepath = self.data_dir / filename
        filepath.write_text("\n".join(video_ids), encoding="utf-8")
        print(f"Saved {len(video_ids)} video IDs to {filepath}")
        return filepath

    def read_video_ids(self, filename: str) -> List[str]:
        """
        Read video IDs from a file.

        Args:
            filename (str): Filename containing video IDs.

        Returns:
            List[str]: List of video IDs.
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist")
        return [line.strip() for line in filepath.read_text(encoding="utf-8").splitlines()]

    def fetch_transcript(self, video_id: str) -> List[dict]:
        """
        Fetch the transcript of a YouTube video using multiple language fallbacks.

        Args:
            video_id (str): Video ID.

        Returns:
            List[dict]: Transcript entries with 'start', 'duration', and 'text'.
        """
        ytt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=os.environ['PROXY_USERNAME'],
                proxy_password=os.environ['PROXY_PASSWORD'],
                filter_ip_locations=["us", "de", "fr"]
            )
        )
        transcript = ytt_api.fetch(video_id, languages=['zh-TW', 'ko', 'en'])
        return transcript

    def make_subtitles(self, transcript: List[dict]) -> str:
        """
        Convert a transcript list into subtitle text.

        Args:
            transcript (List[dict]): List of transcript entries.

        Returns:
            str: Formatted subtitle text with timestamps.
        """
        lines = []
        for entry in transcript:
            ts = self.format_timestamp(entry['start'])
            text = entry['text'].replace('\n', ' ')
            lines.append(f"{ts} {text}")
        return "\n".join(lines)

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Convert seconds to timestamp string (H:MM:SS or M:SS).

        Args:
            seconds (float): Seconds to format.

        Returns:
            str: Formatted timestamp.
        """
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02}:{secs:02}"
        else:
            return f"{minutes}:{secs:02}"

    def download(self, youtuber, video_id: str) -> None:
        """
        Download the transcript for a video and save it to a file.

        Args:
            video_id (str): YouTube video ID.
        """
        result_file = self.data_dir / youtuber / f"{video_id}.txt"
        if result_file.exists():
            print(f"{result_file} already exists, skipping it")
            return

        transcript = self.fetch_transcript(video_id)
        subtitles = self.make_subtitles(transcript)
        if subtitles:
            result_file.write_text(subtitles, encoding="utf-8")
            print(f"Saved subtitles to {result_file}")
        else:
            print(f"No subtitles for {video_id}, skipped.")

    def download_all_transcripts(self, youtuber: str, video_ids: List[str]) -> None:
        """
        Download transcripts for a list of video IDs.

        Args:
            video_ids (List[str]): List of video IDs.
        """
        for vid in tqdm(video_ids):
            try:
                self.download(youtuber, vid)
            except Exception as e:
                print(f"Failed to download {vid}: {e}")
