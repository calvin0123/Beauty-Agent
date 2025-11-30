from pathlib import Path
from typing import List
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from googleapiclient.discovery import build
from tqdm.auto import tqdm
import os
import json
from dotenv import load_dotenv
load_dotenv()


class YoutubeParser:
    """
    A class to parse YouTube channels, fetch video IDs, and download transcripts.

    Attributes:
        youtube (googleapiclient.discovery.Resource): YouTube API client.
        data_dir (Path): Directory to save video IDs and transcripts.
    """

    def __init__(self, api_key: str, youtuber: str = 'heyitsmindy', data_dir: str = "data") -> None:
        """
        Initialize YoutubeParser with YouTube API key and data directory.

        Args:
            api_key (str): YouTube Data API key.
            data_dir (str): Directory to save data files. Defaults to 'data'.
        """
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.youtuber = youtuber

    def get_all_video_id(self, channel_id: str = 'UCYoNeJRnYGcReuBatFl83Cw') -> List[str]:
        """
        Retrieve all video IDs from a YouTube channel.

        Args:
            channel_id (str): The YouTube channel ID.

        Returns:
            List[str]: List of video IDs in the channel.
        """
        channel_response = (
            self.youtube.channels().list(id=channel_id, part="contentDetails").execute()
        )

        uploads_playlist_id = channel_response["items"][0]["contentDetails"][
            "relatedPlaylists"
        ]["uploads"]

        videos = []
        next_page_token = None

        while True:
            playlist_items = (
                self.youtube.playlistItems()
                .list(
                    playlistId=uploads_playlist_id,
                    part="contentDetails",
                    maxResults=50,
                    pageToken=next_page_token,
                )
                .execute()
            )

            for item in playlist_items["items"]:
                video_id = item["contentDetails"]["videoId"]
                published_at = item["contentDetails"]["videoPublishedAt"]
                videos.append({"videoId": video_id, "publishedAt": published_at})
            next_page_token = playlist_items.get("nextPageToken")
            if not next_page_token:
                break

        # sort by timestamp (descending) to get the latest
        videos_sorted = sorted(videos, key=lambda v: v["publishedAt"], reverse=True)
        latest_video = videos_sorted[0] if videos_sorted else None

        # save videos
        file_dir = self.data_dir / f"{self.youtuber}" / 'videos'
        file_dir.mkdir(exist_ok=True)
        filepath = os.path.join(file_dir, f"{self.youtuber}.json")
        with open(filepath, "w") as f:
            json.dump(videos_sorted[:20], f, indent=2)
        
        return videos[:20]

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

        if filename.endswith(".txt"):
            return [
                line.strip()
                for line in filepath.read_text(encoding="utf-8").splitlines()
            ]
        elif filename.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

    def _fetch_transcript(self, video_id: str) -> List[dict]:
        """
        Fetch the transcript of a YouTube video using multiple language fallbacks.

        Args:
            video_id (str): Video ID.

        Returns:
            List[dict]: Transcript entries with 'start', 'duration', and 'text'.
        """
        ytt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=os.environ["PROXY_USERNAME"],
                proxy_password=os.environ["PROXY_PASSWORD"],
                filter_ip_locations=["us", "de", "fr"],
            )
        )
        transcript = ytt_api.fetch(video_id, languages=["zh-TW", "ko", "en"])
        return transcript

    def _make_subtitles(self, transcript: List[dict]) -> str:
        """
        Convert a transcript list into subtitle text.

        Args:
            transcript (List[dict]): List of transcript entries.

        Returns:
            str: Formatted subtitle text with timestamps.
        """
        lines = []

        for entry in transcript:
            ts = self._format_timestamp(entry.start)
            text = entry.text.replace("\n", " ")
            lines.append(ts + " " + text)

        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
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
        result_dir = self.data_dir / youtuber / 'transcript'
        result_dir.mkdir(exist_ok=True)
        result_file = result_dir / f"{video_id}.txt"
        if result_file.exists():
            print(f"{result_file} already exists, skipping it")
            return

        print(f"Start fetching {video_id}")
        transcript = self._fetch_transcript(video_id)
        subtitles = self._make_subtitles(transcript)
        if subtitles:
            result_file.write_text(subtitles, encoding="utf-8")
            print(f"Saved subtitles to {result_file}")
        else:
            print(f"No subtitles for {video_id}, skipped.")

    def download_all_transcripts(self, youtuber: str, video_ids: List[str] = None) -> None:
        """
        Download transcripts for a list of video IDs.

        Args:
            video_ids (List[str]): List of video IDs.
        """

        if not video_ids:
            print('Retrieving the new videos...')
            self.get_all_video_id()
            filepath =  f"{youtuber}/videos/{youtuber}.json"
            video_ids = self.read_video_ids(filepath)


        for vid in tqdm(video_ids):
            try:
                self.download(youtuber, vid['videoId'])
            except Exception as e:
                print(f"Failed to download {youtuber}: {vid}: {e}")


if __name__ == '__main__':
    yp = YoutubeParser(api_key=os.environ['GOOGLE_API_KEY'])
    yp.download_all_transcripts(youtuber='heyitsmindy')
