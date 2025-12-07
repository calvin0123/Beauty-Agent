import asyncio
from typing import Any, Dict
from jaxn import JSONParserHandler, StreamingJSONParser

from service.agent.src.agent import youtube_agent
from service.agent.src.agent import orchestrator_agent
import logfire
from dotenv import load_dotenv

load_dotenv()


logfire.configure()
logfire.instrument_pydantic_ai()


class SearchResultArticleHandler(JSONParserHandler):
    def on_field_start(self, path: str, field_name: str) -> None:
        if field_name == "products":
            print("\n## Recommended Products\n")

    def on_field_end(
        self, path: str, field_name: str, value: str, parsed_value: Any = None
    ) -> None:
        # ############# Clarify search #############
        # print(f"Debug Youtube agent: {path} || Field Name {field_name}")
        if path == "/clarify":
            if path == "/clarify" and field_name == "youtuber":
                print("\n=== Clarify Agent Output ===\n")
                print(f"YouTuber     : {value}")
            if path == "/clarify" and field_name == "user_intent":
                print(f"Intent       : {value}")
            if path == "/clarify" and field_name == "topic":
                print(f"Topic        : {value}")
            if path == "/clarify" and field_name == "video_id":
                print(f"Video ID     : {value}\n")
            if path == "/clarify" and field_name == "in_cache":
                print(f"In Cache  : {value}\n")
        # ############# Clarify search #############

        # ############# Youtuber search #############

        # Handle top-level video fields
        if path == "/youtube" or path == "":
            if field_name == "title":
                # print(f"# {value}\n")
                self.title = value
                print("=== Youtube Beauty Search Output ===")
            if field_name == "youtuber":
                self.youtuber = value
                # print(f"Youtuber: {value}")
            elif field_name == "url":
                self.video_url = value
                # print(f"URL: {value}")
            elif field_name == "category":
                self.category = value
                # print(f"Category: {value}\n")
        # elif path == "" and field_name == "summary":
        #     print(f"Summary:\n{value}\n")
        # ############# Youtuber search #############

    def on_value_chunk(self, path: str, field_name: str, chunk: str) -> None:
        # TODO: how can i pring the recommendation output in stream
        pass

    def on_array_item_end(
        self, path: str, field_name: str, item: Dict[str, Any] = None
    ) -> None:
        # ############# Clarify #############
        if field_name == "available_videos" and item is not None:
            # print(
            #     "Hey I can start answer your question!\nHere are Available Videos (metadata):\n"
            # )
            print(f"   Title  : {item.get('title')}")
            print(f"   Summary: {item.get('summary')}")
            print(f"   URL    : {item.get('url')}\n")
        # ############# Clarify #############

        # ############# Youtuber search #############
        if field_name == "products" and item is not None:
            # Print each product nicely

            print(f"* Product: {item.get('name')}")
            print(f"  ** Recommended: {item.get('recommend')}")
            print(f"  ** Reason: {item.get('reason')}")
            print(f"  ** Transcript: {item.get('transcript')}\n")
            start_time = item.get("start_time")
            if start_time:
                # print(f"  [Video Reference]:")
                print(" [Video Reference]:")
                print(f" Youtuber: {self.youtuber}")
                print(f" Title: {self.title}")
                print(f" URL:  {self.video_url}?t={self.time_to_seconds(start_time)}\n")
            print("======================\n")
        # ############# Youtuber search #############

    @staticmethod
    def time_to_seconds(time_str: str) -> int:
        """Convert a time string MM:SS or HH:MM:SS to total seconds."""
        parts = [int(p) for p in time_str.split(":")]
        if len(parts) == 2:  # MM:SS
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:  # HH:MM:SS
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return 0


async def main(use_agent):
    user_input = "CLIO Pro Eye Palette Air Heritage Edition"
    user_input = "韓國推薦粉底液"

    if use_agent == "youtube":
        agent = youtube_agent.create_youtube_agent()
        callback = youtube_agent.NamedCallback(agent)
    else:
        user_input = "Help me understand  https://www.youtube.com/watch?v=KNR7lyrTThg from heyitsmindy"
        agent = orchestrator_agent.create_orchestration_agent()
        callback = youtube_agent.NamedCallback(agent)

    handler = SearchResultArticleHandler()
    parser = StreamingJSONParser(handler)

    previous_text = ""

    async with agent.run_stream(user_input, event_stream_handler=callback) as result:
        async for item, last in result.stream_responses(debounce_by=0.01):
            for part in item.parts:
                if not hasattr(part, "tool_name"):
                    continue
                if part.tool_name != "final_result":
                    continue

                current_text = part.args
                delta = current_text[len(previous_text) :]
                # print(delta, end="", flush=True)
                parser.parse_incremental(delta)
                previous_text = current_text

            # print(article.format_article())


if __name__ == "__main__":
    use_agent = "youtube"
    use_agent = "orchestrator"

    # if use_agent == 'youtube':
    asyncio.run(main(use_agent))
