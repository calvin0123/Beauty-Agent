import asyncio
import json
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from service.agent.src.agent.youtube_agent import (
    YoutubeSummaryOutput,
    create_youtube_agent,
)

load_dotenv()

st.set_page_config(page_title="YouTube Beauty Chatbot", page_icon=":speech_balloon:")


def init_session_state() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent = create_youtube_agent()
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "stopped" not in st.session_state:
        st.session_state.stopped = False


def reset_conversation() -> None:
    st.session_state.agent = create_youtube_agent()
    st.session_state.message_history = []
    st.session_state.chat_messages = []
    st.session_state.stopped = False


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2)
    except TypeError:
        return str(value)


def format_structured_output(output: Optional[YoutubeSummaryOutput]) -> Optional[str]:
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

    if output is None:
        return None

    lines: List[str] = []
    lines.append(f"**Title:** {output.title}\n")
    lines.append(f"**Youtuber:** {output.youtuber}\n")
    lines.append(f"**Category:** {output.category}\n")
    lines.append(f"**URL:** {output.url}\n")
    lines.append("")
    lines.append(f"**Summary:** {output.summary}")
    lines.append("")
    for idx, product in enumerate(output.products, start=1):
        lines.append(f"{idx}. **{product.name}** (Start: {product.start_time})")
        recommendation = "Yes" if product.recommend else "No"
        lines.append(f"   - Recommended: {recommendation}")
        lines.append(f"   - Reason: {product.reason}")
        lines.append(f"   - Transcript: {product.transcript}")
        lines.append("")

    return "\n".join(lines).strip()


def collect_agent_response(messages: List[Any]) -> tuple[str, List[Dict[str, str]]]:
    text_chunks: List[str] = []
    tool_logs: List[Dict[str, str]] = []
    tool_calls: Dict[str, Any] = {}

    for message in messages:
        for part in getattr(message, "parts", []):
            kind = getattr(part, "part_kind", "")
            if kind == "text":
                text_chunks.append(str(part.content))
            elif kind == "tool-call":
                tool_calls[part.tool_call_id] = part
            elif kind == "tool-return":
                call = tool_calls.get(part.tool_call_id)
                tool_name = getattr(call, "tool_name", "tool") if call else "tool"
                args_value = getattr(call, "args", {}) if call else {}
                tool_logs.append(
                    {
                        "name": tool_name,
                        "args_text": stringify(args_value),
                        "result_text": stringify(part.content),
                    }
                )

    assistant_text = "\n\n".join(
        chunk for chunk in text_chunks if chunk.strip()
    ).strip()
    return assistant_text, tool_logs


def invoke_agent(prompt: str):
    agent = st.session_state.agent
    history = st.session_state.message_history
    return asyncio.run(agent.run(user_prompt=prompt, message_history=history))


def render_chat_history() -> None:
    for message in st.session_state.chat_messages:
        chat = st.chat_message(message["role"])
        chat.markdown(message["content"])
        if message["role"] == "assistant":
            for event in message.get("tool_events", []):
                with chat.expander(f"Tool call: {event['name']}"):
                    st.write("Arguments")
                    st.code(event["args_text"])
                    st.write("Result")
                    st.code(event["result_text"])


def handle_user_prompt(prompt: str) -> None:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        return

    user_entry = {"role": "user", "content": cleaned_prompt}
    st.session_state.chat_messages.append(user_entry)
    st.chat_message("user").markdown(cleaned_prompt)

    if cleaned_prompt.lower() == "stop":
        st.session_state.stopped = True
        stop_message = (
            "Chat ended. Select 'Reset conversation' in the sidebar to start again."
        )
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": stop_message}
        )
        st.chat_message("assistant").markdown(stop_message)
        return

    try:
        with st.spinner("Searching YouTube transcripts..."):
            result = invoke_agent(cleaned_prompt)
    except Exception as exc:  # noqa: BLE001
        error_text = f"An error occurred while calling the agent: {exc}"
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": error_text}
        )
        st.chat_message("assistant").markdown(error_text)
        return

    # print(f'debug: {result}')
    new_messages = result.new_messages()
    st.session_state.message_history.extend(new_messages)

    assistant_text, tool_logs = collect_agent_response(new_messages)

    # Try to parse structured output into YoutubeSummaryOutput
    # structured_output = getattr(result, "data", None)
    structured_output = result.output
    if structured_output:
        # structured_output.display_streamlit()
        combined_text = (
            assistant_text
            or format_structured_output(structured_output)
            or "The agent did not return any text."
        )
        assistant_block = st.chat_message("assistant")
        assistant_block.markdown(combined_text)
        # combined_text = None
    else:
        combined_text = "The agent did not return any text."
        assistant_block.markdown(combined_text)
    # if isinstance(result.output, YoutubeSummaryOutput):
    #     # Display nicely in Streamlit
    #     structured_output.display_streamlit()
    #     combined_text = None  # Already displayed
    # else:
    #     combined_text = assistant_text or format_structured_output(structured_output) or "The agent did not return any text."
    #     assistant_block = st.chat_message("assistant")
    #     assistant_block.markdown(combined_text)

    # structured_text = format_structured_output(getattr(result, "data", None))
    # combined_text = assistant_text or structured_text or "The agent did not return any text."

    assistant_entry: Dict[str, Any] = {"role": "assistant", "content": combined_text}
    if tool_logs:
        assistant_entry["tool_events"] = tool_logs

    st.session_state.chat_messages.append(assistant_entry)

    # assistant_block = st.chat_message("assistant")
    # assistant_block.markdown(combined_text)
    for event in tool_logs:
        with assistant_block.expander(f"Tool call: {event['name']}"):
            st.write("Arguments")
            st.code(event["args_text"])
            st.write("Result")
            st.code(event["result_text"])


def main() -> None:
    init_session_state()

    with st.sidebar:
        st.header("Controls")
        st.write("Ask the agent about beauty products mentioned on YouTube.")
        st.write("Type 'stop' in the chat to end the session.")
        if st.button("Reset conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

    st.title("YouTube Beauty Chatbot")
    st.caption(
        "Converse with the Pydantic AI YouTube agent and keep the history alive until you type 'stop'."
    )

    if st.session_state.stopped:
        st.info("Conversation stopped. Reset to start a new session.")

    render_chat_history()

    prompt = st.chat_input(
        "What beauty topic are you curious about?",
        disabled=st.session_state.stopped,
    )
    if prompt is not None:
        handle_user_prompt(prompt)


if __name__ == "__main__":
    main()
