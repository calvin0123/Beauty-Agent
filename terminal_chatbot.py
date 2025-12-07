import json


class StdOutputInterface:
    def input(self) -> str:
        """
        Get input from the user.
        Returns:
            str: The user's input.
        """
        question = input("You: ")
        return question.strip()

    def display(self, message: str) -> None:
        """
        Display a message.
        Args:
            message: The message to display.
        """
        print("--- Final Answer ---")
        print(message)
        print("-------------------")
        print()

    def display_function_call(
        self, function_name: str, arguments: str, result: str
    ) -> None:
        """
        Display a function call.
        Args:
            function_name: The name of the function to call.
            arguments: The arguments to pass to the function.
            result: The result of the function call.
        """
        print()
        print("--- Function Call ---")
        print(f"Function: {function_name}")
        print(f"Arguments: {arguments}")
        print(f"Result: {result}")
        print("-------------------")
        print()

    def display_response(self, markdown_text: str) -> None:
        """
        Display a response.
        Args:
            markdown_text: The markdown text to display.
        """
        print()
        print(f"Assistant: {markdown_text}\n")

    def display_reasoning(self, markdown_text: str) -> None:
        """
        Display a reasoning.
        Args:
            markdown_text: The markdown text to display.
        """
        print()
        print("--- Reasoning ---")
        print(markdown_text)
        print("---------------")
        print()


class PydanticAIRunner:
    """Runner for Pydantic AI."""

    def __init__(self, chat_interface: StdOutputInterface, agent):
        self.chat_interface = chat_interface
        self.agent = agent

    async def run(self, which_agent: str = "orchestrator") -> None:
        message_history = []

        while True:
            user_input = self.chat_interface.input()
            if user_input.lower() == "stop":
                self.chat_interface.display("Chat ended.")
                break

            result = await self.agent.run(
                user_prompt=user_input, message_history=message_history
            )

            if which_agent == "youtube":
                result.output.print_youtube_summary()
                # output = result.output.format_youtube_summary()
            else:
                if result.output.clarify:
                    # result.output.clarify.print_agent_output()
                    clarify_output = result.output.clarify.format_agent_output()
                    print("\n\n")
                    self.chat_interface.display(clarify_output)

                if result.output.youtube:
                    # result.output.youtube.print_youtube_summary()
                    product_recommend_output = (
                        result.output.youtube.format_youtube_summary()
                    )
                    print("\n\n")
                    self.chat_interface.display(product_recommend_output)

            messages = result.new_messages()

            tool_calls = {}

            for m in messages:
                for part in m.parts:
                    kind = part.part_kind

                    if kind == "text":
                        self.chat_interface.display_response(part.content)

                    if kind == "tool-call":
                        call_id = part.tool_call_id
                        tool_calls[call_id] = part

                    if kind == "tool-return":
                        call_id = part.tool_call_id
                        call = tool_calls[call_id]
                        result = part.content
                        self.chat_interface.display_function_call(
                            call.tool_name, json.dumps(call.args), result
                        )
                        if call.tool_name == "final_result":
                            self.chat_interface.display(part.content)

            message_history.extend(messages)


if __name__ == "__main__":
    from service.agent.src.agent.youtube_agent import (
        create_youtube_agent,
    )
    from service.agent.src.agent.orchestrator_agent import create_orchestration_agent

    import asyncio

    chat_interface = StdOutputInterface()
    agent = "youtube"
    agent = "orchestrator"

    if agent == "youtube":
        youtube_agent = create_youtube_agent()
        agent_runner = PydanticAIRunner(
            chat_interface=chat_interface, agent=youtube_agent
        )
    else:
        orchestrate_agent = create_orchestration_agent()
        agent_runner = PydanticAIRunner(
            chat_interface=chat_interface, agent=orchestrate_agent
        )

    # youtube_callback = NamedCallback(youtube_agent)
    asyncio.run(agent_runner.run(which_agent=agent))
