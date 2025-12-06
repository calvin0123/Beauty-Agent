from pydantic_ai import RunContext
from dotenv import load_dotenv

load_dotenv()
from pydantic_ai import Agent
from pydantic import BaseModel, Field

from service.agent.src.agent.clarify_agent import create_clarify_agent, ClarifyDecision
from service.agent.src.agent.youtube_agent import (
    create_youtube_agent,
    NamedCallback,
    YoutubeSummaryOutput,
)


orchestrator_instructions = """
You are the Orchestrator Agent for a Beauty YouTube Research System.
You understand English and Chinese.

Your responsibilities:
1. ALWAYS call `clarify_agent` first.
2. Use the clarified user_intent to decide next steps.
3. Call `youtube_agent.search_es()` when required **(maximum 2 times)**.
4. Output ONLY the selected agent’s output.
Follow ALL instructions below exactly.

=========================================================
STEP 0 — ALWAYS CALL CLARIFY AGENT FIRST (MANDATORY)
=========================================================
Before doing anything else, you MUST call `clarify_agent`.

Never skip clarification.
Never assume the user intent.
Never answer directly before clarify.

clarify_agent will return:
- user_intent
- video_url (if detected)
- missing_info_needed
- any other structured fields

=========================================================
STEP 1 — HIGH PRIORITY INTENTS (AUTO-YOUTUBE)
=========================================================
If the clarified user_intent is ANY of:

- find_product
- recommendation
- compare
- video_question

→ You MUST call `youtube_agent` immediately.
→ Do NOT stop.
→ Do NOT wait for the user.
→ Do NOT ask follow-up questions.

This rule overrides ALL other rules.

=========================================================
STEP 2 — STOP AFTER CLARIFY (ONLY WHEN allowed)
=========================================================
ONLY stop after clarify_agent IF:

- user_intent is NONE of:
    find_product, recommendation, compare, video_question
  AND
- You cannot proceed without additional user input
  AND
- The user did NOT request YouTube analysis

Then:
→ Return the clarify_agent result and wait for user response.

=========================================================
STEP 3 — COMBINED INTENTS (SUMMARY + PRODUCT)
=========================================================
If clarify_agent detects that the user wants BOTH:
- a summary AND
- a product/recommendation/compare/video_question

Then:
→ Still trigger youtube_agent.
YouTube Agent is always required if ANY high-priority intent is present.

=========================================================
STEP 4 — OUTPUT FORMAT (STRICT)
=========================================================
Your responses must ALWAYS follow the correct tool invocation schema.

Rules:
- Never output free-text when a tool call is required.
- Never mix tool calls with normal messages.
- Only output normal text when NO tool call is required.

1. **If clarify_agent is called:**  
   → Return ONLY `clarify_agent_output`  
   → No added text. No explanation. No commentary.

2. **If youtube_agent is called:**  
   → Return ONLY `youtube_agent_output`  
   → No added text. No explanation. No commentary.

=========================================================
STEP 5 — CACHED DATABASE RULE
=========================================================
If clarify_agent determines that the video is already in the database:
→ youtube_agent may use cached data.
(YouTube agent will decide automatically.)

=========================================================
SUMMARY OF LOGIC FLOW
=========================================================
1. clarify_agent (ALWAYS)
2. If user_intent in {find_product, recommendation, compare, video_question}:
       → youtube_agent
   Else:
       → stop and wait for user

=========================================================
END OF ORCHESTRATION INSTRUCTIONS
=========================================================

"""


class OrchestratorOutput(BaseModel):
    clarify: ClarifyDecision | None = None
    youtube: YoutubeSummaryOutput | None = None
    final_answer: str | None = None


def create_orchestration_agent():

    clarify_agent = create_clarify_agent()
    youtube_agent = create_youtube_agent()

    orchestrator = Agent(
        name="orchestrator",
        instructions=orchestrator_instructions,
        model="gpt-4o-mini",
        output_type=OrchestratorOutput
    )

    @orchestrator.tool
    async def clarify_tool_initial(ctx: RunContext, query: str) -> str:
        """Runs the clarifier once to interpret the user's request.

        Args:
            query: Raw user question.

        Returns:
            A short text summary describing the user's intent.
        """
        print("\n=== Clarifier (Initial) ===")

        clarifier_callback = NamedCallback(clarify_agent)
        results = await clarify_agent.run(
            user_prompt=query, event_stream_handler=clarifier_callback
        )
        return results.output

    @orchestrator.tool
    async def recommendation_tool(ctx: RunContext, query: str):
        """Retrieve the search result and generate"""

        print("\n=== Youtuber Search (Search) ===")

        # prior_outputs = []
        user_questions = []
        for m in ctx.messages:
            for p in m.parts:
                if p.part_kind == "tool-return" and p.tool_name == "clarify_tool_initial":
                    print(p)
                    # prior_outputs.append(p.content)
                    user_questions.append(p.content.user_intent)

        # prior_text = "\n".join(str(x) for x in prior_outputs)
        prior_user_questions_text = "\n".join(str(x) for x in user_questions)
        print(prior_user_questions_text)

        youtube_callback = NamedCallback(youtube_agent)
        results = await youtube_agent.run(
            user_prompt=query, event_stream_handler=youtube_callback
        )

        return results.output
    
    # orchestrator_callback = NamedCallback(orchestrator)

    return orchestrator

