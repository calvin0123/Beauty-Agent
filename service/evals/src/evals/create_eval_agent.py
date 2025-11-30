from enum import Enum
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()


class CheckName(str, Enum):
    instructions_follow = "instructions_follow"
    instructions_avoid = "instructions_avoid"

    answer_relevant = "answer_relevant"
    answer_clear = "answer_clear"
    completeness = "completeness"

    reason_relevant = "reason_relevant"
    reason_reference_relevant = "reason_reference_relevant"
    reason_citations = "reason_citations"
    reason_completeness = "reason_completeness"


CHECK_DESCRIPTIONS = {
    CheckName.instructions_follow: "The agent followed all required instructions in <INSTRUCTIONS>.",
    CheckName.instructions_avoid: "The agent avoided doing anything that was explicitly forbidden.",
    # --- Answer-level evaluation ---
    CheckName.answer_relevant: "The final answer (<ANSWER>) directly addresses the user's question (<QUESTION>) "
    "without drifting into unrelated content.",
    CheckName.answer_clear: "The final answer (<ANSWER>) is clear, logically structured, and correct.",
    CheckName.completeness: "The final answer (<ANSWER>) fully covers all required aspects of the user request.",
    # --- Reasoning / reference-grounding evaluation ---
    CheckName.reason_relevant: "The reasoning in (<DETAILS ANSWER>) logically supports the final answer. "
    "It is relevant, non-contradictory, and stays within the scope of the question.",
    CheckName.reason_reference_relevant: "The reasoning in (<DETAILS ANSWER>) is consistent with the ground-truth (<REFERENCE>). "
    "It mentions the same key products/topics and uses similar reasoning logic. "
    "No hallucinated products or claims appear.",
    CheckName.reason_citations: "The reasoning correctly references the relevant products, items, or transcript sources "
    "that appear in (<REFERENCE>). It does not introduce content not present in <REFERENCE>.",
    CheckName.reason_completeness: "The reasoning in (<DETAILS ANSWER>) covers all products, topics, and key information "
    "present in the ground-truth (<REFERENCE>), without missing important details.",
}


class EvaluationCheck(BaseModel):
    check_name: CheckName = Field(description="The type of evaluation check")
    reasoning: str = Field(description="The reasoning behind the check result")
    check_pass: bool = Field(
        description="Whether the check passed (True) or failed (False)"
    )


class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck] = Field(
        description="List of all evaluation checks"
    )
    summary: str = Field(description="Evaluation summary")


def generate_checklist_text():
    checklist_items = []
    for check_name in CheckName:
        description = CHECK_DESCRIPTIONS[check_name]
        checklist_items.append(f"- {check_name.value}: {description}")
    return "\n".join(checklist_items)


def create_eval_agent():
    eval_instructions = f"""
        You are evaluating the quality of an AI agent’s responses: (<ANSWER>) and (<DETAILS ANSWER>)
        to a user question (<QUESTION>). You are also given:

        - (<LOG>): the full agent interaction log  
        - (<REFERENCE>): the ground-truth detailed answer that the user question was originally derived from

        Your task:
        Evaluate whether (<DETAILS ANSWER>) is correct and SIMILAR based on (<REFERENCE>),
        and evaluate whether (<ANSWER>) correctly answers (<QUESTION>).

        Use the following checklist:

        {generate_checklist_text()}

        Output true/false for each check and provide a short explanation for your judgment.
        """

    judge = Agent(
        name="judge",
        instructions=eval_instructions,
        model="gpt-4o-mini",
        output_type=EvaluationChecklist,
    )

    return judge
