import json
import pickle
import pandas as pd
from service.evals.src.evals.create_eval_agent import create_eval_agent
from service.evals.src.evals.utils import map_progress, process_result
from service.agent.src.agent import youtube_agent
import asyncio


async def run_eval(row):
    original_product = row["original_question"]["product"]
    original_recommend = row["original_question"]["recommend"]
    original_reasone = row["original_question"]["reason"]

    user_prompt = f"""
    <INSTRUCTIONS>{youtube_agent.beauty_instructions}</INSTRUCTIONS>
    <QUESTION>{row["question"]}</QUESTION>
    <FINAL ANSWER>{row["answer"]}</FINAL ANSWER>
    <DETAILS ANSWER>{row["product_answer"]}</DETAILS ANSWER>
    <REFERENCE>{original_product}</REFERENCE>
    <LOG>{json.dumps(row["messages"])}</LOG>
    """.strip()

    judge = create_eval_agent()
    output = await judge.run(user_prompt=user_prompt)

    return row, output


async def main():
    with open("data/ground_truth/eval-run-v2-2025-11-30-11-15.bin", "rb") as f_in:
        rows = pickle.load(f_in)

    results = await map_progress(rows, run_eval, max_concurrency=10)

    all_checks = []

    for original_row, result in results:
        checks = result.output.checklist
        checks_formatted = {"question": original_row["question"]}
        for check in checks:
            checks_formatted[check.check_name] = check.check_pass
            # checks_formatted[check.reasoning] = check.reasoning
        all_checks.append(checks_formatted)

    df_eval = pd.DataFrame(all_checks)
    print(df_eval[df_eval.columns[1:]].mean())


if __name__ == "__main__":
    asyncio.run(main())
