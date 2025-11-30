from utils import map_progress, process_result
from service.agent.src.agent.youtube_agent import NamedCallback, create_youtube_agent
import pandas as pd
import asyncio
import traceback
import pickle


questions = [
    "What are the must-buy foundations at Sephora?",
    "必買的 Sephora 粉底液",
    # "What are the must-buy concealers at Sephora?",
    # "必買的 Sephora 遮暇",
    "What are the don't-buy foundations at Sephora?",
    "不該買的 sephora 粉底液",
    # "What foundations would you recommend from Korea?",
    # "必買得韓國粉底液"
    # "What are the don't-buy blushes at Sephora?",
]


async def run_agent(q):
    try:
        # print(q['user_question'])
        youtube_agent = create_youtube_agent()
        result = await youtube_agent.run(q["user_question"])
        return (q, result)
    except:
        print(f"error processing {q}")
        traceback.print_exc()
        return (None, None)


async def main():
    gd = pd.read_csv(
        "/Users/yenchunchen/Desktop/Project/health-agent/data/ground_truth/ground-truth-mindy.csv"
    )
    ground_truth = gd.to_dict(orient="records")

    all_results = await map_progress(ground_truth, run_agent, max_concurrency=6)

    rows = []
    for q, r in all_results:
        if r is None:
            continue

        usage = r.usage()
        row = process_result(q, r)
        rows.append(row)

    with open("data/ground_truth/eval-run-v2-2025-11-30-12-00.bin", "wb") as f_out:
        pickle.dump(rows, f_out)


if __name__ == "__main__":
    asyncio.run(main())
